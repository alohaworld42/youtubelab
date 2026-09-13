"""
kidsong.review — Automated per-shot quality gate for rendered i2v clips.

Distilled image-to-video models fail in a handful of predictable ways: a
black/near-white dud frame, a blurry mush, a "static image" (the model just
re-emits the input frame with no motion), strobing/glitchy frame-to-frame
noise, or a crowd-collapse where the model hallucinates duplicate copies of
the cast. `HeuristicReviewer` samples a handful of frames per shot (cv2) and
rejects on any of those signatures, with retry hints telling the caller how
to re-roll the shot (bump the seed, simplify the motion prompt, or fall back
to a slower/steadier backend).

`ExternalReviewer` defers judgment to a human/Claude session: it writes a
contact sheet + request JSON and polls for a matching response JSON, so an
orchestrator can pause a batch for manual review without blocking the
process.

cv2/numpy are imported at module load (both are lightweight, pure-CPU deps
already required elsewhere in this repo — unlike torch/diffusers, there's no
reason to defer them).
"""
import json
import logging
import os

import cv2
import numpy as np

# OpenCV opportunistically offloads to the GPU via OpenCL. During a render the
# LTX model already fills ~11 of 12 GB, so a buffer upload for the tiny Haar
# face-detection here failed with CL_MEM_OBJECT_ALLOCATION_FAILURE and took a
# whole episode down (job 13, "Washing Little Hands"). Face detection is
# trivial and has no business competing with the video model for VRAM — pin it
# to the CPU. This is a global cv2 setting; kept in both cv2-using modules so
# neither can re-enable it by import order.
cv2.ocl.setUseOpenCL(False)

log = logging.getLogger("kidsong.review")

_DEFAULT_FRAMES_PER_SHOT = 5

# Score penalty per reject reason; heavier for artifacts that make a clip
# unusable outright (black frame, strobing) vs. milder ones.
_REJECT_WEIGHTS = {
    "black/blank": 0.4,
    "blurry": 0.3,
    "static": 0.3,
    "strobe/glitch": 0.35,
    "duplicate characters": 0.25,
}

_EMPTY_HINTS = {"seed_bump": False, "simplify_action": False, "force_i2v": False}

# Per-shot review PREFILTER. The external (human) review queue was flooding —
# every shot that cleared the hard heuristic wrote a request and waited for a
# person, so a batch of episodes piled up hundreds of per-shot requests that no
# one could realistically look at. But the heuristic already REJECTS the failure
# modes worth a retry (black/blank, static, strobe, crowd-collapse), and the one
# remaining soft flag ("blurry") is documented below as untrustworthy for this
# soft-shaded toon style. So a shot the heuristic ACCEPTS, whose only soft flags
# are ones we've chosen to ignore, does not need human eyes at the shot level —
# the assembled-episode CUT review and the pre-upload human review are the real
# nets. `review.shot_auto_accept` controls this:
#   enabled (default True)      — auto-accept qualifying shots without queueing.
#   surface_reasons (default []) — soft reasons that STILL surface a shot to the
#       human even when the heuristic accepts. Empty = auto-accept every
#       heuristic-cleared shot (the aggressive default); add e.g. "blurry" to
#       route those to the queue instead. A hard-rejected shot is never affected
#       (it never reached this path). A future soft reason not listed here is
#       auto-accepted too — list it to surface novel concerns.
_DEFAULT_SHOT_AUTO_ACCEPT_ENABLED = True
_DEFAULT_SHOT_SURFACE_REASONS = ()


# The auto-accept prefilter never applies to shots staging MORE children than
# this. Rationale (measured live, pixar-full episode): every multi-child group
# shot sailed through auto-accept at score 1.0 with zero inspection — and the
# heuristic gate cannot see clone-collapse, invented non-cast children or
# identity at all, so the single shot class where those defects concentrate
# was exactly the one never looked at. 0/None = no cap (documented opt-out).
_DEFAULT_SHOT_AUTO_ACCEPT_MAX_CHILDREN = 1


def _shot_auto_accept_policy(cfg):
    """Return (enabled, surface_reasons_set, max_children) for the per-shot
    review prefilter. `max_children` caps how many expected on-screen children
    a shot may stage and still be eligible for blind auto-accept; 0 or None
    disables the cap entirely (pre-cap behavior)."""
    review_cfg = (cfg.get("kidsong", {}) or {}).get("review", {}) or {}
    pol = review_cfg.get("shot_auto_accept")
    if not isinstance(pol, dict):
        return (_DEFAULT_SHOT_AUTO_ACCEPT_ENABLED,
                set(_DEFAULT_SHOT_SURFACE_REASONS),
                _DEFAULT_SHOT_AUTO_ACCEPT_MAX_CHILDREN)
    enabled = bool(pol.get("enabled", _DEFAULT_SHOT_AUTO_ACCEPT_ENABLED))
    raw = pol.get("surface_reasons", _DEFAULT_SHOT_SURFACE_REASONS)
    if isinstance(raw, str):
        raw = [raw]
    surface = {str(r).strip() for r in (raw or ()) if str(r).strip()}
    try:
        max_children = pol.get("max_children", _DEFAULT_SHOT_AUTO_ACCEPT_MAX_CHILDREN)
        max_children = int(max_children) if max_children is not None else 0
    except (TypeError, ValueError):
        max_children = _DEFAULT_SHOT_AUTO_ACCEPT_MAX_CHILDREN
    return enabled, surface, max_children


# ------------------------------------------------------------------ frames ---
def _even_indices(total, n):
    if total <= 0:
        return []
    n = max(1, min(n, total))
    if n == 1:
        return [total // 2]
    return sorted({int(round(i * (total - 1) / (n - 1))) for i in range(n)})


def _extract_frames(video_path, n):
    """Grab `n` evenly-spaced frames (BGR numpy arrays) from `video_path`."""
    cap = cv2.VideoCapture(video_path)
    try:
        if not cap.isOpened():
            return []
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        if total > 0:
            frames = []
            for idx in _even_indices(total, n):
                cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                ok, frame = cap.read()
                if ok and frame is not None:
                    frames.append(frame)
            if frames:
                return frames
        # Fallback for containers that don't report a reliable frame count:
        # read sequentially and pick evenly-spaced frames from what we got.
        all_frames = []
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            all_frames.append(frame)
        if not all_frames:
            return []
        idxs = _even_indices(len(all_frames), n)
        return [all_frames[i] for i in idxs]
    finally:
        cap.release()


# ------------------------------------------------------------------ scoring ---
def _expected_face_count(shot):
    """How many faces a shot should show. Reads `cast.expected_child_count`
    (the single source of truth also used by `_resolve_cast_enrichment` for
    the external-review payload) so the heuristic reviewer and the vision
    reviewer can never disagree about the headcount; falls back to the old
    hardcoded-3-for-['all'] heuristic only if the cast bible is unavailable.

    The old hardcoded `3` meant a future 4th cast member would silently break
    this check without anyone touching review.py at all.
    """
    characters = shot.get("characters") or ["all"]
    if isinstance(characters, str):
        characters = [characters]
    try:
        from pipeline.kidsong import cast

        return cast.expected_child_count(characters)
    except Exception:
        log.debug(
            "kidsong review: cast.expected_child_count unavailable, using "
            "heuristic fallback",
            exc_info=True,
        )
    if list(characters) == ["all"]:
        return 3
    return max(1, len(characters))


# --------------------------------------------------------------- cast enrichment ---
def _resolve_cast_enrichment(shot, cast_text=None, song=None):
    """Best-effort cast ground-truth for a shot's external-review request.

    The vision reviewer (scene-plausibility.md, rubric dimensions 3/4/7 --
    CAST INTEGRITY / WARDROBE / TEMPORAL COHERENCE) needs to know what each
    named character actually looks like, not just their name. `shot` only
    ever carries names (e.g. `characters: ["Zuri"]`), so this resolves the
    matching appearance text against `pipeline.kidsong.cast` -- a sibling
    module (skin tone/hair/outfit per child) that may not exist on disk yet
    and, even once it lands, must never be allowed to break a review.

    Resolution order for the descriptive text: an explicit `cast_text`
    override, then `cast.cast_sentence(shot's characters)`, then (if the
    cast module is unavailable/broken) any `song["characters"]` sentence
    already in hand, then nothing -- in which case the key is simply
    omitted rather than shipping a placeholder.

    Returns a dict meant to be merged into the request payload:
    `expected_children` is always present (falls back to
    `_expected_face_count`); `cast_text` / `cast_version` are only present
    when something actually resolved them.
    """
    names = shot.get("characters")
    resolved_text = cast_text
    cast_version = None
    expected_children = None

    try:
        from pipeline.kidsong import cast
    except Exception:
        # ImportError (module doesn't exist yet) is the expected case while
        # a sibling agent is still landing cast.py, but *any* failure here
        # (broken module, bad data file, etc.) must degrade the same way --
        # a cast lookup problem is never a reason to fail a shot review.
        log.debug("kidsong review: cast module unavailable, degrading gracefully", exc_info=True)
        cast = None

    if cast is not None:
        if resolved_text is None:
            try:
                resolved_text = cast.cast_sentence(names)
            except Exception:
                log.debug("kidsong review: cast.cast_sentence(%r) failed", names, exc_info=True)
        try:
            cast_version = cast.version()
        except Exception:
            log.debug("kidsong review: cast.version() failed", exc_info=True)
        try:
            expected_children = cast.expected_child_count(names)
        except Exception:
            log.debug("kidsong review: cast.expected_child_count(%r) failed", names, exc_info=True)

    if resolved_text is None and song:
        fallback = song.get("characters") if isinstance(song, dict) else None
        if fallback:
            resolved_text = str(fallback)

    if expected_children is None:
        # cast.py unavailable, or it errored on the count specifically --
        # fall back to the existing heuristic-face-count logic so this key
        # is always present even without the cast module.
        expected_children = _expected_face_count(shot)

    enrichment = {"expected_children": expected_children}
    if resolved_text:
        enrichment["cast_text"] = resolved_text
    if cast_version:
        enrichment["cast_version"] = cast_version
    return enrichment


def _analyze(frames):
    """Return (lumas, lap_vars, diffs) for a list of BGR frames."""
    grays = [cv2.cvtColor(f, cv2.COLOR_BGR2GRAY) for f in frames]
    lumas = [float(g.mean()) for g in grays]
    lap_vars = [float(cv2.Laplacian(g, cv2.CV_64F).var()) for g in grays]
    diffs = [
        float(np.mean(np.abs(a.astype(np.int16) - b.astype(np.int16))))
        for a, b in zip(grays, grays[1:])
    ]
    return grays, lumas, lap_vars, diffs


class HeuristicReviewer:
    """cv2/numpy-based automatic reject filter — the default reviewer."""

    def __init__(self, cfg=None):
        cfg = cfg or {}
        review_cfg = (cfg.get("kidsong", {}) or {}).get("review", {}) or {}
        self.frames_per_shot = int(review_cfg.get("frames_per_shot", _DEFAULT_FRAMES_PER_SHOT))
        self._face_cascade = self._load_face_cascade()

    @staticmethod
    def _load_face_cascade():
        """Best-effort Haar cascade loader.

        Some opencv-python-headless builds (e.g. the 5.x wheel this repo's
        venv has) drop the bundled `objdetect`/`CascadeClassifier` Python
        bindings and cascade XML data entirely (`cv2.data.haarcascades` is an
        empty directory, `cv2.CascadeClassifier` doesn't exist). Rather than
        hard-failing every review, fall back to skipping the duplicate-face
        check in that case — it already only catches gross crowd-collapse
        failures and is deliberately lenient.
        """
        classifier = getattr(cv2, "CascadeClassifier", None)
        if classifier is None:
            return None
        cascade_path = os.path.join(cv2.data.haarcascades, "haarcascade_frontalface_default.xml")
        if not os.path.exists(cascade_path):
            return None
        try:
            cascade = classifier(cascade_path)
            return cascade if not cascade.empty() else None
        except Exception:
            return None

    def review(self, shot, video_path):
        frames = _extract_frames(video_path, self.frames_per_shot)
        if not frames:
            return {
                "accept": False,
                "score": 0.0,
                "reasons": ["unreadable video"],
                "retry_hints": {**_EMPTY_HINTS, "seed_bump": True},
            }

        grays, lumas, lap_vars, diffs = _analyze(frames)
        reasons = []       # every flag, hard or soft — kept for telemetry/log visibility
        hard_reasons = []  # only these force accept=False
        hints = dict(_EMPTY_HINTS)

        if any(l < 12 or l > 243 for l in lumas):
            reasons.append("black/blank")
            hard_reasons.append("black/blank")
            hints["seed_bump"] = True

        # Laplacian variance measures high-frequency edge density, which this
        # soft-shaded 3D toon style (gentle rim lighting, smooth skin shading —
        # see generate.py's shot prompts) naturally suppresses even in a
        # perfectly crisp render. Confirmed by direct visual inspection: takes
        # with median variance as low as ~12 have been sharp, correctly
        # rendered frames, not blur. This check is no longer trustworthy
        # enough to hard-reject on its own — flag as advisory only (visible in
        # reasons/score, doesn't block) and let the vision reviewer judge
        # actual focus. Revisit if a style-appropriate metric is found.
        if float(np.median(lap_vars)) < 20:
            reasons.append("blurry")
            hints["seed_bump"] = True

        mean_diff = float(np.mean(diffs)) if diffs else 0.0
        max_diff = float(np.max(diffs)) if diffs else 0.0

        if mean_diff < 1.0:
            reasons.append("static")
            hard_reasons.append("static")
            hints["simplify_action"] = True

        if max_diff > 60:
            reasons.append("strobe/glitch")
            hard_reasons.append("strobe/glitch")
            hints["seed_bump"] = True

        # Cartoon faces detect unreliably with a photoreal Haar cascade —
        # this check is deliberately lenient, aimed only at gross
        # crowd-collapse failures (the model hallucinating extra kids). If
        # this environment's opencv build has no cascade available at all
        # (see _load_face_cascade), skip the check rather than guessing.
        if self._face_cascade is not None:
            expected = _expected_face_count(shot)
            face_counts = [
                len(self._face_cascade.detectMultiScale(g, scaleFactor=1.1, minNeighbors=5))
                for g in grays
            ]
            median_faces = float(np.median(face_counts)) if face_counts else 0.0
            if median_faces > expected + 1:
                reasons.append("duplicate characters")
                hard_reasons.append("duplicate characters")
                hints["force_i2v"] = True

        penalty = sum(_REJECT_WEIGHTS.get(r, 0.2) for r in reasons)
        score = max(0.0, min(1.0, 1.0 - penalty))

        # QM-033 telemetry (additive optional key — the accept/score/reasons/
        # retry_hints contract above is untouched, and NO threshold moves here).
        # Why: the 2026-07-25 audit measured strobe/glitch on 5/35 (14.3%) of
        # i2v-anchored takes vs 0/80 (0.0%) of non-anchored takes in the same
        # episodes. Two hypotheses fit that, and the reasons list alone cannot
        # tell them apart:
        #   (a) REAL flicker introduced by the i2v/keyframe path, which would
        #       show up as a large diff somewhere in the MIDDLE of the clip;
        #   (b) a heuristic FALSE POSITIVE — i2v starts from a still keyframe,
        #       so the still->motion transition concentrates change in the
        #       FIRST sampled frame pair (`max_diff_at == 0`), which trips the
        #       `max_diff > 60` check without anything being wrong. This is the
        #       same shape as QM-007's blurry story (style-normal signal read as
        #       a defect), and loosening a gate requires exactly that evidence.
        # `max_diff_at` is the index into `diffs`, i.e. the sampled frame PAIR
        # (i, i+1) out of `frames_per_shot` evenly-spaced frames — 0 means the
        # change is concentrated at the head of the clip.
        metrics = {
            "frames": len(frames),
            "mean_frame_diff": mean_diff,
            "max_frame_diff": max_diff,
            "max_diff_at": int(np.argmax(diffs)) if diffs else None,
            "frame_diffs": [round(d, 3) for d in diffs],
            "median_lap_var": float(np.median(lap_vars)) if lap_vars else None,
            "luma_min": float(np.min(lumas)) if lumas else None,
            "luma_max": float(np.max(lumas)) if lumas else None,
        }

        return {
            "accept": not hard_reasons,
            "score": score,
            "reasons": reasons,
            "retry_hints": hints,
            "metrics": metrics,
        }


# --------------------------------------------------------------- contact sheet ---
def contact_sheet(shot, video_path, out_dir, name=None):
    """Write a 5-frame PIL contact-sheet grid PNG for `shot`; return its path.

    `name` overrides the output basename (e.g. a take id like "s03_a1") so
    every retry gets its own sheet instead of overwriting the shot's.
    """
    from PIL import Image

    os.makedirs(out_dir, exist_ok=True)
    shot_id = name or shot.get("id") or shot.get("shot_id") or os.path.splitext(os.path.basename(video_path))[0]
    out_path = os.path.join(out_dir, f"{shot_id}.png")

    frames = _extract_frames(video_path, 5)
    if not frames:
        Image.new("RGB", (160, 284), (32, 32, 32)).save(out_path)
        return out_path

    thumbs = []
    for f in frames:
        img = Image.fromarray(cv2.cvtColor(f, cv2.COLOR_BGR2RGB))
        img.thumbnail((320, 569))
        thumbs.append(img)

    pad = 6
    tile_w = max(t.width for t in thumbs)
    tile_h = max(t.height for t in thumbs)
    sheet = Image.new(
        "RGB", (tile_w * len(thumbs) + pad * (len(thumbs) + 1), tile_h + pad * 2), (20, 20, 20)
    )
    x = pad
    for t in thumbs:
        sheet.paste(t, (x, pad + (tile_h - t.height) // 2))
        x += t.width + pad
    sheet.save(out_path)
    return out_path


# --------------------------------------------------------------- external ---
def poll_response(request_path, response_path, payload, timeout, poll_seconds=5.0):
    """Write `payload` as JSON to `request_path`, then poll for `response_path`
    up to `timeout` seconds. Returns the parsed response dict, or `None` if it
    never appeared (or was unreadable/invalid JSON) — the caller should fall
    back to its own default verdict in that case.

    Shared polling primitive for every external (human/Claude-in-the-loop)
    review gate in this package — `ExternalReviewer.review` predates this
    helper and keeps its own inline copy of the same logic to avoid changing
    its behavior; `script_qc`/`cut_qc` build on this one instead.
    """
    import time

    # Fixed-name gates (script/cut) reuse the same paths every run: a stale
    # response from a previous video would instantly (and wrongly) decide this
    # one. A new request always invalidates any old response.
    try:
        os.remove(response_path)
    except OSError:
        pass

    try:
        with open(request_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)
    except OSError:
        pass

    deadline = time.time() + timeout
    while time.time() < deadline:
        if os.path.exists(response_path):
            try:
                with open(response_path, "r", encoding="utf-8") as f:
                    return json.load(f)
            except (OSError, json.JSONDecodeError):
                return None
        time.sleep(poll_seconds)
    return None


# ----------------------------------------------------------- run sessions ---
# A kidsong run passes through FOUR external-review gates, in this order:
#
#     script (script_qc.review_script)
#       -> shotlist (script_qc.review_shotlist)
#         -> shot   (ExternalReviewer.review, once per rendered take)
#           -> cut  (cut_qc.review_cut)
#
# "Is anybody actually answering these?" is a property of the RUN, not of any
# one gate: if nobody answered the script gate, nobody is going to answer the
# shot gate either. Before this existed, only the shot loop learned that, and
# the other three gates each paid `external_timeout` seconds of dead wall clock
# on every single episode (measured: 4 gate-waits x 120s = 8 minutes per run,
# GPU idle throughout).
#
# `ReviewSession` is that run-scoped "is anyone home?" state, shared by all
# four gates. It ONLY ever decides HOW LONG WE WAIT for a person. Every
# programmatic gate still runs in full and its verdict is returned unchanged —
# a bad script is still rejected, a bad cut is still held. Requests and contact
# sheets are still written even when we skip the wait, so a human coming back
# later can still see and answer them.
#
# --- how the four gates find the same session, and why it is done this way ---
# The natural mechanism (a run-scoped object threaded through the call chain)
# is not available: `script_qc.review_script(song, cfg)` and
# `cut_qc.review_cut(..., cfg, review_dir)` are module-level functions whose
# signatures are fixed by `pipeline.kidsong.generate`, and there is no run id
# in `cfg`. Keying a registry on the review directory does not work either —
# the script/shotlist gates share ONE process-stable directory
# (`<output>/_kidsong_review`) while the shot/cut gates use a per-run one, so
# no single directory identifies a run across all four. Keying on the cfg
# object is worse: `studio.scheduler` hands the *same* cfg dict to every job it
# runs, so all runs in the process would share one entry.
#
# So the registry is keyed on the output directory (one production line) and a
# session is ROTATED — replaced with a fresh one — as soon as a gate arrives
# whose stage rank is LOWER than the highest rank this session has seen. Gates
# only ever run in ascending rank within a run, so a lower rank can only mean a
# new run has started:
#
#   run 1: script(0) shotlist(1) shot(2) cut(3)   -> one session
#   run 2: script(0)  ... 0 < 3, rotate           -> a fresh session
#   resumed run (script/shotlist skipped): shot(2) ... 2 < 3, rotate -> fresh
#
# Retries within a stage repeat the same rank (`0 < 0` is false), so a script
# regenerate or a cut recut correctly stays inside its own run's session.
#
# Known limit, stated plainly: if a run aborts at its FIRST gate and the next
# run starts at that same gate (script(0) after a session that only ever saw
# script(0)), no rotation fires and the second run inherits the first's
# counter. The consequence is bounded — it can only make the second run stop
# waiting sooner, never change a verdict — and the very next answered request
# resets it. Every other run boundary rotates correctly.
_STAGE_RANKS = {"script": 0, "shotlist": 1, "shot": 2, "cut": 3}

_SESSIONS = {}


class ReviewSession:
    """Run-scoped "is anyone answering external review requests?" state.

    Shared by all four gates of one run. See the module comment above for the
    sharing mechanism. Nothing here touches a verdict — only wait durations.
    """

    def __init__(self, unattended_after=2, first_probe=0.0):
        # Clamped to >= 1 so a misconfigured 0 cannot engage unattended mode
        # before a single request has even been offered for review.
        self.unattended_after = max(1, int(unattended_after or 0))
        self.first_probe = max(0.0, float(first_probe or 0.0))
        self.consecutive_timeouts = 0
        self.unattended = False
        # response paths of requests we wrote but did not wait for, not yet
        # seen answered — cleared in full the moment one of them turns up,
        # which is what flips us back to waiting.
        self.pending_responses = []
        # True once anybody has answered anything in this run. Gates the
        # optional short first probe (see `wait_seconds`).
        self.human_seen = False
        self.max_stage_rank = -1

    # ------------------------------------------------------------ bookkeeping --
    def note_stage(self, stage):
        self.max_stage_rank = max(self.max_stage_rank, _STAGE_RANKS.get(stage, 99))

    def wait_seconds(self, timeout):
        """How long THIS gate should block waiting for a human."""
        if self.unattended:
            return 0.0
        if self.first_probe and not self.human_seen:
            # Nobody has answered anything yet in this run, so the full
            # timeout is a bet on a human we have no evidence for. Probe
            # briefly first; the moment anyone answers ANY request, this
            # branch is dead for the rest of the run and every later gate
            # gets the full timeout.
            return min(self.first_probe, float(timeout))
        return float(timeout)

    def resume_if_answered(self):
        """A human clearing part of the skipped backlog is a live person
        again — resume full waits from this gate onward. Returns True if the
        session flipped back to attended."""
        if not self.unattended or not self.pending_responses:
            return False
        if not any(os.path.exists(p) for p in self.pending_responses):
            return False
        log.info(
            "kidsong review: response found for a backlogged request — "
            "unattended mode OFF, resuming full external-review waits for the "
            "rest of this run"
        )
        self.unattended = False
        self.consecutive_timeouts = 0
        self.pending_responses = []
        self.human_seen = True
        return True

    def note_timeout(self, stage):
        """A gate waited and nobody answered."""
        self.consecutive_timeouts += 1
        if not self.unattended and self.consecutive_timeouts >= self.unattended_after:
            self.unattended = True
            log.info(
                "kidsong review: %s consecutive unanswered external review "
                "requests (through the '%s' gate) — engaging unattended mode "
                "for the rest of this run. Remaining gates (script, shotlist, "
                "shots, cut) write their request and use their own "
                "programmatic verdict instead of waiting. No gate is "
                "weakened; only the wait is skipped.",
                self.consecutive_timeouts,
                stage,
            )

    def note_skipped(self, response_path):
        """A gate wrote its request and did NOT wait (unattended)."""
        self.consecutive_timeouts += 1
        if response_path and response_path not in self.pending_responses:
            self.pending_responses.append(response_path)

    def note_response(self, stage="?"):
        """Somebody answered — the run is attended."""
        if self.unattended:
            log.info(
                "kidsong review: '%s' gate was answered — unattended mode OFF, "
                "resuming full external-review waits for the rest of this run",
                stage,
            )
        self.consecutive_timeouts = 0
        self.unattended = False
        self.pending_responses = []
        self.human_seen = True


def _session_key(cfg):
    """One entry per production line (output directory)."""
    try:
        from pipeline.config import abspath

        return os.path.abspath(abspath(cfg, cfg["paths"]["output_dir"]))
    except Exception:
        # cfg without paths (unit tests, ad-hoc callers) — a single shared
        # slot is fine: it is still rotated per run by the stage-rank rule,
        # and `reset_sessions()` clears it between tests.
        return "__default__"


def session_for(cfg, stage):
    """The `ReviewSession` for the run that `stage` belongs to.

    Rotates in a fresh session when `stage` ranks below the highest stage the
    current session has seen — see the module comment for why that identifies
    a run boundary.
    """
    cfg = cfg or {}
    key = _session_key(cfg)
    rank = _STAGE_RANKS.get(stage, 99)
    session = _SESSIONS.get(key)
    if session is None or rank < session.max_stage_rank:
        if session is not None:
            log.info(
                "kidsong review: new run detected at the '%s' gate — "
                "unattended state reset (was unattended=%s, %s consecutive "
                "unanswered)",
                stage,
                session.unattended,
                session.consecutive_timeouts,
            )
        review_cfg = (cfg.get("kidsong", {}) or {}).get("review", {}) or {}
        session = ReviewSession(
            unattended_after=review_cfg.get("unattended_after_timeouts", 2),
            first_probe=review_cfg.get("external_first_probe", 0),
        )
        _SESSIONS[key] = session
    session.note_stage(stage)
    return session


def reset_sessions():
    """Drop all run sessions. For tests; never needed in production."""
    _SESSIONS.clear()


def external_gate_poll(stage, request_path, response_path, payload, timeout, cfg,
                       poll_seconds=5.0):
    """`poll_response`, but honoring the run's unattended state.

    Returns `(response_or_None, skipped)`. `skipped` is True when unattended
    mode was already engaged and we wrote the request without waiting — the
    caller must still return its own programmatic verdict in that case,
    exactly as it does on a timeout.
    """
    session = session_for(cfg, stage)
    session.resume_if_answered()

    if session.unattended:
        # Write the request (and clear any stale response) so a human coming
        # back later can still see and answer it — just don't idle here.
        poll_response(request_path, response_path, payload, 0.0, poll_seconds)
        session.note_skipped(response_path)
        return None, True

    wait = session.wait_seconds(timeout)
    # Never sleep past the deadline: `poll_response` sleeps a fixed
    # `poll_seconds` between checks, so a short wait (a first probe, or a
    # test's fractional timeout) would otherwise still cost a full 5s tick.
    poll_seconds = max(0.01, min(poll_seconds, wait / 2.0)) if wait > 0 else poll_seconds
    response = poll_response(request_path, response_path, payload, wait, poll_seconds)
    if response is None:
        session.note_timeout(stage)
    else:
        session.note_response(stage)
    return response, False


class ExternalReviewer:
    """Blocking hybrid gate: heuristics first, then a vision reviewer's verdict.

    The technical heuristics run first and can reject outright (no point
    asking a human/Claude about a glitched clip). If they pass, `review`
    writes a contact sheet plus `<out_dir>/<shot_id>.request.json` and then
    POLLS for `<shot_id>.response.json` — the producer (a Claude session with
    vision, or a human) looks at the sheet and writes the verdict file. This
    is what "don't ship until it's actually been looked at" means.

    If no verdict arrives within `review.external_timeout` seconds (default
    600), falls back to the heuristic verdict with a warning reason, so an
    unattended batch degrades to automatic QC instead of hanging forever.

    Adaptive unattended detection: waiting the full timeout on *every* shot
    is fine when someone's actually reviewing, but on a machine nobody is
    watching it burns ~timeout seconds of idle GPU per shot for nothing (16
    shots x 120s default timeout = ~32 minutes wasted on one video). After
    `review.unattended_after_timeouts` (default 2) consecutive unanswered
    requests, the run stops waiting — remaining shots go straight to the
    heuristic verdict (the request/contact sheet are still written, so a
    returning human can review them later). If a response later shows up for
    one of those skipped requests, that's a live human again: waiting
    resumes for the next shot onward. This only ever changes how long we
    wait for a person — the heuristic criteria and the accept/reject logic
    are untouched, so no QC gate is weakened.

    That counter is NOT private to this class: it lives in the run's shared
    `ReviewSession` (see the "run sessions" section above), so the script,
    shotlist and cut gates and this shot loop all learn "nobody is home" from
    each other instead of each rediscovering it at the cost of a full timeout.
    """

    def __init__(self, cfg=None):
        self.cfg = cfg or {}
        review_cfg = (self.cfg.get("kidsong", {}) or {}).get("review", {})
        self.timeout = float(review_cfg.get("external_timeout", 600))
        self.poll_seconds = 5.0
        self._heuristic = HeuristicReviewer(cfg)
        self._auto_accept = _shot_auto_accept_policy(self.cfg)

        self.unattended_after = max(1, int(review_cfg.get("unattended_after_timeouts", 2)))
        # Bound at construction (which happens after the script/shotlist gates
        # and before the cut gate) so this reviewer stays attached to its own
        # run's session for the whole shot loop.
        self.session = session_for(self.cfg, "shot")

    # Back-compat views onto the shared session — these were instance
    # attributes before the state became run-scoped.
    @property
    def _unattended(self):
        return self.session.unattended

    @property
    def _consecutive_timeouts(self):
        return self.session.consecutive_timeouts

    @property
    def _pending_unattended_responses(self):
        return self.session.pending_responses

    def review(self, shot, video_path, out_dir=None, cast_text=None, song=None):
        """`cast_text` and `song` are optional, additive keyword args (existing
        positional callers -- e.g. generate.py's `reviewer.review(shot, path,
        out_dir=review_dir)` -- are unaffected). `cast_text` overrides the
        auto-resolved cast ground-truth outright; `song` (a dict with a
        `characters` sentence) is only consulted as a fallback if the
        `pipeline.kidsong.cast` module is unavailable. See
        `_resolve_cast_enrichment` for the full resolution order.
        """
        import time

        shot_id = shot.get("id") or shot.get("shot_id") or os.path.splitext(os.path.basename(video_path))[0]
        out_dir = out_dir or os.path.dirname(os.path.abspath(video_path))
        os.makedirs(out_dir, exist_ok=True)

        # Stage 1: technical heuristics — a glitched clip never reaches vision.
        heuristic = self._heuristic.review(shot, video_path)
        if not heuristic.get("accept"):
            return heuristic

        # Stage 1b: PREFILTER. If the heuristic cleared this shot and none of its
        # soft flags are ones we've chosen to surface, auto-accept it here — no
        # request file, no queue entry, no wait. This keeps the human review
        # queue down to items that actually need eyes (the cut gate, novel soft
        # concerns) instead of one request per shot per episode. See
        # `_shot_auto_accept_policy`. The verdict still flows through the normal
        # return path (runlog/telemetry unchanged), tagged so it's auditable.
        enabled, surface_reasons, max_children = self._auto_accept
        if enabled and max_children:
            # Multi-child shots (clone-collapse / invented-children territory
            # the heuristic cannot see) are never blind-auto-accepted: they
            # proceed to the request/response vision path below. Fail-open by
            # design — if the count can't be resolved, keep today's behavior.
            try:
                from pipeline.kidsong import cast as _cast

                expected = _cast.expected_child_count(
                    shot.get("characters"), shot.get("shot_type")
                )
                if expected is not None and int(expected) > max_children:
                    enabled = False
            except Exception:
                pass
        if enabled and not (surface_reasons & set(heuristic.get("reasons") or [])):
            verdict = dict(heuristic)
            verdict["auto_accepted"] = True
            verdict.setdefault("reasons", [])
            verdict["reasons"] = list(verdict["reasons"]) + [
                "auto-accepted at shot level (heuristic clear, no surfaced concerns) "
                "— reviewed at the cut gate"
            ]
            return verdict

        # A human clearing the backlog of requests we skipped while
        # unattended is a live person again — resume waiting from this shot
        # onward, before deciding this shot's own wait policy.
        session = self.session
        session.resume_if_answered()

        # Stage 2: vision gate. Take-specific request/response names so retries
        # of the same shot get fresh verdicts.
        take = os.path.splitext(os.path.basename(video_path))[0]
        sheet_path = contact_sheet(shot, video_path, out_dir, name=take)
        request_path = os.path.join(out_dir, f"{take}.request.json")
        response_path = os.path.join(out_dir, f"{take}.response.json")

        enrichment = _resolve_cast_enrichment(shot, cast_text=cast_text, song=song)
        request = {
            "shot_id": shot_id,
            "take": take,
            "shot": shot,
            "video_path": os.path.abspath(video_path),
            "contact_sheet": sheet_path,
            "heuristic": heuristic,
            **enrichment,
        }
        try:
            with open(request_path, "w", encoding="utf-8") as f:
                json.dump(request, f, indent=2)
        except OSError:
            pass

        if session.unattended:
            # Nobody has answered `unattended_after` requests in a row — stop
            # idling the GPU waiting on every remaining shot of this run.
            # The request above still stands for a later manual look.
            session.note_skipped(response_path)
            heuristic = dict(heuristic)
            heuristic.setdefault("reasons", []).append(
                "unattended mode: external review skipped — heuristic verdict used"
            )
            return heuristic

        deadline = time.time() + session.wait_seconds(self.timeout)
        while time.time() < deadline:
            if os.path.exists(response_path):
                try:
                    with open(response_path, "r", encoding="utf-8") as f:
                        verdict = json.load(f)
                    verdict.setdefault("reasons", [])
                    verdict.setdefault("retry_hints", dict(_EMPTY_HINTS))
                    verdict.setdefault("score", 0.5)
                    verdict["accept"] = bool(verdict.get("accept"))
                    session.note_response("shot")
                    return verdict
                except (OSError, json.JSONDecodeError):
                    break  # unreadable response — fall through to heuristic
            time.sleep(self.poll_seconds)

        session.note_timeout("shot")
        heuristic = dict(heuristic)
        heuristic.setdefault("reasons", []).append(
            "external review timed out — heuristic verdict used"
        )
        return heuristic


# ------------------------------------------------------------------ registry ---
_REGISTRY = {
    "heuristic": HeuristicReviewer,
    "external": ExternalReviewer,
}


def get_reviewer(cfg):
    review_cfg = (cfg.get("kidsong", {}) or {}).get("review", {}) or {}
    # Accept both key names — config uses "reviewer"; "backend" kept for compat.
    backend = str(review_cfg.get("reviewer", review_cfg.get("backend", "heuristic"))).lower()
    cls = _REGISTRY.get(backend, HeuristicReviewer)
    return cls(cfg)


# --------------------------------------------------------------------- log ---
def review_shots_log(path, entries):
    """Append-write a JSON list of review entries to `path` (creates it if needed)."""
    existing = []
    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                existing = json.load(f)
            if not isinstance(existing, list):
                existing = []
        except (OSError, json.JSONDecodeError):
            existing = []
    existing.extend(entries if isinstance(entries, (list, tuple)) else [entries])
    with open(path, "w", encoding="utf-8") as f:
        json.dump(existing, f, indent=2)
    return path


# --------------------------------------------------------------------- gate ---
def vision_coverage(review_dir, accepted_takes):
    """Fraction of `accepted_takes` (take ids, e.g. "s03_a1" -- one per shot
    actually used in the cut, duplicates included when a shot's slot is
    covered by another shot's render) that were genuinely vision-reviewed.

    A take counts as covered iff `<review_dir>/<take>.response.json` exists
    AND its `accept` is true. This is deliberately independent of WHEN the
    response was written: a response backfilled after the run ended (the
    auto-reviewer catching up, or a human answering the browser queue later)
    counts exactly the same as one that arrived during the run — that is the
    intentional recovery path for QM-010's promote gate (see generate.py):
    answer the queue, then re-run --resume/--finalize, which calls this again
    and reads the (now updated) responses straight off disk. A take whose
    only verdict is the in-run heuristic fallback (no response file at all —
    see the "unattended mode: external review skipped" / "external review
    timed out" reasons `ExternalReviewer.review` appends above) is uncovered,
    and so is a take whose response file exists but rejected it
    (accept=false): a rejected take should never count as vision-approved
    just because it ended up covering the shot's slot anyway.

    Returns `(coverage, uncovered)`. `coverage` is a float in [0, 1] — 1.0 for
    an empty `accepted_takes` (vacuously: nothing to fail). `uncovered` is the
    subset of `accepted_takes` (original order, duplicates preserved) that did
    not qualify.
    """
    accepted_takes = list(accepted_takes)
    if not accepted_takes:
        return 1.0, []

    uncovered = []
    for take in accepted_takes:
        response_path = os.path.join(review_dir, f"{take}.response.json")
        covered = False
        if os.path.exists(response_path):
            try:
                with open(response_path, "r", encoding="utf-8") as f:
                    verdict = json.load(f)
                covered = bool(verdict.get("accept"))
            except (OSError, json.JSONDecodeError):
                covered = False
        if not covered:
            uncovered.append(take)

    coverage = (len(accepted_takes) - len(uncovered)) / len(accepted_takes)
    return coverage, uncovered


# ------------------------------------------------------------------ test ---
if __name__ == "__main__":
    import numpy as np
    from moviepy import ImageSequenceClip
    from PIL import Image, ImageDraw

    from pipeline.config import abspath, load_config

    cfg = load_config()
    out_dir = abspath(cfg, cfg["paths"]["output_dir"])
    scratch = os.path.join(out_dir, "_kidsong_review_test")
    os.makedirs(scratch, exist_ok=True)

    def make_mp4(path, frames, clip_fps):
        seq = ImageSequenceClip([np.array(f) for f in frames], fps=clip_fps)
        seq.write_videofile(path, fps=clip_fps, codec="libx264", audio=False, logger=None)
        seq.close()

    # (a) moving rectangle -> should ACCEPT
    n_frames = 48
    moving_frames = []
    for k in range(n_frames):
        frame = Image.new("RGB", (480, 832), (25, 25, 25))
        d = ImageDraw.Draw(frame)
        x = int((k / n_frames) * 380)
        d.rectangle((x, 380, x + 90, 470), fill=(230, 70, 70))
        d.ellipse((60, 120, 260, 320), outline=(240, 240, 240), width=8)
        moving_frames.append(frame)
    moving_path = os.path.join(scratch, "moving.mp4")
    make_mp4(moving_path, moving_frames, n_frames / 2.0)

    # (b) 5 identical frames (with texture, so it's rejected for "static" and
    # not merely "blurry") -> should REJECT "static"
    still = Image.new("RGB", (480, 832), (60, 60, 60))
    d = ImageDraw.Draw(still)
    d.rectangle((150, 380, 240, 470), fill=(230, 70, 70))
    d.ellipse((60, 120, 260, 320), outline=(240, 240, 240), width=8)
    static_frames = [still] * 48
    static_path = os.path.join(scratch, "static.mp4")
    make_mp4(static_path, static_frames, 24)

    # Force heuristic mode for the self-test regardless of the user's config
    # (the real config may select the blocking external reviewer).
    test_cfg = dict(cfg)
    test_cfg["kidsong"] = dict(cfg.get("kidsong", {}))
    test_cfg["kidsong"]["review"] = dict(test_cfg["kidsong"].get("review", {}), reviewer="heuristic")
    reviewer = get_reviewer(test_cfg)
    assert isinstance(reviewer, HeuristicReviewer)

    shot = {"id": "s0", "characters": ["kid_a", "kid_b"]}

    verdict_moving = reviewer.review(shot, moving_path)
    print("moving verdict:", verdict_moving)
    assert verdict_moving["accept"] is True, verdict_moving
    assert "static" not in verdict_moving["reasons"], verdict_moving

    verdict_static = reviewer.review(shot, static_path)
    print("static verdict:", verdict_static)
    assert verdict_static["accept"] is False, verdict_static
    assert "static" in verdict_static["reasons"], verdict_static

    sheet = contact_sheet(shot, moving_path, scratch)
    print("contact sheet:", sheet)
    assert os.path.exists(sheet)

    log_path = os.path.join(scratch, "review_log.json")
    review_shots_log(log_path, [{"shot_id": "s0", "verdict": verdict_moving}])
    review_shots_log(log_path, [{"shot_id": "s0", "verdict": verdict_static}])
    with open(log_path, "r", encoding="utf-8") as f:
        logged = json.load(f)
    assert len(logged) == 2, logged
    print("review log:", log_path)

    print("PASS")
