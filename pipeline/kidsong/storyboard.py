"""
kidsong.storyboard — The STORYBOARD BRAIN: turn a song into concrete per-verse
action beats with gaze targets, so characters do purposeful things tied to the
song's own topic instead of the old canned filler ("sings along with a happy
face, moving to the beat").

This sits BEFORE the shot list: `director.plan_shots` turns a song directly
into shots, with no notion of a story arc across the verses. This module adds
that arc as its own artifact — one entry per verse, each carrying a
`beat_goal` and 4-6 concrete `beats` (action + gaze + optional prop +
performer) — which a later change wires into `director`/`generate` so shot
actions are DERIVED from a beat instead of invented per shot. Nothing in this
module is wired into the render pipeline yet; it only builds and gates the
storyboard artifact.

Chain of attempts (first success wins; every exception is swallowed and this
never crashes the pipeline) — the same shape as
`pipeline.kidsong.lyrics.generate_song` / `pipeline.kidsong.director.plan_shots`:
  1. Whatever backend is configured in config.json -> llm.backend (groq/ollama/gemini),
     reusing the same call helpers as script_gen.py.
  2. A plain Ollama /api/chat call against llm.ollama_host (skipped if step 1 already
     was an Ollama call — no point retrying the identical request).
  3. An OpenAI-compatible /v1/chat/completions call against the same host, useful when
     a local llama-server (not real Ollama) is listening on that port instead and only
     speaks the OpenAI API shape. The model id is discovered via GET /v1/models.
  4. `fallback_storyboard` — a deterministic, LLM-free discover/try/grow/celebrate
     arc built from `director.song_story_subject` (or a generic topic noun pulled
     from the title when the song has no detectable non-child subject).

Each LLM attempt is run through the deterministic gate `_storyboard_verdict`
(never the external/human review — this module has no review.py wiring, by
design: it is not yet on the render path). A rejected attempt's reasons are
fed back into the next attempt's prompt (see `_retry_feedback_block`, mirroring
`lyrics._generate_llm_song_gated`'s retry-with-feedback loop). If no attempt
passes after `kidsong.storyboard.attempts` tries (default 3), or the LLM
backend is entirely unreachable, `fallback_storyboard(song)` ships instead —
itself always gate-passing (see its own docstring and
tests/test_kidsong_storyboard.py), so `build_storyboard` always returns a
directable storyboard and never raises.

Returns strict JSON shaped like:
  {
    "verses": [
      {
        "verse": 0,
        "beat_goal": "<one sentence describing what this verse's beats accomplish>",
        "beats": [
          {
            "action": "{name} kneels at the flower bed and pats soil around the seedling",
            "gaze": "down at the seedling",
            "prop": "a small green watering can",       # optional
            "performer": "child",                        # "child" | "all" | "story_subject"
            "camera_address": false
          },
          ... 4-6 beats total ...
        ]
      },
      ... one entry per song verse, same order ...
    ]
  }

Beats ALWAYS use the literal placeholder "{name}" for the acting child —
never a concrete cast name. A later change substitutes the planner's own
cast rotation onto that placeholder; this module never picks who acts.
"""
import json
import logging
import re

from pipeline.script_gen import (
    _call_gemini,
    _call_groq,
    _call_ollama,
    _extract_json,
    _load_prompt,
)

log = logging.getLogger("kidsong.storyboard")

SYSTEM_PROMPT = (
    "You are a storyboard artist for a wholesome preschool TV show, planning concrete, "
    "visible, gentle per-verse action beats for toddlers to watch and copy along with. "
    "You ALWAYS reply with a single valid JSON object and nothing else."
)

# How many times `build_storyboard` asks the LLM for a storyboard, feeding each
# rejection's reasons back into the prompt, before giving up and shipping
# `fallback_storyboard`. Overridable via `kidsong.storyboard.attempts`. 3
# mirrors `lyrics.DEFAULT_LYRICS_LLM_ATTEMPTS` — enough for the gate's
# actionable reasons to be applied without stalling a run.
DEFAULT_STORYBOARD_ATTEMPTS = 3

_VALID_PERFORMERS = {"child", "all", "story_subject"}
_NAME_PLACEHOLDER = "{name}"

# Story-driven casting (kidsong.staging="story"): beats optionally carry a
# "role" so the PLANNER can cast from the STORY instead of round-robin
# rotation — the owner's red-thread rule: one protagonist child carries the
# arc through every verse, a partner appears only where a beat genuinely
# needs a second child, and "all" only at story-gathering moments (opening,
# one mid-song shared moment, finale). Ensemble mode emits no role keys and
# is byte-identical to the pre-story board.
_VALID_ROLES = {"protagonist", "partner", "all"}
# The acting partner child in a role="partner" beat — substituted by the
# planner alongside "{name}", never a concrete cast name (same reasoning as
# _NAME_PLACEHOLDER: the board never picks who acts).
_PARTNER_PLACEHOLDER = "{partner}"

# A beat's own "camera_address" may be true on at most this many beats across
# the WHOLE storyboard — direct camera address is the exception, not the rule
# (see the prompt's ARC MANDATE and `_storyboard_verdict`).
_MAX_CAMERA_ADDRESS_BEATS = 1

_MIN_BEATS_PER_VERSE, _MAX_BEATS_PER_VERSE = 4, 6

# Gaze phrases that only make sense on a beat that actually addresses the
# camera — anywhere else they are exactly the "smiles at the camera" filler
# this module exists to replace.
_CAMERA_GAZE_PHRASES = ("at the camera", "into the lens")

# Generic filler the gate REJECTS in beat actions (case-insensitive substring).
# These are the exact canned phrases the storyboard exists to replace — the
# prompt already forbids them, but a local 8B routinely ignores prompt-level
# bans and every one of these is verb-ish, so only a hard gate catches them.
# The deterministic fallback board contains none of these (asserted in tests).
_FILLER_PHRASES = (
    "sings along",
    "sing along",
    "moving to the beat",
    "moves to the beat",
    "with a happy face",
    "with happy faces",
    "dances happily",
    "claps along",
    "clap along",
    "smiles at the camera",
    "smiles brightly at the camera",
)


# ----------------------------------------------------------------- public -----
def build_storyboard(song, cfg=None):
    """Return this episode's storyboard dict. Never raises.

    Orchestrates the LLM attempt loop -> deterministic gate -> retry-with-
    feedback -> `fallback_storyboard` chain described in the module docstring.
    """
    try:
        if cfg is None:
            from pipeline.config import load_config

            cfg = load_config()

        ks = (cfg.get("kidsong", {}) or {}) if isinstance(cfg, dict) else {}
        storyboard_cfg = ks.get("storyboard", {}) or {}
        try:
            attempts = int(storyboard_cfg.get("attempts", DEFAULT_STORYBOARD_ATTEMPTS))
        except (TypeError, ValueError):
            attempts = DEFAULT_STORYBOARD_ATTEMPTS
        attempts = max(1, attempts)

        feedback = None
        for i in range(attempts):
            try:
                board = (
                    _generate_llm_storyboard(song, cfg, feedback)
                    if feedback
                    else _generate_llm_storyboard(song, cfg)
                )
            except Exception:
                log.exception(
                    "storyboard: LLM attempt %d/%d raised — stopping the attempt "
                    "loop early.", i + 1, attempts,
                )
                break

            if isinstance(board, dict) and board.pop("_backend_dead", False):
                log.warning(
                    "storyboard: LLM backend unavailable on attempt %d/%d — no "
                    "attempt produced; falling back to the deterministic "
                    "storyboard.", i + 1, attempts,
                )
                break

            try:
                verdict = _storyboard_verdict(board, song)
            except Exception:
                log.exception(
                    "storyboard: the gate raised on attempt %d/%d — treating "
                    "the attempt as rejected.", i + 1, attempts,
                )
                verdict = {"accept": False, "reasons": ["the gate itself raised an exception"]}

            if verdict.get("accept"):
                if i:
                    log.info("storyboard accepted on attempt %d/%d.", i + 1, attempts)
                return board

            feedback = list(verdict.get("reasons") or [])
            log.info(
                "storyboard attempt %d/%d rejected: %s",
                i + 1, attempts, "; ".join(feedback),
            )
    except Exception:
        log.exception("storyboard: build_storyboard failed unexpectedly")

    log.warning(
        "storyboard: no LLM attempt passed the gate — using the deterministic "
        "discover/try/grow/celebrate fallback."
    )
    return fallback_storyboard(song, cfg)


# ---------------------------------------------------------------- backends ----
def _generate_llm_storyboard(song, cfg, feedback=None):
    """Ask the configured LLM for one storyboard attempt (backend cascade
    mirroring `director.plan_shots`); never raises.

    Returns a NORMALIZED storyboard dict (see `_normalize_storyboard`). When
    every backend fails, the private key `_backend_dead` is set True so
    `build_storyboard` can stop retrying a dead backend immediately instead of
    burning through every remaining attempt on the same failure — mirroring
    `lyrics._generate_llm_song`'s `_backend_dead` flag. Callers must `.pop`
    that key before treating the result as a real storyboard; it never ships.

    This is the function tests monkeypatch to stub the LLM (see
    tests/test_kidsong_storyboard.py), the same way lyrics tests monkeypatch
    `lyrics._generate_llm_song`.
    """
    from pipeline.kidsong import director

    root = cfg["_root"]
    template = _load_prompt("kidsong_storyboard", root)

    try:
        story_subject = director.song_story_subject(song)
    except Exception:
        story_subject = None
    story_subject_text = (
        story_subject
        if isinstance(story_subject, str) and story_subject.strip()
        else "none detected — build the arc around the song's own topic and props instead"
    )

    user = (
        template
        .replace("{{SONG_JSON}}", json.dumps(song, ensure_ascii=False))
        .replace("{{STORY_SUBJECT}}", story_subject_text)
    )
    if feedback:
        user += _retry_feedback_block(feedback)

    llm = cfg.get("llm", {}) or {}
    backend = llm.get("backend")
    host = llm.get("ollama_host", "http://localhost:11434")

    data = None

    # 1. Whatever backend config.json points at.
    try:
        if backend == "groq":
            raw = _call_groq(SYSTEM_PROMPT, user, llm["groq_model"])
        elif backend == "ollama":
            raw = _call_ollama(SYSTEM_PROMPT, user, llm["ollama_model"], host)
        elif backend == "gemini":
            raw = _call_gemini(SYSTEM_PROMPT, user, llm["gemini_model"])
        else:
            raise ValueError(f"Unknown llm.backend: {backend}")
        data = _extract_json(raw)
    except Exception:
        data = None

    # 2. Plain Ollama /api/chat, unless step 1 already made that exact call.
    if data is None and backend != "ollama":
        try:
            raw = director._post_ollama_chat(SYSTEM_PROMPT, user, llm.get("ollama_model", ""), host)
            data = _extract_json(raw)
        except Exception:
            data = None

    # 3. OpenAI-compatible endpoint (e.g. a local llama-server sharing the Ollama port).
    if data is None:
        try:
            raw = director._post_openai_compatible_chat(SYSTEM_PROMPT, user, host)
            data = _extract_json(raw)
        except Exception:
            data = None

    storyboard = _normalize_storyboard(data, song)
    if data is None:
        storyboard["_backend_dead"] = True
    return storyboard


def _retry_feedback_block(reasons):
    """Prompt appendix that turns the gate's rejection reasons into an explicit
    fix-list for the next LLM attempt — mirrors `lyrics._retry_feedback_block`."""
    bullets = "\n".join(f"- {r}" for r in reasons)
    return (
        "\n\nYOUR PREVIOUS STORYBOARD WAS REJECTED by the automatic checker for "
        "the reasons below. Write a NEW storyboard that fixes EVERY one of "
        "them:\n"
        f"{bullets}\n"
        "Keep the same number of verses and the same \"{name}\" placeholder "
        "convention for the acting child — only fix the problems listed above."
    )


# ------------------------------------------------------------- normalization ---
def _clean_line(text):
    if text is None:
        return ""
    return re.sub(r"\s+", " ", str(text)).strip()


def _coerce_bool(value):
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in ("true", "yes", "1")
    if isinstance(value, (int, float)):
        return bool(value)
    return False


def _coerce_performer(value):
    performer = str(value or "").strip().lower()
    return performer if performer in _VALID_PERFORMERS else "child"


def _coerce_beat(raw):
    """One beat dict, defensively coerced, or None when it has no usable action."""
    if not isinstance(raw, dict):
        return None
    action = _clean_line(raw.get("action"))
    if not action:
        return None
    beat = {
        "action": action,
        "gaze": _clean_line(raw.get("gaze")),
        "performer": _coerce_performer(raw.get("performer")),
        "camera_address": _coerce_bool(raw.get("camera_address")),
    }
    prop = _clean_line(raw.get("prop"))
    if prop:
        beat["prop"] = prop
    # Optional story-casting role: passed through only when valid, dropped
    # otherwise — so a future LLM board can carry roles, but a malformed one
    # degrades to a role-less beat the verdict imposes no role rules on.
    role = str(raw.get("role") or "").strip().lower()
    if role in _VALID_ROLES:
        beat["role"] = role
    return beat


def _coerce_verse(raw, index):
    raw = raw if isinstance(raw, dict) else {}
    try:
        verse = int(raw.get("verse", index))
    except (TypeError, ValueError):
        verse = index

    raw_beats = raw.get("beats")
    raw_beats = raw_beats if isinstance(raw_beats, (list, tuple)) else []
    beats = []
    for b in raw_beats:
        coerced = _coerce_beat(b)
        if coerced:
            beats.append(coerced)
        if len(beats) >= _MAX_BEATS_PER_VERSE:
            break

    return {"verse": verse, "beat_goal": _clean_line(raw.get("beat_goal")), "beats": beats}


def _normalize_storyboard(data, song=None):
    """Validate + coerce raw LLM/parsed storyboard JSON into the strict shape,
    the same defensive way `lyrics._normalize_song` does: never trust an LLM's
    types, presence, or bounds.

    `song` (optional) supplies how many verse entries are expected — one slot
    per song verse, filled POSITIONALLY from whatever the raw data offers at
    that index (never trusting the LLM's own "verse" number for indexing,
    only using it — coerced to an int, defaulting back to the position — as
    the stored "verse" field). When `song` is omitted, the raw data's own
    length is used instead, so this function is still usable standalone.

    This function ONLY coerces types/shape; whether the result is actually
    GOOD (right beat counts, no filler, gaze/camera rules, full verse
    coverage) is decided entirely by `_storyboard_verdict`, never here — same
    separation of concerns as `lyrics._normalize_song` vs `script_qc.review_script`.
    """
    if not isinstance(data, dict):
        data = {}
    raw_verses = data.get("verses")
    raw_verses = raw_verses if isinstance(raw_verses, (list, tuple)) else []

    # `song` gets the same treatment as `data` one line up. Every other input in
    # this module is coerced before use ("never trust an LLM's types, presence,
    # or bounds" — the docstring above), but this one call trusted `song` to be
    # a dict and raised AttributeError on anything else. Not reachable from the
    # pipeline, where `song` always comes from lyrics.generate_song — but the
    # docstring advertises this function as usable standalone, and a lone
    # trusting line in a defensive module is how the next caller gets caught.
    song = song if isinstance(song, dict) else None
    num_song_verses = len(((song or {}).get("verses")) or []) if song is not None else 0
    target = num_song_verses if num_song_verses else len(raw_verses)

    verses = [_coerce_verse(raw_verses[i] if i < len(raw_verses) else {}, i) for i in range(target)]
    return {"verses": verses}


# --------------------------------------------------------------------- gate ---
def _storyboard_verdict(storyboard, song):
    """Deterministic craft/safety gate for a storyboard attempt.

    Reuses the EXACT downstream predicates a shot list derived from this
    storyboard would later be judged by (imported, never copied), so a board
    that passes here cannot flunk that later QC for the same reason:
      * `script_qc._is_verbish` — every beat's action must read as doing
        something (pipeline/kidsong/script_qc.py ~340).
      * `director.martial_terms` / `director.scary_terms` — no lockstep/martial
        staging, nothing frightening, in any beat's action or gaze (the same
        vocabulary `script_qc._check_staging` applies to a shot's action/setting).
      * `director.verse_activity_progression` — a synthetic per-verse check:
        build minimal fake shots from the verse's own beats (the beat's action
        as the shot's action, one setting shared by construction) and require
        the result not be "scattered", so a verse's beats already read as one
        continuous moment before any shot list exists.

    Plus storyboard-specific structural rules this module owns:
      * every song verse is covered by exactly one storyboard verse entry,
        each carrying 4-6 beats;
      * every beat carries a non-empty gaze target;
      * a gaze may only reference the camera/lens when that SAME beat's
        `camera_address` is true, and at most one beat in the whole board may
        set `camera_address` true;
      * every `performer == "child"` beat's action contains the literal
        "{name}" placeholder;
      * every `performer == "story_subject"` beat's action contains neither
        "{name}" nor a concrete cast name;
      * NO beat, regardless of performer, names a concrete cast child — the
        planner substitutes its own rotation onto "{name}"; a beat that
        already hardcodes "Zuri"/"Kofi"/"Nala" defeats that rotation.

    Returns `{"accept": bool, "reasons": [...]}` — no score, unlike
    `script_qc`'s verdict shape: `build_storyboard` has no "ship the
    best-effort attempt" tier, only accept-or-retry-or-fallback, so nothing
    ever reads a score off this.
    """
    from pipeline.kidsong import director
    from pipeline.kidsong.script_qc import _is_verbish

    reasons = []

    verses = storyboard.get("verses") if isinstance(storyboard, dict) else None
    verses = verses if isinstance(verses, (list, tuple)) else []
    song_verses = (song or {}).get("verses") if isinstance(song, dict) else None
    song_verses = song_verses if isinstance(song_verses, (list, tuple)) else []

    if not song_verses or len(verses) != len(song_verses):
        reasons.append(
            f"verse coverage: storyboard has {len(verses)} verse entries but the "
            f"song has {len(song_verses)} verses — "
            "every song verse needs exactly one storyboard entry"
        )

    try:
        cast_names = []
        try:
            from pipeline.kidsong import cast

            cast_names = [n for n in cast.names() if n]
        except Exception:
            cast_names = []

        camera_address_count = 0
        for v in verses:
            v = v if isinstance(v, dict) else {}
            idx = v.get("verse")
            beats = v.get("beats") if isinstance(v.get("beats"), (list, tuple)) else []

            if not (_MIN_BEATS_PER_VERSE <= len(beats) <= _MAX_BEATS_PER_VERSE):
                reasons.append(
                    f"beat count: verse {idx} has {len(beats)} beats (need "
                    f"{_MIN_BEATS_PER_VERSE}-{_MAX_BEATS_PER_VERSE})"
                )

            for b in beats:
                b = b if isinstance(b, dict) else {}
                action = str(b.get("action") or "").strip()
                gaze = str(b.get("gaze") or "").strip()
                performer = str(b.get("performer") or "").strip().lower()
                camera_address = bool(b.get("camera_address"))
                if camera_address:
                    camera_address_count += 1

                if not action or not _is_verbish(action):
                    reasons.append(
                        f"action not verb-ish: verse {idx} beat action={action!r} "
                        "does not read as doing something visible"
                    )

                # Anti-filler: the whole point of the board is that characters
                # DO something tied to the topic. The prompt forbids these, but
                # a local 8B routinely ignores prompt-level bans — and every one
                # of them is verb-ish, so the check above cannot catch them.
                # Measured live: an LLM board shipped "{name} sings along with a
                # happy face, moving to the beat" beats straight through this
                # gate into the plan (plan-A/B, sunflower song) — the exact
                # canned filler the storyboard exists to kill.
                filler = [p for p in _FILLER_PHRASES if p in action.lower()]
                if filler:
                    reasons.append(
                        f"filler action: verse {idx} beat action={action!r} is "
                        f"generic filler ({', '.join(filler)}) — describe a "
                        "concrete, topic-tied thing the performer does instead"
                    )

                martial = set(director.martial_terms(action)) | set(director.martial_terms(gaze))
                if martial:
                    reasons.append(
                        f"martial staging: verse {idx} beat uses martial/lockstep "
                        f"language ({', '.join(sorted(martial))}) — action={action!r}"
                    )

                scary = set(director.scary_terms(action)) | set(director.scary_terms(gaze))
                if scary:
                    reasons.append(
                        f"frightening content: verse {idx} beat contains language "
                        f"that is not wholesome for toddlers "
                        f"({', '.join(sorted(scary))}) — action={action!r}"
                    )

                if not gaze:
                    reasons.append(f"gaze missing: verse {idx} beat has no gaze target — action={action!r}")
                else:
                    gaze_lower = gaze.lower()
                    if any(p in gaze_lower for p in _CAMERA_GAZE_PHRASES) and not camera_address:
                        reasons.append(
                            f"gaze at camera: verse {idx} gaze={gaze!r} addresses the "
                            "camera/lens but camera_address is not true on this beat"
                        )

                if performer == "child" and _NAME_PLACEHOLDER not in action:
                    reasons.append(
                        f"missing name placeholder: verse {idx} performer=child beat "
                        f"action={action!r} does not contain \"{_NAME_PLACEHOLDER}\""
                    )

                # A performer=child beat whose ACTION stages the group is a
                # mislabel: the wiring gives that shot ONE child, the action
                # says "the kids", and the downstream head-count repair
                # flattens the contradiction back to canned filler (measured
                # live: one such beat per board survived every other rule).
                # Either the performer should be "all" or the action should be
                # singular — reject and let the feedback retry fix the label.
                if performer == "child" and any(
                    g in action.lower() for g in ("the kids", "the children", "everyone")
                ):
                    reasons.append(
                        f"performer/action mismatch: verse {idx} performer=child "
                        f"beat action={action!r} stages the whole group — set "
                        "performer to \"all\" or write a single-child action"
                    )

                if performer == "story_subject" and _NAME_PLACEHOLDER in action:
                    reasons.append(
                        f"story_subject uses placeholder: verse {idx} performer="
                        f"story_subject beat action={action!r} must not contain "
                        f"\"{_NAME_PLACEHOLDER}\""
                    )

                # Story-casting role rules — ADDITIVE: they only apply to a
                # beat that actually carries a role, so a role-less (ensemble
                # or legacy LLM) board sees zero new requirements.
                role = str(b.get("role") or "").strip().lower()
                if role:
                    if role == "partner" and (
                        _NAME_PLACEHOLDER not in action
                        or _PARTNER_PLACEHOLDER not in action
                    ):
                        reasons.append(
                            f"partner beat placeholders: verse {idx} role=partner "
                            f"beat action={action!r} must contain BOTH "
                            f"\"{_NAME_PLACEHOLDER}\" and \"{_PARTNER_PLACEHOLDER}\""
                        )
                    if role == "all" and performer != "all":
                        reasons.append(
                            f"role/performer mismatch: verse {idx} role=all beat "
                            f"has performer={performer!r} — a gathering beat must "
                            "stage the whole group"
                        )
                    if performer == "all" and role != "all":
                        reasons.append(
                            f"role/performer mismatch: verse {idx} performer=all "
                            f"beat has role={role!r} — an ensemble beat's role "
                            "must be \"all\""
                        )

                hit_names = sorted(
                    n for n in cast_names
                    if re.search(r"\b" + re.escape(n) + r"\b", action, re.IGNORECASE)
                )
                if hit_names:
                    reasons.append(
                        f"concrete cast name: verse {idx} beat action={action!r} "
                        f"names a cast child ({', '.join(hit_names)}) instead of using "
                        f"the \"{_NAME_PLACEHOLDER}\" placeholder"
                    )

        if camera_address_count > _MAX_CAMERA_ADDRESS_BEATS:
            reasons.append(
                f"camera_address overbudget: {camera_address_count} beats address the "
                f"camera (at most {_MAX_CAMERA_ADDRESS_BEATS} allowed in the whole storyboard)"
            )

        for v in verses:
            v = v if isinstance(v, dict) else {}
            beats = v.get("beats") if isinstance(v.get("beats"), (list, tuple)) else []
            fake_shots = [
                {
                    "action": str((b or {}).get("action") or ""),
                    "setting": "a shared setting",
                    "reuse_of": None,
                    "story_subject": (str((b or {}).get("prop") or "") or None),
                }
                for b in beats
                if isinstance(b, dict)
            ]
            if not fake_shots:
                continue
            progression = director.verse_activity_progression(fake_shots)
            if progression == "scattered":
                reasons.append(
                    f"scattered progression: verse {v.get('verse')}'s beats do not "
                    "read as one continuous moment — give the verse one shared "
                    "activity, or a small setup/event/reaction arc, not unrelated beats"
                )
    except Exception:
        reasons.append("the gate raised an exception while scoring individual beats")

    return {"accept": not reasons, "reasons": reasons}


# ------------------------------------------------------ deterministic fallback ---
# Middle-verse stages, in assignment order. FOUR distinct stages (was two:
# "try"/"grow"): the old pair made verse 1 and verse 3 of every 5-verse song
# byte-identical, because the middle picker cycles and (1-1)%2 == (3-1)%2.
# Measured live: the "Rain, Rain, Go Away" fallback replayed its entire
# umbrella "try" verse verbatim as verse 3, which read as the cut looping a
# scene. The channel's songs run 3-6 verses, so four stages give every real
# song a DISTINCT beat set per middle verse (n<=6); the cross-verse uniqueness
# test in tests/test_kidsong_no_scene_repeat.py guards the guarantee.
_ARC_MIDDLE_STAGES = ("try", "grow", "explore", "share")

_STAGE_GOAL_TEMPLATES = {
    "discover": "The kids notice {subject} for the first time and get curious.",
    "try": "The kids try a hands-on way to explore {subject}.",
    "grow": "The kids see {subject} change for the better, step by step.",
    "explore": "The kids explore {subject} together in a playful new way.",
    "share": "The kids share the fun and take gentle turns together.",
    "celebrate": "The kids celebrate together with {subject}.",
}

# Generic, universally-safe held prop for the TRY stage when the song names no
# real prop of its own — always grammatical, never martial/scary, never
# containing "{name}" or a cast name.
_GENERIC_FALLBACK_PROP = "a favorite toy"

# Curated topic-keyword -> held-prop table for the TRY stage. Every phrase must
# read naturally in ALL the try-stage templates ("picks up {prop} and tries it
# out", "tips {prop} carefully", "hands {prop} to a friend", "down at {prop}"),
# be a real graspable object (the naive alternative — gluing the topic noun into
# a prop phrase — yields nonsense like picking up "a small sunflower garden"),
# and respect the channel floor (no bath/tub items, no jewellery, nothing
# martial/scary). Measured motivation: the "Sunflower Family Tree" fallback
# board had the kids exploring "a favorite toy" instead of anything garden-like,
# which read as random rather than topical. First keyword hit wins, in table
# order; no hit -> _GENERIC_FALLBACK_PROP exactly as before.
_TOPIC_PROPS = (
    (("plant", "flower", "sunflower", "garden", "tree", "seed", "grow", "bloom"),
     "a small green watering can"),
    # Props carry a PINNED COLOR wherever one reads naturally: the prop phrase
    # is the only object identity the per-shot keyframes share, and a
    # color-less "bright little umbrella" rendered as a DIFFERENT colorway in
    # every shot of the storymode render (rainbow -> lime -> pastel between
    # adjacent shots) — an object-continuity break the channel owner flags.
    (("rain", "puddle", "cloud", "weather", "storm"), "a bright yellow umbrella"),
    (("bubble",), "a bubble wand"),
    (("wash", "hands", "soap", "brush", "teeth"), "a little bar of soap"),
    (("bike", "ride", "wheel"), "a shiny bicycle bell"),
    (("breakfast", "food", "cook", "bake", "table", "eat"), "a little mixing spoon"),
    (("color", "colour", "paint", "draw", "rainbow"), "a chunky crayon"),
    (("music", "dance", "clap", "drum", "beat"), "a small toy drum"),
    (("star", "night", "moon", "sleep", "dream"), "a small plush star"),
    (("tidy", "clean", "toys"), "a little toy basket"),
    (("animal", "duck", "dog", "cat", "bird", "bunny", "sheep", "mouse"),
     "a soft animal plushie"),
)

_TITLE_STOPWORDS = {
    "a", "an", "the", "and", "or", "with", "in", "on", "at", "of", "to", "for",
    "song", "sing", "singalong", "along", "kids", "kid", "toddler", "toddlers",
    "day", "time", "adventure", "adventures", "fun", "happy", "little", "big",
    "my", "our", "we", "yay", "new", "join", "let", "lets", "sunny", "great",
    "wonderful", "amazing", "playtime",
}

# Bare particles/adverbs/verbs that survive `_TITLE_STOPWORDS` (they are >2
# letters and not scenery) but are NOT topic nouns — a title like "Rain, Rain,
# Go Away" must not resolve the topic to the adverb "away". Dropped before the
# first-three-words topic pick in `_topic_noun`.
_TOPIC_DROP_WORDS = {
    "go", "goes", "going", "gone", "away", "again", "today", "come", "comes",
    "coming", "around", "round", "back", "here", "there", "now", "off",
}


def _stage_for_verse(i, n):
    """The arc stage ("discover" | "try" | "grow" | "celebrate") for verse `i`
    of an `n`-verse song. First verse discovers, last celebrates; everything
    between cycles through try/grow so a 3-verse song compresses to
    discover -> try -> celebrate and a 5-verse song repeats try/grow once."""
    if n <= 1:
        return "celebrate"
    if i == 0:
        return "discover"
    if i == n - 1:
        return "celebrate"
    return _ARC_MIDDLE_STAGES[(i - 1) % len(_ARC_MIDDLE_STAGES)]


def _topic_noun(song):
    """Best-effort topic noun phrase pulled from the song's title, used only
    when no `story_subject` is detected — the "generic shared-play arc around
    the topic noun from the title" fallback. Never raises; falls back to a
    safe, generic phrase when nothing usable survives."""
    song = song if isinstance(song, dict) else {}
    title = str(song.get("title") or "")

    cast_lower = set()
    try:
        from pipeline.kidsong import cast

        cast_lower = {n.lower() for n in cast.names()}
    except Exception:
        cast_lower = {"zuri", "kofi", "nala"}

    words = re.findall(r"[A-Za-z]+", title)
    keep = []
    seen = set()
    for w in words:
        lw = w.lower()
        # Drop cast names, stopwords, bare particles/adverbs, and repeats. The
        # last two matter: an imperative/onomatopoeic title ("Rain, Rain, Go
        # Away") otherwise yields "rain rain away" — a non-noun the fallback
        # then stages as a physical object ("gather around the rain rain away"),
        # the exact incoherence this de-dupe kills. After the filter it is just
        # "rain", a real topic the beats can reference safely.
        if lw in cast_lower or lw in _TITLE_STOPWORDS or lw in _TOPIC_DROP_WORDS:
            continue
        if len(lw) <= 2 or lw in seen:
            continue
        seen.add(lw)
        keep.append(lw)
    if not keep:
        return "favorite game"
    return " ".join(keep[:3])


def _fallback_prop(topic):
    """A safe held prop for the TRY stage, topic-matched via `_TOPIC_PROPS`.

    Deliberately NOT built by gluing the topic noun into a prop phrase — a
    title like "Sunflower Garden" would yield the nonsensical "a small
    sunflower garden" as something a child "picks up". Instead a curated
    keyword table maps common toddler topics to a real, graspable, on-theme
    object (sunflower -> a watering can); anything unmatched keeps the
    universally-coherent generic toy."""
    words = set(re.findall(r"[a-z]+", str(topic or "").lower()))
    for keywords, prop in _TOPIC_PROPS:
        if words & set(keywords):
            return prop
    return _GENERIC_FALLBACK_PROP


def _tag_roles(beats):
    """Stamp the default story-mode role onto each beat IN PLACE and return
    `beats`: performer "all" -> role "all", performer "child" -> role
    "protagonist" (a beat that already carries a role — the partner beat —
    keeps it), story_subject beats get no role. Only ever called in story
    mode; ensemble boards must stay byte-identical (no role keys at all)."""
    for b in beats:
        if "role" in b:
            continue
        performer = b.get("performer")
        if performer == "all":
            b["role"] = "all"
        elif performer == "child":
            b["role"] = "protagonist"
    return beats


def _staging_mode(cfg):
    """The kidsong.staging mode from `cfg`: "story" or "ensemble" (default).
    Never raises — any unreadable/unknown value is ensemble, so a malformed
    config can only ever produce today's behavior."""
    try:
        ks = cfg.get("kidsong") if isinstance(cfg, dict) else None
        raw = (ks or {}).get("staging") if isinstance(ks, dict) else None
        mode = str(raw or "").strip().lower()
        return "story" if mode == "story" else "ensemble"
    except Exception:
        return "ensemble"


def _stage_beats(stage, subject_phrase, prop_phrase, has_subject, story=False):
    """4-6 beats for one arc stage, in one of two strictly separate regimes.

    `story=True` (kidsong.staging="story") applies the red-thread casting
    deltas on top of whichever regime is active, and tags every beat with a
    "role" (see _tag_roles): the MIDDLE stages lose their per-verse group
    beat — try's group beat becomes the PARTNER beat (a planned two-child
    moment: "{name} passes {prop} gently to {partner}"), grow/explore's group
    beats become protagonist solos — while discover (opening gathering),
    share (the one mid-song shared moment) and celebrate (finale) keep their
    group beats. Ensemble mode (story=False) is byte-identical to the
    pre-story board: same beats, no role keys.

    has_subject=True — the song has a real, concrete non-child hero (a
    sunflower, a lamb, a toy boat). `subject_phrase` is that hero and beats MAY
    stage it as a physical object the kids gather around, tend and celebrate
    with, because it IS one; `performer="story_subject"` beats show the hero
    acting on its own.

    has_subject=False — there is no concrete hero, only an abstract topic
    (rain, colours, clapping). `subject_phrase` is then a "the {topic}" phrase
    that must NEVER be staged as a graspable object — "the kids gather around
    the rain" / "wave to the rain one last time" / "cheering on the rain" is
    the exact incoherence this regime exists to prevent (measured live: the
    "Rain, Rain, Go Away" fallback had the kids circling and cheering on "the
    rain rain away"). Instead the kids simply play together — a real, coherent
    preschool-video moment — gazing at each other and their own hands, and the
    topic is named only in the (internal, unrendered) beat_goal. No
    `story_subject` beat is ever emitted here: there is nothing concrete to show.

    Every beat keeps at most one activity verb and a verse keeps at most four
    distinct ones, so `director.verse_activity_progression` never reads the
    verse as "scattered" (see `_storyboard_verdict`)."""
    if stage == "discover":
        if has_subject:
            beats = [
                {"action": f"{_NAME_PLACEHOLDER} spots {subject_phrase} and points, eyes wide",
                 "gaze": f"wide-eyed at {subject_phrase}", "performer": "child", "camera_address": False},
                {"action": f"{_NAME_PLACEHOLDER} crouches down to get a better look at {subject_phrase}",
                 "gaze": f"down at {subject_phrase}", "performer": "child", "camera_address": False},
                {"action": f"The kids gather around {subject_phrase}, leaning in together",
                 "gaze": f"at {subject_phrase}", "performer": "all", "camera_address": False},
                {"action": f"{subject_phrase} rests quietly nearby, waiting to be found",
                 "gaze": f"on {subject_phrase} itself", "performer": "story_subject", "camera_address": False},
            ]
            return _tag_roles(beats) if story else beats
        # SOLO beats must never mention other children in the ACTION or the
        # GAZE. Measured live (pixar-full render): "waves hello to the other
        # kids" / gaze "toward the other kids" put "The child looks toward the
        # other kids" into a single-child i2v prompt — and LTX painted those
        # kids IN, as non-cast children invading the solo shot (5 of 16 takes).
        # Off-screen friends are implied by the group shots around it; a solo
        # beat keeps its focus on the child's own body, prop or surroundings.
        beats = [
            {"action": f"{_NAME_PLACEHOLDER} skips in and looks around with a big excited grin",
             "gaze": "around the sunny yard", "performer": "child", "camera_address": False},
            {"action": f"{_NAME_PLACEHOLDER} waves hello with both hands held high",
             "gaze": "out across the sunny yard", "performer": "child", "camera_address": False},
            {"action": "The kids gather in a happy little group, ready to begin",
             "gaze": "at each other", "performer": "all", "camera_address": False},
            {"action": f"{_NAME_PLACEHOLDER} bounces on their toes, eager to start",
             "gaze": "ahead across the yard", "performer": "child", "camera_address": False},
        ]
        return _tag_roles(beats) if story else beats

    if stage == "try":
        if story:
            # The story-mode partner moment: the ONE beat in the middle verses
            # where a second child belongs. It replaces the ensemble
            # "take turns" beat, so the planner stages a deliberate TWO-child
            # shot ([protagonist, partner]) instead of a group — or, worse,
            # a solo shot whose text implies invisible friends.
            third = {
                "action": f"{_NAME_PLACEHOLDER} passes {prop_phrase} gently to {_PARTNER_PLACEHOLDER}",
                "gaze": f"on {prop_phrase}", "prop": prop_phrase,
                "performer": "child", "camera_address": False, "role": "partner",
            }
        else:
            third = {
                "action": f"The kids take turns with {prop_phrase} together",
                "gaze": f"on {prop_phrase}", "prop": prop_phrase,
                "performer": "all", "camera_address": False,
            }
        beats = [
            {"action": f"{_NAME_PLACEHOLDER} picks up {prop_phrase} and tries it out",
             "gaze": f"down at {prop_phrase}", "prop": prop_phrase, "performer": "child", "camera_address": False},
            {"action": f"{_NAME_PLACEHOLDER} tips {prop_phrase} carefully, watching what happens",
             "gaze": f"on {prop_phrase}", "prop": prop_phrase, "performer": "child", "camera_address": False},
            third,
        ]
        if has_subject:
            beats.append({
                "action": f"{subject_phrase} leans a little nearer, as if curious",
                "gaze": f"on {subject_phrase}", "performer": "story_subject", "camera_address": False,
            })
        else:
            beats.append({
                "action": f"{_NAME_PLACEHOLDER} holds {prop_phrase} out in front, offering a turn",
                "gaze": f"on {prop_phrase}", "prop": prop_phrase, "performer": "child", "camera_address": False,
            })
        return _tag_roles(beats) if story else beats

    if stage == "grow":
        if has_subject:
            group_or_solo = (
                {"action": f"{_NAME_PLACEHOLDER} leans in close to see how far it has come",
                 "gaze": f"at {subject_phrase}", "performer": "child", "camera_address": False}
                if story else
                {"action": "The kids lean in together to see how far it has come",
                 "gaze": f"at {subject_phrase}", "performer": "all", "camera_address": False}
            )
            beats = [
                # STATE, not process: "changes ..." made the i2v stage morph the
                # subject mid-shot (measured: s06 of the 2026-07-25 baa-baa render
                # started as a WHITE sheep and turned black within one take).
                # Describe the grown RESULT; never an ongoing transformation.
                {"action": f"{subject_phrase} looks a little fuller and happier than before",
                 "gaze": f"on {subject_phrase}", "performer": "story_subject", "camera_address": False},
                {"action": f"{_NAME_PLACEHOLDER} claps once in delight",
                 "gaze": f"on {subject_phrase}", "performer": "child", "camera_address": False},
                group_or_solo,
                {"action": f"{_NAME_PLACEHOLDER} pats the air gently, cheering on {subject_phrase}",
                 "gaze": f"on {subject_phrase}", "performer": "child", "camera_address": False},
            ]
            return _tag_roles(beats) if story else beats
        # "swings both arms" made LTX invent a playground SWING (vertical ropes
        # appeared around the child, measured twice on the storymode render) —
        # "swing" as a solo verb pulls the object prior. Rock/raise carry the
        # same energy with no object attached.
        group_or_solo = (
            {"action": f"{_NAME_PLACEHOLDER} rocks side to side with both arms raised high",
             "gaze": "up at their own raised hands", "performer": "child", "camera_address": False}
            if story else
            {"action": "The kids swing their arms together, having fun",
             "gaze": "at each other", "performer": "all", "camera_address": False}
        )
        beats = [
            {"action": f"{_NAME_PLACEHOLDER} claps out the beat, delighted",
             "gaze": "down at their own clapping hands", "performer": "child", "camera_address": False},
            {"action": f"{_NAME_PLACEHOLDER} jumps up with a big happy smile",
             "gaze": "up into the sunny sky", "performer": "child", "camera_address": False},
            group_or_solo,
            {"action": f"{_NAME_PLACEHOLDER} spins around once, giggling",
             "gaze": "out across the yard", "performer": "child", "camera_address": False},
        ]
        return _tag_roles(beats) if story else beats

    if stage == "explore":
        if has_subject:
            group_or_solo = (
                {"action": f"{_NAME_PLACEHOLDER} points out a tiny detail on {subject_phrase}",
                 "gaze": f"at {subject_phrase}", "performer": "child", "camera_address": False}
                if story else
                {"action": f"The kids point out little details on {subject_phrase}",
                 "gaze": f"at {subject_phrase}", "performer": "all", "camera_address": False}
            )
            beats = [
                {"action": f"{_NAME_PLACEHOLDER} peeks at {subject_phrase} from a new angle",
                 "gaze": f"at {subject_phrase}", "performer": "child", "camera_address": False},
                {"action": f"{_NAME_PLACEHOLDER} tiptoes slowly around {subject_phrase}",
                 "gaze": f"at {subject_phrase}", "performer": "child", "camera_address": False},
                group_or_solo,
                {"action": f"{_NAME_PLACEHOLDER} reaches out to gently touch {subject_phrase}",
                 "gaze": f"at {subject_phrase}", "performer": "child", "camera_address": False},
            ]
            return _tag_roles(beats) if story else beats
        group_or_solo = (
            {"action": f"{_NAME_PLACEHOLDER} wiggles all ten fingers, giggling",
             "gaze": "down at their wiggling fingers", "performer": "child", "camera_address": False}
            if story else
            {"action": "The kids wiggle their fingers together, giggling",
             "gaze": "at each other", "performer": "all", "camera_address": False}
        )
        beats = [
            {"action": f"{_NAME_PLACEHOLDER} peeks around with a curious smile",
             "gaze": "off to one side", "performer": "child", "camera_address": False},
            {"action": f"{_NAME_PLACEHOLDER} tiptoes in a happy little circle",
             "gaze": "down at their feet", "performer": "child", "camera_address": False},
            group_or_solo,
            {"action": f"{_NAME_PLACEHOLDER} reaches up high with both hands",
             "gaze": "up at their hands", "performer": "child", "camera_address": False},
        ]
        return _tag_roles(beats) if story else beats

    if stage == "share":
        # share keeps its group beat in BOTH modes — story mode's one allowed
        # mid-song shared moment (the red thread gathers, then returns to the
        # protagonist).
        if has_subject:
            beats = [
                {"action": f"{_NAME_PLACEHOLDER} skips a happy little loop around {subject_phrase}",
                 "gaze": f"at {subject_phrase}", "performer": "child", "camera_address": False},
                {"action": f"{_NAME_PLACEHOLDER} points at {subject_phrase} with a proud smile",
                 "gaze": f"at {subject_phrase}", "performer": "child", "camera_address": False},
                {"action": f"The kids take gentle turns beside {subject_phrase}",
                 "gaze": f"at {subject_phrase}", "performer": "all", "camera_address": False},
                {"action": f"{_NAME_PLACEHOLDER} pats {subject_phrase} softly and smiles",
                 "gaze": f"at {subject_phrase}", "performer": "child", "camera_address": False},
            ]
            return _tag_roles(beats) if story else beats
        beats = [
            {"action": f"{_NAME_PLACEHOLDER} beams a big friendly smile, rocking side to side",
             "gaze": "off to one side with a smile", "performer": "child", "camera_address": False},
            {"action": f"{_NAME_PLACEHOLDER} skips forward with happy little steps",
             "gaze": "ahead across the yard", "performer": "child", "camera_address": False},
            {"action": "The kids hold hands together in a happy circle",
             "gaze": "at each other", "performer": "all", "camera_address": False},
            {"action": f"{_NAME_PLACEHOLDER} waves both hands in a big friendly hello",
             "gaze": "up at their waving hands", "performer": "child", "camera_address": False},
        ]
        return _tag_roles(beats) if story else beats

    # celebrate
    if has_subject:
        beats = [
            {"action": f"The kids join hands and circle {subject_phrase}, cheering happily",
             "gaze": f"at {subject_phrase}", "performer": "all", "camera_address": False},
            {"action": f"{_NAME_PLACEHOLDER} hops in place, beaming with pride",
             "gaze": f"at {subject_phrase}", "performer": "child", "camera_address": False},
            {"action": f"The kids wave to {subject_phrase} one last time as the song ends",
             "gaze": f"at {subject_phrase}", "performer": "all", "camera_address": False},
            {"action": f"{subject_phrase} seems to sway along happily in the celebration",
             "gaze": f"on {subject_phrase}", "performer": "story_subject", "camera_address": False},
        ]
    else:
        beats = [
            {"action": "The kids join hands and dance in a circle together, cheering happily",
             "gaze": "at each other", "performer": "all", "camera_address": False},
            {"action": f"{_NAME_PLACEHOLDER} hops in place, beaming with pride",
             "gaze": "up at the sky, mid-hop", "performer": "child", "camera_address": False},
            {"action": "The kids wave their hands high in the air as the song ends",
             "gaze": "at each other", "performer": "all", "camera_address": False},
            {"action": f"{_NAME_PLACEHOLDER} takes a bow with a big cheerful grin",
             "gaze": "down toward the grass in a bow", "performer": "child", "camera_address": False},
        ]
    # Two more distinct celebrate beats: the final verse is routinely the
    # longest (it absorbs the outro), and a 6-7-shot celebration over the
    # 4 beats above used to exhaust the pool — the planner's per-verse dedupe
    # then falls back to canned filler for the overflow slots, so the richer
    # this pool, the more of the celebration stays on the story's arc.
    beats += [
        {"action": f"{_NAME_PLACEHOLDER} twirls once with both arms out, giggling",
         "gaze": "out across the yard", "performer": "child", "camera_address": False},
        {"action": "The kids trade gentle high-fives all around",
         "gaze": "on each other's hands", "performer": "all", "camera_address": False},
    ]
    return _tag_roles(beats) if story else beats


def fallback_storyboard(song, cfg=None):
    """Deterministic, LLM-free storyboard: a fixed discover -> try -> grow ->
    celebrate arc (see `_stage_for_verse`), tied to the song's own
    `director.song_story_subject` when it has one, else a generic
    shared-play arc built around a topic noun pulled from the title
    (`_topic_noun`).

    `cfg` (optional) selects the staging mode: `kidsong.staging="story"`
    emits the story-driven casting board — beats tagged with roles
    (protagonist / partner / all), group beats only at gathering moments —
    while `cfg=None` or any other value keeps the ensemble board BYTE-
    IDENTICAL to the pre-story behavior (load-bearing: existing callers and
    fixtures never see a role key).

    This is `build_storyboard`'s safety net when the LLM cannot produce a
    passing attempt, so it must itself always pass `_storyboard_verdict` —
    asserted directly in tests/test_kidsong_storyboard.py for both a
    story-subject song and a subject-less song, in both staging modes — and
    never raises. Deterministic: the same `song` + mode always yields the
    byte-identical board.
    """
    try:
        song = song if isinstance(song, dict) else {}
        story = _staging_mode(cfg) == "story"
        song_verses = song.get("verses") or []
        n = len(song_verses) if song_verses else 1

        try:
            from pipeline.kidsong import director

            story_subject = director.song_story_subject(song)
        except Exception:
            story_subject = None
        has_subject = isinstance(story_subject, str) and bool(story_subject.strip())
        story_subject = story_subject.strip() if has_subject else None

        topic = _topic_noun(song)
        subject_phrase = story_subject if has_subject else f"the {topic}"
        prop_phrase = _fallback_prop(topic)

        verses_out = []
        for i in range(n):
            stage = _stage_for_verse(i, n)
            verses_out.append({
                "verse": i,
                "beat_goal": _STAGE_GOAL_TEMPLATES[stage].format(subject=subject_phrase),
                "beats": _stage_beats(stage, subject_phrase, prop_phrase, has_subject, story=story),
            })
        return {"verses": verses_out}
    except Exception:
        log.exception("storyboard: fallback_storyboard failed unexpectedly")
        return {"verses": []}


if __name__ == "__main__":
    import sys

    demo_song = {
        "title": sys.argv[1] if len(sys.argv) > 1 else "Zuri's Sunflower Garden Day",
        "description": "test",
        "tags": ["kids song"],
        "characters": "Three adorable Black toddlers: Zuri, Kofi and Nala.",
        "verses": [
            {"lines": ["A little seed goes in the ground"], "scene": "A sunny backyard garden with a small sunflower seedling."},
            {"lines": ["Water it each sunny day"], "scene": "The kids water the sunflower seedling with little watering cans."},
            {"lines": ["Up it grows so tall and bright"], "scene": "The sunflower has grown much taller, its face turned to the sun."},
            {"lines": ["Sing and dance around the sunflower"], "scene": "The kids dance around the tall blooming sunflower."},
        ],
    }
    board = fallback_storyboard(demo_song)
    print(json.dumps(board, indent=2, ensure_ascii=False))
    verdict = _storyboard_verdict(board, demo_song)
    print(json.dumps(verdict, indent=2, ensure_ascii=False))
    assert verdict["accept"], verdict["reasons"]
    print("PASS")
