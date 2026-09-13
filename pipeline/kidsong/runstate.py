"""
kidsong.runstate — persisted run state so a crash costs one shot, not a run.

A director-mode run renders ~16 shots at ~2.5 minutes each. When the render
loop died at shot 8 the eight finished takes were orphaned: a re-run minted a
brand-new ``<base>`` and re-rendered everything from scratch, throwing away 40
minutes of GPU time.

This module makes a run resumable by treating ``output/<base>-shots.json`` as
the run's ledger rather than a write-once plan. Each shot carries::

    status   "planned" | "rendered" | "rejected" | "failed"
    take     path to the chosen take (absolute), once one is accepted
    attempts how many takes have been rendered for this shot, ever
    score    the chosen take's review score
    verdict  "accepted" | "rejected" | "unknown" — what QC said about `take`

Resume reads that ledger, keeps every shot that already has an *accepted* take,
and renders only what is missing.

Three rules are load-bearing:

* **Only QC decides "done".** ``rendered`` means a take passed the producer's
  gate. A shot whose every take was rejected is recorded as ``rejected`` — its
  best take is still on record (so the cut is never blocked) but the shot stays
  pending, so a resume renders it again instead of silently shipping footage the
  producer turned down.

* **Nothing is ever deleted.** New takes are written to the next free
  ``_a<N>.mp4`` index; existing media is only ever read (channel-inventory
  rule).
* **Failures are bounded.** A shot that has already burned ``max_attempts``
  takes is marked ``failed`` and is not retried again on the next resume, so an
  unrenderable shot can never loop a scheduler forever.
"""
import json
import logging
import os
import re
import time

log = logging.getLogger("kidsong.runstate")

SHOTS_SUFFIX = "-shots.json"
SONG_SUFFIX = "-song.json"
SHOTS_DIR_SUFFIX = "-shots"

STATUS_PLANNED = "planned"
STATUS_RENDERED = "rendered"
#: A take exists and is the best we have, but QC rejected every attempt. The
#: shot is NOT satisfied: resume retries it (until `max_attempts`), while the
#: recorded take remains available so the cut can still be assembled.
STATUS_REJECTED = "rejected"
STATUS_FAILED = "failed"

VERDICT_ACCEPTED = "accepted"
VERDICT_REJECTED = "rejected"
#: No verdict survives on disk for this take (pre-ledger run, or a run whose
#: review log was lost). Adopted rather than re-rendered, but recorded as
#: unverified rather than pretending it passed.
VERDICT_UNKNOWN = "unknown"

#: Hard ceiling on takes per shot across all resumes of a run. Beyond this the
#: shot is marked failed and resume stops retrying it.
DEFAULT_MAX_ATTEMPTS = 6

_TAKE_RE = re.compile(r"^(?P<shot>.+)_a(?P<n>\d+)\.mp4$", re.IGNORECASE)


# ------------------------------------------------------------------ paths ---
def shots_json_path(out_dir, base):
    return os.path.join(str(out_dir), f"{base}{SHOTS_SUFFIX}")


def song_json_path(out_dir, base):
    return os.path.join(str(out_dir), f"{base}{SONG_SUFFIX}")


def shots_dir_path(out_dir, base):
    return os.path.join(str(out_dir), f"{base}{SHOTS_DIR_SUFFIX}")


def wav_path(out_dir, base):
    return os.path.join(str(out_dir), f"{base}.wav")


#: Highest version suffix we will mint before giving up. A run needing more
#: than this is a bug, and silently clobbering would be the worse answer.
MAX_VERSIONS = 500


def reserve_output_path(path, companions=(), max_versions=MAX_VERSIONS):
    """Return `path`, or the next free ``<stem>-v<N><ext>`` sibling if it exists.

    Generated media in ``output/`` is channel inventory: a finished, approved
    episode must never be replaced by a later re-cut. A resume re-assembles the
    same ``<base>``, so its promotion target is exactly the name an earlier,
    already-approved cut occupies — hence every write into ``output/`` that is
    not a ledger goes through here first.

    Versions start at ``-v2`` (the plain name is v1) and are scanned upward, so
    the choice is deterministic and matches the ``-v2``/``-v3`` files a human
    already produced by hand for this reason.

    `companions` are suffixes that must be free as well (e.g. ``".intro.json"``
    for prepend_intro's sidecar), so a chosen name never lands on top of an
    orphaned sidecar from an earlier cut.
    """
    path = str(path)
    if _free(path, companions):
        return path
    stem, ext = os.path.splitext(path)
    for n in range(2, int(max_versions) + 1):
        candidate = f"{stem}-v{n}{ext}"
        if _free(candidate, companions):
            return candidate
    raise RuntimeError(
        f"Refusing to overwrite {path}: every version up to -v{max_versions} "
        "is already taken."
    )


def _free(path, companions=()):
    # TRAP: a name counts as taken if ANY companion exists, even when `path`
    # itself is gone — e.g. a Final/<base>.mp4 that was hand-deleted for disk
    # space but left its `<base>.mp4.intro.json` sidecar behind permanently
    # burns that base name (this happened to 3 finals). Do not hand-delete a
    # shipped final; remove one, mp4 + sidecars together, only via
    # `remove_final_episode()` below. This function's behavior is unchanged.
    if os.path.exists(path):
        return False
    return not any(os.path.exists(path + str(s)) for s in companions)


#: Sidecar suffixes that travel with a Final/<base>.mp4 and must be removed
#: alongside it — see `remove_final_episode` and the `_free` trap above.
FINAL_COMPANION_SUFFIXES = (".intro.json",)


def remove_final_episode(mp4_path):
    """Remove a shipped Final/<base>.mp4 AND its companion sidecar(s) together.

    This is the ONLY sanctioned way to remove a shipped final. Deleting just
    the mp4 by hand (as happened to 3 finals, to reclaim disk space) strands
    orphaned sidecars (currently `<mp4>.intro.json`) that `_free()` treats as
    proof the name is still taken — permanently burning that base name even
    though the video is gone. Reclaim disk space from shots dirs, staging
    files, or wavs first; only reach for this once a final truly needs to go.

    Raises `ValueError` if `mp4_path` is not inside a directory named "Final",
    refusing anything that isn't obviously a shipped episode.
    """
    mp4_path = str(mp4_path)
    parent = os.path.basename(os.path.dirname(os.path.abspath(mp4_path)))
    if parent != "Final":
        raise ValueError(
            f"refusing to remove {mp4_path!r}: not inside a directory named "
            f"'Final' (parent is {parent!r})"
        )

    removed = []
    for target in (mp4_path, *(mp4_path + s for s in FINAL_COMPANION_SUFFIXES)):
        if os.path.exists(target):
            os.remove(target)
            removed.append(target)
            log.info("removed %s", target)
    return removed


# ------------------------------------------------------------------- I/O ----
def _read_json(path, default=None):
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        return default


def save_shotlist(out_dir, base, shotlist):
    """Write the shot ledger atomically.

    Atomic because this file is rewritten after *every* shot: a crash during
    the write would otherwise leave truncated JSON and make the run
    unresumable — precisely the failure this module exists to prevent.
    """
    path = shots_json_path(out_dir, base)
    os.makedirs(str(out_dir), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(shotlist, f, indent=2, ensure_ascii=False)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)
    return path


def load_shotlist(out_dir, base):
    return _read_json(shots_json_path(out_dir, base))


def load_song(out_dir, base):
    return _read_json(song_json_path(out_dir, base))


# ------------------------------------------------------------------ takes ---
def existing_takes(shots_dir, shot_id):
    """Every take already on disk for `shot_id`, oldest attempt index first."""
    out = []
    try:
        names = os.listdir(str(shots_dir))
    except OSError:
        return out
    for name in names:
        m = _TAKE_RE.match(name)
        if m and m.group("shot") == shot_id:
            out.append((int(m.group("n")), os.path.join(str(shots_dir), name)))
    return [p for _, p in sorted(out)]


def next_attempt_index(shots_dir, shot_id):
    """First unused ``_a<N>`` index — new takes never overwrite old ones."""
    takes = existing_takes(shots_dir, shot_id)
    if not takes:
        return 0
    highest = max(int(_TAKE_RE.match(os.path.basename(p)).group("n")) for p in takes)
    return highest + 1


def _usable(path):
    """A take counts only if it is present and non-empty.

    A zero-byte file is what a render interrupted mid-write leaves behind —
    exactly the shot the crash happened on — so it must not be mistaken for
    completed work.
    """
    try:
        return os.path.isfile(path) and os.path.getsize(path) > 0
    except OSError:
        return False


def shot_is_done(shot, shots_dir):
    """True if `shot` has a usable take that QC actually accepted.

    "Has a file on disk" is not the same as "is finished". A shot whose every
    take was rejected must come back as pending on the next resume, or the
    resume ships footage the producer already turned down.
    """
    if shot.get("reuse_of"):
        return False  # reuse slots are resolved from their source, not rendered
    if shot.get("status") != STATUS_RENDERED:
        return False
    # `last_verdict` is what the buggy build wrote instead of a status: ledgers
    # from that build say status=rendered *and* last_verdict=rejected. Honour
    # the verdict, so those runs re-render rather than reuse a rejected take.
    if shot.get("last_verdict") == VERDICT_REJECTED:
        return False
    if shot.get("verdict") == VERDICT_REJECTED:
        return False
    return _usable(shot.get("take") or "")


def mark_rendered(shot, take_path, score=None, attempts=None, verdict=VERDICT_ACCEPTED):
    """Record `take_path` as this shot's best take.

    `verdict` decides whether the shot counts as *satisfied*. Only
    ``VERDICT_ACCEPTED`` sets ``status = rendered``; a rejected take is recorded
    (the cut still needs something to show) but leaves the shot pending.
    """
    verdict = verdict if verdict in (VERDICT_ACCEPTED, VERDICT_REJECTED) else VERDICT_UNKNOWN
    shot["status"] = STATUS_REJECTED if verdict == VERDICT_REJECTED else STATUS_RENDERED
    shot["take"] = take_path
    shot["verdict"] = verdict
    # Superseded by `status` + `verdict`; drop it so no reader sees a stale value.
    shot.pop("last_verdict", None)
    if score is not None:
        shot["score"] = round(float(score), 4)
    if attempts is not None:
        shot["attempts"] = int(attempts)
    return shot


def mark_failed(shot, reason=None, attempts=None):
    shot["status"] = STATUS_FAILED
    if reason:
        shot["failure_reason"] = str(reason)[:500]
    if attempts is not None:
        shot["attempts"] = int(attempts)
    return shot


def shot_is_exhausted(shot, max_attempts=DEFAULT_MAX_ATTEMPTS):
    """True if this shot must not be retried again on resume."""
    if shot.get("status") == STATUS_FAILED:
        return True
    return int(shot.get("attempts", 0) or 0) >= int(max_attempts)


def load_take_verdicts(shots_dir):
    """Every QC verdict that survives on disk, keyed by take name (``s09_a1``).

    Two sources, both under ``<shots-dir>/review``:

    * ``review_log.json`` — the composite verdict the gate applied AT RUN
      TIME, one entry per attempt. Read first.
    * ``<take>.response.json`` — the vision reviewer's own verdict file. Read
      LAST, so it WINS where the two disagree: a response backfilled after an
      unattended run (the documented QM-010 recovery path — the auto-reviewer
      or a human answering the queue later) is by definition newer than the
      run's own log entry, which for an unanswered take is only the timeout
      heuristic fallback. This matches `review.vision_coverage`, which already
      treats response files as the authority; before this alignment a
      backfilled vision REJECTION was laundered back to accepted on resume
      (measured live: three morphing takes vision-rejected post-run were
      re-adopted as accepted and never re-rendered).

    Neither is guaranteed to exist (pre-ledger runs, and runs killed before the
    log was flushed), hence the {} fallback everywhere.
    """
    verdicts = {}
    review_dir = os.path.join(str(shots_dir), "review")

    def _record(take, verdict):
        if not take or not isinstance(verdict, dict):
            return
        if "accept" not in verdict and "score" not in verdict:
            return
        verdicts[take] = {
            "accept": bool(verdict.get("accept")),
            "score": _as_float(verdict.get("score")),
        }

    entries = _read_json(os.path.join(review_dir, "review_log.json"), default=[])
    if isinstance(entries, list):
        for entry in entries:
            if not isinstance(entry, dict) or entry.get("stage"):
                continue  # script/shotlist/cut verdicts are not per-take
            shot_id = entry.get("shot") or entry.get("shot_id")
            attempt = entry.get("attempt")
            if shot_id is None or attempt is None:
                continue
            _record(f"{shot_id}_a{int(attempt)}", entry.get("verdict"))

    try:
        names = os.listdir(review_dir)
    except OSError:
        names = []
    for name in names:
        if name.endswith(".response.json"):
            _record(name[: -len(".response.json")], _read_json(os.path.join(review_dir, name)))
    return verdicts


def _as_float(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _take_name(path):
    return os.path.splitext(os.path.basename(path))[0]


def best_take(shots_dir, shot_id, verdicts=None):
    """The take a resume should adopt for `shot_id`: ``(path, verdict, score)``.

    "Newest" is the wrong answer: the last take on disk is usually the LAST
    REJECTED attempt of a shot the run died on. Preference order:

    1. an accepted take — highest score among them;
    2. otherwise the highest-scoring take with a known verdict;
    3. otherwise (no verdicts survive at all) the highest index, flagged
       ``VERDICT_UNKNOWN`` rather than pretending it passed QC.

    Returns ``(None, None, None)`` when the shot has no usable media.
    """
    takes = [p for p in existing_takes(shots_dir, shot_id) if _usable(p)]
    if not takes:
        return None, None, None
    if verdicts is None:
        verdicts = load_take_verdicts(shots_dir)

    scored = [(p, verdicts.get(_take_name(p))) for p in takes]
    known = [(p, v) for p, v in scored if v is not None]

    accepted = [(p, v) for p, v in known if v["accept"]]
    if accepted:
        path, v = max(accepted, key=lambda pv: (pv[1]["score"], _attempt_index(pv[0])))
        return path, VERDICT_ACCEPTED, v["score"]
    if known:
        path, v = max(known, key=lambda pv: (pv[1]["score"], _attempt_index(pv[0])))
        return path, VERDICT_REJECTED, v["score"]
    return takes[-1], VERDICT_UNKNOWN, None


def _attempt_index(path):
    m = _TAKE_RE.match(os.path.basename(path))
    return int(m.group("n")) if m else -1


def recover_takes(shotlist, shots_dir):
    """Adopt takes found on disk that the ledger does not know about.

    The crash that motivated this module happened *before* per-shot ledger
    writes existed, so a pre-existing run directory has good takes but a ledger
    that still says "planned". Rather than re-render them, adopt the *best*
    take (see `best_take`) — never simply the newest, which is typically the
    attempt QC had just rejected when the run died.

    A shot whose surviving verdicts are all rejections is adopted as
    ``rejected``: its media is on record for the cut, but the shot stays
    pending so this resume re-renders it. Returns the number of shots recovered.
    """
    recovered = 0
    verdicts = load_take_verdicts(shots_dir)
    for shot in shotlist.get("shots", []):
        if shot.get("reuse_of") or shot_is_done(shot, shots_dir):
            continue
        if shot.get("status") == STATUS_FAILED:
            continue
        path, verdict, score = best_take(shots_dir, shot.get("id", ""), verdicts)
        if not path:
            continue
        # The ledger is a verdict source too, and the only one that survives
        # when the review log was lost. "No verdict on disk" must never be
        # allowed to launder a shot we already know QC turned down.
        if verdict == VERDICT_UNKNOWN and VERDICT_REJECTED in (
            shot.get("status"), shot.get("verdict"), shot.get("last_verdict")
        ):
            verdict = VERDICT_REJECTED
        attempts = max(
            len([p for p in existing_takes(shots_dir, shot.get("id", "")) if _usable(p)]),
            int(shot.get("attempts", 0) or 0),
        )
        mark_rendered(shot, path, score=score, attempts=attempts, verdict=verdict)
        if verdict == VERDICT_ACCEPTED or verdict == VERDICT_UNKNOWN:
            recovered += 1
    return recovered


# -------------------------------------------------------------- discovery ---
def find_resumable(out_dir, max_attempts=DEFAULT_MAX_ATTEMPTS):
    """List interrupted runs in `out_dir`, newest first.

    A run is resumable when it has a shot ledger and at least one shot still
    needing work that has not exhausted its attempts. Runs whose video already
    reached ``Final/`` are excluded — they finished.
    """
    out_dir = str(out_dir)
    runs = []
    try:
        names = sorted(os.listdir(out_dir))
    except OSError:
        return runs

    final_dir = os.path.join(out_dir, "Final")
    for name in names:
        if not name.endswith(SHOTS_SUFFIX):
            continue
        base = name[: -len(SHOTS_SUFFIX)]
        shotlist = _read_json(os.path.join(out_dir, name))
        if not isinstance(shotlist, dict) or not shotlist.get("shots"):
            continue

        sdir = shots_dir_path(out_dir, base)
        shots = shotlist["shots"]
        unique = [s for s in shots if not s.get("reuse_of")]
        done, exhausted = 0, 0
        verdicts = load_take_verdicts(sdir)
        for s in unique:
            if shot_is_done(s, sdir) or _adoptable_take(s, sdir, verdicts):
                done += 1
            elif shot_is_exhausted(s, max_attempts):
                exhausted += 1
        pending = len(unique) - done - exhausted

        finished = os.path.exists(os.path.join(final_dir, base + ".mp4"))
        runs.append(
            {
                "base": base,
                "title": (load_song(out_dir, base) or {}).get("title"),
                "shots_total": len(shots),
                "unique_total": len(unique),
                "unique_done": done,
                "unique_pending": pending,
                "unique_exhausted": exhausted,
                "has_audio": os.path.exists(wav_path(out_dir, base)),
                "finished": finished,
                "resumable": bool(pending and not finished),
                "modified": _mtime(os.path.join(out_dir, name)),
            }
        )
    runs.sort(key=lambda r: r["modified"] or "", reverse=True)
    return runs


def _adoptable_take(shot, shots_dir, verdicts=None):
    """Media a resume would count as satisfying `shot` without re-rendering.

    A take QC rejected is *not* adoptable: the shot still needs work, so it
    belongs in the pending column, not the done column.
    """
    if shot.get("status") == STATUS_REJECTED:
        return None
    path, verdict, _ = best_take(shots_dir, shot.get("id", ""), verdicts)
    return path if verdict in (VERDICT_ACCEPTED, VERDICT_UNKNOWN) else None


def _mtime(path):
    try:
        return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(os.path.getmtime(path)))
    except OSError:
        return None


def format_resumable(runs):
    """Human/scheduler-readable table for ``--list-resumable``."""
    if not runs:
        return "No interrupted kidsong runs found."
    lines = [
        f"{'BASE':<58} {'DONE':>9}  {'PEND':>4}  {'FAIL':>4}  {'AUDIO':>5}  STATUS",
        "-" * 104,
    ]
    for r in runs:
        status = (
            "finished" if r["finished"]
            else "RESUMABLE" if r["resumable"]
            else "blocked (all remaining shots exhausted)" if r["unique_exhausted"]
            else "nothing pending"
        )
        lines.append(
            f"{r['base']:<58} {r['unique_done']:>4}/{r['unique_total']:<4} "
            f"{r['unique_pending']:>4}  {r['unique_exhausted']:>4}  "
            f"{'yes' if r['has_audio'] else 'no':>5}  {status}"
        )
    lines.append("")
    lines.append("Resume with:  python -m pipeline.kidsong.generate --resume <BASE>")
    return "\n".join(lines)
