"""
kidsong.status — Filesystem scanner for the live production dashboard.

Kidsong videos are produced by DETACHED processes (director_run / batch
runners spawned outside Flask), so the web UI has no in-process job state to
read. Instead, this module infers progress by walking `output/` (cfg
paths.output_dir) for the artifacts each pipeline stage leaves behind — see
`pipeline.kidsong.generate._generate_director` for the stage order this
mirrors:

  1. `<base>.wav`                        -> song recorded
  2. `<base>-shots.json`                 -> shots planned
  3. `<base>-shots/*.mp4`                -> rendering shots (takes s00_a0..)
  4. `Final/<base>.mp4`                  -> done

Every helper here is best-effort and never raises: a missing/partial/corrupt
file just means that piece of data degrades to `None`/empty rather than
blowing up the whole scan (one bad video shouldn't take down the dashboard
for every other one). No Flask imports — this is pure data-gathering so it
can be unit-tested and reused from a CLI if needed.
"""
import json
import os
import re
import subprocess
import time

from pipeline.config import abspath

_SHOT_TAKE_RE = re.compile(r"^(?P<shot_id>s\d+)_a(?P<attempt>\d+)\.mp4$")
_BATCH_HEADER_RE = re.compile(r"^===\s*BATCH:\s*(.*?)\s*===\s*$")


# ------------------------------------------------------------------ small utils ---
def _safe_read_json(path):
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError):
        return None


def _safe_stat(path):
    try:
        return os.stat(path)
    except OSError:
        return None


def _tail_lines(path, n=15):
    """Return the last `n` lines of a text file, or [] if unreadable."""
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            lines = f.readlines()
    except OSError:
        return []
    return [ln.rstrip("\n") for ln in lines[-n:]]


def _human_size(num_bytes):
    if num_bytes is None:
        return None
    size = float(num_bytes)
    for unit in ("B", "KB", "MB", "GB"):
        if size < 1024 or unit == "GB":
            return f"{size:.1f}{unit}" if unit != "B" else f"{int(size)}{unit}"
        size /= 1024
    return f"{size:.1f}GB"


# ---------------------------------------------------------------- video grouping ---
def _discover_bases(out_dir):
    """Find every `<stamp>-kidsong-<slug>` base name under out_dir.

    A base is identified by any of its telltale artifacts existing: the
    voice wav, the shots.json, the shots dir, or a Final/<base>.mp4. Returns
    a sorted (newest-stamp-first) list of base names.
    """
    bases = set()
    try:
        entries = os.listdir(out_dir)
    except OSError:
        entries = []

    for name in entries:
        if not re.match(r"^\d{8}-\d{6}-kidsong-", name):
            continue
        if name.endswith(".wav"):
            bases.add(name[: -len(".wav")])
        elif name.endswith("-shots.json"):
            bases.add(name[: -len("-shots.json")])
        elif name.endswith("-shots") and os.path.isdir(os.path.join(out_dir, name)):
            bases.add(name[: -len("-shots")])
        elif name.endswith("-music.wav"):
            bases.add(name[: -len("-music.wav")])

    final_dir = os.path.join(out_dir, "Final")
    try:
        for name in os.listdir(final_dir):
            if name.endswith(".mp4") and re.match(r"^\d{8}-\d{6}-kidsong-", name):
                bases.add(name[: -len(".mp4")])
    except OSError:
        pass

    return sorted(bases, reverse=True)


def _shot_takes(shots_dir):
    """Map shot_id -> sorted list of {attempt, path, mtime} from `<id>_a<n>.mp4` files."""
    takes = {}
    try:
        names = os.listdir(shots_dir)
    except OSError:
        return takes
    for name in names:
        m = _SHOT_TAKE_RE.match(name)
        if not m:
            continue
        shot_id = m.group("shot_id")
        path = os.path.join(shots_dir, name)
        st = _safe_stat(path)
        takes.setdefault(shot_id, []).append(
            {
                "attempt": int(m.group("attempt")),
                "path": path,
                "mtime": st.st_mtime if st else None,
            }
        )
    for shot_id in takes:
        takes[shot_id].sort(key=lambda t: t["attempt"])
    return takes


def _latest_verdicts(review_log_path):
    """Return {shot_id: latest verdict entry} from a review_log.json, or {} if absent/bad."""
    entries = _safe_read_json(review_log_path)
    if not isinstance(entries, list):
        return {}
    latest = {}
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        shot_id = entry.get("shot")
        if shot_id is None:
            continue
        latest[shot_id] = entry  # later entries overwrite — log is append-order
    return latest


def _pending_gate_count(review_dir):
    """How many review requests in `review_dir` have no response yet.

    Mirrors `review_queue._pending_requests`: a gate is pending exactly when
    `<name>.request.json` exists and `<name>.response.json` does not.
    """
    try:
        names = os.listdir(review_dir)
    except OSError:
        return 0
    pending = 0
    for name in names:
        if not name.endswith(".request.json"):
            continue
        response = name[: -len(".request.json")] + ".response.json"
        if response not in names:
            pending += 1
    return pending


def _relpath(out_dir, path):
    try:
        return os.path.relpath(path, out_dir).replace(os.sep, "/")
    except ValueError:
        return None


# --------------------------------------------------------------------- per-video ---
def _scan_one(base, out_dir):
    """Build the status dict for a single video base name. Never raises."""
    info = {
        "base": base,
        "stage": "unknown",
        "label": "Unknown",
        "final_video": None,
        "final_size": None,
        "final_mtime": None,
        "final_mtime_str": None,
        "audio": None,
        "bpm": None,
        "shots_total": None,
        "shots_unique": None,
        "shots": [],
        "awaiting_review": 0,
        "mtime": 0.0,
    }

    try:
        wav_path = os.path.join(out_dir, base + ".wav")
        shots_json_path = os.path.join(out_dir, base + "-shots.json")
        shots_dir = os.path.join(out_dir, base + "-shots")
        review_dir = os.path.join(shots_dir, "review")
        final_path = os.path.join(out_dir, "Final", base + ".mp4")

        latest_mtime = 0.0

        wav_stat = _safe_stat(wav_path)
        if wav_stat:
            latest_mtime = max(latest_mtime, wav_stat.st_mtime)
            info["audio"] = _relpath(out_dir, wav_path)
            beats = _safe_read_json(wav_path + ".beats.json")
            if isinstance(beats, dict):
                info["bpm"] = beats.get("bpm")

        shotlist = _safe_read_json(shots_json_path)
        shots_json_stat = _safe_stat(shots_json_path)
        if shots_json_stat:
            latest_mtime = max(latest_mtime, shots_json_stat.st_mtime)

        if isinstance(shotlist, dict) and isinstance(shotlist.get("shots"), list):
            shots = shotlist["shots"]
            info["shots_total"] = len(shots)
            unique_shots = [s for s in shots if not s.get("reuse_of")]
            info["shots_unique"] = len(unique_shots)

            takes_by_id = _shot_takes(shots_dir)
            verdicts_by_id = _latest_verdicts(os.path.join(review_dir, "review_log.json"))

            rendered_unique = 0
            for shot in shots:
                shot_id = shot.get("id")
                is_unique = not shot.get("reuse_of")
                takes = takes_by_id.get(shot_id, [])
                if takes:
                    latest_mtime = max(latest_mtime, max((t["mtime"] or 0) for t in takes))
                    if is_unique:
                        rendered_unique += 1
                # review_log entries look like {"shot": id, "attempt": n,
                # "seed": n, "verdict": {"accept", "score", "reasons", ...}}
                # (see pipeline.kidsong.generate._generate_director).
                verdict_entry = verdicts_by_id.get(shot_id)
                verdict_body = (verdict_entry or {}).get("verdict") or {}
                sheet_path = os.path.join(review_dir, f"{shot_id}.png")
                sheet_rel = _relpath(out_dir, sheet_path) if os.path.exists(sheet_path) else None
                info["shots"].append(
                    {
                        "id": shot_id,
                        "reuse_of": shot.get("reuse_of"),
                        "shot_type": shot.get("shot_type"),
                        "action": shot.get("action"),
                        "take_count": len(takes),
                        "takes": [
                            {"attempt": t["attempt"], "path": _relpath(out_dir, t["path"])}
                            for t in takes
                        ],
                        "verdict": {
                            "accept": verdict_body.get("accept"),
                            "reasons": verdict_body.get("reasons") or [],
                            "attempt": verdict_entry.get("attempt"),
                            # review_log.json is append-only and outlives the
                            # takes it judged: delete/prune the .mp4 files and
                            # the verdicts stay. The dashboard was then showing
                            # a green "accept" for every shot of an episode
                            # whose shots dir holds nothing but review/ — a
                            # human reads "all approved" where the truth is
                            # "nothing is there". A verdict with no take behind
                            # it, and no other shot to inherit one from, is
                            # ORPHANED and must not be shown as a live verdict.
                            "orphaned": not takes and not shot.get("reuse_of"),
                        }
                        if verdict_entry
                        else None,
                        "thumbnail": sheet_rel,
                    }
                )

            total_takes = sum(len(v) for v in takes_by_id.values())

        final_stat = _safe_stat(final_path)
        if final_stat:
            latest_mtime = max(latest_mtime, final_stat.st_mtime)
            info["final_video"] = _relpath(out_dir, final_path)
            info["final_size"] = _human_size(final_stat.st_size)
            info["final_mtime"] = final_stat.st_mtime
            info["final_mtime_str"] = time.strftime(
                "%Y-%m-%d %H:%M", time.localtime(final_stat.st_mtime)
            )

        # Cut-QC rejects stay in staging awaiting the producer's decision.
        staging_stat = None
        for suffix in (".staging.mp4", ".needs_review.mp4"):
            staging_stat = _safe_stat(os.path.join(out_dir, base + suffix))
            if staging_stat:
                info["staging_video"] = _relpath(out_dir, os.path.join(out_dir, base + suffix))
                latest_mtime = max(latest_mtime, staging_stat.st_mtime)
                break

        # --- stage inference (later stages win) ---
        if final_stat:
            info["stage"] = "done"
            info["label"] = "DONE"
        elif staging_stat:
            info["stage"] = "needs_review"
            info["label"] = "NEEDS REVIEW — cut assembled but not approved"
        elif isinstance(shotlist, dict) and info["shots_total"]:
            info["stage"] = "rendering"
            # Stated as a COUNT, not as an activity. "rendering shots (0/16
            # unique, 0 total takes)" claimed work was happening on an episode
            # that had been sitting untouched for hours — and it sat directly
            # under a badge that (correctly) said the episode was waiting for a
            # human. Two lines of the same card contradicting each other.
            info["label"] = (
                f"{rendered_unique}/{info['shots_unique']} unique shots rendered, "
                f"{total_takes} takes on disk"
            )
        elif shots_json_stat:
            info["stage"] = "planned"
            info["label"] = (
                f"shots planned ({info.get('shots_unique')} unique / "
                f"{info.get('shots_total')} total)"
            )
        elif wav_stat:
            info["stage"] = "recorded"
            info["label"] = "song recorded"

        # Gates this episode is BLOCKED ON A HUMAN for: a *.request.json in its
        # review dir with no *.response.json beside it — the same rule
        # review_queue.list_pending uses, so the dashboard and the queue can
        # never disagree about what is waiting. Without this the dashboard
        # showed 16 episodes sitting on an unanswered gate as plain "rendering",
        # indistinguishable from the 16 that genuinely have nothing to do: the
        # work the operator could clear in one click was invisible on the page
        # whose job is to say what to do next.
        info["awaiting_review"] = _pending_gate_count(review_dir)

        info["mtime"] = latest_mtime
        # How long since ANY file of this episode last changed. The stage badge
        # says "rendering" for everything that has a shot list and no final cut,
        # which is true of an episode the GPU is working on right now AND of one
        # abandoned last week — the dashboard showed 32 episodes "rendering"
        # with nothing running. The stage cannot tell them apart; this can, so
        # the reader gets the fact instead of a badge that overstates.
        info["idle_seconds"] = max(0.0, time.time() - latest_mtime) if latest_mtime else None
    except Exception as e:  # pragma: no cover - defensive catch-all
        info["stage"] = "error"
        info["label"] = f"scan error: {e}"
    return info


# --------------------------------------------------------------------- logs / gpu ---
def _parse_batch_log(lines):
    """Extract the most recent '=== BATCH: ... ===' status from log tail lines."""
    last_batch = None
    for line in lines:
        m = _BATCH_HEADER_RE.match(line.strip())
        if m:
            last_batch = m.group(1)
    return last_batch


def _gpu_status():
    try:
        out = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=memory.used,memory.total,utilization.gpu",
                "--format=csv,noheader",
            ],
            capture_output=True,
            text=True,
            timeout=3,
        )
        if out.returncode != 0 or not out.stdout.strip():
            return None
        line = out.stdout.strip().splitlines()[0]
        parts = [p.strip() for p in line.split(",")]
        if len(parts) != 3:
            return {"raw": line}
        return {"memory_used": parts[0], "memory_total": parts[1], "utilization": parts[2]}
    except (OSError, subprocess.SubprocessError):
        return None


# --------------------------------------------------------------------- entrypoint ---
def scan_output(cfg):
    """Scan `output/` for all kidsong artifacts and return a plain-data dashboard dict.

    Never raises: any per-video failure degrades that one entry to
    {"stage": "error", ...} rather than aborting the whole scan.
    """
    out_dir = abspath(cfg, cfg["paths"]["output_dir"])

    bases = _discover_bases(out_dir)
    videos = [_scan_one(base, out_dir) for base in bases]
    videos.sort(key=lambda v: v.get("mtime") or 0, reverse=True)

    done = [v for v in videos if v["stage"] == "done"]
    in_progress = [v for v in videos if v["stage"] != "done"]

    director_log = _tail_lines(os.path.join(out_dir, "director_run.log"), 15)
    batch_log = _tail_lines(os.path.join(out_dir, "batch_run.log"), 15)

    return {
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "videos": videos,
        "done": done,
        "in_progress": in_progress,
        "director_log": director_log,
        "batch_log": batch_log,
        "batch_status": _parse_batch_log(batch_log),
        "gpu": _gpu_status(),
        "comfy_url": "http://127.0.0.1:8188",
    }


if __name__ == "__main__":
    from pipeline.config import load_config

    _cfg = load_config()
    result = scan_output(_cfg)
    print(json.dumps(result, indent=2, default=str))
