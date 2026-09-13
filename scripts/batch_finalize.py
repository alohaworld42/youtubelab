"""Re-render the channel inventory at final quality via the i2v-hires pipeline.

One `--finalize` per DISTINCT episode (newest draft iteration of each song;
older iterations of the same song are historical experiments, not inventory).
Sequential on purpose: one GPU, and generate.py's GPU lock serializes anyway.

Idempotent: a base is skipped when the state file says done OR a post-fix
Final/ cut for it already exists, so the batch can be re-launched after any
interruption and continues where it stopped. Failures are recorded and the
batch moves on — one bad episode must not strand the rest of the inventory.

Progress: output/batch_finalize.log (one line per event) and
output/batch_finalize_state.json (machine-readable per-base status).
"""
import datetime as dt
import json
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "output")
PYTHON = os.path.join(ROOT, "venv", "Scripts", "python.exe")
STATE_PATH = os.path.join(OUT, "batch_finalize_state.json")
LOG_PATH = os.path.join(OUT, "batch_finalize.log")

# Finals rendered before this moment predate the full fix chain and do not
# count as "already done". Two fixes gate it: the i2v-hires graph (commits
# 5a5af17/025ed80) AND reference_anchor OFF (style-mixing fix, 2026-07-24
# ~16:30 — earlier same-day finals still cut vinyl-ref solo shots against
# film-CGI ensembles).
FIX_CUTOFF = dt.datetime(2026, 7, 24, 16, 25, 0).timestamp()

# Newest draft iteration per distinct song, newest episodes first. The
# rain-rain storyfix goes LAST: a finalize of it was already launched
# separately today, so by the time the batch reaches it the guard below
# usually skips it (and re-finalizes only if that run died).
BASES = [
    "20260722-181532-kidsong-painting-a-rainbow-together-fun",
    "20260722-174255-kidsong-baa-baa-black-sheep-a-classic-sing-along-for-toddl",
    "20260722-155606-kidsong-bubble-friends-dance-and-play",
    "20260722-145643-kidsong-rainy-day-umbrella-parade",
    "20260722-144143-kidsong-alle-vögel-sind-schon-da-ein-fröhliches-frühlingsl",
    "20260722-142656-kidsong-rainy-day-friends",
    "20260722-130751-kidsong-little-gardeners-grow-sunflowers",
    "20260721-194700-kidsong-colors-on-our-table",
    "20260721-190129-kidsong-skip-to-my-lou-a-skipping-and-clapping-sing-along-",
    "20260721-180915-kidsong-sunflower-family-tree",
    "20260721-170859-kidsong-pat-a-cake-a-baking-sing-along-song-for-toddlers",
    "20260721-170208-kidsong-clap-your-hands-a-happy-sing-along-song-for-toddle",
    "20260721-131710-kidsong-are-you-sleeping-a-gentle-wake-up-sing-along-for-t",
    "20260721-115254-kidsong-here-we-go-round-the-mulberry-bush-an-action-sing-",
    "20260720-201849-kidsong-hickory-dickory-dock-a-fun-clock-sing-along-for-to",
    "20260720-200946-kidsong-row-row-row-your-boat-a-sing-along-song-for-toddle",
    "20260720-200056-kidsong-zuri-had-a-little-lamb-a-happy-sing-along-song-for",
    "20260720-191135-kidsong-twinkle-twinkle-little-star-a-gentle-sing-along-fo",
    "20260720-170942-kidsong-zuri-kofi-nala-feel-the-breeze",
    "20260720-110129-kidsong-zuris-watering-blooms-day",
    "20260720-092939-kidsong-rainbow-friends-color-fun",
    "20260720-085055-kidsong-splish-splash-fun-with-friends",
    "20260719-195556-kidsong-counting-friends-fun-day",
    "20260723-105443-kidsong-rain-rain-go-away-storyfix",
]


def log(msg):
    line = f"{dt.datetime.now():%Y-%m-%d %H:%M:%S} {msg}"
    print(line, flush=True)
    with open(LOG_PATH, "a", encoding="utf-8") as fh:
        fh.write(line + "\n")


def load_state():
    try:
        with open(STATE_PATH, encoding="utf-8") as fh:
            return json.load(fh)
    except (OSError, ValueError):
        return {}


def save_state(state):
    tmp = STATE_PATH + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(state, fh, indent=2, ensure_ascii=False)
    os.replace(tmp, STATE_PATH)


def post_fix_final_cut(base):
    """Path of a finished cut for any -final* base of `base` rendered AFTER
    the pipeline fix, or None. Pre-fix finals rendered the soft 896x512 path.

    A cut counts whether it was PROMOTED (output/Final/<b>.mp4) or held in
    staging by the cut-review gate (output/<b>.staging*.mp4) — a held cut is
    fully rendered; a human reviews every video before upload regardless."""
    candidates = []
    final_dir = os.path.join(OUT, "Final")
    if os.path.isdir(final_dir):
        for name in os.listdir(final_dir):
            if name.startswith(base + "-final") and name.endswith(".mp4"):
                candidates.append(os.path.join(final_dir, name))
    for name in os.listdir(OUT):
        if (name.startswith(base + "-final") and ".staging" in name
                and name.endswith(".mp4")):
            candidates.append(os.path.join(OUT, name))
    for path in candidates:
        if os.path.getmtime(path) >= FIX_CUTOFF and os.path.getsize(path) > 0:
            return path
    return None


def main():
    state = load_state()
    log(f"batch start: {len(BASES)} episodes queued")
    done = failed = skipped = 0
    for i, base in enumerate(BASES, 1):
        entry = state.get(base) or {}
        existing = post_fix_final_cut(base)
        if entry.get("status") == "done" or existing:
            log(f"[{i}/{len(BASES)}] SKIP {base} (already finalized post-fix"
                f"{': ' + os.path.basename(existing) if existing else ''})")
            state[base] = entry or {"status": "done", "cut": existing}
            save_state(state)
            skipped += 1
            continue
        log(f"[{i}/{len(BASES)}] FINALIZE {base}")
        state[base] = {"status": "running", "started": dt.datetime.now().isoformat()}
        save_state(state)
        try:
            proc = subprocess.run(
                [PYTHON, "-m", "pipeline.kidsong.generate", "--finalize", base],
                cwd=ROOT, capture_output=True, text=True, encoding="utf-8",
                errors="replace",
            )
        except Exception as exc:  # noqa: BLE001 - keep the batch alive
            state[base] = {"status": "failed", "error": repr(exc)}
            save_state(state)
            log(f"[{i}/{len(BASES)}] FAILED {base}: {exc!r}")
            failed += 1
            continue
        cut = post_fix_final_cut(base)
        if proc.returncode == 0 and cut:
            state[base] = {"status": "done", "cut": cut,
                           "finished": dt.datetime.now().isoformat()}
            done += 1
            log(f"[{i}/{len(BASES)}] DONE {base} -> {os.path.basename(cut)}")
        else:
            tail = (proc.stdout or "")[-800:] + (proc.stderr or "")[-800:]
            state[base] = {"status": "failed", "returncode": proc.returncode,
                           "tail": tail}
            failed += 1
            log(f"[{i}/{len(BASES)}] FAILED {base} rc={proc.returncode} "
                f"(see output/*.log for the run itself)")
        save_state(state)
    log(f"batch complete: {done} done, {skipped} skipped, {failed} failed")


if __name__ == "__main__":
    main()
