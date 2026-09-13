"""Regenerate the channel inventory as COMPLETELY NEW episodes.

Supersedes scripts/batch_finalize.py (user call 2026-07-24): finalize reused
the old drafts' shot lists, which predate the storyboard/coherence fixes —
fresh runs get new lyrics/song, a storyboard arc, a new shot list from the
current director, style-consistent keyframes and the i2v-hires renders.

One `--topic` per distinct episode of the old inventory (the topic maps to
the same public-domain song via prompts/pd_songs.json where one exists; the
rotation guard may deterministically swap a just-used classic for a fresh
one — that is channel programming, not an error). The German episode runs
with kidsong.language temporarily set to 'de' and restored afterwards.

Idempotent via output/batch_fresh_state.json — re-launch after any
interruption and it continues. Sequential behind the pipeline's GPU lock.
Progress: output/batch_fresh.log.
"""
import datetime as dt
import json
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "output")
PYTHON = os.path.join(ROOT, "venv", "Scripts", "python.exe")
CONFIG = os.path.join(ROOT, "config.json")
STATE_PATH = os.path.join(OUT, "batch_fresh_state.json")
LOG_PATH = os.path.join(OUT, "batch_fresh.log")

# (slug, --topic, language) — ONE item per public-domain library song.
#
# Trimmed from the original 24-episode inventory list (2026-07-24): the
# lyrics LLM (ollama) is down and no cloud key is configured, so every
# original topic falls back to PD-library rotation — and the EN library
# holds only 10 songs. More items than songs = duplicate-song episodes
# (measured: 'painting a rainbow together' produced zuri-had-a-little-lamb).
# The ~13 original-song episodes (bubble friends, sunflower family tree,
# rainbow colors, …) are BLOCKED until the lyrics LLM is back; run them as
# a second batch then. little-lamb is pre-marked done in the state file
# (episode 1 of the untrimmed batch already produced it).
ITEMS = [
    ("baa-baa-black-sheep", "baa baa black sheep", "en"),
    ("twinkle-twinkle", "twinkle twinkle little star", "en"),
    ("rain-rain-go-away", "rain rain go away", "en"),
    ("row-your-boat", "row row row your boat", "en"),
    ("hickory-dickory-dock", "hickory dickory dock", "en"),
    ("are-you-sleeping", "are you sleeping brother john", "en"),
    ("pat-a-cake", "pat a cake", "en"),
    ("mulberry-bush", "here we go round the mulberry bush", "en"),
    ("skip-to-my-lou", "skip to my lou", "en"),
    ("little-lamb", "zuri had a little lamb", "en"),
    ("alle-voegel", "alle Vögel sind schon da", "de"),
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


def set_language(lang):
    """Set kidsong.language in config.json; returns the previous value.

    The pipeline has no CLI/env override for language, so the German episode
    needs the live config flipped for its run (and restored right after)."""
    with open(CONFIG, encoding="utf-8") as fh:
        cfg = json.load(fh)
    prev = cfg.setdefault("kidsong", {}).get("language")
    cfg["kidsong"]["language"] = lang
    tmp = CONFIG + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(cfg, fh, indent=2, ensure_ascii=False)
    os.replace(tmp, CONFIG)
    return prev


def restore_language(prev):
    with open(CONFIG, encoding="utf-8") as fh:
        cfg = json.load(fh)
    if prev is None:
        cfg.get("kidsong", {}).pop("language", None)
    else:
        cfg.setdefault("kidsong", {})["language"] = prev
    tmp = CONFIG + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(cfg, fh, indent=2, ensure_ascii=False)
    os.replace(tmp, CONFIG)


def video_path_from(stdout):
    """The promoted/staging cut path from the CLI's Result JSON, or None."""
    for line in (stdout or "").splitlines():
        line = line.strip()
        if line.startswith("Result: {"):
            try:
                return json.loads(line[len("Result: "):]).get("video_path")
            except ValueError:
                pass
    return None


def main():
    state = load_state()
    log(f"fresh batch start: {len(ITEMS)} episodes queued")
    done = failed = skipped = 0
    for i, (slug, topic, lang) in enumerate(ITEMS, 1):
        if (state.get(slug) or {}).get("status") == "done":
            log(f"[{i}/{len(ITEMS)}] SKIP {slug} (already done: "
                f"{os.path.basename(state[slug].get('video') or '?')})")
            skipped += 1
            continue
        log(f"[{i}/{len(ITEMS)}] GENERATE {slug} — topic {topic!r} lang={lang}")
        state[slug] = {"status": "running", "topic": topic,
                       "started": dt.datetime.now().isoformat()}
        save_state(state)
        prev_lang = None
        try:
            if lang != "en":
                prev_lang = set_language(lang)
            proc = subprocess.run(
                [PYTHON, "-m", "pipeline.kidsong.generate", "--topic", topic],
                cwd=ROOT, capture_output=True, text=True, encoding="utf-8",
                errors="replace",
            )
        except Exception as exc:  # noqa: BLE001 - keep the batch alive
            state[slug] = {"status": "failed", "topic": topic, "error": repr(exc)}
            save_state(state)
            log(f"[{i}/{len(ITEMS)}] FAILED {slug}: {exc!r}")
            failed += 1
            continue
        finally:
            if lang != "en":
                restore_language(prev_lang)
        video = video_path_from(proc.stdout)
        if proc.returncode == 0 and video:
            state[slug] = {"status": "done", "topic": topic, "video": video,
                           "finished": dt.datetime.now().isoformat()}
            done += 1
            log(f"[{i}/{len(ITEMS)}] DONE {slug} -> {video}")
        else:
            tail = (proc.stdout or "")[-600:] + (proc.stderr or "")[-600:]
            state[slug] = {"status": "failed", "topic": topic,
                           "returncode": proc.returncode, "tail": tail}
            failed += 1
            log(f"[{i}/{len(ITEMS)}] FAILED {slug} rc={proc.returncode} "
                f"(run log under output/, see tail in state file)")
        save_state(state)
    log(f"fresh batch complete: {done} done, {skipped} skipped, {failed} failed")


if __name__ == "__main__":
    main()
