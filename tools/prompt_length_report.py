"""Measure composed shot-prompt length across the shipped corpus.

`docs/quality/SOURCES.md` records a ~200-word cap for LTX-2 prompts (the
RunDiffusion guide) and a measurement of how far past it this channel's prompts
run. That measurement was taken by hand once and then quoted for weeks while the
prompt kept changing underneath it. This script IS the measurement, so re-running
it costs nothing and the number in the docs can never silently go stale again.

It recomposes every shot in `output/*-shots.json` through the real
`generate._shot_prompt` at the CURRENT code and config, so it reports what today's
pipeline would send — not what was sent when the shotlist was written.

    python tools/prompt_length_report.py                 # active style
    python tools/prompt_length_report.py pixar_toon pixar_toon_concise
                                                         # A/B two styles

CPU only: no GPU, no ComfyUI, no network. Reads local JSON and composes strings.
"""
import glob
import json
import logging
import os
import statistics
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pipeline.config import load_config                       # noqa: E402
from pipeline.kidsong.generate import _identity_block, _shot_prompt  # noqa: E402
from pipeline.kidsong.render_style import resolve_style       # noqa: E402

CAP = 200  # words; the RunDiffusion LTX-2 guide's recommendation


def _episodes():
    """(shotlist, song) pairs for every episode with both files on disk."""
    for shots_path in sorted(glob.glob("output/*-shots.json")):
        song_path = shots_path[:-len("-shots.json")] + "-song.json"
        if not os.path.exists(song_path):
            continue
        try:
            with open(shots_path, encoding="utf-8") as f:
                shots = json.load(f)
            with open(song_path, encoding="utf-8") as f:
                song = json.load(f)
        except (OSError, ValueError):
            continue
        if isinstance(shots, dict):
            shots = shots.get("shots", [])
        yield [s for s in shots if isinstance(s, dict)], song


def _cfg_for(style_name, base):
    return {"kidsong": {"render_style": style_name,
                        "render_styles": base["kidsong"]["render_styles"],
                        "decreep": base["kidsong"].get("decreep", {})}}


def measure(cfg):
    """Word count + head count for every child-bearing shot in the corpus."""
    rows = []
    for shots, song in _episodes():
        for shot in shots:
            try:
                words = len(_shot_prompt(shot, song, cfg).split())
                count = _identity_block(shot, song, cfg)[5]
            except Exception:  # a malformed historical shot must not stop the report
                continue
            rows.append((words, count))
    return rows


def report(label, rows):
    words = sorted(w for w, _ in rows)
    n = len(words)
    if not n:
        print(f"{label}: no shots found (run from the repo root)")
        return
    over = sum(1 for w in words if w > CAP)
    print(f"{label:24s} n={n:5d}  min={words[0]:3d}  median={int(statistics.median(words)):3d}  "
          f"p90={words[int(0.9 * (n - 1))]:3d}  max={words[-1]:3d}  "
          f"over {CAP}: {over} ({100 * over / n:.1f}%)")
    for count in sorted({c for _, c in rows}, key=lambda c: (c is None, c)):
        group = [w for w, c in rows if c == count]
        hits = sum(1 for w in group if w > CAP)
        print(f"    {str(count):>4s} children  n={len(group):4d}  "
              f"mean={statistics.mean(group):6.1f}  over-cap {100 * hits / len(group):5.1f}%")


def main(argv):
    logging.disable(logging.WARNING)  # the corpus has known off-bible names
    base = load_config()
    names = argv[1:] or [base["kidsong"].get("render_style", "pixar_toon")]
    for name in names:
        style = resolve_style(_cfg_for(name, base))
        report(f"{name} (tail {len(str(style['prompt_tail']).split())}w)",
               measure(_cfg_for(name, base)))


if __name__ == "__main__":
    main(sys.argv)
