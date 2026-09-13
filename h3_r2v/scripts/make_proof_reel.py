#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
make_proof_reel.py - Baut aus den gerenderten Proof-Shots einen Reel + Handy-Version.
"""

import argparse
import json
import os
import shutil
import subprocess
import sys


def log(msg):
    print(msg, flush=True)


def die(msg, code=2):
    print("FEHLER: " + str(msg), flush=True)
    sys.exit(code)


def find_ffmpeg(extra_roots=()):
    p = shutil.which("ffmpeg")
    if p:
        return p
    for root in extra_roots:
        if root and os.path.isdir(root):
            root = os.path.abspath(root)
            for dirpath, dirnames, filenames in os.walk(root):
                if dirpath[len(root):].count(os.sep) > 5:
                    dirnames[:] = []
                    continue
                dirnames[:] = [d for d in dirnames if d.lower() not in (".git", "node_modules", "models")]
                if "ffmpeg.exe" in filenames:
                    return os.path.join(dirpath, "ffmpeg.exe")
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shots-dir", required=True)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--exclude", action="append", default=[],
                    help="Shots mit diesem ID-Praefix ueberspringen (mehrfach angeben)")
    ap.add_argument("--ffmpeg-root", default="")
    args = ap.parse_args()

    shots_dir = os.path.abspath(args.shots_dir)
    manifest_path = os.path.abspath(args.manifest)
    out_dir = os.path.abspath(args.out_dir)
    os.makedirs(out_dir, exist_ok=True)

    if not os.path.isfile(manifest_path):
        die("Manifest nicht gefunden: %s" % manifest_path)

    manifest = json.load(open(manifest_path, "r", encoding="utf-8"))
    shot_ids = sorted([k for k, v in manifest.items() if v.get("status") == "done"
                       and not any(k.startswith(p) for p in args.exclude)])
    if not shot_ids:
        die("Keine fertigen Shots im Manifest.")

    mp4s = []
    for sid in shot_ids:
        p = os.path.join(shots_dir, sid + ".mp4")
        if os.path.isfile(p) and os.path.getsize(p) > 0:
            mp4s.append(p)
        else:
            log("WARNUNG: Shot %s fehlt oder leer: %s" % (sid, p))

    if not mp4s:
        die("Keine nutzbaren MP4s in %s" % shots_dir)

    # Concat list
    lst = os.path.join(out_dir, "concat.txt")
    with open(lst, "w", encoding="utf-8") as f:
        for p in mp4s:
            f.write("file '%s'\n" % p.replace("\\", "/").replace("'", "'\\''"))

    reel_full = os.path.join(out_dir, "proof_reel_full.mp4")
    reel_phone = os.path.join(out_dir, "proof_reel_phone.mp4")

    ff = find_ffmpeg([args.ffmpeg_root] if args.ffmpeg_root else [])
    if not ff:
        die("ffmpeg nicht gefunden. --ffmpeg-root angeben oder ffmpeg in PATH.")

    # Full quality concat (copy mode)
    log("Concat full quality -> %s" % reel_full)
    subprocess.run([ff, "-y", "-f", "concat", "-safe", "0", "-i", lst, "-c", "copy", reel_full],
                   check=True)

    # Phone version: 720p, CRF 23, ~<30 MB
    log("Phone version (720p, CRF 23) -> %s" % reel_phone)
    subprocess.run([ff, "-y", "-i", reel_full, "-vf", "scale=-2:720", "-c:v", "libx264",
                    "-crf", "23", "-preset", "fast", "-pix_fmt", "yuv420p",
                    "-c:a", "aac", "-b:a", "128k", reel_phone],
                   check=True)

    sz = os.path.getsize(reel_phone)
    log("Fertig: %s (%.1f MB)" % (reel_phone, sz / 1024 / 1024))
    if sz > 30 * 1024 * 1024:
        log("WARNUNG: Phone-Version > 30 MB. Nochmal mit hoeherem CRF rendern.")

    log("Full: %s" % reel_full)
    log("Phone: %s" % reel_phone)


if __name__ == "__main__":
    main()