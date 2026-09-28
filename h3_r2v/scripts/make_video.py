#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
make_video.py - KI-Schnittsystem: baut aus gerenderten H3-Shots ein komplettes
Video (Storyboard) mit Xfade-Uebergaengen, optionaler Musik und Untertiteln.

Storyboard-JSON (minimal):
{
  "title": "Mein Gartenlied",
  "music": "pfad/zum/song.mp3",            # optional
  "music_volume": 0.6,                      # optional
  "fps": 24,
  "shots": [
    {"id": "shot_01", "transition": "fade", "duration": 5.1, "text": "Zuerst kommt Zuri"},
    {"id": "shot_02", ...}
  ]
}
transition: "cut" | "fade" | "wipeleft" | "circleopen" | "dissolve"
"""

import argparse
import json
import os
import subprocess
import sys


def log(msg):
    print(msg, flush=True)


def die(msg, code=2):
    print("FEHLER: " + str(msg), flush=True)
    sys.exit(code)


def find_ffmpeg(extra_roots=()):
    import shutil
    p = shutil.which("ffmpeg")
    if p:
        return p
    for root in extra_roots:
        if root and os.path.isdir(root):
            for dirpath, dirnames, filenames in os.walk(root):
                if "ffmpeg.exe" in filenames:
                    return os.path.join(dirpath, "ffmpeg.exe")
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        return None


def get_duration(ff, path):
    r = subprocess.run([ff, "-i", path], capture_output=True, text=True)
    for line in (r.stderr or "").splitlines():
        if "Duration:" in line:
            try:
                d = line.split("Duration:")[1].split(",")[0].strip()
                h, m, s = d.split(":")
                return int(h) * 3600 + int(m) * 60 + float(s)
            except Exception:
                return None
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--storyboard", required=True)
    ap.add_argument("--shots-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--ffmpeg-root", default="")
    ap.add_argument("--staging", default="")
    args = ap.parse_args()

    sb = json.load(open(args.storyboard, "r", encoding="utf-8"))
    shots = sb.get("shots", [])
    if not shots:
        die("Storyboard enthaelt keine Shots.")

    ff = find_ffmpeg([args.ffmpeg_root] if args.ffmpeg_root else [])
    if not ff:
        die("ffmpeg nicht gefunden.")

    staging = args.staging or os.path.join(os.path.dirname(os.path.abspath(args.out)), ".staging")
    os.makedirs(staging, exist_ok=True)

    # 1. Clips normalisieren (gleiche Aufloesung/FPS, kein Audio-Aussetzer) in staging
    norm = []
    fps = int(sb.get("fps", 24))
    for i, s in enumerate(shots):
        src = os.path.join(args.shots_dir, s["id"] + ".mp4")
        if not os.path.isfile(src) or os.path.getsize(src) == 0:
            die("Shot fehlt/leer: %s" % src)
        dur = s.get("duration") or get_duration(ff, src) or 5.0
        s["_orig"] = dur
        s["_dur"] = dur
        nxt = os.path.join(staging, "norm_%02d.mp4" % i)
        # re-encode fuer gleichmaessige Konkatenation; filter erhaelt Audio
        subprocess.run([ff, "-y", "-i", src, "-vf", "fps=" + str(fps),
                        "-c:v", "libx264", "-crf", "18", "-preset", "fast",
                        "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "192k", nxt],
                       check=True)
        norm.append(nxt)
        log("normalisiert %s (%.2fs)" % (s["id"], dur))

    # 2. Uebergaenge: sequenziell falten (mit realen Datei-Dauern; Offset darf
    #    nie ueber das Ende von Eingang A hinausgehen, sonst wird gekuerzt)
    trans_sec = float(sb.get("transition_sec", 0.4))

    def merge_pair(a, b, t, step_idx):
        da = get_duration(ff, a) or 5.0
        db = get_duration(ff, b) or 5.0
        out = os.path.join(staging, "chain_0_%02d.mp4" % step_idx)
        off = max(0.0, da - trans_sec)
        # beide Pfade mit Startzeit versehen, dann verschmelzen
        vf = "[0:v]fps=%d,setpts=PTS-STARTPTS[va];[1:v]fps=%d,setpts=PTS-STARTPTS+%.3f/TB[vb];[va][vb]xfade=transition=%s:duration=%.2f:offset=%.2f[vout]" % (
            fps, fps, off, t, trans_sec, off)
        af = "[0:a]atrim=0:%.2f[a0];[1:a]atrim=0:%.2f[a1];[a0][a1]amix=inputs=2:duration=first,atrim=0:%.2f[aout]" % (
            da, db, da + db - trans_sec)
        subprocess.run([ff, "-y", "-i", a, "-i", b, "-filter_complex", vf + ";" + af,
                        "-map", "[vout]", "-map", "[aout]",
                        "-c:v", "libx264", "-crf", "18", "-preset", "fast", out], check=True)
        return out

    med = len(norm)
    chain = norm[0]
    for i in range(1, med):
        chain = merge_pair(chain, norm[i], shots[i].get("transition", "fade"), i + 1)
    total = get_duration(ff, chain) or 0.0

    # 3. Musik untermischen (optional)
    music = sb.get("music")
    tmp_mix = os.path.join(staging, "mix_audio.mp4")
    if music and os.path.isfile(music):
        vol = float(sb.get("music_volume", 0.6))
        mv = os.path.join(staging, "music_norm.m4a")
        subprocess.run([ff, "-y", "-i", music, "-t", str(total), "-ar", "48000", "-ac", "2",
                        "-c:a", "aac", mv], check=True)
        cmd = [ff, "-y", "-i", chain, "-i", mv,
               "-filter_complex", "[1:a]volume=%.2f[m];[0:a][m]amix=inputs=2:duration=first[aout]" % vol,
               "-map", "0:v", "-map", "[aout]",
               "-c:v", "copy", "-c:a", "aac", tmp_mix]
        subprocess.run(cmd, check=True)
        chain = tmp_mix
    else:
        log("Keine Musik angegeben - nutze reine Shot-Audio.")

    # 4. Untertitel einbrennen (optional, ..text-Alt auf Shots als Burn-in)
    has_text = any(s.get("text") for s in shots)
    if has_text:
        # korrekte Startzeit je Shot im xfade-Output: kumulierte Oiginaldauern
        # minus je einem Uebergang zwischen den Shots
        starts = []
        acc = 0.0
        for i, s in enumerate(shots):
            starts.append(acc)
            acc += s["_orig"] - (trans_sec if i < len(shots) - 1 else 0.0)

        def fmt_ts(sec):
            ms = int(round(sec * 1000))
            h, rem = divmod(ms, 3600000)
            m, rem = divmod(rem, 60000)
            s, ms = divmod(rem, 1000)
            return "%02d:%02d:%02d,%03d" % (h, m, s, ms)

        srt_path = os.path.join(staging, "subs.srt")
        with open(srt_path, "w", encoding="utf-8-sig", newline="\r\n") as fh:
            n = 1
            for i, s in enumerate(shots):
                txt = s.get("text")
                if not txt:
                    continue
                t0 = starts[i] + 0.3
                t1 = min(t0 + 2.8, starts[i] + s["_orig"] - 0.1)
                fh.write("%d\n%s --> %s\n%s\n\n" % (n, fmt_ts(t0), fmt_ts(t1), txt))
                n += 1

        # subtitles-Filter (libass) statt drawtext - kein Windows-Font-escaping
        sub_arg = srt_path.replace("\\", "/").replace(":", r"\:")
        vf = "fps=%d,subtitles='%s':force_style='FontName=Arial,FontSize=22,PrimaryColour=&H00FFFFFF,OutlineColour=&H00000000,Outline=2,Shadow=0,BorderStyle=1,Alignment=2,MarginV=60'" % (fps, sub_arg)
        out_sub = os.path.join(staging, "subtitled.mp4")
        try:
            subprocess.run([ff, "-y", "-i", chain, "-vf", vf,
                            "-c:v", "libx264", "-crf", "18", "-preset", "fast",
                            "-c:a", "copy", out_sub], check=True)
            chain = out_sub
            log("Untertitel eingebrannt (%d Texte)" % n)
        except subprocess.CalledProcessError:
            log("WARNUNG: Untertitel-Schritt fehlgeschlagen - Video ohne Texte.")

    # 5. Final encoden
    out = os.path.abspath(args.out)
    final = os.path.join(staging, "final.mp4")
    subprocess.run([ff, "-y", "-i", chain, "-c:v", "libx264", "-crf", "19", "-preset", "medium",
                    "-c:a", "aac", "-b:a", "192k", "-movflags", "+faststart", final], check=True)
    os.replace(final, out)

    log("FERTIG: %s (%.1f MB, %.1fs)" % (out, os.path.getsize(out) / 1024 / 1024, total))


if __name__ == "__main__":
    main()