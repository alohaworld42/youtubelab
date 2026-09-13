"""Agent 4: Clips auf 1280x720 konformieren, Handy-Version, Side-by-Side.

    python assemble.py --run proof01 [--sbs s01_V2.mp4,s01_norefs_V2.mp4]

final/<clip>.mp4 : 1344x768 -> crop 1344x756 -> scale 1280x720 (lanczos), libx264 crf 18, aac
phone/<clip>.mp4 : crf 28, <= final.phone_max_mb (assert)
final/<a>__vs__<b>.mp4 : hstack 640x360 je Seite
"""
import argparse
import glob
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "common"))
import comfy_lab as L  # noqa: E402


def probe_size(ffmpeg, path):
    r = subprocess.run([ffmpeg, "-hide_banner", "-i", path], capture_output=True, text=True)
    for line in r.stderr.splitlines():
        if "Video:" in line:
            for tok in line.replace(",", " ").split():
                if "x" in tok and tok.replace("x", "").isdigit():
                    w, h = tok.split("x")
                    return int(w), int(h)
    return None


def vf_for(w, h, W, H):
    """Crop to target aspect (center), then scale."""
    if (w, h) == (W, H):
        return "null"
    if abs(w / h - W / H) < 1e-3:
        return f"scale={W}:{H}:flags=lanczos"
    if w / h > W / H:      # too wide -> crop width
        cw = int(round(h * W / H / 2)) * 2
        return f"crop={cw}:{h}:{(w - cw) // 2}:0,scale={W}:{H}:flags=lanczos"
    ch = int(round(w * H / W / 2)) * 2   # too tall -> crop height
    return f"crop={w}:{ch}:0:{(h - ch) // 2},scale={W}:{H}:flags=lanczos"


def run(cmd):
    subprocess.run(cmd, check=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--sbs", default=None, help="comma pairs a.mp4,b.mp4 (relative to clips_V2)")
    a = ap.parse_args()
    cfg = L.lab_config()
    paths = L.run_paths(cfg, a.run)
    log = L.setup_logging(paths["root"], "ASSEMBLE")
    ff, fc = cfg["ffmpeg"], cfg["final"]
    W, H, fps = fc["width"], fc["height"], fc["fps"]
    for clip in sorted(glob.glob(os.path.join(paths["clips"], "*.mp4"))):
        name = os.path.basename(clip)
        size = probe_size(ff, clip)
        if not size:
            log.warning("%s: cannot probe, skip", name)
            continue
        vf = vf_for(size[0], size[1], W, H)
        final = os.path.join(paths["final"], name)
        phone = os.path.join(paths["phone"], name)
        if not L.done(final):
            run([ff, "-hide_banner", "-loglevel", "error", "-y", "-i", clip, "-vf", vf, "-r", str(fps),
                 "-c:v", "libx264", "-preset", "slow", "-crf", str(fc["crf"]), "-pix_fmt", "yuv420p",
                 "-c:a", "aac", "-b:a", "128k", "-movflags", "+faststart", final])
        if not L.done(phone):
            run([ff, "-hide_banner", "-loglevel", "error", "-y", "-i", final, "-c:v", "libx264", "-preset", "medium",
                 "-crf", str(fc["phone_crf"]), "-maxrate", "4M", "-bufsize", "8M", "-pix_fmt", "yuv420p",
                 "-c:a", "aac", "-b:a", "96k", "-movflags", "+faststart", phone])
        mb = os.path.getsize(phone) / 1e6
        assert mb <= fc["phone_max_mb"], f"phone version too big: {mb:.1f} MB"
        log.info("%s: %dx%d -> %s | final %.1f MB | phone %.1f MB", name, size[0], size[1], vf,
                 os.path.getsize(final) / 1e6, mb)
    pairs = []
    if a.sbs:
        toks = a.sbs.split(",")
        pairs = [(toks[i], toks[i + 1]) for i in range(0, len(toks) - 1, 2)]
    else:
        fin = {os.path.basename(p) for p in glob.glob(os.path.join(paths["final"], "*.mp4"))}
        for n in sorted(fin):
            if "_norefs" in n:
                other = n.replace("_norefs", "")
                if other in fin:
                    pairs.append((other, n))
    for a_, b_ in pairs:
        out = os.path.join(paths["final"], f"{os.path.splitext(a_)[0]}__vs__{os.path.splitext(b_)[0]}.mp4")
        if L.done(out):
            continue
        run([ff, "-hide_banner", "-loglevel", "error", "-y", "-i", os.path.join(paths["final"], a_),
             "-i", os.path.join(paths["final"], b_), "-filter_complex",
             "[0:v]scale=640:360[a];[1:v]scale=640:360[b];[a][b]hstack", "-map", "0:a?",
             "-c:v", "libx264", "-crf", "20", "-pix_fmt", "yuv420p", "-movflags", "+faststart", out])
        log.info("side-by-side -> %s", out)


if __name__ == "__main__":
    main()
