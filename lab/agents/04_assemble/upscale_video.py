"""ML-Upscale eines Videos mit spandrel (RealESRGAN/UltraSharp), dann auf Zielgroesse.

    python upscale_video.py --in clip.mp4 --out clip_up.mp4 --model 4x-UltraSharp.pth
        [--target 1280x720] [--device cuda|cpu] [--roundtrip 864x480]

--roundtrip WxH: Testmodus — Eingang erst auf WxH herunterskalieren (simuliert einen
                 kleinen Render), dann per Modell hochskalieren. Fuer Qualitaets-A/B.
Frames via ffmpeg raus/rein; Audio wird uebernommen.
"""
import argparse
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "common"))
import comfy_lab as L  # noqa: E402


def load_model(path, device):
    import torch
    from spandrel import ModelLoader
    m = ModelLoader().load_from_file(path)
    m.model.eval().to(device)
    return m, m.scale


def upscale_frame(model, img, device):
    import torch
    import numpy as np
    t = torch.from_numpy(np.asarray(img)).float().div(255.0).permute(2, 0, 1).unsqueeze(0).to(device)
    with torch.no_grad():
        out = model.model(t)
    out = out.clamp(0, 1).squeeze(0).permute(1, 2, 0).mul(255).round().byte().cpu().numpy()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default="4x-UltraSharp.pth")
    ap.add_argument("--target", default="1280x720")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--roundtrip", default=None, help="downscale to WxH first (A/B test)")
    a = ap.parse_args()
    cfg = L.lab_config()
    ff = cfg["ffmpeg"]
    import torch
    from PIL import Image
    dev = a.device if (a.device == "cpu" or torch.cuda.is_available()) else "cpu"
    model_path = os.path.join(cfg["models_dir"], "upscale_models", a.model)
    tmp = os.path.join(cfg["temp_dir"], "upscale_" + os.path.splitext(os.path.basename(a.out))[0])
    os.makedirs(tmp, exist_ok=True)
    tw, th = map(int, a.target.lower().split("x"))

    vf = f"-vf scale={a.roundtrip.replace('x', ':')}:flags=area" if a.roundtrip else ""
    subprocess.run([ff, "-hide_banner", "-loglevel", "error", "-y", "-i", a.inp] + (vf.split() if vf else []) +
                   [os.path.join(tmp, "in_%04d.png")], check=True)
    frames = sorted(f for f in os.listdir(tmp) if f.startswith("in_"))
    model, scale = load_model(model_path, dev)
    print(f"model {a.model} scale {scale}x on {dev}, {len(frames)} frames, roundtrip={a.roundtrip}")
    import time
    t0 = time.time()
    for i, fn in enumerate(frames):
        img = Image.open(os.path.join(tmp, fn)).convert("RGB")
        up = upscale_frame(model, img, dev)
        Image.fromarray(up).resize((tw, th), Image.LANCZOS).save(os.path.join(tmp, f"out_{i:04d}.png"))
    dt = time.time() - t0
    print(f"upscaled {len(frames)} frames in {dt:.0f}s ({dt/max(len(frames),1):.2f}s/frame)")
    # reassemble, carry audio from source
    subprocess.run([ff, "-hide_banner", "-loglevel", "error", "-y", "-framerate", "24",
                    "-i", os.path.join(tmp, "out_%04d.png"), "-i", a.inp,
                    "-map", "0:v", "-map", "1:a?", "-c:v", "libx264", "-preset", "slow", "-crf", "18",
                    "-pix_fmt", "yuv420p", "-r", "24", "-movflags", "+faststart", "-shortest", a.out], check=True)
    print("wrote", a.out, f"{os.path.getsize(a.out)/1e6:.1f}MB")
    for f in os.listdir(tmp):
        os.remove(os.path.join(tmp, f))


if __name__ == "__main__":
    main()
