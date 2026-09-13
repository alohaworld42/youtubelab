"""Agent 3: Identitaets-QC mit DINOv2-small (Crop-Cosine), Kontaktbogen, report.md.

    python identity_qc.py --run proof01 [--what keyframes|clips|all] [--warmup]

Scores (0..1, Cosine):
  id_<kind>  = max ueber Crops (Vollbild + 3x3 Fenster 50 %, Schritt 25 %) der Aehnlichkeit zur canonical-Ref
               (Ref-Embedding = Mittel aus ganzer Ref und oberen 60 % = Gesicht+Haar), Mittel ueber Frames
  drift      = min_t cos(frame_t, frame_0)  (nur Clips)
  cross      = cos(mean-emb Shot A, mean-emb Shot B) fuer Zuri-Shots (s01/s02/s03)
  neg        = Negativkontrolle: altes v4-s05-Keyframe (Fremdkinder) vs. jede canonical-Ref
Ausgabe: qc/scores.json, qc/report.md, qc/contact_<what>.png, qc/frames/
"""
import argparse
import glob
import json
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "common"))
import comfy_lab as L  # noqa: E402
import shots as S  # noqa: E402

NEG_KEYFRAME = ("C:/Users/Aloha/Desktop/Projects/youtubegenerator/output/"
                "20260903-125326-kidsong-farben-lernen-rot-blau-gelb-und-grün-für-kleinkind-style-wan22-toon-v4-shots/keyframes/s05_1280x704.png")


class Dino:
    def __init__(self, cfg):
        os.environ["HF_HOME"] = cfg["hf_home"]
        import torch
        from transformers import AutoImageProcessor, AutoModel
        self.torch = torch
        self.dev = "cuda" if torch.cuda.is_available() else "cpu"
        self.proc = AutoImageProcessor.from_pretrained("facebook/dinov2-small")
        self.model = AutoModel.from_pretrained("facebook/dinov2-small").to(self.dev).eval()
        self.cache = {}

    def embed(self, pil):
        key = (id(pil), pil.size)
        with self.torch.no_grad():
            inp = self.proc(images=pil, return_tensors="pt").to(self.dev)
            out = self.model(**inp)
            v = out.pooler_output[0]
            v = v / v.norm()
        return v.cpu()

    @staticmethod
    def cos(a, b):
        return float((a * b).sum())


def crops(pil):
    w, h = pil.size
    yield pil
    cw, ch = int(w * 0.5), int(h * 0.5)
    for fy in (0.0, 0.25, 0.5):
        for fx in (0.0, 0.25, 0.5):
            x0, y0 = int(w * fx), int(h * fy)
            yield pil.crop((x0, y0, x0 + cw, y0 + ch))


def ref_embedding(dino, path):
    from PIL import Image
    im = Image.open(path).convert("RGB")
    w, h = im.size
    top = im.crop((0, 0, w, int(h * 0.6)))
    v = dino.embed(im) + dino.embed(top)
    return v / v.norm()


def frame_score(dino, pil, ref_emb):
    return max(dino.cos(dino.embed(c), ref_emb) for c in crops(pil))


def extract_frames(cfg, clip, out_dir, n=6):
    """n frames evenly spaced -> list of paths."""
    os.makedirs(out_dir, exist_ok=True)
    stem = os.path.splitext(os.path.basename(clip))[0]
    pattern = os.path.join(out_dir, f"{stem}_%02d.png")
    existing = sorted(glob.glob(os.path.join(out_dir, f"{stem}_*.png")))
    if len(existing) >= n:
        return existing[:n]
    # count frames via ffmpeg (null muxer)
    r = subprocess.run([cfg["ffmpeg"], "-hide_banner", "-i", clip, "-map", "0:v", "-c", "copy", "-f", "null", "-"],
                       capture_output=True, text=True)
    frames = 0
    for line in r.stderr.splitlines():
        if "frame=" in line:
            try:
                frames = int(line.split("frame=")[1].split()[0])
            except ValueError:
                pass
    frames = max(frames, n)
    idx = [round(i * (frames - 1) / (n - 1)) for i in range(n)]
    sel = "+".join(f"eq(n\\,{i})" for i in idx)
    subprocess.run([cfg["ffmpeg"], "-hide_banner", "-loglevel", "error", "-y", "-i", clip,
                    "-vf", f"select='{sel}'", "-vsync", "vfr", pattern], check=True)
    return sorted(glob.glob(os.path.join(out_dir, f"{stem}_*.png")))[:n]


def contact_sheet(rows, out_path, cell=320, refs=None):
    """rows: list of (label, [image paths]); refs: dict label->ref path shown in col 0."""
    from PIL import Image, ImageDraw
    ncols = 1 + max(len(r[1]) for r in rows)
    W, H = ncols * cell, len(rows) * cell + 24 * len(rows)
    sheet = Image.new("RGB", (W, H), (30, 30, 30))
    d = ImageDraw.Draw(sheet)
    y = 0
    for label, paths in rows:
        d.text((4, y + 4), label, fill=(255, 255, 255))
        y += 24
        col_paths = ([refs.get(label)] if refs and refs.get(label) else [None]) + list(paths)
        for i, p in enumerate(col_paths):
            if not p:
                continue
            im = Image.open(p).convert("RGB")
            im.thumbnail((cell - 4, cell - 4))
            sheet.paste(im, (i * cell + 2, y + 2))
        y += cell
    sheet.save(out_path)
    return out_path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--what", default="all", choices=["keyframes", "clips", "all"])
    ap.add_argument("--warmup", action="store_true")
    a = ap.parse_args()
    cfg = L.lab_config()
    paths = L.run_paths(cfg, a.run)
    log = L.setup_logging(paths["root"], "QC")
    from PIL import Image
    dino = Dino(cfg)
    if a.warmup:
        log.info("dino ready on %s", dino.dev)
        return
    cast, by_id = S.load_cast(cfg["lab_dir"])
    shots_all = S.load_shots(paths["shots"])
    shot_by_id = {s["id"]: s for s in shots_all}
    refs = {c["id"]: ref_embedding(dino, os.path.join(cfg["lab_dir"], c["canonical"])) for c in cast["characters"]}
    scores = {"keyframes": {}, "clips": {}, "cross": {}, "neg": {}}

    # negative control
    if os.path.isfile(NEG_KEYFRAME):
        im = Image.open(NEG_KEYFRAME).convert("RGB")
        for cid, emb in refs.items():
            scores["neg"][cid] = round(frame_score(dino, im, emb), 4)
        log.info("negative control (v4 s05 strangers): %s", scores["neg"])
    # ref vs ref (upper bound / separation)
    scores["ref_vs_ref"] = {f"{x}-{y}": round(dino.cos(refs[x], refs[y]), 4) for x in refs for y in refs if x < y}

    frames_dir = os.path.join(paths["qc"], "frames")
    mean_emb = {}   # (engine, shot) -> embedding of best zuri crop per frame averaged

    def score_images(engine, shot_id, images):
        shot = shot_by_id.get(shot_id.replace("_probe_refs", "").replace("_probe_norefs", "").replace("_norefs", "")) or shot_by_id.get("s01")
        per_child = {}
        embs = []
        for cid in shot["cast"]:
            vals = []
            for p in images:
                im = Image.open(p).convert("RGB")
                vals.append(frame_score(dino, im, refs[cid]))
            per_child[cid] = round(sum(vals) / len(vals), 4)
        # whole-frame embeddings for drift + cross
        for p in images:
            embs.append(dino.embed(Image.open(p).convert("RGB")))
        drift = min(dino.cos(embs[0], e) for e in embs[1:]) if len(embs) > 1 else 1.0
        m = sum(embs) / len(embs)
        mean_emb[(engine, shot_id)] = m / m.norm()
        return {"id": per_child, "drift": round(drift, 4), "frames": len(images)}

    rows_kf, rows_cl = [], []
    if a.what in ("keyframes", "all"):
        for engine in sorted(os.listdir(paths["keyframes"])):
            for p in sorted(glob.glob(os.path.join(paths["keyframes"], engine, "*.png"))):
                sid = os.path.basename(p).split("_")[0]
                sc = score_images(engine, sid, [p])
                scores["keyframes"][f"{engine}/{sid}"] = sc
                rows_kf.append((f"{engine} {sid} id={sc['id']}", [p]))
                log.info("keyframe %s/%s: %s", engine, sid, sc)
    if a.what in ("clips", "all"):
        for p in sorted(glob.glob(os.path.join(paths["clips"], "*.mp4"))):
            name = os.path.splitext(os.path.basename(p))[0]
            sid = name.split("_")[0]
            fr = extract_frames(cfg, p, frames_dir)
            sc = score_images("clip", name, fr)
            scores["clips"][name] = sc
            rows_cl.append((f"{name} id={sc['id']} drift={sc['drift']}", fr))
            log.info("clip %s: %s", name, sc)
    # cross-shot (zuri present in all 3)
    for (e1, s1), m1 in mean_emb.items():
        for (e2, s2), m2 in mean_emb.items():
            if e1 == e2 and s1 < s2:
                scores["cross"][f"{e1}:{s1}~{s2}"] = round(dino.cos(m1, m2), 4)

    canon = {}
    for label, _ in rows_kf + rows_cl:
        canon[label] = os.path.join(cfg["lab_dir"], by_id["zuri"]["canonical"])
    if rows_kf:
        contact_sheet(rows_kf, os.path.join(paths["qc"], "contact_keyframes.png"), refs=canon)
    if rows_cl:
        contact_sheet(rows_cl, os.path.join(paths["qc"], "contact_clips.png"), refs=canon)

    with open(os.path.join(paths["qc"], "scores.json"), "w", encoding="utf-8") as f:
        json.dump(scores, f, indent=2)
    lines = [f"# QC {a.run}", "", f"Negativkontrolle (Fremdkinder v4-s05): {scores['neg']}",
             f"Ref-vs-Ref (Trennschaerfe der Refs untereinander): {scores['ref_vs_ref']}", "",
             "## Keyframes", "| engine/shot | id-scores | ", "|---|---|"]
    for k, v in scores["keyframes"].items():
        lines.append(f"| {k} | {v['id']} |")
    lines += ["", "## Clips", "| clip | id-scores | drift |", "|---|---|---|"]
    for k, v in scores["clips"].items():
        lines.append(f"| {k} | {v['id']} | {v['drift']} |")
    lines += ["", "## Cross-Shot (gleiche Engine)", "| pair | cos |", "|---|---|"]
    for k, v in scores["cross"].items():
        lines.append(f"| {k} | {v} |")
    with open(os.path.join(paths["qc"], "report.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    log.info("report -> %s", os.path.join(paths["qc"], "report.md"))


if __name__ == "__main__":
    main()
