"""kidsong.lipsync — real Wav2Lip lip-sync as a GATED post-process step.

Viability is pixel-proven (investigation at ``D:\\brainrot\\lipsync-test``): the
gold-standard torch ``wav2lip_gan.pth`` (96px) drives the MOUTH of our stylized
CoComelon-style toddlers straight from the song audio while the toon look
survives, because our characters keep human face topology so SCRFD/insightface
detects them. This module wires that into the kidsong edit as an OPTIONAL step
that is OFF by default (``kidsong.lipsync.enabled = false``) — with it off, the
pipeline is byte-identical to today (``apply`` returns its inputs unchanged and
generate.py never even imports the heavy path).

Two halves, deliberately split across two Python environments (same pattern as
ACE-Step, which the repo venv also never imports in-process):

  * ORCHESTRATION (this file, imported by the repo venv): pure-stdlib. Given the
    beat-cut ``cut_list`` + ``renders`` map, it works out each qualifying cut's
    SONG-TIME window, slices ``output/<base>.wav`` for that window, shells the
    per-shot take + slice into the worker, and — on success — points that cut's
    ``src`` at the freshly lip-synced clip. Any failure passes the cut through
    UNCHANGED (never corrupts an episode).

  * WORKER (``--worker`` in this same file, run under
    ``kidsong.lipsync.python`` — the isolated lipsync venv that carries torch +
    the ``wav2lip-onnx-HQ`` repo + the checkpoint): detect the centered/largest
    singing face, FFHQ-align the mouth region, run wav2lip mouth-gen from the
    audio, elliptical-mask blend the new mouth back. Body/camera/background/
    animation are untouched — only the mouth changes. Frames where detection
    drops (fast cut, extreme head turn) pass through unchanged. The inference
    math is copied verbatim from the proven investigation harness
    (``orig_run.py`` ``--mode sync``, the run that produced
    ``out/orig_sync_toon/synced.mp4``).

The repo venv NEVER imports cv2/torch/numpy from here — those live only inside
the worker functions, reached solely by the ``--worker`` subprocess.
"""
import argparse
import json
import os
import subprocess
import sys
import wave

# --------------------------------------------------------------- config ---
#: Defaults point at the proven investigation harness. The gold checkpoint is
#: the TORCH ``wav2lip_gan.pth`` (96px) — the onnx exports in that tree have a
#: near-dead audio branch (silent-corruption trap) and are deliberately NOT the
#: default. ``model`` is 96px because that is the only checkpoint that exists in
#: the harness and the only one pixel-verified on our toon faces; "384px" is
#: accepted for forward-compat but needs a 384 checkpoint that is not shipped.
_DEFAULTS = {
    "enabled": False,
    "python": "D:/brainrot/lipsync-test/venv/Scripts/python.exe",
    # The dir carrying audio.py + hparams.py + models/wav2lip.py (torch model).
    "wav2lip_src": "D:/brainrot/lipsync-test/orig",
    # The wav2lip-onnx-HQ clone carrying utils/retinaface + utils/face_alignment
    # + utils/scrfd_2.5g_bnkps.onnx (the SCRFD detector that finds toon faces).
    "repo": "D:/brainrot/lipsync-test/wav2lip-onnx-HQ",
    "checkpoint": "D:/brainrot/lipsync-test/orig/wav2lip_gan.pth",
    "model": "96px",             # 96px | 384px (only 96px is shipped/proven)
    "enhancer": "none",          # none | light (light == a gentle unsharp, NOT GFPGAN)
    # Apply lip-sync only where a single singing face is prominent. Wides and
    # inserts (object closeups, subject-only shots) have no clear singing face.
    "shot_types": ["closeup", "medium"],
    "pads": 4,                   # vertical pad on the aligned mouth crop
    "det_thresh": 0.3,           # SCRFD detection threshold
    "timeout_seconds": 1800,     # per-shot worker wall-clock cap (CPU is slow)
}


def resolve_config(cfg):
    """The effective ``kidsong.lipsync`` settings, defaults filled in.

    A plain dict merge over ``_DEFAULTS`` — never mutates ``cfg``. ``shot_types``
    is normalized to a lower-cased set for gating.
    """
    ks = (cfg or {}).get("kidsong", {}) if isinstance(cfg, dict) else {}
    raw = (ks or {}).get("lipsync", {}) or {}
    out = dict(_DEFAULTS)
    for k, v in raw.items():
        if not str(k).startswith("_"):
            out[k] = v
    out["enabled"] = bool(out.get("enabled", False))
    types = out.get("shot_types") or []
    out["shot_types_set"] = {str(t).strip().lower() for t in types}
    return out


# ------------------------------------------------------- window + gating ---
def _shot_type_map(shotlist):
    """``shot_id -> shot_type`` from a shotlist (dict-with-shots or bare list)."""
    shots = shotlist.get("shots") if isinstance(shotlist, dict) else shotlist
    out = {}
    for s in shots or []:
        sid = s.get("id")
        if sid is not None:
            out[sid] = str(s.get("shot_type", "")).strip().lower()
    return out


def plan_lipsync(cut_list, shotlist, ls_cfg):
    """Which cuts get lip-synced, and over what SONG-TIME window.

    Returns a list of ``{"index", "shot_id", "src", "start", "end", "window"}``
    — one per cut whose shot's ``shot_type`` is in the configured set and whose
    window is positive. ``start``/``end`` are the cut's own song-time boundaries
    (``build_cut_list`` already resolved them), so the audio slice is exactly the
    audio that plays under that cut; the shot take is played from its own t=0, so
    frame ``i`` (at ``i/fps`` from the cut start) is driven by ``song[start +
    i/fps]``. Gating keys on the cut's ``shot_id`` (the shot actually on screen),
    falling back to ``src`` for a reused render.
    """
    types = _shot_type_map(shotlist)
    want = ls_cfg.get("shot_types_set")
    if want is None:
        want = {str(t).strip().lower() for t in (ls_cfg.get("shot_types") or [])}
    plans = []
    for i, cut in enumerate(cut_list):
        sid = cut.get("shot_id")
        src = cut.get("src", sid)
        stype = types.get(sid) or types.get(src) or ""
        if stype not in want:
            continue
        start = float(cut.get("start", 0.0))
        end = float(cut.get("end", 0.0))
        if end - start <= 0:
            continue
        plans.append({
            "index": i,
            "shot_id": sid,
            "src": src,
            "start": start,
            "end": end,
            "window": round(end - start, 4),
        })
    return plans


# --------------------------------------------------------- audio slicing ---
def slice_wav(src_wav, start, end, out_wav):
    """Write ``src_wav``'s ``[start, end)`` seconds to ``out_wav`` (PCM path).

    Stdlib ``wave`` only, format-preserving (channels/width/rate carried over) so
    it is testable with no ffmpeg/network/model. ``end`` is clamped to the file
    length; a window that starts past the end yields a single frame rather than
    an empty file (an empty wav would make the worker's mel empty). Raises on a
    non-PCM/compressed wav — the caller falls back to ffmpeg via ``_slice``.
    """
    with wave.open(src_wav, "rb") as wf:
        nch = wf.getnchannels()
        width = wf.getsampwidth()
        rate = wf.getframerate()
        total = wf.getnframes()
        comp = wf.getcomptype()
        if comp != "NONE":
            raise ValueError(f"{src_wav}: compressed wav ({comp}) not sliceable by stdlib")
        s = max(0, int(round(float(start) * rate)))
        e = min(total, int(round(float(end) * rate)))
        s = min(s, max(0, total - 1))
        if e <= s:
            e = min(total, s + 1)
        wf.setpos(s)
        frames = wf.readframes(e - s)
    with wave.open(out_wav, "wb") as ow:
        ow.setnchannels(nch)
        ow.setsampwidth(width)
        ow.setframerate(rate)
        ow.writeframes(frames)
    return out_wav


def _slice(src_wav, start, end, out_wav):
    """Slice ``[start, end)`` of ``src_wav`` to ``out_wav`` — stdlib first, then
    ffmpeg for anything ``wave`` cannot open (float wavs, odd codecs)."""
    try:
        return slice_wav(src_wav, start, end, out_wav)
    except Exception:
        subprocess.run(
            ["ffmpeg", "-y", "-v", "error", "-ss", f"{float(start):.4f}",
             "-to", f"{float(end):.4f}", "-i", src_wav, out_wav],
            check=True,
        )
        return out_wav


# ------------------------------------------------------------- subprocess ---
def _worker_cmd(ls_cfg, face, audio, out_path):
    """The ``[python, this_file, --worker, ...]`` command driving one shot."""
    return [
        ls_cfg["python"], os.path.abspath(__file__), "--worker",
        "--face", os.path.abspath(face),
        "--audio", os.path.abspath(audio),
        "--out", os.path.abspath(out_path),
        "--wav2lip-src", ls_cfg["wav2lip_src"],
        "--repo", ls_cfg["repo"],
        "--checkpoint", ls_cfg["checkpoint"],
        "--model", str(ls_cfg["model"]),
        "--enhancer", str(ls_cfg["enhancer"]),
        "--pads", str(ls_cfg["pads"]),
        "--det-thresh", str(ls_cfg["det_thresh"]),
    ]


def _default_runner(ls_cfg, face, audio, out_path, log=None):
    """Run the worker subprocess. Returns True iff it wrote a non-trivial file.

    Never raises: a dead interpreter, a missing model, a timeout or a crash all
    return False so the caller passes the shot through un-synced. The worker's
    stderr tail is logged for diagnosis.
    """
    cmd = _worker_cmd(ls_cfg, face, audio, out_path)
    try:
        proc = subprocess.run(
            cmd, capture_output=True, text=True,
            timeout=float(ls_cfg.get("timeout_seconds", 1800)),
        )
    except Exception as exc:  # timeout, interpreter missing, OSError, …
        if log:
            log(f"    lipsync worker could not run ({exc}) — passing shot through")
        return False
    if proc.returncode != 0:
        if log:
            tail = (proc.stderr or proc.stdout or "").strip().splitlines()[-4:]
            log(f"    lipsync worker failed (rc={proc.returncode}): {' | '.join(tail)}")
        return False
    if not (os.path.exists(out_path) and os.path.getsize(out_path) > 0):
        if log:
            log("    lipsync worker returned 0 but wrote no output — passing through")
        return False
    return True


# ---------------------------------------------------------------- apply ---
def apply(cfg, cut_list, renders, shotlist, voice_path, build_dir,
          on_progress=None, runner=None):
    """Lip-sync the qualifying cuts and return ``(cut_list, renders)`` for edit.

    DISABLED (``kidsong.lipsync.enabled`` false): returns the SAME ``cut_list``
    and ``renders`` objects, untouched — the byte-identical guarantee.

    ENABLED: for each cut ``plan_lipsync`` selects, slice ``voice_path`` to that
    cut's window, run the worker on that cut's source take, and — only on success
    — mint a per-cut ``src`` key pointing at the synced clip (so
    ``edit.assemble`` plays the lip-synced take for that cut while every other
    cut, and every reuse of the same source elsewhere, is unaffected). The synced
    clip has the SAME dimensions and frame count as the source render, so the
    cut's duration/loop behaviour in ``assemble`` is unchanged — only the mouth
    moves. Any per-shot failure leaves that cut exactly as it was.

    Writes only under ``build_dir`` (a NEW directory) — never touches existing
    takes/finals. ``runner`` is injectable for tests; it defaults to the real
    subprocess.
    """
    def log(msg):
        if on_progress:
            on_progress(msg)

    ls = resolve_config(cfg)
    if not ls["enabled"]:
        return cut_list, renders

    plans = plan_lipsync(cut_list, shotlist, ls)
    if not plans:
        log("Lip-sync: no closeup/medium cuts to sync — nothing to do.")
        return cut_list, renders

    os.makedirs(build_dir, exist_ok=True)
    runner = runner or _default_runner

    new_cuts = [dict(c) for c in cut_list]
    new_renders = dict(renders)
    synced = 0
    for p in plans:
        idx = p["index"]
        src = p["src"]
        src_render = new_renders.get(src)
        if not src_render or not os.path.exists(src_render):
            log(f"  cut {idx} ({p['shot_id']}): no source render for {src!r} — passing through")
            continue
        slice_path = os.path.join(build_dir, f"cut{idx:02d}_{p['shot_id']}.slice.wav")
        out_path = os.path.join(build_dir, f"cut{idx:02d}_{p['shot_id']}.synced.mp4")
        log(f"  cut {idx} ({p['shot_id']}, {p['window']:.2f}s): lip-syncing…")
        try:
            _slice(voice_path, p["start"], p["end"], slice_path)
        except Exception as exc:
            log(f"    could not slice audio ({exc}) — passing shot through")
            continue
        ok = False
        try:
            ok = runner(ls, src_render, slice_path, out_path, log=log)
        except TypeError:
            # A test/legacy runner without the log kwarg.
            ok = runner(ls, src_render, slice_path, out_path)
        except Exception as exc:
            log(f"    lipsync worker raised ({exc}) — passing shot through")
            ok = False
        if ok and os.path.exists(out_path) and os.path.getsize(out_path) > 0:
            key = f"__lipsync__{idx:02d}_{p['shot_id']}"
            new_renders[key] = out_path
            new_cuts[idx]["src"] = key
            new_cuts[idx]["lipsynced"] = True
            synced += 1
        else:
            # Pass-through: leave the cut pointing at its original render.
            log(f"    cut {idx} ({p['shot_id']}): not synced — using original take")

    if synced == 0:
        # Nothing actually changed — hand back the originals so a downstream
        # identity check still holds and no stray keys leak into assemble.
        log("Lip-sync: no cut was synced — using the original takes.")
        return cut_list, renders

    log(f"Lip-sync: {synced}/{len(plans)} eligible cut(s) synced.")
    return new_cuts, new_renders


# =========================================================================
# WORKER — runs ONLY under the isolated lipsync interpreter (--worker).
# Heavy imports (cv2/numpy/torch/onnxruntime + the wav2lip repo) live here so
# the repo venv importing this module for orchestration never pulls them in.
# =========================================================================
def _worker_main(args):
    import cv2
    import numpy as np

    # The wav2lip repo + the torch-model/audio source dir go on sys.path; chdir
    # into the repo so its relative asset lookups (temp/, hparams) resolve — this
    # mirrors the proven harness (orig_run.py) exactly.
    sys.path.insert(0, args.wav2lip_src)
    sys.path.insert(0, args.repo)
    os.chdir(args.repo)

    import onnxruntime
    onnxruntime.set_default_logger_severity(3)
    import audio
    import torch
    from models.wav2lip import Wav2Lip
    from utils.face_alignment import get_cropped_head_256
    from utils.retinaface import RetinaFace

    img_size = 96 if str(args.model) != "384px" else 384
    if img_size != 96:
        # Only the 96px torch gold checkpoint is shipped/proven; a 384 run needs
        # a 384 checkpoint + matching architecture that this tree does not carry.
        print("ERROR: only the 96px model is available in this environment "
              "(no 384px checkpoint shipped).", file=sys.stderr)
        return 3
    pad_y = max(-15, min(int(args.pads), 15))

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # --- detector + torch wav2lip gold model -----------------------------
    det = RetinaFace(os.path.join(args.repo, "utils/scrfd_2.5g_bnkps.onnx"),
                     provider=["CPUExecutionProvider"])
    model = Wav2Lip()
    ckpt = torch.load(args.checkpoint, map_location=device)
    sd = ckpt["state_dict"] if isinstance(ckpt, dict) and "state_dict" in ckpt else ckpt
    sd = {k.replace("module.", ""): v for k, v in sd.items()}
    model.load_state_dict(sd)
    model = model.to(device).eval()

    # --- read the face video --------------------------------------------
    cap = cv2.VideoCapture(args.face)
    fps = cap.get(cv2.CAP_PROP_FPS) or 24.0
    frames = []
    while True:
        ok, f = cap.read()
        if not ok:
            break
        frames.append(f)
    cap.release()
    if not frames:
        print("ERROR: no frames read from face video", file=sys.stderr)
        return 4
    frame_h, frame_w = frames[0].shape[:2]
    n = len(frames)

    # --- audio -> 16k mono -> mel chunks, padded to n frames -------------
    import tempfile
    tmp16 = os.path.join(tempfile.gettempdir(),
                         f"lipsync_{os.getpid()}_16k.wav")
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", args.audio,
                    "-ac", "1", "-ar", "16000", tmp16], check=True)
    wav = audio.load_wav(tmp16, 16000)
    mel = audio.melspectrogram(wav)
    if np.isnan(mel.reshape(-1)).sum():
        print("ERROR: mel has NaNs (bad audio)", file=sys.stderr)
        return 5
    mel_step = 16
    mim = 80. / fps
    mel_chunks = []
    i = 0
    while True:
        s = int(i * mim)
        if s + mel_step > len(mel[0]):
            mel_chunks.append(mel[:, len(mel[0]) - mel_step:])
            break
        mel_chunks.append(mel[:, s:s + mel_step])
        i += 1
    # Pad/truncate the mel timeline to EXACTLY the video's frame count so the
    # synced clip keeps the source render's duration (frames past the audio
    # window — beyond this cut's slice — get the last chunk and are trimmed away
    # by the editor's per-cut subclip anyway).
    while len(mel_chunks) < n:
        mel_chunks.append(mel_chunks[-1])
    mel_chunks = mel_chunks[:n]
    try:
        os.remove(tmp16)
    except OSError:
        pass

    # --- blend masks (verbatim from the proven harness) ------------------
    static_face_mask = np.zeros((224, 224), dtype=np.uint8)
    static_face_mask = cv2.ellipse(static_face_mask, (112, 162), (62, 54), 0, 0, 360, (255, 255, 255), -1)
    static_face_mask = cv2.ellipse(static_face_mask, (112, 122), (46, 23), 0, 0, 360, (0, 0, 0), -1)
    static_face_mask = cv2.resize(static_face_mask, (256, 256))
    static_face_mask = cv2.rectangle(static_face_mask, (0, 246), (246, 246), (0, 0, 0), -1)
    static_face_mask = cv2.cvtColor(static_face_mask, cv2.COLOR_GRAY2RGB) / 255
    static_face_mask = cv2.GaussianBlur(static_face_mask, (19, 19), cv2.BORDER_DEFAULT)

    sub_face_mask = np.zeros((256, 256), dtype=np.uint8)
    sub_face_mask = cv2.rectangle(sub_face_mask, (42, 65 - pad_y), (214, 249), (255, 255, 255), -1)
    sub_face_mask = cv2.GaussianBlur(sub_face_mask.astype(np.uint8), (29, 29), cv2.BORDER_DEFAULT)
    sub_face_mask = cv2.cvtColor(sub_face_mask, cv2.COLOR_GRAY2RGB) / 255

    def pick_center_face(kpss, bboxes):
        """The face closest to frame centre — our singing subject is centred;
        edge/background faces are left untouched (their frames pass through the
        blend as the ORIGINAL pixels because only the centre face is aligned)."""
        if kpss is None or len(kpss) == 0:
            return None
        cx, cy = frame_w / 2.0, frame_h / 2.0
        best, bd = None, 1e18
        for k in range(len(kpss)):
            x1, y1, x2, y2 = bboxes[k][:4]
            d = ((x1 + x2) / 2 - cx) ** 2 + ((y1 + y2) / 2 - cy) ** 2
            if d < bd:
                bd, best = d, kpss[k]
        return best

    def infer(sub, m):
        ib = np.asarray([sub]).astype(np.float32)
        masked = ib.copy()
        masked[:, img_size // 2:] = 0
        ib = np.concatenate((masked, ib), axis=3) / 255.
        ib = torch.FloatTensor(ib.transpose(0, 3, 1, 2)).to(device)
        mb = torch.FloatTensor(
            np.reshape(np.asarray([m]), [1, m.shape[0], m.shape[1], 1]).transpose(0, 3, 1, 2)
        ).to(device)
        with torch.no_grad():
            p = model(mb, ib)[0].cpu().numpy().transpose(1, 2, 0) * 255.
        return p.astype(np.uint8)

    light = str(args.enhancer) == "light"

    def enhance(pred):
        # A GENTLE unsharp only — NOT GFPGAN/GPEN (those de-stylize toward
        # realism and break the toon look). Recovers a little of the lower-face
        # softening wav2lip introduces without changing the style.
        blur = cv2.GaussianBlur(pred, (0, 0), 1.0)
        return cv2.addWeighted(pred, 1.4, blur, -0.4, 0)

    silent = args.out + ".silent.mp4"
    vw = cv2.VideoWriter(silent, cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame_w, frame_h))
    ndet = 0
    for fr, m in zip(frames, mel_chunks):
        bboxes, kpss = det.detect(fr, input_size=(320, 320), det_thresh=float(args.det_thresh))
        kps = pick_center_face(kpss, bboxes)
        if kps is None:
            vw.write(fr)  # detection dropped — pass the frame through UNCHANGED
            continue
        ndet += 1
        crop, mat = get_cropped_head_256(fr, kps, size=256, scale=1.0)
        mat_rev = cv2.invertAffineTransform(mat)
        aligned_orig = crop.copy()
        sub = cv2.resize(crop[65 - pad_y:241 - pad_y, 62:194], (img_size, img_size))
        pred = infer(sub, m)
        if light:
            pred = enhance(pred)
        p = cv2.resize(pred, (132, 176))
        p_aligned = crop.copy()
        p_aligned[65 - pad_y:241 - pad_y, 62:194] = p
        aligned = (sub_face_mask * p_aligned + (1 - sub_face_mask) * aligned_orig).astype(np.uint8)
        mask = cv2.warpAffine(static_face_mask, mat_rev, (frame_w, frame_h))
        dealigned = cv2.warpAffine(aligned, mat_rev, (frame_w, frame_h))
        res = (mask * dealigned + (1 - mask) * fr).astype(np.uint8)
        vw.write(res)
    vw.release()

    # Re-encode to h264/yuv420p (video only — the editor strips take audio and
    # muxes the song itself). mp4v out reads fine in moviepy but h264 keeps the
    # take library uniform and avoids any odd reader edge cases.
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", silent,
                    "-c:v", "libx264", "-pix_fmt", "yuv420p", args.out], check=True)
    try:
        os.remove(silent)
    except OSError:
        pass

    print(json.dumps({"frames": n, "faces_driven": ndet, "fps": fps,
                      "device": device, "enhancer": args.enhancer}))
    return 0


def _build_worker_parser():
    p = argparse.ArgumentParser(description="Wav2Lip lip-sync worker (isolated env)")
    p.add_argument("--worker", action="store_true")
    p.add_argument("--face")
    p.add_argument("--audio")
    p.add_argument("--out")
    p.add_argument("--wav2lip-src", dest="wav2lip_src")
    p.add_argument("--repo")
    p.add_argument("--checkpoint")
    p.add_argument("--model", default="96px")
    p.add_argument("--enhancer", default="none")
    p.add_argument("--pads", type=int, default=4)
    p.add_argument("--det-thresh", dest="det_thresh", type=float, default=0.3)
    return p


if __name__ == "__main__":
    _args = _build_worker_parser().parse_args()
    if _args.worker:
        raise SystemExit(_worker_main(_args))
    print("pipeline.kidsong.lipsync: nothing to do without --worker "
          "(orchestration is imported, not run as a script).", file=sys.stderr)
    raise SystemExit(2)
