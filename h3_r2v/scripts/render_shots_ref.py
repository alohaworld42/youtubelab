#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
render_shots_ref.py - Reference-conditioned Shots (MiniMax H3 Reference-to-Video) via ComfyUI-API.

Prinzip: derselbe fixe Cast-Referenzsatz (canonical.png) + feste Seeds in JEDEN Shot.
Kein Keyframe-Wuerfeln pro Shot. Checkpointing via output/manifest.json, Resume per --resume.
Nur Python-Stdlib. Node-Schemas werden zur Laufzeit von /object_info gelesen (kein Raten).
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
import traceback
import urllib.error
import urllib.parse
import urllib.request
import uuid


def log(msg):
    print(time.strftime("[%H:%M:%S] ") + str(msg), flush=True)


def die(msg, code=2):
    log("FEHLER: " + str(msg))
    sys.exit(code)


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path, data):
    d = os.path.dirname(path)
    if d:
        os.makedirs(d, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


# ---------------- HTTP ----------------


def http_json(base, path, payload=None, timeout=60):
    url = base.rstrip("/") + path
    data = None
    headers = {"Accept": "application/json"}
    if payload is not None:
        data = json.dumps(payload).encode("utf-8")
        headers["Content-Type"] = "application/json"
    req = urllib.request.Request(url, data=data, headers=headers)
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode("utf-8", "replace"))


def http_download(base, item, dest):
    q = urllib.parse.urlencode({
        "filename": item["filename"],
        "subfolder": item.get("subfolder", ""),
        "type": item.get("type", "output"),
    })
    url = base.rstrip("/") + "/view?" + q
    with urllib.request.urlopen(url, timeout=600) as r, open(dest, "wb") as f:
        shutil.copyfileobj(r, f)


def upload_image(base, path, subfolder, arcname):
    bnd = "----kv" + uuid.uuid4().hex
    with open(path, "rb") as f:
        blob = f.read()
    parts = [("--" + bnd + "\r\n").encode("ascii"),
             ('Content-Disposition: form-data; name="image"; filename="' + arcname + '"\r\n').encode("utf-8"),
             b"Content-Type: application/octet-stream\r\n\r\n", blob, b"\r\n"]
    for k, v in (("overwrite", "true"), ("subfolder", subfolder)):
        parts += [("--" + bnd + "\r\n").encode("ascii"),
                  ('Content-Disposition: form-data; name="' + k + '"\r\n\r\n').encode("ascii"),
                  v.encode("ascii"), b"\r\n"]
    parts.append(("--" + bnd + "--\r\n").encode("ascii"))
    req = urllib.request.Request(base.rstrip("/") + "/upload/image", data=b"".join(parts),
                                 headers={"Content-Type": "multipart/form-data; boundary=" + bnd})
    with urllib.request.urlopen(req, timeout=300) as r:
        j = json.loads(r.read().decode("utf-8", "replace"))
    sub = (j.get("subfolder") or subfolder or "").strip("/")
    name = j.get("name") or arcname
    return (sub + "/" + name) if sub else name


# ---------------- Template / Nodes ----------------


def find_ref_node(wf):
    for nid, node in wf.items():
        ct = str(node.get("class_type", ""))
        ctl = ct.lower()
        if "minimax" in ctl and "h3" in ctl:
            return nid, ct
    for nid, node in wf.items():
        ct = str(node.get("class_type", ""))
        ctl = ct.lower()
        if "reference" in ctl and "video" in ctl:
            return nid, ct
    return None, None


def load_template(path):
    if not path or not os.path.isfile(path):
        die("Kein Workflow-Template gefunden ('%s'). Loesungen:\n"
            "  1) run_proof.ps1 -CaptureTemplate  (holt den letzten H3-Run aus der ComfyUI-Historie)\n"
            "  2) -Template <pfad> auf eine API-Export-JSON zeigen" % path)
    wf = load_json(path)
    if isinstance(wf, dict) and isinstance(wf.get("nodes"), list):
        die("Template ist im UI-Format. In ComfyUI: Dev Mode -> 'Export (API)', "
            "oder run_proof.ps1 -CaptureTemplate.")
    if not isinstance(wf, dict) or not wf:
        die("Template unlesbar: %s" % path)
    return wf


def capture_template(base, dest):
    try:
        h = http_json(base, "/history?max_items=200", timeout=120)
    except Exception as e:
        die("ComfyUI-Historie nicht lesbar (%s). Laeuft ComfyUI?" % e)
    graph = None
    pid_used = None
    for pid, entry in h.items():
        pr = entry.get("prompt") or []
        g = pr[2] if len(pr) > 2 else None
        if isinstance(g, dict) and find_ref_node(g)[0]:
            graph, pid_used = g, pid
    if graph is None:
        die("Kein Reference-to-Video-Run in der Historie gefunden. "
            "Erst den alten H3-Ref2Video-Test laufen lassen, dann -CaptureTemplate erneut.")
    save_json(dest, graph)
    log("Template gesichert (prompt_id=%s) -> %s" % (pid_used, dest))


def get_class_info(base, cls):
    q = urllib.parse.quote(cls, safe="")
    try:
        j = http_json(base, "/object_info/" + q, timeout=120)
        if j and cls in j:
            return j[cls]
    except urllib.error.HTTPError:
        pass
    full = http_json(base, "/object_info", timeout=300)
    keys = [w[:4] for w in cls.lower().split()] or [cls.lower()[:4]]
    similar = [k for k in full.keys() if any(s in k.lower() for s in keys)]
    die("Node-Klasse '%s' nicht geladen. Aehnliche: %s" % (cls, similar[:12]))


def input_spec(oinfo):
    spec = oinfo.get("input") or {}
    ins = {}
    ins.update(spec.get("required") or {})
    ins.update(spec.get("optional") or {})
    return ins


def input_type(v):
    if isinstance(v, list) and v:
        first = v[0]
        if isinstance(first, str):
            meta = v[1] if len(v) > 1 and isinstance(v[1], dict) else {}
            return first, meta
        if isinstance(first, list):
            return "COMBO", {"choices": first}
    return "ANY", {}


def set_widget(ins, inputs, names, value, label, prefix=""):
    for n in names:
        if n not in ins or value is None:
            continue
        t, meta = input_type(ins[n])
        if t == "COMBO":
            choices = [str(c) for c in (meta.get("choices") or [])]
            if choices and str(value) not in choices:
                log("%s  %s: Wert %r nicht in Optionen %s - uebersprungen" % (prefix, label, value, choices[:10]))
                return False
        inputs[n] = value
        log("%s  %s: %s = %r" % (prefix, label, n, value))
        return True
    log("%s  %s: kein passender Input (%s) - uebersprungen" % (prefix, label, ", ".join(names)))
    return False


def find_upstream_text_node(wf, h3_id):
    h3 = wf.get(h3_id) or {}
    for k, v in (h3.get("inputs") or {}).items():
        if isinstance(v, list) and len(v) == 2:
            up = wf.get(str(v[0]))
            if not up:
                continue
            ct = str(up.get("class_type", "")).lower()
            if "textencode" in ct or "text_encode" in ct or "qwen" in ct or "prompt" in ct:
                for tk in ("text", "prompt", "caption"):
                    if isinstance(up.get("inputs", {}).get(tk), str):
                        return str(v[0]), tk
    return None, None


SAVE_HINTS = ("SaveVideo", "VHS_VideoCombine", "SaveAnimatedWEBP", "SaveAnimatedPNG", "SaveWEBM")


def fill_required(cls, oinfo, inputs, label=""):
    req = ((oinfo.get("input") or {}).get("required")) or {}
    for k, v in req.items():
        if k in inputs:
            continue
        t, meta = input_type(v)
        if t == "COMBO":
            choices = meta.get("choices") or []
            if not choices:
                die("%s: Combo-Input '%s' ohne Optionen" % (cls, k))
            inputs[k] = choices[0]
            log("%s  %s.%s = %r (erste Option)" % (label, cls, k, inputs[k]))
        elif isinstance(meta, dict) and "default" in meta:
            inputs[k] = meta["default"]
        else:
            die("%s: Pflicht-Input '%s' (Typ %s) nicht automatisch setzbar" % (cls, k, t))


def ensure_saver(base, wf, h3_id, h3_oinfo, prefix, label=""):
    for nid, node in wf.items():
        ct = str(node.get("class_type", ""))
        if not any(h.lower() in ct.lower() for h in SAVE_HINTS):
            continue
        inputs = node.setdefault("inputs", {})
        for k in ("filename_prefix", "filename"):
            if isinstance(inputs.get(k), str):
                inputs[k] = prefix
                log("%s saver: %s (%s), prefix='%s'" % (label, nid, ct, prefix))
                return
        log("%s saver vorhanden: %s (%s)" % (label, nid, ct))
        return
    outs = [str(o) for o in (h3_oinfo.get("output") or [])]
    if "VIDEO" not in outs:
        die("%s Template hat keinen Video-Saver und H3-Output ist %r. "
            "Einmal im WebUI laufen lassen, dann -CaptureTemplate." % (label, outs))
    cv = get_class_info(base, "CreateVideo")
    sv = get_class_info(base, "SaveVideo")
    cv_inputs = {}
    for k, v in input_spec(cv).items():
        if input_type(v)[0] == "IMAGE":
            cv_inputs[k] = [h3_id, 0]
            break
    fill_required("CreateVideo", cv, cv_inputs, label)
    sv_inputs = {"filename_prefix": prefix}
    for k, v in input_spec(sv).items():
        if input_type(v)[0] == "VIDEO":
            sv_inputs[k] = ["__cvnode__", 0]
            break
    fill_required("SaveVideo", sv, sv_inputs, label)
    wf["__cvnode__"] = {"class_type": "CreateVideo", "inputs": cv_inputs}
    wf["__svnode__"] = {"class_type": "SaveVideo", "inputs": sv_inputs}
    log("%s CreateVideo+SaveVideo angehaengt (prefix='%s')" % (label, prefix))


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


def retrieve_outputs(base, entry, dest_mp4, label, fps=24, ffmpeg_roots=()):
    cands = []
    for _nid, outs in (entry.get("outputs") or {}).items():
        if not isinstance(outs, dict):
            continue
        for _k, val in outs.items():
            if isinstance(val, list):
                for item in val:
                    if isinstance(item, dict) and item.get("filename"):
                        cands.append(item)

    def rank(c):
        n = str(c.get("filename", "")).lower()
        for i, ext in enumerate((".mp4", ".webm", ".mov", ".mkv", ".gif", ".webp")):
            if n.endswith(ext):
                return i
        return 90 if n.endswith(".png") or n.endswith(".jpg") else 95

    vids = sorted([c for c in cands if rank(c) < 90], key=rank)
    if vids:
        http_download(base, vids[0], dest_mp4)
        log("%s geladen: %s (%d Bytes)" % (label, dest_mp4, os.path.getsize(dest_mp4)))
        return True

    frames = sorted([c for c in cands if rank(c) == 90], key=lambda c: str(c.get("filename", "")))
    if frames:
        fdir = os.path.join(os.path.dirname(dest_mp4), "_frames_" + label)
        os.makedirs(fdir, exist_ok=True)
        local = []
        for i, fr in enumerate(frames):
            p = os.path.join(fdir, "f%05d.png" % i)
            http_download(base, fr, p)
            local.append(p)
        ff = find_ffmpeg(ffmpeg_roots)
        if not ff:
            log("%s nur Einzel frames erhalten und kein ffmpeg gefunden - Frames: %s" % (label, fdir))
            return False
        lst = os.path.join(fdir, "list.txt")
        with open(lst, "w", encoding="utf-8") as f:
            for p in local:
                f.write("file '%s'\n" % p.replace("\\", "/").replace("'", "'\\''"))
        subprocess.run([ff, "-y", "-f", "concat", "-safe", "0", "-i", lst, "-vf", "fps=%d" % fps,
                        "-c:v", "libx264", "-crf", "18", "-pix_fmt", "yuv420p", dest_mp4],
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return os.path.isfile(dest_mp4) and os.path.getsize(dest_mp4) > 0
    return False


def submit_and_wait(base, wf, timeout, label):
    cid = uuid.uuid4().hex
    resp = http_json(base, "/prompt", {"prompt": wf, "client_id": cid}, timeout=120)
    if resp.get("node_errors"):
        raise RuntimeError("Node-Fehler beim Queueing: " + json.dumps(resp["node_errors"])[:3000])
    pid = resp["prompt_id"]
    log("%s queued: %s" % (label, pid))
    t0 = time.time()
    last = 0.0
    while True:
        time.sleep(5)
        if time.time() - t0 > timeout:
            raise TimeoutError("Timeout nach %ds (prompt_id=%s)" % (timeout, pid))
        try:
            h = http_json(base, "/history/" + pid, timeout=60)
        except Exception:
            continue
        e = h.get(pid)
        if not e:
            if time.time() - last > 60:
                last = time.time()
                log("%s laeuft... %ds" % (label, int(time.time() - t0)))
            continue
        st = e.get("status") or {}
        for msg in (st.get("messages") or []):
            if isinstance(msg, list) and msg and msg[0] == "execution_error":
                d = msg[1] if len(msg) > 1 else {}
                raise RuntimeError("Node-Error %s: %s" % (d.get("node_type"), d.get("exception_message")))
        if st.get("completed"):
            log("%s fertig nach %ds" % (label, int(time.time() - t0)))
            return pid


def fetch_history(base, pid):
    h = http_json(base, "/history/" + pid, timeout=60)
    return h.get(pid) or {}


def fps_for(shot, wf):
    v = shot.get("fps")
    if v:
        return int(v)
    for node in wf.values():
        for k, val in (node.get("inputs") or {}).items():
            if "fps" in str(k).lower() and isinstance(val, (int, float)):
                return int(val)
    return 24


# ---------------- Main ----------------


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shots")
    ap.add_argument("--cast")
    ap.add_argument("--out")
    ap.add_argument("--template", default="")
    ap.add_argument("--comfy", default="http://127.0.0.1:8188")
    ap.add_argument("--timeout", type=int, default=2400)
    ap.add_argument("--only", default="")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--capture-template", dest="capture_template", default="")
    args = ap.parse_args()

    if args.capture_template:
        capture_template(args.comfy, args.capture_template)
        return

    for req in ("shots", "cast", "out"):
        if not getattr(args, req):
            die("--%s wird benoetigt" % req)

    cast_cfg = load_json(args.cast)
    shots_cfg = load_json(args.shots)
    cast_dir = os.path.dirname(os.path.abspath(args.cast))
    for c in cast_cfg["cast"]:
        if c.get("ref") and not os.path.isabs(c["ref"]):
            c["ref"] = os.path.abspath(os.path.join(cast_dir, c["ref"]))
    cast_by_id = {c["id"]: c for c in cast_cfg["cast"]}
    style = cast_cfg.get("style", "")
    no_extra = cast_cfg.get("no_extra", "")

    shots = shots_cfg["shots"]
    if args.only:
        want = [s.strip() for s in args.only.split(",") if s.strip()]
        shots = [s for s in shots if str(s.get("id")) in want]
        if not shots:
            die("Keine Shots matchen --only=%s" % args.only)

    for c in cast_cfg["cast"]:
        if not c.get("ref") or not os.path.isfile(c["ref"]):
            die("Referenzbild fehlt fuer '%s': %s" % (c.get("id"), c.get("ref")))

    out_dir = os.path.abspath(args.out)
    shots_dir = os.path.join(out_dir, "shots")
    debug_dir = os.path.join(out_dir, "debug")
    os.makedirs(shots_dir, exist_ok=True)
    os.makedirs(debug_dir, exist_ok=True)
    manifest_path = os.path.join(out_dir, "manifest.json")
    manifest = load_json(manifest_path) if (args.resume and os.path.isfile(manifest_path)) else {}

    try:
        st = http_json(args.comfy, "/system_stats", timeout=10)
        dev = (st.get("devices") or [{}])[0]
        log("ComfyUI OK: v%s | %s" % (st.get("system", {}).get("comfyui_version", "?"), dev.get("name", "?")))
    except Exception as e:
        die("ComfyUI nicht erreichbar (%s): %s" % (args.comfy, e))

    template = args.template or os.path.join(os.path.dirname(os.path.abspath(args.shots)), "h3_template.json")
    wf0 = load_template(template)
    h3_id, h3_cls = find_ref_node(wf0)
    if not h3_id:
        die("Kein Reference-to-Video-Node im Template. Klassen: %s" %
            sorted({str(n.get("class_type")) for n in wf0.values()})[:20])
    log("H3-Node: id=%s class=%s (template=%s)" % (h3_id, h3_cls, template))

    oinfo = get_class_info(args.comfy, h3_cls)
    ins = input_spec(oinfo)
    opt_keys = set(((oinfo.get("input") or {}).get("optional") or {}).keys())
    log("H3-Inputs: %s" % sorted(ins.keys()))

    prompt_key = next((k for k in ("prompt", "text", "positive", "caption") if k in ins), None)
    text_node_id = None
    text_key = None
    if not prompt_key:
        text_node_id, text_key = find_upstream_text_node(wf0, h3_id)
        if not text_node_id:
            die("Weder Prompt-Input am Node '%s' noch upstream Text-Node. Inputs: %s"
                % (h3_cls, sorted(ins.keys())))
        log("Prompt wird am upstream Node gesetzt: %s.%s" % (text_node_id, text_key))

    def is_img_type(v):
        t, meta = input_type(v)
        if t == "IMAGE":
            return True
        if t == "COMFY_AUTOGROW_V3":
            inner = (((meta.get("template") or {}).get("input") or {}).get("required") or {})
            return any(input_type(iv)[0] == "IMAGE" for iv in inner.values())
        return False

    img_inputs = [k for k, v in ins.items() if is_img_type(v)]
    if not img_inputs:
        die("Kein IMAGE/Autogrow-Referenz-Input am Node '%s'. Inputs: %s" % (h3_cls, sorted(ins.keys())))
    ref_key = next((k for k in img_inputs if "ref" in k.lower()), img_inputs[0])
    ref_is_autogrow = input_type(ins[ref_key])[0] == "COMFY_AUTOGROW_V3"
    ref_prefix = "ref_image_"
    if ref_is_autogrow:
        ref_prefix = (((input_type(ins[ref_key])[1].get("template") or {}).get("prefix")) or "ref_image_")
    log("Ref-Input: %s.%s (autogrow=%s prefix=%s)" % (h3_cls, ref_key, ref_is_autogrow, ref_prefix))

    ref_values = {}
    for c in cast_cfg["cast"]:
        if args.dry_run:
            ref_values[c["id"]] = "kidsvid2_refs/%s_canonical.png" % c["id"]
        else:
            ref_values[c["id"]] = upload_image(args.comfy, c["ref"], "kidsvid2_refs",
                                               "%s_canonical.png" % c["id"])
            log("Upload %s -> %s" % (c["id"], ref_values[c["id"]]))

    failed = 0
    for shot in shots:
        sid = str(shot.get("id", "shot"))
        label = "[%s]" % sid
        dest = os.path.join(shots_dir, sid + ".mp4")
        m = manifest.get(sid, {})
        if args.resume and m.get("status") == "done" and os.path.isfile(dest) and os.path.getsize(dest) > 0:
            log("%s fertig (Resume ueberspringt)" % label)
            continue

        members = [cast_by_id[c] for c in (shot.get("cast") or []) if c in cast_by_id]
        if not members:
            log("%s FEHLER: cast unbekannt/leer" % label)
            failed += 1
            continue
        action = str(shot.get("action", "")).strip()
        prompt_text = "%s Exactly %d children appear in this scene: %s. %s %s" % (
            action, len(members), ", ".join(x["name"] for x in members), no_extra, style)
        seed = int(shot.get("seed") or 123456789)

        wf = json.loads(json.dumps(wf0))
        h = wf[h3_id].setdefault("inputs", {})

        if prompt_key:
            h[prompt_key] = prompt_text
        else:
            wf[text_node_id].setdefault("inputs", {})[text_key] = prompt_text
        log("%s prompt: %s" % (label, prompt_text))

        seeded = False
        for name in ("seed", "noise_seed"):
            if name in ins:
                h[name] = seed
                log("%s %s = %d" % (label, name, seed))
                seeded = True
                break
        if not seeded:
            for nid, node in wf.items():
                if str(node.get("class_type", "")) == "RandomNoise":
                    ni = node.setdefault("inputs", {})
                    for name in ("noise_seed", "seed"):
                        if name in ni or name in (input_spec(get_class_info(args.comfy, "RandomNoise")) or {}):
                            ni[name] = seed
                            log("%s RandomNoise(%s).%s = %d" % (label, nid, name, seed))
                            seeded = True
                            break
                if seeded:
                    break
        if not seeded:
            log("%s WARNUNG: kein Seed-Input am Node - fester Seed nicht setzbar" % label)

        for key, names in (("frames", ("length", "num_frames", "frames", "video_length")),
                           ("fps", ("fps", "frame_rate")),
                           ("width", ("width",)), ("height", ("height",))):
            if shot.get(key) is not None:
                set_widget(ins, h, list(names), shot.get(key), key, label)

        for k, v in (shot.get("node_overrides") or {}).items():
            if k not in ins:
                log("%s WARNUNG: node_override '%s' existiert nicht - ignoriert" % (label, k))
                continue
            t, meta = input_type(ins[k])
            if t == "COMBO":
                choices = [str(x) for x in (meta.get("choices") or [])]
                if choices and str(v) not in choices:
                    log("%s WARNUNG: override %s=%r nicht in %s - ignoriert" % (label, k, v, choices[:10]))
                    continue
            h[k] = v
            log("%s override %s = %r" % (label, k, v))

        for k in img_inputs:
            if k != ref_key and k in opt_keys and k in h:
                del h[k]
                log("%s optionalen IMAGE-Input '%s' entfernt (keine alten Bilder mischen)" % (label, k))

        for x in members:
            wf["refload_" + x["id"]] = {"class_type": "LoadImage", "inputs": {"image": ref_values[x["id"]]}}
        if ref_is_autogrow:
            for _k in [k for k in list(h.keys()) if str(k).startswith(ref_key + ".")]:
                del h[_k]
            for i, x in enumerate(members):
                h["%s.%s%d" % (ref_key, ref_prefix, i)] = ["refload_" + x["id"], 0]
        else:
            h[ref_key] = [["refload_" + x["id"], 0] for x in members]
        log("%s refs -> %s.%s : %s" % (label, h3_cls, ref_key, [x["id"] for x in members]))

        ensure_saver(args.comfy, wf, h3_id, oinfo, sid, label)
        save_json(os.path.join(debug_dir, sid + ".json"), wf)

        if args.dry_run:
            log("%s DRY-RUN OK (nichts gesendet). Workflow: %s" % (label, os.path.join(debug_dir, sid + ".json")))
            continue

        try:
            pid = submit_and_wait(args.comfy, wf, args.timeout, label)
            entry = fetch_history(args.comfy, pid)
            ok = retrieve_outputs(args.comfy, entry, dest, sid, fps_for(shot, wf),
                                  (os.path.dirname(template),))
            if not ok:
                raise RuntimeError("kein Video in den ComfyUI-Outputs gefunden")
            manifest[sid] = {"status": "done", "file": dest, "seed": seed, "prompt_id": pid,
                             "cast": [x["id"] for x in members]}
            save_json(manifest_path, manifest)
            log("%s FERTIG -> %s" % (label, dest))
        except Exception as e:
            failed += 1
            manifest[sid] = {"status": "failed", "error": str(e)[:500], "seed": seed}
            save_json(manifest_path, manifest)
            log("%s FEHLER: %s" % (label, e))
            traceback.print_exc()

    if args.dry_run:
        log("DRY-RUN fertig: %d Shot(s) validiert." % len(shots))
        return
    log("Ende: %d Shot(s), %d Fehler. Manifest: %s" % (len(shots), failed, manifest_path))
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()