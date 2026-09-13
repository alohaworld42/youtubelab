"""Schreibt lab.json fuer DIESEN Rechner (Pfade aus Argumenten/Autoerkennung).

    <python> configure.py --comfy "<ComfyUI>" --models "<models_dir>" --orca "<youtubegenerator-clone>"

ffmpeg wird im ComfyUI-venv oder im Repo-venv gesucht; sonst --ffmpeg angeben.
Der GPU-Lock zeigt auf <orca>/output/gpu_render.lock (dieselbe Datei, die die
Pipeline nutzt). Alles andere (Aufloesungen, Seeds, Downloads) bleibt wie im
eingecheckten lab.json.
"""
import argparse
import glob
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
LAB = os.path.dirname(HERE)


def find_ffmpeg(comfy, orca):
    for root in (comfy, orca, os.path.dirname(comfy)):
        hits = glob.glob(os.path.join(root, "**", "ffmpeg*win*x86_64*.exe"), recursive=True)
        hits += glob.glob(os.path.join(root, "**", "imageio_ffmpeg", "binaries", "ffmpeg*.exe"), recursive=True)
        if hits:
            return hits[0].replace("\\", "/")
    return "ffmpeg"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--comfy", required=True)
    ap.add_argument("--models", required=True)
    ap.add_argument("--orca", required=True)
    ap.add_argument("--python", default=None, help="ComfyUI venv python (default: das aktuelle)")
    ap.add_argument("--ffmpeg", default=None)
    a = ap.parse_args()
    comfy = a.comfy.replace("\\", "/")
    models = a.models.replace("\\", "/")
    orca = a.orca.replace("\\", "/")
    py = (a.python or sys.executable).replace("\\", "/")
    ff = (a.ffmpeg or find_ffmpeg(a.comfy, a.orca)).replace("\\", "/")

    # bestehendes lab.json als Vorlage, nur Pfade ersetzen
    cfg = json.load(open(os.path.join(LAB, "lab.json"), encoding="utf-8"))
    cfg["lab_dir"] = LAB.replace("\\", "/")
    cfg["python"] = py
    cfg["comfy_dir"] = comfy
    cfg["models_dir"] = models
    cfg["orca_root"] = orca
    cfg["ffmpeg"] = ff
    cfg["lock_file"] = orca + "/output/gpu_render.lock"
    cfg["hf_home"] = LAB.replace("\\", "/") + "/.hf_cache"
    cfg["temp_dir"] = LAB.replace("\\", "/") + "/Claudetempfiles"
    json.dump(cfg, open(os.path.join(LAB, "lab.json"), "w", encoding="utf-8"), indent=2, ensure_ascii=False)
    print("lab.json geschrieben:")
    for k in ("python", "comfy_dir", "models_dir", "orca_root", "ffmpeg", "lock_file"):
        ok = os.path.exists(cfg[k]) or k in ("lock_file",)
        print(("  OK  " if ok else "  !!  ") + k + " = " + cfg[k])
    # Warnung fehlende Pfade
    for k in ("python", "comfy_dir", "models_dir", "orca_root"):
        if not os.path.exists(cfg[k]):
            print("  ! WARNUNG: existiert nicht:", cfg[k])
    print("\nStart: ComfyUI-Server hochfahren (siehe SETUP.md), dann")
    print('  & "<powershell>" -ExecutionPolicy Bypass -File "%s\\run_proof.ps1" -Stage 3 -RunId proof01' % LAB)


if __name__ == "__main__":
    main()
