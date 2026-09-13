"""Rekonstruiert das Konsistenz-Lab auf einem neuen Rechner OHNE Claude.

Voraussetzung (siehe SETUP.md): ComfyUI Desktop installiert, dessen venv-Python
bekannt, torch als +cu126 (Treiber-kompatibel). Dann:

    <comfy_venv_python> bootstrap.py --comfy "<ComfyUI-Ordner>" --models "<models_dir>" [--orca "<clone-ziel>"] [--skip-models] [--skip-nodes] [--skip-pip]

Macht der Reihe nach:
  1. pip-Deps in das AKTUELLE Python (= das, mit dem du dieses Skript startest).
  2. Custom-Nodes klonen (ComfyUI-GGUF + h3_cond_cache) nach <ComfyUI>/custom_nodes.
  3. Code-Repo youtubegenerator klonen (liefert pipeline.* + GGUF-Patch).
  4. GGUF-Patch anwenden (tools/patch_comfyui_gguf_h3.py).
  5. Alle Modelle aus models.json nach <models_dir>/<dest>/ laden (Resume, nie ueberschreiben).
  6. dinov2-small vorcachen.
  7. h3_model_paths.yaml schreiben.
Danach: python bootstrap/configure.py  (schreibt lab.json fuer diesen Rechner).
"""
import argparse
import json
import os
import shutil
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
LAB = os.path.dirname(HERE)


def sh(cmd, **kw):
    print("+", " ".join(cmd) if isinstance(cmd, list) else cmd)
    return subprocess.run(cmd, **kw)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--comfy", required=True, help="ComfyUI-Ordner (enthaelt main.py, custom_nodes/)")
    ap.add_argument("--models", required=True, help="models_dir (enthaelt unet/ text_encoders/ vae/ loras/ ...)")
    ap.add_argument("--orca", default=os.path.join(os.path.dirname(LAB), "youtubegenerator"), help="Klonziel fuer den Code-Repo")
    ap.add_argument("--skip-models", action="store_true")
    ap.add_argument("--skip-nodes", action="store_true")
    ap.add_argument("--skip-pip", action="store_true")
    a = ap.parse_args()
    man = json.load(open(os.path.join(HERE, "models.json"), encoding="utf-8"))
    py = sys.executable
    os.makedirs(a.models, exist_ok=True)

    if not a.skip_pip:
        print("\n== 1. pip deps ==")
        sh([py, "-m", "pip", "install", "-q", *man["pip"]])

    if not a.skip_nodes:
        print("\n== 2. custom nodes ==")
        cn = os.path.join(a.comfy, "custom_nodes")
        os.makedirs(cn, exist_ok=True)
        for node in man["custom_nodes"]:
            dst = os.path.join(cn, node["dir"])
            if os.path.isdir(dst):
                print("have", node["dir"])
            else:
                sh(["git", "clone", "--depth", "1", node["repo"], dst])
        print("\n== 3. code repo (pipeline.*) ==")
        if os.path.isdir(os.path.join(a.orca, "pipeline")):
            print("have", a.orca)
        else:
            sh(["git", "clone", "--branch", man["code_repo"]["branch"], man["code_repo"]["url"], a.orca])
        print("\n== 4. GGUF-Patch ==")
        patch = os.path.join(a.orca, "tools", "patch_comfyui_gguf_h3.py")
        gguf = os.path.join(cn, "ComfyUI-GGUF")
        if os.path.isfile(patch):
            sh([py, patch, gguf])
        else:
            print("! Patch nicht gefunden:", patch, "- H3-GGUFs laden sonst nicht. Repo pruefen.")

    if not a.skip_models:
        print("\n== 5. Modelle ==")
        from huggingface_hub import hf_hub_download
        tmp = os.path.join(LAB, "Claudetempfiles", "hf_dl")
        os.makedirs(tmp, exist_ok=True)
        for m in man["models"]:
            dest_dir = os.path.join(a.models, m["dest"])
            os.makedirs(dest_dir, exist_ok=True)
            final = os.path.join(dest_dir, m.get("save_as", os.path.basename(m["file"])))
            exp = float(m.get("gb", 0)) * 1e9
            if os.path.isfile(final) and (exp == 0 or os.path.getsize(final) >= 0.95 * exp):
                print("have  %-55s %.1f GB" % (os.path.basename(final), os.path.getsize(final) / 1e9))
                continue
            print("fetch %s/%s (%.1f GB) [%s]" % (m["repo"], os.path.basename(m["file"]), m.get("gb", 0), m.get("role", "")))
            got = hf_hub_download(repo_id=m["repo"], filename=m["file"], local_dir=tmp)
            if not os.path.isfile(final):
                shutil.move(got, final)
            print("  ->", final)
        print("\n== 6. dinov2 vorcachen ==")
        os.environ["HF_HOME"] = os.path.join(LAB, ".hf_cache")
        try:
            from transformers import AutoImageProcessor, AutoModel
            for mm in man["hf_models_precache"]:
                AutoImageProcessor.from_pretrained(mm); AutoModel.from_pretrained(mm)
                print("cached", mm)
        except Exception as e:
            print("! dino precache skipped:", str(e)[:100])

    print("\n== 7. h3_model_paths.yaml ==")
    yaml = os.path.join(a.comfy, "h3_model_paths.yaml")
    with open(yaml, "w", encoding="utf-8") as f:
        f.write("h3_repo_models:\n  base_path: %s\n  unet: unet\n  diffusion_models: diffusion_models\n"
                "  text_encoders: text_encoders\n  clip: text_encoders\n  vae: vae\n  loras: loras\n" % a.models.replace("\\", "/"))
    print("wrote", yaml)

    print("\n== fertig. Jetzt: ==")
    print("  %s %s --comfy \"%s\" --models \"%s\" --orca \"%s\"" % (py, os.path.join(HERE, "configure.py"), a.comfy, a.models, a.orca))
    print("  (schreibt lab.json fuer diesen Rechner)")


if __name__ == "__main__":
    main()
