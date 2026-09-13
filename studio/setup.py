"""Setup wizard backend: status checks, .env writer, LLM test, starter assets.

Everything the user previously did by hand (setx, copying files) goes through
here so the browser UI can do it instead.
"""
import glob
import json
import os
import re
import shutil
import threading

import requests

from studio import models

VALID_BACKENDS = ("groq", "ollama", "gemini")
KEY_FOR_BACKEND = {"groq": "GROQ_API_KEY", "gemini": "GEMINI_API_KEY"}

# Known no-copyright gameplay loops for the one-click starter pack.
DEFAULT_GAMEPLAY_URLS = [
    "https://www.youtube.com/watch?v=n_Dv4JMiwK8",  # Subway Surfers (no copyright)
    "https://www.youtube.com/watch?v=intRX7BRA90",  # Minecraft parkour (no copyright)
]

_dl_state = {"status": "idle", "done": 0, "total": 0, "error": None}
_dl_lock = threading.Lock()


# ------------------------------------------------------------------- .env ----
def _env_path(root):
    return os.path.join(root, ".env")


def read_env(root):
    env = {}
    path = _env_path(root)
    if not os.path.exists(path):
        return env
    with open(path, "r", encoding="utf-8-sig") as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith("#") or "=" not in s:
                continue
            k, _, v = s.partition("=")
            env[k.strip()] = v.strip().strip('"').strip("'")
    return env


_ENV_KEY_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def _clean_env_value(value):
    """A .env value is one line. Anything else is an injection.

    `.env` is KEY=VALUE per line, so a value containing a newline does not
    store a multi-line value — it writes extra lines that `read_env` and
    `pipeline.config.load_dotenv` then parse as further settings. Every value
    here arrives from a browser field: pasting
    `abc<newline>STUDIO_PASSWORD=hunter2` into the API-key box on the setup page
    would have set the studio's login password. The views only `.strip()`, which
    catches a trailing newline and nothing in the middle.
    """
    text = str(value if value is not None else "")
    # Collapse every line/paragraph separator, not just \n — a key copied from a
    # web page can carry \r or U+2028.
    for sep in ("\r\n", "\r", "\n", " ", " ", "\x00"):
        text = text.replace(sep, " ")
    return text.strip()


def write_env(root, updates):
    """Update KEY=VALUE entries, keeping unrelated lines/comments intact.
    Values also go straight into os.environ so they work without restart.

    Keys must look like environment variables and values must be single-line;
    see `_clean_env_value`.
    """
    cleaned = {}
    for key, value in (updates or {}).items():
        key = str(key or "").strip()
        if not key:
            continue
        if not _ENV_KEY_RE.match(key):
            raise ValueError(f"Ungültiger Variablenname: {key!r}")
        cleaned[key] = _clean_env_value(value)
    updates = cleaned
    if not updates:
        return
    path = _env_path(root)
    lines = []
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8-sig") as f:
            lines = f.read().splitlines()

    remaining = dict(updates)
    out = []
    for line in lines:
        s = line.strip()
        if s and not s.startswith("#") and "=" in s:
            key = s.partition("=")[0].strip()
            if key in remaining:
                out.append(f"{key}={remaining.pop(key)}")
                continue
        out.append(line)
    for k, v in remaining.items():
        out.append(f"{k}={v}")

    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(out) + "\n")
    # .env holds every API key and the studio password. Best-effort owner-only
    # on POSIX; a no-op on Windows, where it is created with the user's ACL.
    try:
        os.chmod(path, 0o600)
    except OSError:
        pass

    for k, v in updates.items():
        os.environ[k] = v


# ----------------------------------------------------------------- checks ----
def _check_ffmpeg():
    ffmpeg = shutil.which("ffmpeg")
    ffprobe = shutil.which("ffprobe")
    return {"ok": bool(ffmpeg and ffprobe), "ffmpeg": ffmpeg, "ffprobe": ffprobe}


def _check_llm(cfg):
    llm = cfg.get("llm") or {}
    backend = llm.get("backend", "groq")
    result = {"backend": backend, "ok": False, "detail": ""}
    if backend == "ollama":
        host = llm.get("ollama_host", "http://localhost:11434")
        try:
            resp = requests.get(f"{host}/api/tags", timeout=2)
            resp.raise_for_status()
            names = [m.get("name") for m in resp.json().get("models") or []]
            result["ok"] = bool(names)
            result["detail"] = ", ".join(names) if names else "Ollama läuft, aber kein Modell (ollama pull llama3.1)"
        except Exception:
            result["detail"] = "Ollama nicht erreichbar (App starten / ollama serve)"
    elif backend in KEY_FOR_BACKEND:
        key = os.environ.get(KEY_FOR_BACKEND[backend])
        result["ok"] = bool(key)
        result["detail"] = "Key gesetzt" if key else f"{KEY_FOR_BACKEND[backend]} fehlt"
    else:
        result["detail"] = f"Unbekanntes Backend: {backend}"
    return result


def _check_gameplay(cfg):
    d = os.path.join(cfg["_root"], "assets", "gameplay")
    clips = []
    for ext in ("*.mp4", "*.mov", "*.mkv", "*.webm"):
        clips.extend(glob.glob(os.path.join(d, ext)))
    real = [c for c in clips if not os.path.basename(c).startswith("_placeholder")]
    return {"ok": bool(real), "count": len(real)}


def setup_status(cfg):
    root = cfg["_root"]
    ffmpeg = _check_ffmpeg()
    llm = _check_llm(cfg)
    channels = models.list_channels()
    status = {
        "ffmpeg": ffmpeg,
        "llm": llm,
        "client_secret": {"ok": os.path.exists(os.path.join(root, "client_secret.json"))},
        "channels": {"ok": bool(channels), "count": len(channels)},
        "gameplay": _check_gameplay(cfg),
        "stock_keys": {
            "pexels": bool(os.environ.get("PEXELS_API_KEY")),
            "pixabay": bool(os.environ.get("PIXABAY_API_KEY")),
        },
    }
    status["complete"] = ffmpeg["ok"] and llm["ok"]
    return status


# ------------------------------------------------------------ llm backend ----
def set_llm_backend(cfg, backend, api_key=None):
    if backend not in VALID_BACKENDS:
        raise ValueError(f"Backend muss eins sein von: {', '.join(VALID_BACKENDS)}")
    root = cfg["_root"]
    cfg_path = os.path.join(root, "config.json")
    if not os.path.exists(cfg_path):
        shutil.copyfile(os.path.join(root, "config.example.json"), cfg_path)
    with open(cfg_path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    raw.setdefault("llm", {})["backend"] = backend
    # config.json is the app's entire configuration and this rewrites all of it
    # to change one string. Truncating it mid-write (crash, full disk) leaves
    # the studio unable to start; publish it atomically instead.
    from pipeline.atomicio import atomic_write_json

    atomic_write_json(cfg_path, raw, indent=2, ensure_ascii=False)

    if api_key and backend in KEY_FOR_BACKEND:
        write_env(root, {KEY_FOR_BACKEND[backend]: api_key.strip()})


def test_llm(cfg):
    """Round-trip against the configured backend. Returns (ok, detail)."""
    from pipeline.script_gen import complete

    try:
        raw = complete(
            'Reply with exactly this JSON and nothing else: {"status": "OK"}', cfg
        )
        return ("OK" in (raw or ""), (raw or "")[:200])
    except Exception as e:
        return (False, str(e))


# ---------------------------------------------------------- client secret ----
def save_client_secret(root, file_bytes):
    try:
        data = json.loads(file_bytes.decode("utf-8"))
    except (ValueError, UnicodeDecodeError):
        raise ValueError("Datei ist kein gültiges JSON.")
    if not isinstance(data, dict) or not ({"installed", "web"} & set(data)):
        raise ValueError(
            'Falsches Format — erwartet OAuth-Client-JSON mit "installed" '
            "(Desktop-App-Client aus der Google Cloud Console)."
        )
    path = os.path.join(root, "client_secret.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f)
    return path


# ------------------------------------------------------- gameplay starter ----
def gameplay_download_status():
    with _dl_lock:
        return dict(_dl_state)


def start_gameplay_download(cfg, urls=None):
    """Download starter gameplay clips in a background thread. Returns False
    if a download is already running."""
    with _dl_lock:
        if _dl_state["status"] == "running":
            return False
        # A whitespace-only URL in the box used to survive the `urls or DEFAULT`
        # check (a list holding "  " is truthy) and then filter down to nothing,
        # so the run finished instantly with status "error" and no message —
        # "Download fehlgeschlagen" for a box the user just left blank-ish.
        urls = [u.strip() for u in (urls or []) if u and u.strip()]
        if not urls:
            urls = list(DEFAULT_GAMEPLAY_URLS)
        _dl_state.update({"status": "running", "done": 0, "total": len(urls), "error": None})

    t = threading.Thread(target=_download_gameplay, args=(cfg, urls), daemon=True)
    t.start()
    return True


def _download_gameplay(cfg, urls):
    from studio.assets import _ytdlp_fetch

    out_dir = os.path.join(cfg["_root"], "assets", "gameplay")
    max_minutes = int((cfg.get("studio") or {}).get("gameplay_max_minutes", 8))
    errors = []
    for url in urls:
        try:
            path = _ytdlp_fetch(url, out_dir, max_minutes=max_minutes)
            if path:
                models.record_asset(
                    "gameplay", "ytdlp", path, source_url=url,
                    keywords="gameplay,starter", size_bytes=os.path.getsize(path),
                )
        except Exception as e:
            errors.append(f"{url}: {e}")
        with _dl_lock:
            _dl_state["done"] += 1
    with _dl_lock:
        _dl_state["status"] = "error" if len(errors) == len(urls) else "done"
        _dl_state["error"] = "; ".join(errors)[:500] if errors else None
