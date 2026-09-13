"""Loads config.json (falling back to config.example.json) and exposes it as a dict."""
import json
import os

_HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load_dotenv(root=None):
    """Load KEY=VALUE lines from a .env file in the project root into os.environ.

    Existing environment variables win. No external dependency; the setup UI
    writes secrets here so users never need setx.
    """
    path = os.path.join(root or _HERE, ".env")
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8-sig") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, _, value = line.partition("=")
                key, value = key.strip(), value.strip().strip('"').strip("'")
                if key and key not in os.environ:
                    os.environ[key] = value

    # Also accept a keys/ directory: one file per secret, filename = variable name
    # (e.g. keys/pixabay_api_key -> PIXABAY_API_KEY). Empty files are skipped.
    keys_dir = os.path.join(root or _HERE, "keys")
    if os.path.isdir(keys_dir):
        for name in os.listdir(keys_dir):
            fp = os.path.join(keys_dir, name)
            if not os.path.isfile(fp):
                continue
            var = os.path.splitext(name)[0].upper()
            if not var or var in os.environ:
                continue
            try:
                with open(fp, "r", encoding="utf-8-sig") as f:
                    value = f.read().strip()
            except OSError:
                continue
            if value:
                os.environ[var] = value


def _strip_comments(obj):
    """Remove keys starting with '_' (used for inline docs in the JSON)."""
    if isinstance(obj, dict):
        return {k: _strip_comments(v) for k, v in obj.items() if not k.startswith("_")}
    if isinstance(obj, list):
        return [_strip_comments(v) for v in obj]
    return obj


def load_config():
    load_dotenv()
    example = os.path.join(_HERE, "config.example.json")
    path = os.path.join(_HERE, "config.json")
    if not os.path.exists(path):
        # first boot: materialize a real config so the user can edit it later
        try:
            import shutil

            shutil.copyfile(example, path)
        except OSError:
            pass

    # The example is always the base so new config keys (added in updates) work
    # even when the user's config.json predates them; user values win.
    with open(example, "r", encoding="utf-8") as f:
        cfg = _strip_comments(json.load(f))
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            cfg = merge_overrides(cfg, _strip_comments(json.load(f)))
    cfg["_root"] = _HERE
    return cfg


def abspath(cfg, rel):
    """Resolve a path from config relative to the project root."""
    if os.path.isabs(rel):
        return rel
    return os.path.join(cfg["_root"], rel)


def apply_style(cfg, style_name):
    """Merge a style preset from cfg["styles"] over the config.

    Styles bundle look & feel (captions, music, background sources, encode
    settings) so a kids channel and a brainrot channel can share one pipeline.
    Unknown/empty style names return cfg unchanged.
    """
    if not style_name:
        return cfg
    preset = (cfg.get("styles") or {}).get(style_name)
    if not preset:
        return cfg
    return merge_overrides(cfg, preset)


def merge_overrides(cfg, overrides):
    """Deep-merge `overrides` over `cfg` and return a new dict.

    Used to apply per-channel settings (voices, privacy, …) on top of the
    global config without mutating it.
    """
    if not overrides:
        return cfg
    merged = dict(cfg)
    for key, val in overrides.items():
        if isinstance(val, dict) and isinstance(merged.get(key), dict):
            merged[key] = merge_overrides(merged[key], val)
        else:
            merged[key] = val
    return merged
