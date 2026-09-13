"""Per-channel OAuth connect flow.

`start_connect()` runs InstalledAppFlow in a background thread (it opens the
local browser); the UI polls `connect_status()`. On success the token is saved
to tokens/<yt_channel_id>.json and a channels row is created/updated.
"""
import os
import threading

from pipeline.youtube_upload import SCOPES
from studio import models

_state = {"status": "idle", "error": None, "channel_id": None}
_lock = threading.Lock()


def _tokens_dir(cfg):
    d = os.path.join(cfg["_root"], "tokens")
    os.makedirs(d, exist_ok=True)
    return d


def connect_status():
    with _lock:
        return dict(_state)


def start_connect(cfg, secret_path=None):
    """Kick off the browser consent flow. Returns False if one is already running."""
    with _lock:
        if _state["status"] == "running":
            return False
        _state.update({"status": "running", "error": None, "channel_id": None})

    secret = secret_path or os.path.join(cfg["_root"], "client_secret.json")
    t = threading.Thread(target=_run_flow, args=(cfg, secret), daemon=True)
    t.start()
    return True


def _run_flow(cfg, secret_path):
    try:
        if not os.path.exists(secret_path):
            raise FileNotFoundError(
                "client_secret.json fehlt im Projektordner. Siehe README (YouTube Setup)."
            )
        from google_auth_oauthlib.flow import InstalledAppFlow
        from googleapiclient.discovery import build

        flow = InstalledAppFlow.from_client_secrets_file(secret_path, SCOPES)
        creds = flow.run_local_server(port=0)

        service = build("youtube", "v3", credentials=creds)
        resp = service.channels().list(part="snippet", mine=True).execute()
        items = resp.get("items") or []
        if not items:
            raise RuntimeError(
                "Kein YouTube-Kanal auf diesem Google-Konto gefunden. "
                "Beim Consent-Screen den richtigen Kanal/Brand-Account wählen."
            )
        yt_id = items[0]["id"]
        title = items[0]["snippet"]["title"]

        token_path = os.path.join(_tokens_dir(cfg), f"{yt_id}.json")
        # Stored RELATIVE to the project root so the committed studio.db works
        # on every checkout, not only the machine that ran the flow. The
        # absolute path is still what we write the file with.
        token_rel = os.path.relpath(token_path, cfg["_root"]).replace(os.sep, "/")
        with open(token_path, "w", encoding="utf-8") as f:
            f.write(creds.to_json())
        # A refresh token is a live credential for somebody's YouTube channel —
        # anyone holding this file can upload as them. Owner-only on POSIX,
        # best-effort; a no-op on Windows, where it inherits the user's ACL.
        # Same treatment as .env in studio/setup.py.
        try:
            os.chmod(token_path, 0o600)
        except OSError:
            pass

        existing = models.get_channel_by_yt_id(yt_id)
        if existing:
            models.update_channel(
                existing["id"],
                token_path=token_rel,
                yt_channel_title=title,
                status="active",
            )
            channel_id = existing["id"]
        else:
            channel_id = models.create_channel(
                name=title, yt_channel_id=yt_id, yt_channel_title=title,
                token_path=token_rel,
            )
        with _lock:
            _state.update({"status": "done", "channel_id": channel_id})
    except Exception as e:
        with _lock:
            _state.update({"status": "error", "error": str(e)})


def _resolve_under_root(value, cfg, fallback):
    """Absolute paths are honoured; relative ones resolve against the project
    root; empty falls back to the root-level default.

    `studio.db` is shared between machines (it is committed), and it used to
    store token_path as an ABSOLUTE path — whatever the machine that ran the
    OAuth flow happened to be. On any other checkout that path does not exist,
    so `get_service` raised "Token ... is missing or invalid" and the channel
    was flipped to needs_reauth even though a perfectly good token sat in
    `tokens/` right there. Relative is the portable form; absolute stays
    supported so an existing row or a deliberate out-of-tree token keeps
    working.
    """
    if not value:
        return os.path.join(cfg["_root"], fallback)
    if os.path.isabs(value):
        return value
    return os.path.join(cfg["_root"], value)


def channel_upload_paths(channel, cfg):
    """Resolve (secret_path, token_path) for a channel row, with root fallbacks."""
    secret = _resolve_under_root(channel.get("secret_path"), cfg, "client_secret.json")
    token = _resolve_under_root(channel.get("token_path"), cfg, "token.json")
    return secret, token
