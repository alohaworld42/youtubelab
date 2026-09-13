"""
youtube_upload.py — Upload a finished video to YouTube as a Short.

One-time setup (see README):
  1. Create a Google Cloud project, enable "YouTube Data API v3".
  2. Create an OAuth client of type "Desktop app", download the JSON.
  3. Save it as  client_secret.json  in the project root.

Legacy single-channel mode: token cached in token.json in the project root.
Multi-channel mode (studio): pass explicit `secret_path`/`token_path` — each
channel keeps its own token under tokens/<yt_channel_id>.json.
"""
import os

SCOPES = [
    "https://www.googleapis.com/auth/youtube.upload",
    "https://www.googleapis.com/auth/youtube.readonly",
]


def get_service(secret_path, token_path, allow_interactive=True):
    """Build an authorized YouTube API client for one credential pair.

    `allow_interactive=False` raises instead of opening a browser — used by the
    scheduler so a background upload never blocks on a consent screen.
    """
    from google.auth.transport.requests import Request
    from google.oauth2.credentials import Credentials
    from googleapiclient.discovery import build

    creds = None
    if os.path.exists(token_path):
        creds = Credentials.from_authorized_user_file(token_path, SCOPES)
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            if not allow_interactive:
                raise RuntimeError(
                    f"Token {token_path} is missing or invalid and interactive "
                    "re-auth is not allowed here. Reconnect the channel in the UI."
                )
            if not os.path.exists(secret_path):
                raise FileNotFoundError(
                    f"OAuth client secret not found: {secret_path}. "
                    "Follow the YouTube setup steps in the README."
                )
            from google_auth_oauthlib.flow import InstalledAppFlow

            flow = InstalledAppFlow.from_client_secrets_file(secret_path, SCOPES)
            creds = flow.run_local_server(port=0)
        os.makedirs(os.path.dirname(os.path.abspath(token_path)), exist_ok=True)
        with open(token_path, "w", encoding="utf-8") as f:
            f.write(creds.to_json())

    return build("youtube", "v3", credentials=creds)


def _get_service(root):
    """Legacy helper: root-level client_secret.json / token.json."""
    return get_service(
        os.path.join(root, "client_secret.json"),
        os.path.join(root, "token.json"),
    )


DESCRIPTION_LIMIT = 4900


def upload(video_path, title, description, tags, cfg,
           token_path=None, secret_path=None, privacy=None, allow_interactive=True,
           made_for_kids=None, thumbnail_path=None, video_type=None,
           synthetic_media=None):
    """Upload one video. Without token_path/secret_path this behaves exactly like
    the original single-channel version (root token.json + config privacy).

    Every caller reaches YouTube through here, which is why the AI disclosure is
    applied at this seam rather than at the three call sites: `studio.scheduler`
    (the scheduled path), `pipeline.generate` and the kidsong CLI would each have
    had to remember, and the one that forgot would publish undisclosed.
    """
    from googleapiclient.http import MediaFileUpload

    from pipeline import ai_disclosure

    root = cfg["_root"]
    yt = cfg["youtube"]
    privacy = privacy or yt.get("privacy", "private")
    if made_for_kids is None:
        made_for_kids = bool(yt.get("made_for_kids", False))

    if yt.get("append_shorts_tag", True) and "#shorts" not in (description or "").lower():
        description = (description or "").rstrip() + "\n\n#Shorts"

    # Truncation happens INSIDE append_note, against the operator's text only —
    # slicing afterwards would let a long description push the disclosure off
    # the end and publish an undisclosed video with no error anywhere.
    description = ai_disclosure.append_note(description, cfg, limit=DESCRIPTION_LIMIT)

    if token_path or secret_path:
        secret_path = secret_path or os.path.join(root, "client_secret.json")
        token_path = token_path or os.path.join(root, "token.json")
        service = get_service(secret_path, token_path, allow_interactive=allow_interactive)
    else:
        service = _get_service(root)

    body = {
        "snippet": {
            "title": title[:100],
            "description": description[:DESCRIPTION_LIMIT],
            "tags": tags[:15],
            "categoryId": str(yt.get("category_id", "24")),
        },
        "status": {
            "privacyStatus": privacy,
            "selfDeclaredMadeForKids": bool(made_for_kids),
            "containsSyntheticMedia": ai_disclosure.declare_synthetic_media(
                cfg, video_type=video_type, override=synthetic_media
            ),
        },
    }
    media = MediaFileUpload(video_path, chunksize=-1, resumable=True, mimetype="video/mp4")
    request = service.videos().insert(part="snippet,status", body=body, media_body=media)

    response = None
    while response is None:
        _status, response = request.next_chunk()
    video_id = response["id"]

    thumbnail_set = False
    if thumbnail_path and os.path.exists(thumbnail_path):
        # Custom thumbnails need a phone-verified account; never fail the upload.
        try:
            service.thumbnails().set(
                videoId=video_id,
                media_body=MediaFileUpload(thumbnail_path, mimetype="image/jpeg"),
            ).execute()
            thumbnail_set = True
        except Exception:
            pass

    return {
        "video_id": video_id,
        "url": f"https://youtube.com/watch?v={video_id}",
        "privacy": privacy,
        "thumbnail_set": thumbnail_set,
    }
