"""Data access functions. All JSON columns are parsed/serialized here so the
rest of the app only ever sees dicts/lists.
"""
import json
import time

from studio.db import get_db

TRANSIENT_JOB_STATES = ("scripting", "tts", "captions", "rendering")
WORKING_JOB_STATES = TRANSIENT_JOB_STATES + ("uploading",)


def now():
    return time.strftime("%Y-%m-%d %H:%M:%S")


def today():
    return time.strftime("%Y-%m-%d")


def _loads(s, default):
    try:
        v = json.loads(s) if s else default
        return v if v is not None else default
    except (ValueError, TypeError):
        return default


# ---------------------------------------------------------------- channels ----
def _normalize_language(lang):
    """Lowercase, first subtag only ('de-DE' -> 'de'), blank/None -> 'en'.

    Not a hard BCP-47 validator — just enough normalization so 'de', 'DE' and
    'de-DE' all compare equal.
    """
    if not lang:
        return "en"
    lang = str(lang).strip().lower()
    if not lang:
        return "en"
    return lang.split("-")[0]


def channel_language(channel):
    """Normalized language tag for a channel dict (or None -> 'en')."""
    if not channel:
        return "en"
    return _normalize_language(channel.get("language"))


_BRANDING_DEFAULTS = {
    "title_template": "",
    "description_template": "",
    "tags": [],
    "intro_path": "",
    "outro_path": "",
    "logo_path": "",
    "primary_color": "",
    "secondary_color": "",
    "font": "",
}


def channel_branding(channel):
    """Full branding dict for a channel: every key always present, missing or
    malformed data falls back to sensible defaults instead of raising.

    Accepts a parsed channel dict (with a "branding" dict), a raw DB row
    (with a "branding_json" string), or None.
    """
    result = {k: (list(v) if isinstance(v, list) else v) for k, v in _BRANDING_DEFAULTS.items()}
    if not channel:
        return result
    raw = channel.get("branding", channel.get("branding_json"))
    if isinstance(raw, str):
        raw = _loads(raw, {})
    if not isinstance(raw, dict):
        raw = {}
    for k in _BRANDING_DEFAULTS:
        if k in raw and raw[k] is not None:
            result[k] = raw[k]
    return result


def _parse_channel(row):
    if not row:
        return None
    row["voices"] = _loads(row.pop("voices_json", None), {})
    row["schedule"] = _loads(row.pop("schedule_json", None), [])
    row["asset_prefs"] = _loads(row.pop("asset_prefs_json", None), {})
    row["branding"] = channel_branding({"branding_json": row.pop("branding_json", None)})
    row["language"] = channel_language(row)
    return row


def create_channel(name, yt_channel_id=None, yt_channel_title=None, token_path=None,
                   language=None, branding=None, content_profile=None):
    db = get_db()
    fields = {
        "name": name,
        "yt_channel_id": yt_channel_id,
        "yt_channel_title": yt_channel_title,
        "token_path": token_path,
        "created_at": now(),
    }
    if language is not None:
        fields["language"] = _normalize_language(language)
    if branding is not None:
        fields["branding_json"] = json.dumps(branding)
    if content_profile is not None:
        fields["content_profile"] = content_profile
    cols = ", ".join(fields)
    marks = ", ".join("?" for _ in fields)
    cur = db.execute(f"INSERT INTO channels ({cols}) VALUES ({marks})", tuple(fields.values()))
    db.commit()
    return cur.lastrowid


def list_channels(status=None):
    db = get_db()
    if status:
        rows = db.execute("SELECT * FROM channels WHERE status=? ORDER BY id", (status,)).fetchall()
    else:
        rows = db.execute("SELECT * FROM channels ORDER BY id").fetchall()
    return [_parse_channel(r) for r in rows]


def get_channel(channel_id):
    row = get_db().execute("SELECT * FROM channels WHERE id=?", (channel_id,)).fetchone()
    return _parse_channel(row)


def get_channel_by_yt_id(yt_channel_id):
    row = get_db().execute(
        "SELECT * FROM channels WHERE yt_channel_id=?", (yt_channel_id,)
    ).fetchone()
    return _parse_channel(row)


def update_channel(channel_id, **fields):
    for k in ("voices", "schedule", "asset_prefs", "branding"):
        if k in fields:
            fields[k + "_json"] = json.dumps(fields.pop(k))
    if "language" in fields:
        fields["language"] = _normalize_language(fields["language"])
    if not fields:
        return
    cols = ", ".join(f"{k}=?" for k in fields)
    db = get_db()
    db.execute(f"UPDATE channels SET {cols} WHERE id=?", (*fields.values(), channel_id))
    db.commit()


def delete_channel(channel_id):
    db = get_db()
    db.execute("DELETE FROM channels WHERE id=?", (channel_id,))
    db.commit()


# ------------------------------------------------------------------- ideas ----
def add_idea(channel_id, topic, source="user", video_type=None, priority=0, dedupe_hash=None):
    """Returns the new idea id, or None if dedupe_hash already exists for the channel."""
    db = get_db()
    cur = db.execute(
        "INSERT OR IGNORE INTO ideas (channel_id, source, topic, video_type, priority,"
        " dedupe_hash, created_at) VALUES (?,?,?,?,?,?,?)",
        (channel_id, source, topic, video_type, priority, dedupe_hash, now()),
    )
    db.commit()
    return cur.lastrowid if cur.rowcount else None


def list_ideas(channel_id=None, status=None, limit=200):
    q = "SELECT * FROM ideas"
    cond, params = [], []
    if channel_id is not None:
        cond.append("channel_id=?")
        params.append(channel_id)
    if status:
        cond.append("status=?")
        params.append(status)
    if cond:
        q += " WHERE " + " AND ".join(cond)
    q += " ORDER BY id DESC LIMIT ?"
    params.append(limit)
    return get_db().execute(q, params).fetchall()


def pending_ideas_ranked(channel_id):
    """All pending ideas in pick order: user ideas, then priority, then oldest."""
    return get_db().execute(
        "SELECT * FROM ideas WHERE channel_id=? AND status='pending'"
        " ORDER BY (source='user') DESC, priority DESC, id ASC",
        (channel_id,),
    ).fetchall()


def _resolve_job_video_type(channel_id, video_type=None):
    if video_type:
        return video_type
    channel = get_channel(channel_id)
    return channel["default_video_type"] if channel else None


def partition_pending_ideas(channel_id, video_type=None):
    """Split pending ideas into (suitable, rejected) for the intended job type.

    `rejected` is a list of (idea_row, reason). For non-kidsong job types every
    pending idea is suitable — the gate is a children's-content gate, not a
    general quality filter. Rejected rows are NOT modified: they keep their
    status and may still serve a future channel of their own type.
    """
    from studio import ideas as ideas_mod  # lazy: studio.ideas imports models

    rows = pending_ideas_ranked(channel_id)
    if not ideas_mod.is_kidsong_type(_resolve_job_video_type(channel_id, video_type)):
        return list(rows), []

    suitable, rejected = [], []
    for row in rows:
        reason = ideas_mod.kidsong_idea_rejection(row)
        (rejected.append((row, reason)) if reason else suitable.append(row))
    return suitable, rejected


def next_pending_idea(channel_id, video_type=None):
    """User ideas beat auto ideas, then priority, then oldest first.

    When the job this idea would feed is a kidsong job (explicit `video_type`,
    or the channel's default), ideas that are not toddler-appropriate are
    skipped rather than handed out. They stay pending and untouched. This is
    the funnel every idea-driven job creator goes through (cascade and the
    scheduler cron), which is why the filter lives here.
    """
    suitable, _rejected = partition_pending_ideas(channel_id, video_type)
    return suitable[0] if suitable else None


def set_idea_status(idea_id, status):
    db = get_db()
    used = now() if status == "used" else None
    if used:
        db.execute("UPDATE ideas SET status=?, used_at=? WHERE id=?", (status, used, idea_id))
    else:
        db.execute("UPDATE ideas SET status=? WHERE id=?", (status, idea_id))
    db.commit()


def recent_idea_topics(channel_id, limit=50):
    rows = get_db().execute(
        "SELECT topic FROM ideas WHERE channel_id=? ORDER BY id DESC LIMIT ?",
        (channel_id, limit),
    ).fetchall()
    return [r["topic"] for r in rows]


def delete_idea(idea_id):
    db = get_db()
    db.execute("DELETE FROM ideas WHERE id=? AND status='pending'", (idea_id,))
    db.commit()


# -------------------------------------------------------------------- jobs ----
def create_job(channel_id, idea_id=None, video_type=None, topic=None,
               scheduled_for=None, slot_key=None, auto_approve=False, style=None):
    """Returns job id, or None when slot_key already exists (double-enqueue guard)."""
    db = get_db()
    cur = db.execute(
        "INSERT OR IGNORE INTO jobs (channel_id, idea_id, video_type, topic, scheduled_for,"
        " slot_key, auto_approve, style, created_at) VALUES (?,?,?,?,?,?,?,?,?)",
        (channel_id, idea_id, video_type, topic, scheduled_for or now(), slot_key,
         1 if auto_approve else 0, style, now()),
    )
    db.commit()
    return cur.lastrowid if cur.rowcount else None


def get_job(job_id):
    return get_db().execute("SELECT * FROM jobs WHERE id=?", (job_id,)).fetchone()


def slot_key_exists(slot_key):
    return get_db().execute(
        "SELECT 1 FROM jobs WHERE slot_key=?", (slot_key,)
    ).fetchone() is not None


def update_job(job_id, **fields):
    if not fields:
        return
    cols = ", ".join(f"{k}=?" for k in fields)
    db = get_db()
    db.execute(f"UPDATE jobs SET {cols} WHERE id=?", (*fields.values(), job_id))
    db.commit()


def set_job_status(job_id, status, error=None):
    fields = {"status": status}
    if error is not None:
        fields["error"] = error
    if status in TRANSIENT_JOB_STATES and status == "scripting":
        fields["started_at"] = now()
    if status in ("ready", "uploaded", "failed", "rejected"):
        fields["finished_at"] = now()
    update_job(job_id, **fields)


def next_due_job():
    return get_db().execute(
        "SELECT * FROM jobs WHERE status='queued' AND scheduled_for <= ?"
        " ORDER BY scheduled_for ASC, id ASC LIMIT 1",
        (now(),),
    ).fetchone()


def jobs_in_states(states):
    marks = ",".join("?" * len(states))
    return get_db().execute(
        f"SELECT * FROM jobs WHERE status IN ({marks}) ORDER BY id", tuple(states)
    ).fetchall()


def current_job():
    rows = jobs_in_states(WORKING_JOB_STATES)
    return rows[0] if rows else None


def list_jobs(status=None, limit=100):
    if status:
        return get_db().execute(
            "SELECT * FROM jobs WHERE status=? ORDER BY id DESC LIMIT ?", (status, limit)
        ).fetchall()
    return get_db().execute("SELECT * FROM jobs ORDER BY id DESC LIMIT ?", (limit,)).fetchall()


def count_jobs(status):
    row = get_db().execute("SELECT COUNT(*) c FROM jobs WHERE status=?", (status,)).fetchone()
    return row["c"]


# ------------------------------------------------------------------ videos ----
def create_video(job_id, channel_id, path, title, description, tags, duration,
                 script=None, bg_asset_id=None, thumb_path=None):
    db = get_db()
    cur = db.execute(
        "INSERT INTO videos (job_id, channel_id, path, title, description, tags_json,"
        " duration, script_json, bg_asset_id, thumb_path, created_at)"
        " VALUES (?,?,?,?,?,?,?,?,?,?,?)",
        (job_id, channel_id, path, title, description, json.dumps(tags or []),
         duration, json.dumps(script) if script else None, bg_asset_id, thumb_path, now()),
    )
    db.commit()
    return cur.lastrowid


def _parse_video(row):
    if not row:
        return None
    row["tags"] = _loads(row.get("tags_json"), [])
    return row


def get_video(video_id):
    return _parse_video(
        get_db().execute("SELECT * FROM videos WHERE id=?", (video_id,)).fetchone()
    )


def get_video_by_job(job_id):
    return _parse_video(
        get_db().execute(
            "SELECT * FROM videos WHERE job_id=? ORDER BY id DESC LIMIT 1", (job_id,)
        ).fetchone()
    )


def update_video(video_id, **fields):
    if "tags" in fields:
        fields["tags_json"] = json.dumps(fields.pop("tags"))
    if not fields:
        return
    cols = ", ".join(f"{k}=?" for k in fields)
    db = get_db()
    db.execute(f"UPDATE videos SET {cols} WHERE id=?", (*fields.values(), video_id))
    db.commit()


def videos_for_review():
    rows = get_db().execute(
        "SELECT v.*, j.status AS job_status, j.id AS job_id, c.name AS channel_name"
        " FROM videos v JOIN jobs j ON j.id = v.job_id"
        " LEFT JOIN channels c ON c.id = v.channel_id"
        " WHERE j.status='ready' ORDER BY v.id DESC"
    ).fetchall()
    return [_parse_video(r) for r in rows]


# ----------------------------------------------------------------- uploads ----
def record_upload(video_id, channel_id, yt_video_id, url, privacy):
    db = get_db()
    db.execute(
        "INSERT INTO uploads (video_id, channel_id, yt_video_id, url, privacy, uploaded_at)"
        " VALUES (?,?,?,?,?,?)",
        (video_id, channel_id, yt_video_id, url, privacy, now()),
    )
    db.commit()


def uploads_today():
    row = get_db().execute(
        "SELECT COUNT(*) c FROM uploads WHERE uploaded_at LIKE ?", (today() + "%",)
    ).fetchone()
    return row["c"]


def list_uploads(limit=100):
    return get_db().execute(
        "SELECT u.*, v.title, v.path, c.name AS channel_name FROM uploads u"
        " LEFT JOIN videos v ON v.id = u.video_id"
        " LEFT JOIN channels c ON c.id = u.channel_id"
        " ORDER BY u.id DESC LIMIT ?",
        (limit,),
    ).fetchall()


# ------------------------------------------------------------------ assets ----
def record_asset(kind, source, path, source_url=None, keywords="", attribution="",
                 duration=None, size_bytes=0):
    db = get_db()
    cur = db.execute(
        "INSERT INTO assets (kind, source, source_url, path, keywords, attribution,"
        " duration, size_bytes, created_at, last_used_at) VALUES (?,?,?,?,?,?,?,?,?,?)",
        (kind, source, source_url, path, keywords, attribution, duration, size_bytes,
         now(), now()),
    )
    db.commit()
    return cur.lastrowid


def touch_asset(asset_id):
    db = get_db()
    db.execute("UPDATE assets SET last_used_at=? WHERE id=?", (now(), asset_id))
    db.commit()


def list_assets(kind=None):
    if kind:
        return get_db().execute(
            "SELECT * FROM assets WHERE kind=? ORDER BY last_used_at DESC", (kind,)
        ).fetchall()
    return get_db().execute("SELECT * FROM assets ORDER BY last_used_at DESC").fetchall()


def delete_asset(asset_id):
    db = get_db()
    db.execute("DELETE FROM assets WHERE id=?", (asset_id,))
    db.commit()


# -------------------------------------------------------------- job events ----
def add_event(job_id, stage, message):
    db = get_db()
    db.execute(
        "INSERT INTO job_events (job_id, ts, stage, message) VALUES (?,?,?,?)",
        (job_id, now(), stage, message),
    )
    db.commit()


def events_for_job(job_id, limit=100):
    return get_db().execute(
        "SELECT * FROM job_events WHERE job_id=? ORDER BY id ASC LIMIT ?", (job_id, limit)
    ).fetchall()
