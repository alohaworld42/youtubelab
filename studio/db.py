"""SQLite layer: one file (studio.db), WAL mode, user_version migrations.

Connections are per-thread (Flask request threads + the scheduler thread each
get their own). Rows come back as plain dicts.
"""
import os
import sqlite3
import threading

_HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

_local = threading.local()
_init_lock = threading.Lock()
_initialized = False


def db_path():
    return os.environ.get("STUDIO_DB_PATH") or os.path.join(_HERE, "studio.db")


def _dict_factory(cursor, row):
    return {col[0]: row[i] for i, col in enumerate(cursor.description)}


MIGRATIONS = [
    # v1 — full initial schema
    """
    CREATE TABLE channels (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT NOT NULL,
        yt_channel_id TEXT UNIQUE,
        yt_channel_title TEXT,
        token_path TEXT,
        secret_path TEXT,
        niche TEXT NOT NULL DEFAULT '',
        description TEXT NOT NULL DEFAULT '',
        default_video_type TEXT NOT NULL DEFAULT 'facts',
        voices_json TEXT NOT NULL DEFAULT '{}',
        privacy TEXT NOT NULL DEFAULT 'private',
        schedule_json TEXT NOT NULL DEFAULT '[]',
        asset_prefs_json TEXT NOT NULL DEFAULT '{}',
        status TEXT NOT NULL DEFAULT 'active',
        created_at TEXT NOT NULL
    );

    CREATE TABLE ideas (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        channel_id INTEGER REFERENCES channels(id) ON DELETE CASCADE,
        source TEXT NOT NULL DEFAULT 'user',
        topic TEXT NOT NULL,
        video_type TEXT,
        priority INTEGER NOT NULL DEFAULT 0,
        status TEXT NOT NULL DEFAULT 'pending',
        dedupe_hash TEXT,
        created_at TEXT NOT NULL,
        used_at TEXT
    );
    CREATE UNIQUE INDEX idx_ideas_dedupe ON ideas(channel_id, dedupe_hash);

    CREATE TABLE jobs (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        channel_id INTEGER REFERENCES channels(id) ON DELETE SET NULL,
        idea_id INTEGER REFERENCES ideas(id) ON DELETE SET NULL,
        video_type TEXT,
        topic TEXT,
        status TEXT NOT NULL DEFAULT 'queued',
        error TEXT,
        attempts INTEGER NOT NULL DEFAULT 0,
        auto_approve INTEGER NOT NULL DEFAULT 0,
        scheduled_for TEXT,
        slot_key TEXT UNIQUE,
        started_at TEXT,
        finished_at TEXT,
        created_at TEXT NOT NULL
    );
    CREATE INDEX idx_jobs_status ON jobs(status);

    CREATE TABLE videos (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        job_id INTEGER REFERENCES jobs(id) ON DELETE CASCADE,
        channel_id INTEGER,
        path TEXT NOT NULL,
        title TEXT,
        description TEXT,
        tags_json TEXT NOT NULL DEFAULT '[]',
        duration REAL,
        script_json TEXT,
        bg_asset_id INTEGER,
        created_at TEXT NOT NULL
    );

    CREATE TABLE uploads (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        video_id INTEGER REFERENCES videos(id) ON DELETE SET NULL,
        channel_id INTEGER,
        yt_video_id TEXT,
        url TEXT,
        privacy TEXT,
        uploaded_at TEXT NOT NULL
    );

    CREATE TABLE assets (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        kind TEXT NOT NULL,
        source TEXT NOT NULL,
        source_url TEXT,
        path TEXT NOT NULL,
        keywords TEXT NOT NULL DEFAULT '',
        attribution TEXT NOT NULL DEFAULT '',
        duration REAL,
        size_bytes INTEGER NOT NULL DEFAULT 0,
        created_at TEXT NOT NULL,
        last_used_at TEXT
    );

    CREATE TABLE job_events (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        job_id INTEGER REFERENCES jobs(id) ON DELETE CASCADE,
        ts TEXT NOT NULL,
        stage TEXT,
        message TEXT
    );
    """,
    # v2 — style presets, per-channel kids flag, thumbnails
    """
    ALTER TABLE channels ADD COLUMN style TEXT NOT NULL DEFAULT '';
    ALTER TABLE channels ADD COLUMN made_for_kids INTEGER NOT NULL DEFAULT 0;
    ALTER TABLE jobs ADD COLUMN style TEXT;
    ALTER TABLE videos ADD COLUMN thumb_path TEXT;
    """,
    # v3 — resume_base: a kidsong job that died mid-render remembers the base
    # it was rendering so a retry can resume that run (reusing already-
    # accepted takes) instead of starting a brand-new episode from scratch
    # (Contract 3, forensics on runs that died mid-render with zero retries).
    """
    ALTER TABLE jobs ADD COLUMN resume_base TEXT;
    """,
    # v4 — per-channel language, branding and content profile, so one
    # dashboard can drive channels of different language/brand.
    """
    ALTER TABLE channels ADD COLUMN language TEXT NOT NULL DEFAULT 'en';
    ALTER TABLE channels ADD COLUMN branding_json TEXT NOT NULL DEFAULT '{}';
    ALTER TABLE channels ADD COLUMN content_profile TEXT NOT NULL DEFAULT '';
    """,
]


def get_db():
    """Per-thread connection; creates schema on first use."""
    conn = getattr(_local, "conn", None)
    if conn is not None and getattr(_local, "path", None) == db_path():
        return conn
    init_db()
    conn = sqlite3.connect(db_path(), timeout=30)
    conn.row_factory = _dict_factory
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    _local.conn = conn
    _local.path = db_path()
    return conn


def init_db():
    global _initialized
    with _init_lock:
        if _initialized and os.path.exists(db_path()):
            return
        conn = sqlite3.connect(db_path(), timeout=30)
        try:
            conn.execute("PRAGMA journal_mode=WAL")
            version = conn.execute("PRAGMA user_version").fetchone()[0]
            for i, script in enumerate(MIGRATIONS, start=1):
                if version < i:
                    conn.executescript(script)
                    conn.execute(f"PRAGMA user_version = {i}")
                    conn.commit()
            _initialized = True
        finally:
            conn.close()


def reset_for_tests():
    """Drop the cached connection so tests can point STUDIO_DB_PATH elsewhere."""
    global _initialized
    conn = getattr(_local, "conn", None)
    if conn is not None:
        conn.close()
        _local.conn = None
    _initialized = False
