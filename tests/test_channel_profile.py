"""Per-channel language, branding and content-profile columns (migration
that adds `language`, `branding_json`, `content_profile` to `channels`).

Covers: defaults on a fresh channel, the branding dict round-tripping through
the DB, language-tag normalization, resilience to hand-edited/broken
`branding_json`, and incremental migration of a DB created before this
migration existed.
"""
import sqlite3

import pytest

from studio import db, models

DEFAULT_BRANDING = {
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

SAMPLE_BRANDING = {
    "title_template": "{topic} | Lernlieder für Kinder",
    "description_template": "{topic} auf {language} für {channel}",
    "tags": ["kids", "lernlieder"],
    "intro_path": "assets/intro.mp4",
    "outro_path": "assets/outro.mp4",
    "logo_path": "assets/logo.png",
    "primary_color": "#ff0000",
    "secondary_color": "#00ff00",
    "font": "Comic Sans",
}


# ---------------------------------------------------------------- defaults ----
def test_fresh_channel_has_profile_defaults(tmp_db):
    ch_id = models.create_channel("Test")
    ch = models.get_channel(ch_id)

    assert ch["language"] == "en"
    assert ch["content_profile"] == ""
    assert ch["branding"] == DEFAULT_BRANDING


def test_content_profile_is_stored_verbatim_not_interpreted(tmp_db):
    ch_id = models.create_channel("Test", content_profile="lofi-facts-v2")
    assert models.get_channel(ch_id)["content_profile"] == "lofi-facts-v2"


# ------------------------------------------------------------- branding ----
def test_branding_roundtrips_through_create(tmp_db):
    ch_id = models.create_channel("Test", branding=SAMPLE_BRANDING)
    assert models.get_channel(ch_id)["branding"] == SAMPLE_BRANDING


def test_branding_roundtrips_through_update(tmp_db):
    ch_id = models.create_channel("Test")
    models.update_channel(ch_id, branding=SAMPLE_BRANDING)
    assert models.get_channel(ch_id)["branding"] == SAMPLE_BRANDING


def test_branding_partial_dict_keeps_other_keys_at_default(tmp_db):
    ch_id = models.create_channel("Test", branding={"primary_color": "#123456"})
    branding = models.get_channel(ch_id)["branding"]

    assert branding["primary_color"] == "#123456"
    assert branding["tags"] == []
    assert branding["font"] == ""


def test_channel_branding_helper_handles_none_and_missing_channel():
    assert models.channel_branding(None) == DEFAULT_BRANDING
    assert models.channel_branding({}) == DEFAULT_BRANDING


# --------------------------------------------------------------- language ----
@pytest.mark.parametrize("raw,expected", [
    ("de-DE", "de"),
    ("DE", "de"),
    ("de", "de"),
    ("en-US", "en"),
    ("En-us", "en"),
    ("", "en"),
    (None, "en"),
])
def test_channel_language_normalizes(raw, expected):
    assert models.channel_language({"language": raw}) == expected


def test_channel_language_handles_none_channel():
    assert models.channel_language(None) == "en"


def test_language_normalized_on_create_and_update(tmp_db):
    ch_id = models.create_channel("Test", language="de-DE")
    assert models.get_channel(ch_id)["language"] == "de"

    models.update_channel(ch_id, language="EN-us")
    assert models.get_channel(ch_id)["language"] == "en"


def test_language_defaults_to_en_when_omitted(tmp_db):
    ch_id = models.create_channel("Test", language="")
    assert models.get_channel(ch_id)["language"] == "en"


# -------------------------------------------------------- broken branding ----
def test_hand_edited_broken_branding_json_does_not_raise(tmp_db):
    ch_id = models.create_channel("Test")
    conn = db.get_db()
    conn.execute("UPDATE channels SET branding_json=? WHERE id=?", ("{not valid json", ch_id))
    conn.commit()

    ch = models.get_channel(ch_id)  # must not raise

    assert ch["branding"] == DEFAULT_BRANDING


def test_channel_branding_helper_handles_non_object_json():
    assert models.channel_branding({"branding_json": "[1, 2, 3]"}) == DEFAULT_BRANDING
    assert models.channel_branding({"branding_json": "null"}) == DEFAULT_BRANDING
    assert models.channel_branding({"branding_json": ""}) == DEFAULT_BRANDING


# --------------------------------------------------------------- migration ----
def test_migration_adds_profile_columns_to_a_pre_existing_db(tmp_path, monkeypatch):
    """A DB created before this migration existed (all earlier migrations
    applied, this one not yet) must pick up `language`/`branding_json`/
    `content_profile` — with correct defaults on rows that predate them —
    the next time `init_db()` runs, without anyone re-running the earlier
    migrations."""
    path = tmp_path / "old.db"
    conn = sqlite3.connect(str(path))
    for script in db.MIGRATIONS[:-1]:  # every migration except the one just added
        conn.executescript(script)
    conn.execute(
        "INSERT INTO channels (name, created_at) VALUES ('Legacy', '2020-01-01 00:00:00')"
    )
    conn.execute(f"PRAGMA user_version = {len(db.MIGRATIONS) - 1}")
    conn.commit()
    conn.close()

    monkeypatch.setenv("STUDIO_DB_PATH", str(path))
    db.reset_for_tests()
    db.init_db()

    conn = sqlite3.connect(str(path))
    conn.row_factory = sqlite3.Row
    cols = {r[1] for r in conn.execute("PRAGMA table_info(channels)")}
    assert {"language", "branding_json", "content_profile"} <= cols

    row = conn.execute("SELECT * FROM channels WHERE name='Legacy'").fetchone()
    assert row["language"] == "en"
    assert row["branding_json"] == "{}"
    assert row["content_profile"] == ""
    conn.close()
    db.reset_for_tests()

    # And the model layer reads the migrated row cleanly through get_channel.
    monkeypatch.setenv("STUDIO_DB_PATH", str(path))
    db.reset_for_tests()
    ch = models.list_channels()[0]
    assert ch["language"] == "en"
    assert ch["branding"] == DEFAULT_BRANDING
    db.reset_for_tests()
