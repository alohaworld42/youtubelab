import json
import os
import sqlite3

from PIL import Image

from pipeline.config import apply_style, load_dotenv
from pipeline.thumbnail import compose_thumbnail, _wrap_title
from PIL import ImageDraw, ImageFont

BASE = {
    "captions": {"font_path": "x", "font_size": 110, "words_per_group": 1, "uppercase": True,
                 "fill_color": [255, 255, 255], "stroke_color": [0, 0, 0], "stroke_width": 14},
    "video": {"fps": 30, "music_volume": 0.06},
    "assets": {"source_order": ["gameplay", "stock", "generated"], "stock_query_bias": ""},
    "styles": {
        "kids": {
            "captions": {"words_per_group": 3, "uppercase": False},
            "video": {"music_volume": 0.22},
            "assets": {"source_order": ["stock", "generated"]},
        }
    },
}


def test_apply_style_merges_deep():
    merged = apply_style(BASE, "kids")
    assert merged["captions"]["words_per_group"] == 3
    assert merged["captions"]["uppercase"] is False
    assert merged["captions"]["font_size"] == 110  # untouched base value kept
    assert merged["video"]["music_volume"] == 0.22
    assert merged["video"]["fps"] == 30
    assert merged["assets"]["source_order"] == ["stock", "generated"]


def test_apply_style_unknown_or_empty_is_noop():
    assert apply_style(BASE, None) is BASE
    assert apply_style(BASE, "") is BASE
    assert apply_style(BASE, "nope") is BASE


def test_migration_v2_upgrades_v1_db(tmp_path, monkeypatch):
    from studio import db

    path = tmp_path / "old.db"
    conn = sqlite3.connect(str(path))
    conn.executescript(db.MIGRATIONS[0])
    conn.execute("PRAGMA user_version = 1")
    conn.commit()
    conn.close()

    monkeypatch.setenv("STUDIO_DB_PATH", str(path))
    db.reset_for_tests()
    db.init_db()

    conn = sqlite3.connect(str(path))
    cols = {r[1] for r in conn.execute("PRAGMA table_info(channels)")}
    assert "style" in cols and "made_for_kids" in cols
    vcols = {r[1] for r in conn.execute("PRAGMA table_info(videos)")}
    assert "thumb_path" in vcols
    conn.close()
    db.reset_for_tests()


def test_keys_dir_loader(tmp_path, monkeypatch):
    monkeypatch.delenv("PIXABAY_API_KEY", raising=False)
    keys = tmp_path / "keys"
    keys.mkdir()
    (keys / "pixabay_api_key").write_text("k123\n", encoding="utf-8")
    (keys / "empty_key").write_text("", encoding="utf-8")
    load_dotenv(str(tmp_path))
    assert os.environ.get("PIXABAY_API_KEY") == "k123"
    assert "EMPTY_KEY" not in os.environ


def test_wrap_title_respects_width():
    img = Image.new("RGB", (10, 10))
    d = ImageDraw.Draw(img)
    font = ImageFont.load_default()
    lines = _wrap_title(d, "one two three four five six seven eight", font, 60)
    assert 1 < len(lines) <= 4
    for line in lines:
        assert d.textlength(line, font=font) <= 60 or len(line.split()) == 1


def test_compose_thumbnail_writes_jpeg(tmp_path):
    frame = Image.new("RGB", (1080, 1920), (200, 60, 140))
    cfg = {"captions": {"font_path": "C:/Windows/Fonts/arialbd.ttf",
                        "stroke_color": [0, 0, 0], "highlight_color": [255, 222, 0]}}
    out = str(tmp_path / "t.jpg")
    compose_thumbnail(frame, "Super Fun Colors For Happy Kids", cfg, out)
    with Image.open(out) as img:
        assert img.size == (1280, 720)
        assert img.format == "JPEG"


def test_generated_bg_marker_flows_from_assets(tmp_db, monkeypatch):
    from studio import assets

    monkeypatch.setattr(assets, "_from_stock", lambda kw, cfg: (None, None))
    cfg = {"_root": ".", "paths": {"gameplay_dir": "assets/gameplay"},
           "assets": {"source_order": ["stock", "generated"]}}
    path, asset_id = assets.get_background(None, "colors for kids", None, cfg)
    from pipeline.assemble import GENERATED_BG

    assert path == GENERATED_BG
    assert asset_id is None
