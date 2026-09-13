import os

import pytest

from studio import assets, models


CFG = {"_root": ".", "paths": {"gameplay_dir": "assets/gameplay"}, "studio": {}}


def _channel(order):
    return {
        "id": 1,
        "asset_prefs": {"source_order": order, "gameplay_urls": []},
    }


def test_keyword_extraction_prefers_tags():
    kw = assets.extract_keywords("cats vs dogs", {"tags": ["Space Facts", "the moon"]})
    assert "space" in kw and "facts" in kw and "moon" in kw
    assert "the" not in kw


def test_keyword_extraction_falls_back_to_topic():
    kw = assets.extract_keywords("the haunted lighthouse of germany", None)
    assert "haunted" in kw and "lighthouse" in kw
    assert "the" not in kw and "of" not in kw


def test_gameplay_tier_wins_when_available(tmp_db, monkeypatch):
    monkeypatch.setattr(assets, "_from_channel_prefs", lambda ch, cfg: ("gp.mp4", 1))
    monkeypatch.setattr(assets, "_from_stock", lambda kw, cfg: ("stock.mp4", 2))
    path, asset_id = assets.get_background(_channel(["gameplay", "stock"]), "topic", None, CFG)
    assert path == "gp.mp4"


def test_falls_through_to_stock(tmp_db, monkeypatch):
    monkeypatch.setattr(assets, "_from_channel_prefs", lambda ch, cfg: (None, None))
    monkeypatch.setattr(assets, "_from_stock", lambda kw, cfg: ("stock.mp4", 2))
    path, asset_id = assets.get_background(_channel(["gameplay", "stock"]), "topic", None, CFG)
    assert path == "stock.mp4"


def test_stock_first_order_respected(tmp_db, monkeypatch):
    monkeypatch.setattr(assets, "_from_channel_prefs", lambda ch, cfg: ("gp.mp4", 1))
    monkeypatch.setattr(assets, "_from_stock", lambda kw, cfg: ("stock.mp4", 2))
    path, _ = assets.get_background(_channel(["stock", "gameplay"]), "topic", None, CFG)
    assert path == "stock.mp4"


def test_all_tiers_fail_returns_none(tmp_db, monkeypatch):
    monkeypatch.setattr(assets, "_from_channel_prefs", lambda ch, cfg: (None, None))
    monkeypatch.setattr(assets, "_from_stock", lambda kw, cfg: (None, None))
    path, asset_id = assets.get_background(_channel(["gameplay", "stock"]), "topic", None, CFG)
    assert path is None and asset_id is None


def test_no_channel_uses_local_gameplay(tmp_db, monkeypatch, tmp_path):
    """Quick-generate (no channel): a real local clip means None sentinel
    (assemble picks it), not skipping straight to stock."""
    gp = tmp_path / "gameplay"
    gp.mkdir()
    (gp / "clip1.mp4").write_bytes(b"x")
    cfg = {"_root": str(tmp_path), "paths": {"gameplay_dir": "gameplay"},
           "assets": {"source_order": ["gameplay", "stock", "generated"]}}
    monkeypatch.setattr(assets, "_from_stock", lambda kw, cfg: ("stock.mp4", 9))
    path, asset_id = assets.get_background(None, "topic", None, cfg)
    assert path is None and asset_id is None


def test_no_channel_only_placeholder_falls_through_to_stock(tmp_db, monkeypatch, tmp_path):
    """Only the bundled _placeholder present -> skip gameplay, use stock."""
    gp = tmp_path / "gameplay"
    gp.mkdir()
    (gp / "_placeholder.mp4").write_bytes(b"x")
    cfg = {"_root": str(tmp_path), "paths": {"gameplay_dir": "gameplay"},
           "assets": {"source_order": ["gameplay", "stock", "generated"]}}
    monkeypatch.setattr(assets, "_from_stock", lambda kw, cfg: ("stock.mp4", 9))
    path, asset_id = assets.get_background(None, "topic", None, cfg)
    assert path == "stock.mp4"


def test_stock_search_skipped_without_api_keys(tmp_db, monkeypatch):
    monkeypatch.delenv("PEXELS_API_KEY", raising=False)
    monkeypatch.delenv("PIXABAY_API_KEY", raising=False)
    path, asset_id = assets._from_stock(["space"], CFG)
    assert path is None and asset_id is None


def test_cached_stock_reused(tmp_db, monkeypatch, tmp_path):
    clip = tmp_path / "clip.mp4"
    clip.write_bytes(b"x")
    models.record_asset("stock", "pexels", str(clip), keywords="space,facts")
    path, asset_id = assets._from_stock(["space"], CFG)
    assert path == str(clip)
    assert asset_id is not None


# ------------------------------------------------------- atomic stock download --
class _FakeResponse:
    """Minimal stand-in for requests.get(stream=True)'s context manager."""

    def __init__(self, chunks, fail_after=None):
        self._chunks = chunks
        self._fail_after = fail_after

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def raise_for_status(self):
        pass

    def iter_content(self, chunk_size=None):
        for i, chunk in enumerate(self._chunks):
            if self._fail_after is not None and i == self._fail_after:
                raise ConnectionError("connection reset mid-download")
            yield chunk


def test_a_completed_download_lands_at_the_destination(tmp_path, monkeypatch):
    dest = tmp_path / "clip.mp4"
    monkeypatch.setattr(assets.requests, "get",
                        lambda *a, **k: _FakeResponse([b"aaa", b"bbb"]))

    assets._download("https://example.com/v.mp4", str(dest))

    assert dest.read_bytes() == b"aaabbb"
    assert not (tmp_path / "clip.mp4.part").exists()


def test_an_interrupted_download_leaves_no_stump(tmp_path, monkeypatch):
    """Writing straight to `dest` left a TRUNCATED mp4 at the final path — and
    every caller guards with `if not os.path.exists(dest)`, so that stump was
    then reused as the background clip on every later render, with its partial
    size recorded as the real one."""
    dest = tmp_path / "clip.mp4"
    monkeypatch.setattr(assets.requests, "get",
                        lambda *a, **k: _FakeResponse([b"aaa", b"bbb"], fail_after=1))

    with pytest.raises(ConnectionError):
        assets._download("https://example.com/v.mp4", str(dest))

    assert not dest.exists(), "a partial download was cached as if it were complete"
    assert not (tmp_path / "clip.mp4.part").exists(), "the .part file was left behind"


def test_an_interrupt_signal_mid_download_leaves_no_stump(tmp_path, monkeypatch):
    """KeyboardInterrupt/SIGTERM is exactly the case that produced the stump, and
    it is a BaseException — a plain `except Exception` would miss it."""
    dest = tmp_path / "clip.mp4"

    class _Interrupting(_FakeResponse):
        def iter_content(self, chunk_size=None):
            yield b"aaa"
            raise KeyboardInterrupt

    monkeypatch.setattr(assets.requests, "get", lambda *a, **k: _Interrupting([]))

    with pytest.raises(KeyboardInterrupt):
        assets._download("https://example.com/v.mp4", str(dest))

    assert not dest.exists()
    assert not (tmp_path / "clip.mp4.part").exists()


def test_a_retry_after_a_failed_download_actually_refetches(tmp_path, monkeypatch):
    """The point of the fix: the next run must not treat the stump as a hit."""
    dest = tmp_path / "clip.mp4"
    calls = []

    def _get(*a, **k):
        calls.append(1)
        if len(calls) == 1:
            return _FakeResponse([b"aaa", b"bbb"], fail_after=1)
        return _FakeResponse([b"aaa", b"bbb"])

    monkeypatch.setattr(assets.requests, "get", _get)

    with pytest.raises(ConnectionError):
        assets._download("https://example.com/v.mp4", str(dest))
    assert not os.path.exists(str(dest))          # the guard callers use

    assets._download("https://example.com/v.mp4", str(dest))
    assert dest.read_bytes() == b"aaabbb"
