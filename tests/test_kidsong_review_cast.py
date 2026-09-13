"""Tests for the cast-ground-truth enrichment ExternalReviewer.review adds to
its vision-review request payload (`cast_text`, `cast_version`,
`expected_children`) -- fixing the confirmed broken wire where the vision
reviewer received only bare character NAMES (e.g. `characters: ["Zuri"]`)
with no appearance ground truth, making rubric dimensions 3 (CAST
INTEGRITY), 4 (WARDROBE) and 7 (TEMPORAL COHERENCE) unverifiable.

No GPU/network/real vision call needed: `_heuristic.review` is monkeypatched
to always accept (stage 1 heuristics are untouched by this change and are
covered by their own tests) and the external_timeout is set tiny so `review`
falls straight through to the heuristic-timeout fallback almost immediately
-- the request JSON is written to disk *before* that wait loop starts, so we
just read it back off disk afterward to inspect the payload.

`pipeline/kidsong/cast.py` may not exist on disk yet (a sibling module
another agent is landing). Tests that need a working cast module inject a
fake one into `sys.modules` rather than depending on the real file; the
ImportError-degradation test forces the "module genuinely unavailable" case
explicitly via `sys.modules[...] = None` (the standard way to make Python's
import system raise ImportError for a name regardless of what's on disk).
"""
import json
import os
import sys

import pytest

from pipeline.kidsong.review import ExternalReviewer, _EMPTY_HINTS, _resolve_cast_enrichment

_CAST_MODULE_NAME = "pipeline.kidsong.cast"

_FAKE_CAST_SENTENCES = {
    "Zuri": "Zuri (Black skin, braids, yellow dress)",
    "Kofi": "Kofi (Black skin, short black hair, blue shirt)",
    "Nala": "Nala (Black skin, puffs, purple dress)",
}


def _make_fake_cast_module():
    """A minimal stand-in for the not-yet-landed `pipeline.kidsong.cast` API."""
    import types

    mod = types.ModuleType(_CAST_MODULE_NAME)

    def cast_sentence(names):
        names = list(names) if names else ["all"]
        if names == ["all"]:
            return "; ".join(_FAKE_CAST_SENTENCES[n] for n in ("Zuri", "Kofi", "Nala"))
        picked = [_FAKE_CAST_SENTENCES[n] for n in names if n in _FAKE_CAST_SENTENCES]
        return "; ".join(picked) if picked else "; ".join(_FAKE_CAST_SENTENCES.values())

    def describe(name):
        return _FAKE_CAST_SENTENCES.get(name, "")

    def expected_child_count(names):
        names = list(names) if names else ["all"]
        return 3 if names == ["all"] else max(1, len(names))

    def names():
        return ["Zuri", "Kofi", "Nala"]

    def version():
        return "cast-v1-test"

    mod.cast_sentence = cast_sentence
    mod.describe = describe
    mod.expected_child_count = expected_child_count
    mod.names = names
    mod.version = version
    return mod


@pytest.fixture
def fake_cast_module(monkeypatch):
    """Inject a working fake `pipeline.kidsong.cast` for the duration of a test.

    `from pipeline.kidsong import cast` has a fast path: if the `pipeline.kidsong`
    package object already has a `cast` attribute cached (true once the real
    module -- which now exists on disk -- has been imported anywhere in this
    test session, e.g. by its own test file), `from X import Y` uses that
    cached attribute directly and never re-consults `sys.modules['X.Y']`. So
    both the sys.modules entry AND the package attribute must be patched, or
    a real cast.py elsewhere in the session silently shadows the fake one.
    """
    import pipeline.kidsong as kidsong_pkg

    mod = _make_fake_cast_module()
    monkeypatch.setitem(sys.modules, _CAST_MODULE_NAME, mod)
    monkeypatch.setattr(kidsong_pkg, "cast", mod, raising=False)
    return mod


@pytest.fixture
def no_cast_module(monkeypatch):
    """Force `from pipeline.kidsong import cast` to raise ImportError, regardless
    of whether a real cast.py exists on disk. Needs both halves (see
    `fake_cast_module` docstring for why): remove the cached package attribute
    so the `hasattr` fast path can't short-circuit, and set the sys.modules
    entry to None -- the documented way to make the import system treat a
    name as unimportable -- so a fresh import attempt also fails.
    """
    import pipeline.kidsong as kidsong_pkg

    monkeypatch.delattr(kidsong_pkg, "cast", raising=False)
    monkeypatch.setitem(sys.modules, _CAST_MODULE_NAME, None)


def _reviewer(timeout=0.03, poll_seconds=0.005):
    # These tests assert on the REQUEST payload the reviewer writes, so the
    # shot-level auto-accept prefilter (which would skip writing a request for a
    # clean heuristic) must be off here.
    r = ExternalReviewer(cfg={"kidsong": {"review": {
        "external_timeout": timeout, "shot_auto_accept": {"enabled": False},
    }}})
    r.timeout = timeout
    r.poll_seconds = poll_seconds
    r._heuristic.review = lambda shot, video_path: {
        "accept": True, "score": 1.0, "reasons": [], "retry_hints": dict(_EMPTY_HINTS),
    }
    return r


def _shot(characters, shot_id="s01"):
    return {"id": shot_id, "characters": characters, "shot_type": "closeup", "action": "brushing teeth"}


def _video_path(tmp_path, name):
    # Doesn't need to exist: contact_sheet() falls back to a blank placeholder
    # frame when the video can't be opened.
    return str(tmp_path / f"{name}.mp4")


def _read_request(out_dir, take):
    with open(os.path.join(out_dir, f"{take}.request.json"), encoding="utf-8") as f:
        return json.load(f)


# ------------------------------------------------------------- payload contents --
def test_request_contains_cast_text_scoped_to_shot_characters(tmp_path, fake_cast_module):
    r = _reviewer()
    out_dir = str(tmp_path / "review")
    shot = _shot(["Zuri"], shot_id="s01")
    video_path = _video_path(tmp_path, "s01_a0")

    r.review(shot, video_path, out_dir=out_dir)

    req = _read_request(out_dir, "s01_a0")
    assert "cast_text" in req
    assert "Zuri" in req["cast_text"]
    assert "Kofi" not in req["cast_text"]
    assert "Nala" not in req["cast_text"]


def test_request_contains_expected_children_single_name(tmp_path, fake_cast_module):
    r = _reviewer()
    out_dir = str(tmp_path / "review")
    shot = _shot(["Kofi"], shot_id="s02")
    video_path = _video_path(tmp_path, "s02_a0")

    r.review(shot, video_path, out_dir=out_dir)

    req = _read_request(out_dir, "s02_a0")
    assert req["expected_children"] == 1


def test_request_contains_expected_children_all(tmp_path, fake_cast_module):
    r = _reviewer()
    out_dir = str(tmp_path / "review")
    shot = _shot(["all"], shot_id="s03")
    video_path = _video_path(tmp_path, "s03_a0")

    r.review(shot, video_path, out_dir=out_dir)

    req = _read_request(out_dir, "s03_a0")
    assert req["expected_children"] == 3


def test_expected_children_falls_back_without_cast_module(tmp_path, no_cast_module):
    """Even with no cast.py at all, expected_children must still be present and
    correct -- it has its own fallback (_expected_face_count), independent of
    the cast_text resolution path."""
    r = _reviewer()
    out_dir = str(tmp_path / "review")

    r.review(_shot(["Nala"], shot_id="s04"), _video_path(tmp_path, "s04_a0"), out_dir=out_dir)
    assert _read_request(out_dir, "s04_a0")["expected_children"] == 1

    r.review(_shot(["all"], shot_id="s05"), _video_path(tmp_path, "s05_a0"), out_dir=out_dir)
    assert _read_request(out_dir, "s05_a0")["expected_children"] == 3


def test_cast_version_included_when_module_available(tmp_path, fake_cast_module):
    r = _reviewer()
    out_dir = str(tmp_path / "review")
    r.review(_shot(["Zuri"]), _video_path(tmp_path, "s06_a0"), out_dir=out_dir)
    req = _read_request(out_dir, "s06_a0")
    assert req.get("cast_version") == "cast-v1-test"


# ------------------------------------------------------------------- overrides --
def test_explicit_cast_text_kwarg_overrides_auto_resolution(tmp_path, fake_cast_module):
    r = _reviewer()
    out_dir = str(tmp_path / "review")
    shot = _shot(["Zuri"], shot_id="s07")

    r.review(shot, _video_path(tmp_path, "s07_a0"), out_dir=out_dir, cast_text="OVERRIDDEN TEXT")

    req = _read_request(out_dir, "s07_a0")
    assert req["cast_text"] == "OVERRIDDEN TEXT"


def test_song_characters_used_as_fallback_when_cast_module_missing(tmp_path, no_cast_module):
    r = _reviewer()
    out_dir = str(tmp_path / "review")
    shot = _shot(["Zuri"], shot_id="s08")
    song = {"characters": "Three adorable Black toddlers: Zuri (braids, yellow dress); Kofi; Nala."}

    r.review(shot, _video_path(tmp_path, "s08_a0"), out_dir=out_dir, song=song)

    req = _read_request(out_dir, "s08_a0")
    assert req.get("cast_text") == song["characters"]


# ------------------------------------------------------------- graceful failure --
def test_import_failure_does_not_crash_review_and_still_writes_request(tmp_path, no_cast_module):
    r = _reviewer()
    out_dir = str(tmp_path / "review")
    shot = _shot(["Zuri"], shot_id="s09")

    verdict = r.review(shot, _video_path(tmp_path, "s09_a0"), out_dir=out_dir)

    assert isinstance(verdict, dict)  # did not raise
    req = _read_request(out_dir, "s09_a0")
    assert req["shot_id"] == "s09"
    # No cast module and no song/override supplied -- cast_text is omitted
    # entirely rather than shipping a placeholder/None.
    assert "cast_text" not in req
    assert "cast_version" not in req
    assert req["expected_children"] == 1  # fallback heuristic still ran


def test_broken_cast_module_does_not_crash_review(tmp_path, monkeypatch):
    """A `cast.py` that imports fine but raises inside its functions must
    also degrade gracefully, not just an outright missing module."""
    import types

    mod = types.ModuleType(_CAST_MODULE_NAME)

    def _boom(*a, **k):
        raise RuntimeError("cast data file is corrupt")

    mod.cast_sentence = _boom
    mod.expected_child_count = _boom
    mod.version = _boom
    mod.describe = _boom
    mod.names = _boom

    import pipeline.kidsong as kidsong_pkg

    monkeypatch.setitem(sys.modules, _CAST_MODULE_NAME, mod)
    monkeypatch.setattr(kidsong_pkg, "cast", mod, raising=False)

    r = _reviewer()
    out_dir = str(tmp_path / "review")
    verdict = r.review(_shot(["Zuri"], shot_id="s10"), _video_path(tmp_path, "s10_a0"), out_dir=out_dir)

    assert isinstance(verdict, dict)
    req = _read_request(out_dir, "s10_a0")
    assert "cast_text" not in req
    assert "cast_version" not in req
    assert req["expected_children"] == 1  # _expected_face_count fallback


# --------------------------------------------------------------- back-compat --
def test_existing_positional_callers_still_work(tmp_path, no_cast_module):
    """generate.py calls `reviewer.review(render_shot, out_path, out_dir=review_dir)`
    -- no cast_text/song kwargs at all. That call shape must keep working
    unchanged."""
    r = _reviewer()
    out_dir = str(tmp_path / "review")
    shot = _shot(["all"], shot_id="s11")

    verdict = r.review(shot, _video_path(tmp_path, "s11_a0"), out_dir=out_dir)
    assert isinstance(verdict, dict)
    assert "accept" in verdict

    # Also confirm the fully-positional two-arg call shape (no out_dir either).
    verdict2 = r.review(_shot(["all"], shot_id="s12"), _video_path(tmp_path, "s12_a0"))
    assert isinstance(verdict2, dict)
    assert "accept" in verdict2


# ------------------------------------------------------------------ unit-level --
def test_resolve_cast_enrichment_directly_with_fake_module(fake_cast_module):
    enrichment = _resolve_cast_enrichment({"characters": ["Kofi"]})
    assert enrichment["expected_children"] == 1
    assert "Kofi" in enrichment["cast_text"]
    assert enrichment["cast_version"] == "cast-v1-test"


def test_resolve_cast_enrichment_omits_keys_when_nothing_resolves(no_cast_module):
    enrichment = _resolve_cast_enrichment({"characters": ["Kofi"]})
    assert "cast_text" not in enrichment
    assert "cast_version" not in enrichment
    assert enrichment["expected_children"] == 1
