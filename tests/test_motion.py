from pipeline.motion import (
    build_motion_clips,
    _make_component,
    _split_number,
    _fmt_count,
    _timeline,
)
from pipeline.script_gen import _normalize_graphic
from pipeline.highlights import to_stat_specs
from pipeline.generate import _build_graphics


MOTION_CFG = {
    "motion": {
        "enabled": True,
        "auto_stats": True,
        "max_count": 5,
        "min_gap_seconds": 0.8,
        "hold_seconds": 0.5,
        "font_path": "C:/Windows/Fonts/arialbd.ttf",
        "accent_color": [255, 214, 40],
        "accent_b_color": [83, 196, 255],
        "card_color": [22, 20, 30],
    }
}


# --------------------------------------------------------------- primitives ---
def test_split_number_variants():
    assert _split_number("3") == (3, "3")
    assert _split_number("90%") == (90, "90%")
    assert _split_number("1,000") == (1000, "1,000")
    assert _split_number("none")[0] is None


def test_fmt_count_rolls_up():
    assert _fmt_count(100, "100", 0.0) == "0"
    assert _fmt_count(100, "100", 0.5) == "50"
    assert _fmt_count(100, "100%", 1.0) == "100%"


def test_timeline_bounds():
    # alpha starts near 0, peaks mid, returns near 0 at the very end
    _, a0, _ = _timeline(0.0, 3.0)
    _, amid, _ = _timeline(1.5, 3.0)
    _, aend, _ = _timeline(3.0, 3.0)
    assert a0 < 0.2
    assert amid > 0.9
    assert aend < 0.2


# --------------------------------------------------------------- components ---
def test_each_component_builds_and_renders():
    specs = [
        {"type": "stat", "value": "3", "label": "hearts"},
        {"type": "card", "text": "it never forgets"},
        {"type": "compare", "a_label": "cats", "a_value": 90, "b_label": "dogs", "b_value": 40},
    ]
    for s in specs:
        comp = _make_component(s, MOTION_CFG, 1080)
        assert comp is not None
        img = comp.render(0.5, 2.0)
        assert img.mode == "RGBA"
        assert img.size == comp.canvas


def test_unknown_component_type_is_none():
    assert _make_component({"type": "bogus"}, MOTION_CFG, 1080) is None


def test_build_motion_clips_disabled():
    cfg = {"motion": {"enabled": False}}
    g = [{"type": "stat", "value": "3", "start": 0.0, "end": 2.0}]
    assert build_motion_clips(cfg, g, (1080, 1920), 30) == []


def test_build_motion_clips_positions_and_times():
    g = [{"type": "stat", "value": "3", "label": "x", "start": 1.0, "end": 3.0}]
    clips = build_motion_clips(MOTION_CFG, g, (1080, 1920), 30)
    assert len(clips) == 1
    assert clips[0].start == 1.0
    assert clips[0].mask is not None


# ---------------------------------------------------------- spec normalizing ---
def test_normalize_graphic_valid_and_invalid():
    assert _normalize_graphic({"type": "stat", "value": 3, "label": "hearts"})["value"] == "3"
    assert _normalize_graphic({"type": "card", "text": "hi"})["type"] == "card"
    assert _normalize_graphic({"type": "card", "text": ""}) is None
    assert _normalize_graphic({"type": "compare", "a_label": "a", "a_value": 1, "b_label": "b", "b_value": 2})["type"] == "compare"
    assert _normalize_graphic({"type": "compare", "a_label": "a"}) is None
    assert _normalize_graphic({"type": "stat"}) is None
    assert _normalize_graphic("nope") is None


def test_to_stat_specs_splits_number_and_label():
    specs = to_stat_specs([{"text": "3 HEARTS", "start": 0.9, "end": 2.0}])
    assert specs[0] == {"type": "stat", "value": "3", "label": "HEARTS", "start": 0.9, "end": 2.0}


# ------------------------------------------------------------ merge in build ---
def _cfg():
    return dict(MOTION_CFG, highlights={"enabled": True, "max_count": 4, "min_gap_seconds": 3.0, "hold_seconds": 0.6, "max_extra_words": 1})


def test_build_graphics_prefers_llm_and_caps():
    script = {"lines": [
        {"speaker": "narrator", "text": "big claim", "graphic": {"type": "card", "text": "wow"}},
        {"speaker": "narrator", "text": "second"},
    ]}
    segments = [
        {"speaker": "narrator", "text": "big claim", "start": 0.0, "end": 2.0},
        {"speaker": "narrator", "text": "second", "start": 2.0, "end": 4.0},
    ]
    words = []  # no numbers -> no auto stats
    out = _build_graphics(_cfg(), script, segments, words, duration=10.0)
    assert len(out) == 1
    assert out[0]["type"] == "card"
    assert out[0]["start"] == 0.0
    assert out[0]["end"] == 2.5  # 2.0 + hold_seconds(0.5)


def test_build_graphics_auto_stat_fills_when_no_clash():
    script = {"lines": [{"speaker": "narrator", "text": "octopuses have three hearts"}]}
    segments = [{"speaker": "narrator", "text": "octopuses have three hearts", "start": 0.0, "end": 3.0}]
    words = [
        {"word": "OCTOPUSES", "start": 0.0, "end": 0.6},
        {"word": "HAVE", "start": 0.6, "end": 0.9},
        {"word": "THREE", "start": 0.9, "end": 1.3},
        {"word": "HEARTS", "start": 1.3, "end": 1.7},
    ]
    out = _build_graphics(_cfg(), script, segments, words, duration=10.0)
    assert len(out) == 1
    assert out[0]["type"] == "stat"
    assert out[0]["value"] == "3"


def test_build_graphics_disabled():
    cfg = {"motion": {"enabled": False}}
    script = {"lines": [{"speaker": "narrator", "text": "x", "graphic": {"type": "card", "text": "y"}}]}
    segments = [{"speaker": "narrator", "text": "x", "start": 0.0, "end": 2.0}]
    assert _build_graphics(cfg, script, segments, [], 10.0) == []
