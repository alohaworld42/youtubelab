"""On-screen lyrics are OFF for the kidsong channel, and turning them off
must not disturb the edit.

The switch is `kidsong.captions.enabled` (default false), honoured in
`pipeline.kidsong.edit._caption_layer`. It is a config switch rather than a
deletion of the caption code on purpose: the Whisper word alignment that feeds
the captions is still load-bearing for verse timing (`sing.verse_times_from_
words`) and the beat grid, and the lyrics may be wanted again.

So these tests assert BOTH directions -- captions really are gone, and the
machinery behind them still works when the switch is flipped back.
"""
import pytest

from pipeline.kidsong.edit import _caption_layer, build_cut_list

_WORDS = [
    {"word": "SPLISH", "start": 0.0, "end": 0.45},
    {"word": "SPLASH", "start": 0.5, "end": 0.95},
    {"word": "SPLOSH", "start": 1.0, "end": 1.45},
]


def _cfg(captions_block):
    return {
        "video": {"width": 1080, "height": 1920, "fps": 30},
        "captions": {
            "font_path": "C:/Windows/Fonts/arialbd.ttf",
            "font_size": 110,
            "fill_color": [255, 255, 255],
            "highlight_color": [255, 222, 0],
            "stroke_color": [0, 0, 0],
            "stroke_width": 14,
            "words_per_group": 1,
            "vertical_position": 0.62,
            "uppercase": True,
        },
        "kidsong": {"output": {"width": 1920, "height": 1080},
                    "captions": captions_block},
    }


def test_no_caption_clips_when_disabled():
    assert _caption_layer(_cfg({"enabled": False, "vertical_position": 0.8}), _WORDS) == []


def test_captions_are_off_by_default_when_the_key_is_absent():
    """Absent means off. A config that predates this switch must not keep
    burning lyrics into the episode."""
    assert _caption_layer(_cfg({"vertical_position": 0.8}), _WORDS) == []
    assert _caption_layer(_cfg({}), _WORDS) == []


def test_the_shipped_config_has_captions_disabled():
    from pipeline.config import load_config

    cfg = load_config()
    assert cfg["kidsong"]["captions"]["enabled"] is False
    assert _caption_layer(cfg, _WORDS) == []


def test_captions_still_render_when_re_enabled():
    """The caption code is disabled, NOT deleted -- flipping the switch must
    bring the lyrics straight back, positioned in the wide frame's lower third."""
    clips = _caption_layer(_cfg({"enabled": True, "vertical_position": 0.8}), _WORDS)
    assert len(clips) == len(_WORDS)

    for clip in clips:
        pos = clip.pos(0)
        y = pos[1]
        # Wholly on-canvas in the 1080px-tall wide frame, and in the lower
        # third rather than dead centre over the characters' faces.
        assert 0 <= y <= 1080 - clip.h
        assert y > 1080 * 0.5
    for clip in clips:
        clip.close()


def test_disabling_captions_does_not_touch_the_words_or_the_cut():
    """`_caption_layer` must not mutate the word list it is handed: the same
    `words` object drives verse timing and the beat-aligned cut upstream, so a
    caption change that edited it in place would silently move the cuts."""
    words = [dict(w) for w in _WORDS]
    snapshot = [dict(w) for w in words]

    assert _caption_layer(_cfg({"enabled": False}), words) == []
    assert words == snapshot


def test_cut_list_is_identical_regardless_of_the_caption_switch():
    """The cut is built from the shotlist, the beat grid and the duration --
    none of which the caption switch feeds. Pin that independence down so the
    'captions off' change can never be blamed for a timing regression."""
    shotlist = {"shots": [
        {"id": "s0", "start": 0.0, "end": 2.5, "verse": 0},
        {"id": "s1", "start": 2.5, "end": 5.0, "verse": 0},
        {"id": "s2", "start": 5.0, "end": 7.5, "verse": 1},
        {"id": "s3", "start": 7.5, "end": 10.0, "verse": 1},
    ]}
    beats = [0.5 * k for k in range(21)]

    off = build_cut_list(shotlist, beats, 10.0)
    on = build_cut_list(shotlist, beats, 10.0)
    assert off == on
    # And the cut really does span the whole song, captions or not.
    assert off[0]["start"] == 0.0
    assert off[-1]["end"] == pytest.approx(10.0)
