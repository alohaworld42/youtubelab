"""Regression tests for the video-less-episode bug in pipeline/kidsong/edit.py.

WHAT HAPPENED (run 20260720-092939-kidsong-rainbow-friends-color-fun):

All 16 shots rendered fine, then `assemble` wrote a 1.0 MB staging.mp4 that
ffprobe reports as a single 60.000s AAC stream — ZERO video streams.

Root cause: `pipeline.assemble._caption_clips` computes the caption baseline
as `int(cfg["video"]["height"] * cfg["captions"]["vertical_position"])`.
cfg["video"] is the VERTICAL Shorts frame (1080x1920) but kidsong composites
WIDE (1920x1080, `edit._output_dims`). With the shipped vertical_position of
0.62 that put every caption at y=1190 in a 1080px-tall frame — entirely below
the canvas. MoviePy's `compose_mask` then blends the caption mask into a
zero-row slice of the background:

    ValueError: operands could not be broadcast together with
                shapes (48,272) (0,272)

...raised from inside `write_videofile`, AFTER the audio was muxed. ffmpeg
finalised a valid, playable, completely video-less MP4.

This was invisible to every existing check: the file exists, plays, has the
right duration, and `_post_grade`'s ffmpeg pass fails on it and is swallowed
by design as "grade skipped", which preserves the broken file.

The two guards below:
  1. `_caption_layer` positions captions against the frame actually being
     composited, and clamps them on-canvas.
  2. `verify_render` ffprobes the written file and raises loudly unless it has
     a video stream of the right size and the right duration.
"""
import json
import os
import subprocess

import pytest

from pipeline.kidsong import edit


# The real shipped values that produced the bug.
SHORTS_H = 1920
KIDSONG_W, KIDSONG_H = 1920, 1080
VERTICAL_POSITION = 0.62


def _cfg():
    """A config shaped like the real one: vertical video dims, wide kidsong out.

    `kidsong.captions.enabled` is TRUE here even though the shipped channel
    config now has it false (no on-screen lyrics — see
    tests/test_kidsong_captions_disabled.py). Everything in this module is
    about WHERE a caption lands once it is drawn, so these tests must keep
    exercising the drawing path; the switch itself is covered separately.
    """
    return {
        "video": {"width": 1080, "height": SHORTS_H, "fps": 24},
        "kidsong": {
            "output": {"width": KIDSONG_W, "height": KIDSONG_H},
            "captions": {"enabled": True},
        },
        "captions": {
            "vertical_position": VERTICAL_POSITION,
            "words_per_group": 3,
            "uppercase": True,
            "pop_in": False,
            "font_path": "",           # falls back to Pillow's default font
            "font_size": 64,
            "fill_color": [255, 255, 255],
            "stroke_color": [0, 0, 0],
            "stroke_width": 4,
        },
    }


WORDS = [
    {"word": "RED", "start": 0.0, "end": 0.4},
    {"word": "AND", "start": 0.4, "end": 0.8},
    {"word": "BLUE", "start": 0.8, "end": 1.2},
    {"word": "SING", "start": 1.2, "end": 1.6},
]


# --------------------------------------------------------- caption placement ---
def test_the_bug_the_shorts_baseline_falls_outside_the_kidsong_frame():
    """Pin the arithmetic that caused the crash, so the premise can't drift."""
    bad_y = int(SHORTS_H * VERTICAL_POSITION)
    assert bad_y == 1190
    assert bad_y > KIDSONG_H, (
        "premise of this regression test: the Shorts-derived caption baseline "
        "must land outside the wide kidsong frame"
    )


def test_captions_are_positioned_inside_the_kidsong_frame():
    cfg = _cfg()
    clips = edit._caption_layer(cfg, WORDS)
    assert clips, "expected caption clips for non-empty words"

    for clip in clips:
        y = clip.pos(0)[1]
        assert isinstance(y, (int, float)), f"unexpected caption position {clip.pos(0)!r}"
        assert 0 <= y, f"caption placed above the frame at y={y}"
        assert y + clip.h <= KIDSONG_H, (
            f"caption at y={y} (height {clip.h}) overflows the "
            f"{KIDSONG_W}x{KIDSONG_H} frame — this is the crash"
        )
        try:
            clip.close()
        except Exception:
            pass


def test_caption_layer_scales_the_baseline_to_the_output_frame():
    """The baseline tracks the WIDE frame height, never the vertical one.

    The bug this guards: the baseline used to be computed against
    cfg["video"]["height"] (1920), putting every caption below a 1080px
    canvas. Whatever fraction is in play, the result must be derived from
    the kidsong output height.
    """
    cfg = _cfg()
    clips = edit._caption_layer(cfg, WORDS[:1])
    y = clips[0].pos(0)[1]
    expected = int(KIDSONG_H * edit._KIDSONG_CAPTION_VPOS)
    assert abs(y - expected) <= clips[0].h, f"y={y}, expected about {expected}"
    # And explicitly NOT the vertical-frame computation that caused the bug.
    assert y != int(cfg["video"]["height"] * VERTICAL_POSITION)
    clips[0].close()


def test_caption_layer_keeps_lyrics_out_of_the_faces():
    """Sing-along lyrics belong in the lower third of the wide frame.

    The global captions.vertical_position (0.62) was tuned for 9:16; in 16:9
    it lands at y=669 of 1080 — dead centre, over the characters' mouths.
    """
    cfg = _cfg()
    clips = edit._caption_layer(cfg, WORDS[:1])
    y = clips[0].pos(0)[1]
    assert y > KIDSONG_H * 0.7, f"caption at y={y} sits too high in a {KIDSONG_H}px frame"
    assert y + clips[0].h <= KIDSONG_H
    clips[0].close()


def test_kidsong_caption_position_is_configurable():
    """kidsong.captions.vertical_position overrides the built-in default."""
    cfg = _cfg()
    cfg.setdefault("kidsong", {})["captions"] = {"enabled": True, "vertical_position": 0.5}
    clips = edit._caption_layer(cfg, WORDS[:1])
    y = clips[0].pos(0)[1]
    assert abs(y - int(KIDSONG_H * 0.5)) <= clips[0].h, f"override ignored, y={y}"
    clips[0].close()


def test_caption_layer_clamps_an_oversized_caption():
    """Even a font/config change that makes captions huge must stay on-canvas."""
    cfg = _cfg()
    cfg["captions"]["font_size"] = 300
    cfg["captions"]["vertical_position"] = 0.98
    clips = edit._caption_layer(cfg, WORDS[:1])
    clip = clips[0]
    y = clip.pos(0)[1]
    assert 0 <= y and y + clip.h <= KIDSONG_H, f"y={y} h={clip.h} frame_h={KIDSONG_H}"
    clip.close()


def test_assemble_positions_captions_against_the_output_frame(monkeypatch, tmp_path):
    """Guard the CALL SITE, not just the helper.

    `assemble` must route captions through `_caption_layer`. Calling
    `pipeline.assemble._caption_clips` directly still "works" (MoviePy quietly
    drops a fully off-canvas clip rather than drawing it), so the only visible
    symptom of a regression here is an episode with NO captions on screen —
    which no other test would notice. Spy on the composite instead.
    """
    seen = {}

    real_composite = edit.CompositeVideoClip

    def spy(clips, **kwargs):
        seen["clips"] = list(clips)
        return real_composite(clips, **kwargs)

    monkeypatch.setattr(edit, "CompositeVideoClip", spy)
    monkeypatch.setattr(edit, "_post_grade", lambda *a, **k: None)
    monkeypatch.setattr(edit, "verify_render", lambda *a, **k: True)

    class _FakeFinal:
        def __init__(self, inner):
            self._inner = inner

        def __getattr__(self, name):
            return getattr(self._inner, name)

        def write_videofile(self, *a, **k):
            return None

    W, H = 960, 540
    cfg = _cfg()
    cfg["kidsong"]["output"] = {"width": W, "height": H}
    cfg["video"]["fps"] = 12
    cfg["captions"]["font_size"] = 40

    import numpy as np
    import wave
    from moviepy import ImageSequenceClip

    audio_path = tmp_path / "a.wav"
    sr = 22050
    n = int(sr * 2.0)
    with wave.open(str(audio_path), "w") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(np.zeros(n, dtype=np.int16).tobytes())

    render_path = tmp_path / "s.mp4"
    seq = ImageSequenceClip(
        [np.full((H, 300, 3), 40, dtype=np.uint8) for _ in range(24)], fps=12
    )
    seq.write_videofile(str(render_path), fps=12, codec="libx264", audio=False, logger=None)
    seq.close()

    monkeypatch.setattr(
        edit, "AudioFileClip", lambda p: __import__("moviepy").AudioFileClip(p)
    )

    orig_write = None
    import moviepy.video.compositing.CompositeVideoClip as _cvc  # noqa: F401

    # Swap write_videofile out on the instance the composite returns. `assemble`
    # now writes to (and os.replace()s from) a `.part.mp4` sibling of the
    # requested path — see Fix 2's atomic-write contract — so the stub must
    # actually create that file, same as a real (if instant) encode would.
    def spy_composite(clips, **kwargs):
        seen["clips"] = list(clips)
        comp = real_composite(clips, **kwargs)

        def _fake_write(dest, *a, **k):
            open(dest, "wb").close()

        comp.write_videofile = _fake_write
        return comp

    monkeypatch.setattr(edit, "CompositeVideoClip", spy_composite)

    edit.assemble(
        cfg, [{"shot_id": "s0", "start": 0.0, "end": 2.0, "src": "s0"}],
        {"s0": str(render_path)}, str(audio_path), WORDS, 2.0,
        str(tmp_path / "out.mp4"), on_progress=lambda m: None,
    )

    caption_clips = [
        c for c in seen["clips"]
        if isinstance(c.pos(0), (tuple, list)) and c.pos(0)[0] == "center"
    ]
    assert caption_clips, "assemble composited no caption clips"
    for c in caption_clips:
        y = c.pos(0)[1]
        assert 0 <= y and y + c.h <= H, (
            f"assemble placed a caption at y={y} (h={c.h}) in a {W}x{H} frame — "
            "captions will be invisible; assemble must use _caption_layer"
        )


def test_caption_layer_is_empty_without_words():
    assert edit._caption_layer(_cfg(), []) == []


def test_caption_layer_does_not_mutate_the_callers_config():
    cfg = _cfg()
    before = json.dumps(cfg, sort_keys=True)
    for clip in edit._caption_layer(cfg, WORDS):
        clip.close()
    assert json.dumps(cfg, sort_keys=True) == before


# ------------------------------------------------------------- verify_render ---
def _has_ffmpeg():
    try:
        subprocess.run(["ffmpeg", "-version"], capture_output=True, check=True)
        return True
    except Exception:
        return False


needs_ffmpeg = pytest.mark.skipif(not _has_ffmpeg(), reason="ffmpeg/ffprobe not available")


@pytest.fixture()
def audio_only_mp4(tmp_path):
    """Exactly the artefact the bug produced: AAC, right duration, no video."""
    path = tmp_path / "audio_only.mp4"
    subprocess.run(
        ["ffmpeg", "-y", "-v", "error", "-f", "lavfi",
         "-i", "sine=frequency=440:duration=3", "-c:a", "aac", str(path)],
        check=True,
    )
    return str(path)


@pytest.fixture()
def good_mp4(tmp_path):
    path = tmp_path / "good.mp4"
    subprocess.run(
        ["ffmpeg", "-y", "-v", "error",
         "-f", "lavfi", "-i", f"color=c=blue:s={KIDSONG_W}x{KIDSONG_H}:d=3:r=24",
         "-f", "lavfi", "-i", "sine=frequency=440:duration=3",
         "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac",
         "-shortest", str(path)],
        check=True,
    )
    return str(path)


@needs_ffmpeg
def test_verify_render_rejects_an_audio_only_file(audio_only_mp4):
    """THE regression: the exact shape of the shipped broken episode."""
    with pytest.raises(edit.OutputVerificationError) as exc:
        edit.verify_render(audio_only_mp4, 3.0, (KIDSONG_W, KIDSONG_H))
    assert "NO VIDEO STREAM" in str(exc.value)


@needs_ffmpeg
def test_verify_render_accepts_a_good_file(good_mp4):
    assert edit.verify_render(good_mp4, 3.0, (KIDSONG_W, KIDSONG_H)) is True


@needs_ffmpeg
def test_verify_render_rejects_the_wrong_resolution(good_mp4):
    with pytest.raises(edit.OutputVerificationError) as exc:
        edit.verify_render(good_mp4, 3.0, (1080, 1920))
    assert "resolution" in str(exc.value)


@needs_ffmpeg
def test_verify_render_rejects_the_wrong_duration(good_mp4):
    with pytest.raises(edit.OutputVerificationError) as exc:
        edit.verify_render(good_mp4, 30.0, (KIDSONG_W, KIDSONG_H))
    assert "duration" in str(exc.value)


@needs_ffmpeg
def test_verify_render_reports_every_problem_at_once(audio_only_mp4):
    with pytest.raises(edit.OutputVerificationError) as exc:
        edit.verify_render(audio_only_mp4, 30.0, (KIDSONG_W, KIDSONG_H))
    msg = str(exc.value)
    assert "NO VIDEO STREAM" in msg and "duration" in msg


def test_verify_render_rejects_a_missing_file(tmp_path):
    with pytest.raises(edit.OutputVerificationError) as exc:
        edit.verify_render(str(tmp_path / "nope.mp4"), 3.0, (KIDSONG_W, KIDSONG_H))
    assert "not written" in str(exc.value)


# ----------------------------------- truncated-video-track incident (Fix 1) ---
# Real incident: output/20260720-085055-kidsong-splish-splash-fun-with-friends
# .staging.mp4 — a 0.2s/6-frame video track muxed (without -shortest) with a
# complete 60s audio track. Container-level `format.duration` reports the
# STREAM MAX, i.e. the audio length, so the file PASSED every check that only
# looked at nb_frames==0 and format.duration. verify_render must also check
# the video stream's OWN duration.
@pytest.fixture()
def short_video_long_audio_mp4(tmp_path):
    """Reproduces the incident shape: ~0.2s video, ~10s audio, no -shortest."""
    path = tmp_path / "truncated_video.mp4"
    subprocess.run(
        ["ffmpeg", "-y", "-v", "error",
         "-f", "lavfi", "-i", f"color=c=blue:s={KIDSONG_W}x{KIDSONG_H}:d=0.2:r=24",
         "-f", "lavfi", "-i", "sine=frequency=440:duration=10",
         "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac",
         str(path)],
        check=True,
    )
    return str(path)


@needs_ffmpeg
def test_verify_render_accepts_a_full_length_video_and_audio_pair(tmp_path):
    """(a) 2s video + 2s audio must pass verification for duration=2."""
    path = tmp_path / "full_length.mp4"
    subprocess.run(
        ["ffmpeg", "-y", "-v", "error",
         "-f", "lavfi", "-i", f"color=c=green:s={KIDSONG_W}x{KIDSONG_H}:d=2:r=24",
         "-f", "lavfi", "-i", "sine=frequency=440:duration=2",
         "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac",
         "-shortest", str(path)],
        check=True,
    )
    assert edit.verify_render(str(path), 2.0, (KIDSONG_W, KIDSONG_H)) is True


@needs_ffmpeg
def test_verify_render_catches_a_truncated_video_track_behind_full_length_audio(
    short_video_long_audio_mp4,
):
    """(b) THE incident: format.duration (audio-length) alone passed this file.

    nb_frames==0 doesn't catch it either (the video track has ~6 real frames).
    Only checking the video stream's own duration against the expected
    duration catches it.
    """
    # Confirm the container-level duration is indeed the (long) audio length,
    # i.e. the container-duration check alone would NOT catch this file.
    info = edit._probe_json(short_video_long_audio_mp4)
    container_duration = float(info["format"]["duration"])
    assert container_duration > 5.0, (
        "premise: container duration reflects the long audio track, not the "
        "short video track"
    )

    with pytest.raises(edit.OutputVerificationError) as exc:
        edit.verify_render(short_video_long_audio_mp4, 10.0, (KIDSONG_W, KIDSONG_H))
    msg = str(exc.value)
    assert "video stream duration" in msg, (
        f"expected the new video-stream-duration problem, got: {msg}"
    )


@needs_ffmpeg
def test_verify_render_can_ignore_audio(tmp_path):
    path = tmp_path / "silent.mp4"
    subprocess.run(
        ["ffmpeg", "-y", "-v", "error", "-f", "lavfi",
         "-i", f"color=c=red:s={KIDSONG_W}x{KIDSONG_H}:d=2:r=24",
         "-c:v", "libx264", "-pix_fmt", "yuv420p", str(path)],
        check=True,
    )
    with pytest.raises(edit.OutputVerificationError):
        edit.verify_render(str(path), 2.0, (KIDSONG_W, KIDSONG_H))
    assert edit.verify_render(
        str(path), 2.0, (KIDSONG_W, KIDSONG_H), require_audio=False
    ) is True


# ------------------------------------------------------- end-to-end assemble ---
@needs_ffmpeg
def test_assemble_with_captions_produces_a_video_stream(tmp_path):
    """The full path that crashed: captions + a wide kidsong frame.

    Small (2s, 640x360) so it stays a unit test, but it exercises the exact
    composite that raised `operands could not be broadcast together`.
    """
    import wave

    import numpy as np
    from moviepy import ImageSequenceClip

    W, H = 640, 360
    cfg = _cfg()
    cfg["kidsong"]["output"] = {"width": W, "height": H}
    cfg["kidsong"]["grade"] = None            # skip the ffmpeg grade pass
    cfg["video"]["fps"] = 12
    cfg["captions"]["font_size"] = 40

    audio_path = tmp_path / "song.wav"
    sr = 22050
    n = int(sr * 2.0)
    tone = (np.sin(2 * np.pi * 440 * np.arange(n) / sr) * 16000).astype(np.int16)
    with wave.open(str(audio_path), "w") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(tone.tobytes())

    render_path = tmp_path / "shot.mp4"
    frames = [
        np.full((360, 200, 3), (30, 30 + k * 4, 90), dtype=np.uint8) for k in range(24)
    ]
    seq = ImageSequenceClip(frames, fps=12)
    seq.write_videofile(str(render_path), fps=12, codec="libx264", audio=False, logger=None)
    seq.close()

    cut_list = [{"shot_id": "s0", "start": 0.0, "end": 2.0, "src": "s0"}]
    out_path = tmp_path / "out.mp4"

    edit.assemble(
        cfg, cut_list, {"s0": str(render_path)}, str(audio_path),
        WORDS, 2.0, str(out_path), on_progress=lambda m: None,
    )

    # verify_render already ran inside assemble; assert independently here so a
    # regression in the guard itself cannot hide a regression in the output.
    info = edit._probe_json(str(out_path))
    video = [s for s in info["streams"] if s["codec_type"] == "video"]
    audio = [s for s in info["streams"] if s["codec_type"] == "audio"]
    assert video, "assemble wrote no video stream — the original bug is back"
    assert audio, "assemble wrote no audio stream"
    assert (video[0]["width"], video[0]["height"]) == (W, H)
    assert abs(float(info["format"]["duration"]) - 2.0) <= 1.0
