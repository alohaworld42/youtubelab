"""A published video must say a machine made it — and must not lie about how.

Two failure directions, and both are real:

  * **Silence.** Before `pipeline/ai_disclosure.py` existed, nothing in this repo
    disclosed anything. The `videos.insert` body carried `privacyStatus` and
    `selfDeclaredMadeForKids` and nothing else, the description was passed
    through untouched, the rendered MP4 had no tags.
  * **A wrong label.** YouTube's altered-or-synthetic disclosure is for
    *realistic* content and explicitly not for wholly animated work, so
    declaring a stylised toon as synthetic media is its own defect. `"auto"`
    therefore resolves to False for the animated types and True for everything
    else, and these tests pin both halves — a version that just answered True
    everywhere would pass a naive "is it disclosed?" test.

CPU-only. `tag_file` needs ffmpeg and skips without it.
"""
import os
import shutil
import subprocess

import pytest

from pipeline import ai_disclosure

HAS_FFMPEG = shutil.which("ffmpeg") is not None and shutil.which("ffprobe") is not None
needs_ffmpeg = pytest.mark.skipif(not HAS_FFMPEG, reason="ffmpeg/ffprobe not on PATH")


def _cfg(**over):
    block = dict(over) if over else {}
    return {"ai_disclosure": block} if block else {}


# ------------------------------------------------------------- the defaults --
def test_a_config_with_no_disclosure_block_still_discloses():
    """An operator upgrading from an older config.json must not silently end up
    publishing undisclosed. Absent means defaults, not off."""
    assert ai_disclosure.is_enabled({}) is True
    assert ai_disclosure.note({}) == ai_disclosure.DEFAULT_NOTE


def test_the_shipped_config_discloses_by_default():
    from pipeline.config import load_config

    cfg = load_config()
    assert ai_disclosure.is_enabled(cfg) is True
    assert ai_disclosure.note(cfg)


def test_a_partial_block_keeps_the_other_defaults():
    cfg = _cfg(description_note="Made with AI.")
    assert ai_disclosure.note(cfg) == "Made with AI."
    assert ai_disclosure.metadata_tags(cfg)          # embed_metadata still on


def test_disabling_it_is_possible_and_total():
    cfg = _cfg(enabled=False)
    assert ai_disclosure.note(cfg) == ""
    assert ai_disclosure.append_note("D", cfg) == "D"
    assert ai_disclosure.metadata_tags(cfg) == {}
    assert ai_disclosure.declare_synthetic_media(cfg) is False


@pytest.mark.parametrize("bad", [None, 12, [], {}, object()])
def test_a_malformed_note_falls_back_instead_of_publishing_nothing(bad):
    """config is hand-edited. A wrong type here must not mean an undisclosed
    upload, and must not crash an upload that already has the video encoded."""
    assert ai_disclosure.note(_cfg(description_note=bad)) == ai_disclosure.DEFAULT_NOTE


def test_a_note_wrapped_across_lines_is_collapsed():
    cfg = _cfg(description_note="Made\n  with\tAI.")
    assert ai_disclosure.note(cfg) == "Made with AI."


# ------------------------------------------------------ appending to a body --
def test_the_note_is_appended_to_a_normal_description():
    out = ai_disclosure.append_note("A song about brushing teeth.", {})
    assert out.startswith("A song about brushing teeth.")
    assert out.endswith(ai_disclosure.DEFAULT_NOTE)


def test_an_empty_description_becomes_just_the_note():
    assert ai_disclosure.append_note("", {}) == ai_disclosure.DEFAULT_NOTE
    assert ai_disclosure.append_note(None, {}) == ai_disclosure.DEFAULT_NOTE


def test_appending_twice_does_not_duplicate_it():
    """A re-upload, or a description already carrying the note, must not end up
    with it twice."""
    once = ai_disclosure.append_note("D", {})
    assert ai_disclosure.append_note(once, {}) == once


def test_a_note_already_present_but_re_wrapped_still_counts_as_present():
    wrapped = "D\n\n" + ai_disclosure.DEFAULT_NOTE.replace(" ", "\n", 1)
    assert ai_disclosure.has_note(wrapped, {})
    assert ai_disclosure.append_note(wrapped, {}) == wrapped


def test_a_description_at_the_cap_still_ends_with_the_disclosure():
    """THE defect this ordering exists to prevent: appending first and slicing
    afterwards lets a long description push the disclosure off the end, and
    publishes an undisclosed video with no error anywhere."""
    limit = 4900
    out = ai_disclosure.append_note("x" * 6000, {}, limit=limit)

    assert len(out) <= limit
    assert out.endswith(ai_disclosure.DEFAULT_NOTE), "the disclosure was truncated away"


def test_a_cap_smaller_than_the_note_keeps_the_note_not_the_prose():
    out = ai_disclosure.append_note("x" * 500, {}, limit=20)
    assert len(out) == 20
    assert out == ai_disclosure.DEFAULT_NOTE[:20]


def test_a_short_description_under_the_cap_is_untouched_except_for_the_note():
    out = ai_disclosure.append_note("hello", {}, limit=4900)
    assert out == "hello\n\n" + ai_disclosure.DEFAULT_NOTE


# ------------------------------------------- the YouTube altered-media flag --
def test_an_animated_kidsong_is_not_declared_altered_or_synthetic():
    """YouTube's disclosure is for realistic content and exempts animation.
    Declaring a toon would be a wrong label, not a safe one."""
    assert ai_disclosure.declare_synthetic_media({}, video_type="kidsong") is False


@pytest.mark.parametrize("video_type", ["facts", "brainrot", "", None])
def test_everything_that_is_not_animated_is_declared(video_type):
    assert ai_disclosure.declare_synthetic_media({}, video_type=video_type) is True


def test_the_animated_list_is_configurable():
    cfg = _cfg(animated_video_types=["facts"])
    assert ai_disclosure.declare_synthetic_media(cfg, video_type="facts") is False
    assert ai_disclosure.declare_synthetic_media(cfg, video_type="kidsong") is True


@pytest.mark.parametrize("value,expected", [
    (True, True), (False, False),
    ("true", True), ("false", False), ("on", True), ("off", False),
])
def test_an_explicit_setting_overrides_the_inference(value, expected):
    cfg = _cfg(youtube_synthetic_media=value)
    assert ai_disclosure.declare_synthetic_media(cfg, video_type="kidsong") is expected


def test_the_caller_override_can_only_add_a_declaration():
    """Same asymmetry as `scheduler._made_for_kids`: a caller may add a
    disclosure, never remove one. The failure that matters is the missing one."""
    cfg = _cfg(youtube_synthetic_media=True)

    assert ai_disclosure.declare_synthetic_media(cfg, "kidsong", override=True) is True
    assert ai_disclosure.declare_synthetic_media({}, "kidsong", override=True) is True
    # False does NOT switch a configured declaration back off
    assert ai_disclosure.declare_synthetic_media(cfg, "kidsong", override=False) is True


def test_an_unrecognised_mode_falls_back_to_auto_rather_than_off():
    cfg = _cfg(youtube_synthetic_media="maybe")
    assert ai_disclosure.declare_synthetic_media(cfg, video_type="facts") is True
    assert ai_disclosure.declare_synthetic_media(cfg, video_type="kidsong") is False


# --------------------------------------------------------- container tagging --
def test_metadata_can_be_switched_off_without_losing_the_description_note():
    cfg = _cfg(embed_metadata=False)
    assert ai_disclosure.metadata_tags(cfg) == {}
    assert ai_disclosure.note(cfg)                   # still disclosed on YouTube


def test_tagging_a_file_that_is_not_there_is_a_no_op(tmp_path):
    assert ai_disclosure.tag_file(str(tmp_path / "nope.mp4"), {}) is False
    assert ai_disclosure.tag_file(None, {}) is False


def test_tagging_leaves_no_temp_file_when_ffmpeg_is_missing(tmp_path, monkeypatch):
    """An hour of GPU must not be lost because ffmpeg is not installed."""
    src = tmp_path / "v.mp4"
    src.write_bytes(b"not really an mp4")

    def boom(*a, **k):
        raise FileNotFoundError("ffmpeg")

    monkeypatch.setattr(subprocess, "run", boom)

    assert ai_disclosure.tag_file(str(src), {}) is False
    assert src.read_bytes() == b"not really an mp4", "the episode was damaged"
    assert not os.path.exists(str(src) + ".disclosure.mp4")


@needs_ffmpeg
def test_a_tagged_mp4_really_carries_the_disclosure(tmp_path):
    """Through real ffmpeg, read back with real ffprobe."""
    src = tmp_path / "v.mp4"
    subprocess.run(
        ["ffmpeg", "-y", "-v", "error", "-f", "lavfi", "-i", "color=c=red:s=64x64:d=1",
         "-c:v", "libx264", "-pix_fmt", "yuv420p", str(src)],
        check=True,
    )

    assert ai_disclosure.tag_file(str(src), {}) is True

    tags = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format_tags",
         "-of", "default=nw=1", str(src)],
        capture_output=True, text=True, check=True,
    ).stdout
    assert "künstlicher Intelligenz" in tags or "artificial intelligence" in tags
    assert not os.path.exists(str(src) + ".disclosure.mp4")


@needs_ffmpeg
def test_tagging_preserves_the_video(tmp_path):
    """A stream copy must not change duration or dimensions — `edit.py` runs
    `verify_render` after this call and would fail the whole episode."""
    src = tmp_path / "v.mp4"
    subprocess.run(
        ["ffmpeg", "-y", "-v", "error", "-f", "lavfi", "-i", "color=c=red:s=64x64:d=2",
         "-c:v", "libx264", "-pix_fmt", "yuv420p", str(src)],
        check=True,
    )

    def probe():
        out = subprocess.run(
            ["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
             "stream=width,height", "-show_entries", "format=duration",
             "-of", "default=nw=1:nk=1", str(src)],
            capture_output=True, text=True, check=True,
        ).stdout.split()
        return out[0], out[1], round(float(out[2]))

    before = probe()
    ai_disclosure.tag_file(str(src), {})
    assert probe() == before
