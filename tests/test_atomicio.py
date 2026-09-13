"""A file is published only once it is complete.

Every writer in this repo that streamed straight into its final path had the
same failure: an interruption left a TRUNCATED file where a complete one
belongs, and no reader downstream can tell the difference. `assets.py` cached
the stump as a finished stock clip (QM-023); `kidsong/comfy.py` writes the
rendered TAKE, which then reaches the contact sheet and the editor;
`setup.py` rewrites the whole of `config.json` to change one string.

CPU-only, no network.
"""
import json
import os

import pytest

from pipeline.atomicio import atomic_path, atomic_write_bytes, atomic_write_json


# ------------------------------------------------------------- atomic_path --
def test_the_destination_appears_only_on_success(tmp_path):
    dest = tmp_path / "out.bin"
    seen_during = {}

    with atomic_path(str(dest)) as tmp:
        with open(tmp, "wb") as f:
            f.write(b"payload")
        seen_during["dest_exists"] = dest.exists()

    assert seen_during["dest_exists"] is False, "the destination existed mid-write"
    assert dest.read_bytes() == b"payload"
    assert not (tmp_path / "out.bin.part").exists()


def test_a_raise_inside_the_block_leaves_nothing_behind(tmp_path):
    dest = tmp_path / "out.bin"

    with pytest.raises(RuntimeError):
        with atomic_path(str(dest)) as tmp:
            with open(tmp, "wb") as f:
                f.write(b"half")
            raise RuntimeError("connection reset")

    assert not dest.exists()
    assert not (tmp_path / "out.bin.part").exists()


def test_an_interrupt_is_cleaned_up_too(tmp_path):
    """KeyboardInterrupt/SIGTERM is the interruption that produced the stump,
    and it is a BaseException — a plain `except Exception` would miss it."""
    dest = tmp_path / "out.bin"

    with pytest.raises(KeyboardInterrupt):
        with atomic_path(str(dest)) as tmp:
            with open(tmp, "wb") as f:
                f.write(b"half")
            raise KeyboardInterrupt

    assert not dest.exists()
    assert not (tmp_path / "out.bin.part").exists()


def test_an_existing_destination_survives_a_failed_rewrite(tmp_path):
    """The old file must still be there — replacing it with a truncated one is
    the whole failure mode."""
    dest = tmp_path / "config.json"
    dest.write_text("original", encoding="utf-8")

    with pytest.raises(RuntimeError):
        with atomic_path(str(dest)) as tmp:
            with open(tmp, "w", encoding="utf-8") as f:
                f.write("half-written")
            raise RuntimeError("disk full")

    assert dest.read_text(encoding="utf-8") == "original"


def test_a_stale_part_from_an_earlier_crash_is_not_reused(tmp_path):
    dest = tmp_path / "out.bin"
    (tmp_path / "out.bin.part").write_bytes(b"garbage from last time")

    with atomic_path(str(dest)) as tmp:
        with open(tmp, "wb") as f:
            f.write(b"fresh")

    assert dest.read_bytes() == b"fresh"


def test_writing_nothing_is_an_error_not_an_empty_file(tmp_path):
    dest = tmp_path / "out.bin"
    with pytest.raises(FileNotFoundError):
        with atomic_path(str(dest)):
            pass
    assert not dest.exists()


def test_missing_parent_directories_are_created(tmp_path):
    dest = tmp_path / "deep" / "nested" / "out.bin"
    atomic_write_bytes(str(dest), [b"x"])
    assert dest.read_bytes() == b"x"


# ------------------------------------------------------ atomic_write_bytes --
def test_chunks_are_concatenated(tmp_path):
    dest = tmp_path / "clip.mp4"
    atomic_write_bytes(str(dest), [b"aa", b"", b"bb", None, b"cc"])
    assert dest.read_bytes() == b"aabbcc"


def test_a_generator_that_raises_partway_publishes_nothing(tmp_path):
    dest = tmp_path / "clip.mp4"

    def chunks():
        yield b"aa"
        raise ConnectionError("reset")

    with pytest.raises(ConnectionError):
        atomic_write_bytes(str(dest), chunks())

    assert not dest.exists()
    assert not (tmp_path / "clip.mp4.part").exists()


# ------------------------------------------------------- atomic_write_json --
def test_json_round_trips(tmp_path):
    dest = tmp_path / "shots.json"
    atomic_write_json(str(dest), {"shots": [{"id": "s00"}]}, indent=2)
    assert json.loads(dest.read_text(encoding="utf-8")) == {"shots": [{"id": "s00"}]}


def test_an_unencodable_payload_leaves_the_previous_file_intact(tmp_path):
    """Serialising inside the temp file is deliberate: a payload that cannot
    encode raises before anything is renamed."""
    dest = tmp_path / "config.json"
    dest.write_text('{"llm": {"backend": "groq"}}', encoding="utf-8")

    with pytest.raises(TypeError):
        atomic_write_json(str(dest), {"bad": {1, 2, 3}})

    assert json.loads(dest.read_text(encoding="utf-8")) == {"llm": {"backend": "groq"}}
    assert not (tmp_path / "config.json.part").exists()


# ------------------------------------------------ the callers actually use it --
def test_every_media_download_publishes_atomically():
    """A regression guard on the pattern, not on one call site: these are the
    modules that write a downloaded artifact to its final path."""
    import inspect

    from pipeline.kidsong import comfy as kidsong_comfy
    from pipeline.kidsong import seedance as kidsong_seedance
    from studio import assets, higgsfield
    from studio import comfy as studio_comfy
    from studio import svm

    for module in (assets, studio_comfy, svm, higgsfield, kidsong_comfy, kidsong_seedance):
        src = inspect.getsource(module)
        assert "atomic_write_bytes" in src or "atomic_path" in src, module.__name__


def test_the_config_rewrite_publishes_atomically():
    import inspect

    from studio import setup

    assert "atomic_write_json" in inspect.getsource(setup.set_llm_backend)


def test_the_song_json_resume_needs_is_written_atomically():
    """`runstate.save_shotlist` is atomic and says why: "a crash during the
    write would otherwise leave truncated JSON and make the run unresumable —
    precisely the failure this module exists to prevent." The song JSON written
    on the very next line, which resume needs just as much (re-singing the audio
    reads it), was not."""
    import inspect
    import re

    from pipeline.kidsong import generate, runstate

    src = inspect.getsource(generate)
    # Every place the song JSON is WRITTEN: the fresh run, and the re-render
    # paths (--finalize / --restyle) that copy it under a new base. Matched on
    # the call, not on a variable name, so a rename cannot quietly drop one.
    writes = [m.start() for m in re.finditer(r"song_json_path\(out_dir, \w+\)", src)]
    assert len(writes) >= 2, "expected the fresh-run and re-render song writes"
    written_atomically = 0
    for idx in writes:
        if "atomic_write_json" in src[max(0, idx - 200):idx]:
            written_atomically += 1
    assert written_atomically >= 2, (
        f"only {written_atomically} of {len(writes)} song_json_path uses are preceded by "
        "atomic_write_json — a truncated song JSON makes the run unresumable"
    )

    # and the thing it is being made consistent with
    assert "os.replace" in inspect.getsource(runstate.save_shotlist)


def test_an_interrupted_frame_grab_leaves_no_stump(tmp_path, monkeypatch):
    """`transitions.extract_frame` writes GUIDE IMAGES: the FLF2V first/last
    frame, a chain guide, and (via refs.harvest_reference) a cast reference
    anchor. Every reader checks only `getsize(...) > 0`, so a truncated PNG
    from a killed ffmpeg would be staged into ComfyUI as a real frame."""
    from pipeline.kidsong import transitions

    out_png = tmp_path / "guide.png"
    monkeypatch.setattr(transitions, "_ffmpeg_available", lambda: True)

    def _dying_ffmpeg(cmd, **kwargs):
        # write a partial file at the temp path, then fail the way a killed
        # ffmpeg does
        with open(cmd[-1], "wb") as f:
            f.write(b"\x89PNG\r\n\x1a\n truncated")
        raise transitions.subprocess.CalledProcessError(137, cmd)

    monkeypatch.setattr(transitions.subprocess, "run", _dying_ffmpeg)

    class _DeadClip:
        fps, duration = 24.0, 1.0

        def save_frame(self, path, t=0.0):
            raise RuntimeError("moviepy fallback also unavailable")

        def close(self):
            pass

    import sys
    import types

    fake_moviepy = types.ModuleType("moviepy")
    fake_moviepy.VideoFileClip = lambda p: _DeadClip()
    monkeypatch.setitem(sys.modules, "moviepy", fake_moviepy)

    with pytest.raises(Exception):
        transitions.extract_frame("clip.mp4", "first", str(out_png))

    assert not out_png.exists(), "a truncated guide frame was left at the real path"
    assert not (tmp_path / "guide.png.part").exists()


def test_a_successful_frame_grab_still_lands(tmp_path, monkeypatch):
    from pipeline.kidsong import transitions

    out_png = tmp_path / "guide.png"
    monkeypatch.setattr(transitions, "_ffmpeg_available", lambda: True)

    def _good_ffmpeg(cmd, **kwargs):
        with open(cmd[-1], "wb") as f:
            f.write(b"\x89PNG\r\n\x1a\n real frame bytes")

        class _R:
            returncode = 0
        return _R()

    monkeypatch.setattr(transitions.subprocess, "run", _good_ffmpeg)

    assert transitions.extract_frame("clip.mp4", "first", str(out_png)) == str(out_png)
    assert out_png.read_bytes().startswith(b"\x89PNG")
    assert not (tmp_path / "guide.png.part").exists()


def test_the_cached_intro_and_its_manifest_are_published_atomically():
    """`intro.is_current` trusts the cached mp4 whenever a matching manifest
    sits beside it. A forced rebuild that died mid-encode left a truncated mp4
    that the manifest then vouched for — and it would be prepended to every
    episode from then on."""
    import inspect

    from pipeline.kidsong import intro

    assert "atomic_path" in inspect.getsource(intro._conform)
    assert "atomic_write_json" in inspect.getsource(intro.build_intro)
