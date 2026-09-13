"""pipeline.tts scratch-dir hygiene."""
import glob
import os
import tempfile

import pytest

from pipeline import tts


def _cfg():
    return {"voices": {"narrator": "en-US-AriaNeural", "rate": "+0%"}}


def _tmpdirs():
    return set(glob.glob(os.path.join(tempfile.gettempdir(), "brainrot_tts_*")))


def test_a_failed_synthesis_does_not_leak_its_scratch_dir(tmp_path, monkeypatch):
    """The cleanup used to sit after the loop, so any failure in it — edge-tts
    throttling past its retries is the common one — left the scratch dir and its
    mp3s behind. A retried job leaks one per attempt."""
    before = _tmpdirs()

    def _boom(*a, **k):
        raise RuntimeError("edge-tts 403 (throttled)")

    monkeypatch.setattr(tts, "_synth_line_with_retry", _boom)

    with pytest.raises(RuntimeError):
        tts.synthesize([{"speaker": "narrator", "text": "hi"}], _cfg(),
                       str(tmp_path / "voice.wav"))

    assert _tmpdirs() == before, "a scratch dir was left behind by a failed synthesis"


def test_a_successful_synthesis_also_cleans_up(tmp_path, monkeypatch):
    before = _tmpdirs()

    def _fake_synth(text, voice, rate, out_path, attempts=3):
        with open(out_path, "wb") as f:
            f.write(b"\x00")

    class _Seg:
        def __init__(self, ms=100):
            self.ms = ms

        def __len__(self):
            return self.ms

        def __iadd__(self, other):
            self.ms += len(other)
            return self

        def __add__(self, other):
            return _Seg(self.ms + len(other))

        def export(self, path, format=None):
            open(path, "wb").close()

    monkeypatch.setattr(tts, "_synth_line_with_retry", _fake_synth)
    monkeypatch.setattr(tts.AudioSegment, "silent", staticmethod(lambda duration=0: _Seg(duration)))
    monkeypatch.setattr(tts.AudioSegment, "from_file", staticmethod(lambda p: _Seg(500)))

    out = tts.synthesize([{"speaker": "narrator", "text": "hi"}], _cfg(),
                         str(tmp_path / "voice.wav"))

    assert out["segments"][0]["speaker"] == "narrator"
    assert _tmpdirs() == before
