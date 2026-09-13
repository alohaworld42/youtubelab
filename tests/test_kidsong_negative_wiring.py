"""Integration test for the render-time NEGATIVE prompt wiring in
`pipeline.kidsong.generate._generate_director`.

Mutation-proven test gap: an adversarial verifier deleted the
`"NEGATIVE": {"text": shot_negative},` line from the ComfyUI render patch dict
(generate.py, the `client.render(...)` call inside the per-shot retry loop)
and the ENTIRE suite still passed -- `_shot_negative` itself is unit-tested
(tests/test_kidsong_shot_prompt.py) but nothing asserted the computed value
actually reaches the render call. This file closes that gap: it runs
`generate_kidsong` through a resumed run with ONE PENDING shot (so the render
loop actually executes, unlike test_kidsong_intro_promotion.py's fixture
where every shot is pre-rendered specifically to SKIP the render loop)
against a fake ComfyClient that captures the patch dict passed to `render`.

CPU-only, no GPU, no network, no ffmpeg, no real ComfyUI. `load_workflow`
reads the real repo workflow JSON off local disk (no network) -- same trick
tests/test_kidsong_intro_promotion.py uses -- so the baseline NEGATIVE text
asserted against here is the actual authored baseline, not a fake one.
"""
import json
import os
import struct
import wave

import pytest

from pipeline.kidsong import runstate
from pipeline.kidsong.comfy import ComfyClient as _RealComfyClient

_REAL_LOAD_WORKFLOW = _RealComfyClient.load_workflow
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _write_wav(path, seconds=2.0, rate=8000):
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(rate)
        wf.writeframes(struct.pack("<h", 0) * int(rate * seconds))


@pytest.fixture
def pending_shot_run(tmp_path, monkeypatch):
    """A director run with exactly ONE PENDING shot (a Kofi closeup) so the
    render loop actually runs, plus every other GPU/media stage faked --
    the same fixture shape as test_kidsong_intro_promotion.py's
    `promotable_run`, except that fixture pre-renders its shots specifically
    so the render loop is SKIPPED; this one needs it to run.
    """
    out_dir = tmp_path / "output"
    out_dir.mkdir()
    base = "20260720-093000-kidsong-negative-wiring-test"
    shots_dir = runstate.shots_dir_path(str(out_dir), base)
    os.makedirs(shots_dir, exist_ok=True)

    _write_wav(out_dir / f"{base}.wav")
    song = {
        "title": "Negative Wiring Test Song",
        "description": "d",
        "tags": ["t"],
        "characters": "Three toddlers: Zuri; Kofi; Nala",
        "verses": [{"scene": "bathroom", "lines": ["a", "b"]}],
    }
    (out_dir / f"{base}-song.json").write_text(json.dumps(song), encoding="utf-8")

    shot = {
        "id": "s00", "verse": 0, "start": 0.0, "end": 2.0,
        "shot_type": "closeup", "characters": ["Kofi"],
        "action": "Kofi brushes his teeth with a big smile",
        "setting": "a bright cheerful bathroom", "camera": "static",
        "reuse_of": None, "seed": 1000, "status": "planned",
    }
    runstate.save_shotlist(str(out_dir), base, {"shots": [shot]})

    render_calls = []  # (workflow_name, patches_dict) captured per client.render call

    import pipeline.captions as captions_mod
    import pipeline.kidsong.comfy as comfy_mod
    import pipeline.kidsong.cut_qc as cut_qc_mod
    import pipeline.kidsong.edit as edit_mod
    import pipeline.kidsong.review as review_mod
    import pipeline.kidsong.sing as sing_mod

    class FakeComfyClient:
        """`load_workflow` reads the REAL repo workflow file (no network) so
        the baseline NEGATIVE text is the genuine authored one; `render` is
        faked to capture the patch dict and just write a dummy output file."""

        def __init__(self, cfg):
            self.url = "http://fake:8188"
            self._proc = None

        def load_workflow(self, workflow_name):
            return _REAL_LOAD_WORKFLOW(self, workflow_name)

        def ensure_up(self):
            pass

        def render(self, workflow, patches, out_path):
            render_calls.append((workflow, patches))
            with open(out_path, "wb") as f:
                f.write(b"\0" * 1024)

        def free(self):
            pass

    class FakeReviewer:
        def review(self, shot, video_path, out_dir=None):
            return {"accept": True, "score": 1.0, "reasons": [], "retry_hints": {}}

    def fake_assemble(cfg, cut_list, renders, voice, words, duration, out_path, **kw):
        with open(out_path, "wb") as f:
            f.write(b"\0" * 4096)

    def fake_review_cut(staging_path, cut_list, beats, words, duration, cfg, review_dir):
        return {"accept": True, "score": 1.0, "reasons": [], "retry_hints": {}}

    def fake_prepend_intro(video_path, cfg, on_progress=None):
        return {"applied": False, "intro_seconds": 0.0, "reason": "disabled for test"}

    monkeypatch.setattr(comfy_mod, "ComfyClient", FakeComfyClient)
    monkeypatch.setattr(review_mod, "get_reviewer", lambda cfg: FakeReviewer())
    monkeypatch.setattr(review_mod, "contact_sheet", lambda *a, **k: None)
    monkeypatch.setattr(review_mod, "review_shots_log", lambda path, entries: path)
    monkeypatch.setattr(captions_mod, "get_word_timestamps", lambda *a, **k: [])
    monkeypatch.setattr(sing_mod, "verse_times_from_words", lambda *a, **k: [(0.0, 2.0)])
    monkeypatch.setattr(edit_mod, "beat_grid", lambda path: [0.0, 1.0, 2.0])
    monkeypatch.setattr(edit_mod, "build_cut_list", lambda *a, **k: [])
    monkeypatch.setattr(edit_mod, "assemble", fake_assemble)
    monkeypatch.setattr(cut_qc_mod, "review_cut", fake_review_cut)
    monkeypatch.setattr(edit_mod, "prepend_intro", fake_prepend_intro)

    def _sing_must_not_run(*a, **k):
        raise AssertionError("resume must not re-sing")

    monkeypatch.setattr(sing_mod, "sing_song", _sing_must_not_run)

    cfg = {
        "_root": _REPO_ROOT,
        "paths": {"output_dir": str(out_dir)},
        "video": {"max_seconds": 60},
        "captions": {"uppercase": True},
        "kidsong": {
            "mode": "director",
            "singer": "ace",
            "seed": 20260717,
            "shot": {"fps": 24, "max_frames": 241, "hires_pass": False, "style_trigger": "P1x4r"},
            "review": {"script": False, "cut": True, "max_retries_per_shot": 0},
            "intro": {"enabled": False},
        },
    }

    return {"cfg": cfg, "base": base, "out_dir": str(out_dir), "render_calls": render_calls}


def test_render_patch_includes_negative_with_character_and_baseline_terms(pending_shot_run):
    from pipeline.kidsong.generate import generate_kidsong

    run = pending_shot_run
    result = generate_kidsong(cfg=run["cfg"], resume_base=run["base"])

    assert result["status"] == "approved"
    assert len(run["render_calls"]) == 1
    _workflow, patches = run["render_calls"][0]

    # The gap: NEGATIVE must be present in the patch dict client.render() is
    # actually called with -- not merely computed and discarded.
    assert "NEGATIVE" in patches
    assert isinstance(patches["NEGATIVE"], dict)
    assert set(patches["NEGATIVE"].keys()) == {"text"}
    negative_text = patches["NEGATIVE"]["text"]
    assert isinstance(negative_text, str) and negative_text

    # Kofi's own must_not signature term (see test_kidsong_shot_prompt.py's
    # _shot_negative tests) must have made it all the way into the render call.
    assert "yellow shirt" in negative_text.lower()

    # The workflow's authored baseline terms must still be present -- the
    # per-shot negative is additive, never a replacement.
    assert "wrong shirt color" in negative_text
    assert "shirtless child" in negative_text


def test_render_patch_prompt_and_seed_also_present(pending_shot_run):
    """Sanity check that this fixture actually exercises a real render call
    with the expected other patch keys, so a false-positive "NEGATIVE
    present" (e.g. from an entirely different/stale code path) is unlikely."""
    from pipeline.kidsong.generate import generate_kidsong

    run = pending_shot_run
    generate_kidsong(cfg=run["cfg"], resume_base=run["base"])

    assert len(run["render_calls"]) == 1
    _workflow, patches = run["render_calls"][0]
    assert "PROMPT" in patches and "Kofi" in patches["PROMPT"]["text"]
    assert patches["SEED"]["noise_seed"] == 1000
