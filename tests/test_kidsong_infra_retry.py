"""Regression test for Contract 1 (forensics pass on episodes that died
mid-render): an infra failure from ComfyClient.render() (timeout, dropped
connection, dead server) must retry the SAME shot attempt instead of either
(a) burning a QC attempt (max_retries_per_shot) or (b) propagating to the
outer handler and aborting every remaining shot in the episode.

Built on the same fixture shape as test_kidsong_negative_wiring.py: a resumed
run with exactly ONE PENDING shot so the render loop actually executes, with
every other GPU/media stage faked. CPU-only, no GPU, no network, no ffmpeg,
no real ComfyUI.
"""
import json
import os
import struct
import wave

import pytest

from pipeline.kidsong import runstate
from pipeline.kidsong.comfy import ComfyClient as _RealComfyClient
from pipeline.kidsong.comfy import ComfyUnreachableError

_REAL_LOAD_WORKFLOW = _RealComfyClient.load_workflow
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _write_wav(path, seconds=2.0, rate=8000):
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(rate)
        wf.writeframes(struct.pack("<h", 0) * int(rate * seconds))


def _make_run(tmp_path, monkeypatch, render_side_effects):
    """Same fixture shape as test_kidsong_negative_wiring.py's
    `pending_shot_run`, parameterized on what `render()` does on each call:
    `render_side_effects` is a list where each entry is either an Exception
    instance (raised) or None (a normal successful render)."""
    out_dir = tmp_path / "output"
    out_dir.mkdir()
    base = "20260721-093000-kidsong-infra-retry-test"
    shots_dir = runstate.shots_dir_path(str(out_dir), base)
    os.makedirs(shots_dir, exist_ok=True)

    _write_wav(out_dir / f"{base}.wav")
    song = {
        "title": "Infra Retry Test Song",
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

    render_calls = []
    free_calls = []
    restart_calls = []

    import pipeline.captions as captions_mod
    import pipeline.kidsong.comfy as comfy_mod
    import pipeline.kidsong.cut_qc as cut_qc_mod
    import pipeline.kidsong.edit as edit_mod
    import pipeline.kidsong.review as review_mod
    import pipeline.kidsong.sing as sing_mod

    class FakeComfyClient:
        def __init__(self, cfg):
            self.url = "http://fake:8188"
            self._proc = None

        def load_workflow(self, workflow_name):
            return _REAL_LOAD_WORKFLOW(self, workflow_name)

        def ensure_up(self):
            pass

        def render(self, workflow, patches, out_path):
            i = len(render_calls)
            render_calls.append((workflow, patches))
            effect = render_side_effects[i] if i < len(render_side_effects) else None
            if effect is not None:
                raise effect
            with open(out_path, "wb") as f:
                f.write(b"\0" * 1024)

        def free(self):
            free_calls.append(1)

        def restart_if_hung(self):
            restart_calls.append(1)

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

    # No real sleeping between infra retries.
    monkeypatch.setattr("pipeline.kidsong.generate.time.sleep", lambda s: None)

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
            "review": {
                "script": False, "cut": True,
                # Only ONE QC attempt allowed -- proves the infra retries
                # below did NOT consume this budget.
                "max_retries_per_shot": 0,
                "comfy_infra_retries": 3,
            },
            "intro": {"enabled": False},
        },
    }

    return {
        "cfg": cfg, "base": base, "out_dir": str(out_dir),
        "render_calls": render_calls, "free_calls": free_calls,
        "restart_calls": restart_calls,
    }


def test_infra_timeouts_retry_same_attempt_and_shot_completes(tmp_path, monkeypatch):
    """TimeoutError twice, then a normal render -> the shot still completes,
    on its first (and only) QC attempt."""
    from pipeline.kidsong.generate import generate_kidsong

    run = _make_run(
        tmp_path, monkeypatch,
        render_side_effects=[
            TimeoutError("ComfyUI render exceeded 600s for prompt p1"),
            TimeoutError("ComfyUI render exceeded 600s for prompt p2"),
            None,
        ],
    )

    result = generate_kidsong(cfg=run["cfg"], resume_base=run["base"])

    assert result["status"] == "approved"
    # render() was retried twice before succeeding.
    assert len(run["render_calls"]) == 3
    # client.free() called once between each failed attempt and the retry.
    # 2 infra-retry frees (before each retried attempt) + 1 unconditional
    # `finally: client.free()` at the end of the whole render loop (existing
    # pipeline behaviour, always runs on success or failure -- not part of
    # Contract 1, just something this count has to account for).
    assert len(run["free_calls"]) == 3
    # A plain TimeoutError is not a ComfyUnreachableError -- restart_if_hung
    # must NOT have been invoked for it.
    assert run["restart_calls"] == []

    shotlist = runstate.load_shotlist(run["out_dir"], run["base"])
    shot = shotlist["shots"][0]
    # QC attempt counter unburned: exactly ONE QC attempt was spent, even
    # though the render itself was attempted three times.
    assert shot["attempts"] == 1
    assert shot["status"] == runstate.STATUS_RENDERED
    assert shot["verdict"] == runstate.VERDICT_ACCEPTED


def test_comfy_unreachable_triggers_restart_before_retrying(tmp_path, monkeypatch):
    """A ComfyUnreachableError additionally calls the Contract-2 recovery
    helper (restart_if_hung) before the retried render."""
    from pipeline.kidsong.generate import generate_kidsong

    run = _make_run(
        tmp_path, monkeypatch,
        render_side_effects=[
            ComfyUnreachableError("ComfyUI at http://fake:8188 has not answered for over 30s"),
            None,
        ],
    )

    result = generate_kidsong(cfg=run["cfg"], resume_base=run["base"])

    assert result["status"] == "approved"
    assert len(run["render_calls"]) == 2
    # 1 infra-retry free (before the retried attempt) + 1 unconditional
    # `finally: client.free()` at the end of the render loop (see the other
    # test for why).
    assert len(run["free_calls"]) == 2
    assert len(run["restart_calls"]) == 1


def test_infra_retry_budget_exhausted_reraises(tmp_path, monkeypatch):
    """More infra failures than comfy_infra_retries allows must still
    eventually raise (the outer-abort last resort stays intact)."""
    from pipeline.kidsong.generate import generate_kidsong

    run = _make_run(
        tmp_path, monkeypatch,
        render_side_effects=[TimeoutError("down")] * 10,  # far more than the budget of 3
    )

    with pytest.raises(TimeoutError):
        generate_kidsong(cfg=run["cfg"], resume_base=run["base"])

    # comfy_infra_retries=3 -> up to 4 render attempts (1 + 3 retries) before
    # giving up.
    assert len(run["render_calls"]) == 4
    # 3 infra-retry frees (before each of the first 3 retries; the 4th
    # failure exhausts the budget and raises without one more) + 1
    # unconditional `finally: client.free()` once the exception unwinds out
    # of the render loop (see the other tests for why).
    assert len(run["free_calls"]) == 4
