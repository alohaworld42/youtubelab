"""Tests for pipeline.kidsong.auto_review (QM-010): the headless answerer for
the vision QC gate's request/response files.

No real `claude` CLI calls anywhere here — `_call_claude` (and, for the one
end-to-end test, `subprocess.run` itself) is monkeypatched. Everything else
(JSON extraction, schema validation, atomic writes, request discovery,
skip-already-answered, stale skip, and the shot/cut/script routing) runs for
real against tmp_path fixtures shaped exactly like `review.py` / `cut_qc.py`
/ `script_qc.py` write them.
"""
import json
import os
import time
from types import SimpleNamespace

import pytest

from pipeline.kidsong import auto_review

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _cfg(tmp_path, **auto_overrides):
    return {
        "_root": _ROOT,
        "paths": {"output_dir": str(tmp_path)},
        "kidsong": {"review": {"auto": auto_overrides}},
    }


# ------------------------------------------------------------ json extraction --
def test_extract_json_plain():
    obj = auto_review.extract_json(
        '{"accept": true, "score": 1.0, "reasons": [], "retry_hints": {}}'
    )
    assert obj == {"accept": True, "score": 1.0, "reasons": [], "retry_hints": {}}


def test_extract_json_with_prose_before_and_after():
    text = (
        "Sure, here is my verdict:\n\n"
        '{"accept": false, "score": 0.2, "reasons": ["bad"], "retry_hints": {"seed_bump": true}}'
        "\n\nLet me know if you need anything else!"
    )
    obj = auto_review.extract_json(text)
    assert obj["accept"] is False
    assert obj["reasons"] == ["bad"]


def test_extract_json_code_fence():
    text = '```json\n{"accept": true, "score": 0.9, "reasons": [], "retry_hints": {}}\n```'
    obj = auto_review.extract_json(text)
    assert obj["accept"] is True


def test_extract_json_takes_the_last_object_when_multiple_present():
    text = (
        'draft: {"accept": true, "score": 0.1} '
        'final answer: {"accept": false, "score": 0.9, "reasons": ["x"], "retry_hints": {}}'
    )
    obj = auto_review.extract_json(text)
    assert obj["accept"] is False
    assert obj["score"] == 0.9


def test_extract_json_ignores_braces_inside_string_values():
    text = (
        'Here is my verdict: {"accept": true, "score": 1.0, '
        '"reasons": ["a toy block shows a { symbol, unrelated"], "retry_hints": {}}'
    )
    obj = auto_review.extract_json(text)
    assert obj is not None
    assert obj["accept"] is True
    assert obj["reasons"] == ["a toy block shows a { symbol, unrelated"]


def test_extract_json_returns_none_when_nothing_parses():
    assert auto_review.extract_json("no json here at all") is None
    assert auto_review.extract_json("") is None
    assert auto_review.extract_json(None) is None


# --------------------------------------------------------------- validation --
def test_validate_verdict_accepts_well_formed():
    assert auto_review.validate_verdict(
        {"accept": True, "score": 0.5, "reasons": [], "retry_hints": {}}
    )


@pytest.mark.parametrize(
    "bad",
    [
        {},
        {"accept": "yes", "score": 0.5, "reasons": [], "retry_hints": {}},
        {"accept": True, "score": "high", "reasons": [], "retry_hints": {}},
        {"accept": True, "score": 1.5, "reasons": [], "retry_hints": {}},
        {"accept": True, "score": -0.1, "reasons": [], "retry_hints": {}},
        {"accept": True, "score": 0.5, "reasons": "none", "retry_hints": {}},
        {"accept": True, "score": 0.5, "reasons": [], "retry_hints": ["x"]},
        {"accept": True, "score": True, "reasons": [], "retry_hints": {}},
    ],
)
def test_validate_verdict_rejects_malformed(bad):
    assert not auto_review.validate_verdict(bad)


# ---------------------------------------------------------------- retry logic --
def test_get_verdict_retries_once_then_succeeds(monkeypatch):
    calls = []

    def fake_call(cmd, prompt, model, timeout=None):
        calls.append(1)
        if len(calls) == 1:
            return "not json at all"
        return '{"accept": true, "score": 0.8, "reasons": [], "retry_hints": {}}'

    monkeypatch.setattr(auto_review, "_call_claude", fake_call)
    v = auto_review._get_verdict("claude", "prompt", "model", "label")
    assert v["accept"] is True
    assert len(calls) == 2


def test_get_verdict_gives_up_after_two_failures_and_never_returns_malformed(monkeypatch):
    monkeypatch.setattr(auto_review, "_call_claude", lambda *a, **k: "still not json")
    v = auto_review._get_verdict("claude", "prompt", "model", "label")
    assert v is None


def test_get_verdict_retries_on_schema_violation_not_just_parse_failure(monkeypatch):
    calls = []

    def fake_call(cmd, prompt, model, timeout=None):
        calls.append(1)
        if len(calls) == 1:
            # parses fine but fails schema (score out of range)
            return '{"accept": true, "score": 5.0, "reasons": [], "retry_hints": {}}'
        return '{"accept": true, "score": 0.5, "reasons": [], "retry_hints": {}}'

    monkeypatch.setattr(auto_review, "_call_claude", fake_call)
    v = auto_review._get_verdict("claude", "prompt", "model", "label")
    assert v["score"] == 0.5
    assert len(calls) == 2


# ------------------------------------------------------------------ atomic write --
def test_write_response_atomic_creates_file_and_leaves_no_tmp(tmp_path):
    resp = tmp_path / "sub" / "s00_a0.response.json"
    payload = {"accept": True, "score": 1.0, "reasons": [], "retry_hints": {}}
    auto_review._write_response_atomic(str(resp), payload)

    assert resp.exists()
    data = json.loads(resp.read_text(encoding="utf-8"))
    assert data == payload
    leftovers = list((tmp_path / "sub").glob(".auto_review_*"))
    assert leftovers == []


# --------------------------------------------------------------- fixtures ---
def _write_shot_request(out_dir, base, take="s00_a0", age_seconds=0, extra=None):
    review_dir = os.path.join(out_dir, f"{base}-shots", "review")
    os.makedirs(review_dir, exist_ok=True)
    png_path = os.path.join(review_dir, f"{take}.png")
    with open(png_path, "wb") as f:
        f.write(b"\x89PNG\r\n\x1a\n")  # never actually decoded by auto_review

    req = {
        "shot_id": take.split("_a")[0],
        "take": take,
        "shot": {
            "action": "claps hands happily",
            "characters": ["Kofi"],
            "shot_type": "medium",
            "camera": "static",
            "setting": "a sunny backyard",
        },
        "video_path": os.path.join(review_dir, f"{take}.mp4"),
        "contact_sheet": png_path,
        "heuristic": {"accept": True, "score": 0.9, "reasons": [], "retry_hints": {}},
        "expected_children": 1,
        "cast_text": "Kofi: warm brown skin, short black coily hair, wears a blue t-shirt.",
        "cast_version": "v1",
    }
    if extra:
        req.update(extra)

    req_path = os.path.join(review_dir, f"{take}.request.json")
    with open(req_path, "w", encoding="utf-8") as f:
        json.dump(req, f)

    if age_seconds:
        old = time.time() - age_seconds
        os.utime(req_path, (old, old))
    return req_path


def _write_cut_request(out_dir, base):
    review_dir = os.path.join(out_dir, f"{base}-shots", "review")
    os.makedirs(review_dir, exist_ok=True)
    sheet_path = os.path.join(review_dir, "cut_sheet.png")
    with open(sheet_path, "wb") as f:
        f.write(b"\x89PNG\r\n\x1a\n")

    req = {
        "video_path": os.path.join(review_dir, f"{base}.staging.mp4"),
        "cut_list_summary": [{"shot_id": "s00", "start": 0.0, "end": 2.0, "src": "s00_a0"}],
        "contact_sheet": sheet_path,
        "programmatic_verdict": {
            "accept": True, "score": 1.0, "reasons": [], "retry_hints": {"recut": False, "reshoot": []},
        },
    }
    req_path = os.path.join(review_dir, "cut.request.json")
    with open(req_path, "w", encoding="utf-8") as f:
        json.dump(req, f)
    return req_path


def _write_script_request(out_dir):
    review_dir = os.path.join(out_dir, "_kidsong_review")
    os.makedirs(review_dir, exist_ok=True)
    req = {
        "song": {
            "title": "Clap Your Hands",
            "verses": [{"lines": ["Clap, clap, clap your hands", "Clap them all around"], "scene": "kids clapping"}],
            "characters": "Kofi and Nala play together outside.",
        },
        "programmatic_verdict": {"accept": True, "score": 1.0, "reasons": [], "retry_hints": {"regenerate": False}},
    }
    req_path = os.path.join(review_dir, "script.request.json")
    with open(req_path, "w", encoding="utf-8") as f:
        json.dump(req, f)
    return req_path


# ------------------------------------------------------------ config wiring --
def test_auto_cfg_defaults():
    merged = auto_review._auto_cfg({"kidsong": {"review": {}}})
    assert merged == {
        "enabled": True, "poll_seconds": 10, "model": "claude-sonnet-5",
        "claude_cmd": "claude", "max_age_hours": 48,
    }


def test_auto_cfg_overrides_leave_other_defaults_intact():
    cfg = {"kidsong": {"review": {"auto": {"poll_seconds": 5, "enabled": False}}}}
    merged = auto_review._auto_cfg(cfg)
    assert merged["poll_seconds"] == 5
    assert merged["enabled"] is False
    assert merged["model"] == "claude-sonnet-5"


# ------------------------------------------------------------ discovery/skip --
def test_run_once_skips_request_that_already_has_a_response(tmp_path, monkeypatch):
    req_path = _write_shot_request(str(tmp_path), "mysong")
    resp_path = req_path[: -len(".request.json")] + ".response.json"
    original = {"accept": True, "score": 1.0, "reasons": ["human answered"], "retry_hints": {}}
    with open(resp_path, "w", encoding="utf-8") as f:
        json.dump(original, f)

    called = []
    monkeypatch.setattr(
        auto_review, "_call_claude",
        lambda *a, **k: (called.append(1), '{"accept": false, "score": 0.0, "reasons": [], "retry_hints": {}}')[1],
    )

    n = auto_review.run_once(_cfg(tmp_path))
    assert n == 0
    assert called == []
    with open(resp_path, encoding="utf-8") as f:
        assert json.load(f) == original  # untouched


def test_run_once_skips_stale_requests_by_default(tmp_path, monkeypatch):
    _write_shot_request(str(tmp_path), "oldsong", age_seconds=100 * 3600)  # 100h old
    called = []
    monkeypatch.setattr(auto_review, "_call_claude", lambda *a, **k: called.append(1))

    n = auto_review.run_once(_cfg(tmp_path, max_age_hours=48))
    assert n == 0
    assert called == []


def test_run_once_include_stale_answers_old_requests(tmp_path, monkeypatch):
    _write_shot_request(str(tmp_path), "oldsong", age_seconds=100 * 3600)
    monkeypatch.setattr(
        auto_review, "_call_claude",
        lambda *a, **k: '{"accept": true, "score": 1.0, "reasons": [], '
                        '"retry_hints": {"seed_bump": false, "simplify_action": false, "force_i2v": false}}',
    )

    n = auto_review.run_once(_cfg(tmp_path, max_age_hours=48), include_stale=True)
    assert n == 1


def test_run_once_leaves_request_pending_when_claude_never_produces_a_valid_verdict(tmp_path, monkeypatch):
    req_path = _write_shot_request(str(tmp_path), "brokensong")
    resp_path = req_path[: -len(".request.json")] + ".response.json"
    monkeypatch.setattr(auto_review, "_call_claude", lambda *a, **k: "I refuse to answer in JSON.")

    n = auto_review.run_once(_cfg(tmp_path))
    assert n == 0
    assert not os.path.exists(resp_path)  # never write a malformed/no response


# --------------------------------------------------------- routing by kind --
def test_cut_request_is_answered(tmp_path, monkeypatch):
    req_path = _write_cut_request(str(tmp_path), "cutsong")
    resp_path = req_path[: -len(".request.json")] + ".response.json"
    monkeypatch.setattr(
        auto_review, "_call_claude",
        lambda *a, **k: '{"accept": true, "score": 1.0, "reasons": [], '
                        '"retry_hints": {"recut": false, "reshoot": []}}',
    )

    n = auto_review.run_once(_cfg(tmp_path))
    assert n == 1
    with open(resp_path, encoding="utf-8") as f:
        data = json.load(f)
    assert data["accept"] is True


def test_script_request_is_answered(tmp_path, monkeypatch):
    req_path = _write_script_request(str(tmp_path))
    resp_path = req_path[: -len(".request.json")] + ".response.json"
    monkeypatch.setattr(
        auto_review, "_call_claude",
        lambda *a, **k: '{"accept": true, "score": 1.0, "reasons": [], "retry_hints": {"regenerate": false}}',
    )

    n = auto_review.run_once(_cfg(tmp_path))
    assert n == 1
    assert os.path.exists(resp_path)


# ------------------------------------------------------------------ end to end --
def test_end_to_end_shot_review_via_mocked_subprocess(tmp_path, monkeypatch):
    """Fake review dir with request+png -> mocked claude reply -> response.json
    appears with the correct schema. This is the only test that goes through
    `subprocess.run` itself (mocked) rather than `_call_claude`, so it also
    exercises the envelope parsing (`result` field) and the exact CLI flags."""
    req_path = _write_shot_request(str(tmp_path), "sunnysong")
    resp_path = req_path[: -len(".request.json")] + ".response.json"

    verdict_text = (
        "Looking closely at the five frames...\n\n"
        '{"accept": true, "score": 0.95, '
        '"reasons": ["clean single-child shot, matches cast_text"], '
        '"retry_hints": {"seed_bump": false, "simplify_action": false, "force_i2v": false}}'
    )
    envelope = json.dumps({"type": "result", "subtype": "success", "is_error": False, "result": verdict_text})

    seen_cmds = []

    def fake_run(cmd, **kwargs):
        seen_cmds.append(cmd)
        assert kwargs.get("capture_output") is True
        assert kwargs.get("text") is True
        return SimpleNamespace(returncode=0, stdout=envelope, stderr="")

    monkeypatch.setattr(auto_review.subprocess, "run", fake_run)

    cfg = _cfg(tmp_path, claude_cmd="claude", model="claude-sonnet-5")
    n = auto_review.run_once(cfg)

    assert n == 1
    assert os.path.exists(resp_path)
    with open(resp_path, encoding="utf-8") as f:
        data = json.load(f)
    assert data["accept"] is True
    assert data["score"] == 0.95
    assert isinstance(data["reasons"], list)
    assert isinstance(data["retry_hints"], dict)

    cmd = seen_cmds[0]
    assert os.path.basename(cmd[0]).split(".")[0] == "claude"  # resolved path on Windows
    assert "-p" in cmd
    assert "--model" in cmd and "claude-sonnet-5" in cmd
    assert "--allowedTools" in cmd and "Read" in cmd
    assert "--output-format" in cmd and "json" in cmd


def test_call_claude_reports_none_on_non_json_stdout(monkeypatch):
    monkeypatch.setattr(
        auto_review.subprocess, "run",
        lambda cmd, **k: SimpleNamespace(returncode=1, stdout="not an envelope", stderr="boom"),
    )
    assert auto_review._call_claude("claude", "prompt", "model") is None


def test_call_claude_returns_result_field_not_text_field():
    """Regression guard for the field name: a live `claude -p ... --output-format
    json` call was verified to put the final message in `result`, not `text`."""
    import subprocess as _sp

    class FakeCompleted:
        returncode = 0
        stdout = json.dumps({"type": "result", "is_error": False, "result": "hello", "text": "WRONG FIELD"})
        stderr = ""

    orig_run = _sp.run
    try:
        auto_review.subprocess.run = lambda *a, **k: FakeCompleted()
        out = auto_review._call_claude("claude", "prompt", "model")
        assert out == "hello"
    finally:
        auto_review.subprocess.run = orig_run


# ---------------------------------------------------------------- watch loop --
def test_run_watch_respects_stop_after(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(auto_review, "run_once", lambda cfg, include_stale=False: (calls.append(1), 0)[1])
    monkeypatch.setattr(auto_review.time, "sleep", lambda s: None)

    auto_review.run_watch(_cfg(tmp_path, poll_seconds=0), stop_after=3)
    assert len(calls) == 3
