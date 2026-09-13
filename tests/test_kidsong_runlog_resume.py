"""Tests for per-run logging (pipeline.kidsong.runlog) and crash-resume
(pipeline.kidsong.runstate + generate_kidsong(resume_base=...)).

Motivating incident: a run rendered 8 of 16 shots and died. Nothing was logged
(the pipeline only printed to stdout, and the batch runner's shell redirection
had silently detached), and the re-run started a fresh <base> and re-rendered
everything.

No GPU, no ComfyUI, no network: the render loop's ComfyClient, reviewer,
Whisper, beat grid, editor and cut QC are all monkeypatched. The "renders" are
byte-writes to tmp_path, which is exactly what resume inspects.
"""
import json
import os
import struct
import sys
import wave

import pytest

from pipeline.kidsong import runstate
from pipeline.kidsong.review import review_shots_log as _real_review_shots_log
from pipeline.kidsong.runlog import ERROR_LOG_NAME, STARTUP_PREFIX, RunLog, install_excepthook


# ============================================================ runlog =========
def test_log_file_is_created_and_populated_before_base_is_known(tmp_path):
    """A crash during config/lyrics/singing must still leave a readable log."""
    rl = RunLog(str(tmp_path), stream=False)
    try:
        assert os.path.basename(rl.path).startswith(STARTUP_PREFIX)
        rl.stage("script")
        rl.info("  -> Writing song lyrics…")
        text = open(rl.path, encoding="utf-8").read()
    finally:
        rl.close()

    assert "STAGE script" in text
    assert "Writing song lyrics" in text
    # Timestamps and levels are what make the file diagnosable.
    assert "INFO" in text
    assert text.lstrip()[:4].isdigit(), f"expected a leading timestamp, got {text[:40]!r}"


def test_rebind_moves_the_startup_log_to_base_log_keeping_early_records(tmp_path):
    rl = RunLog(str(tmp_path), stream=False)
    try:
        rl.info("early record before base existed")
        startup_path = rl.path
        rl.rebind("20260720-085055-kidsong-splish-splash")
        rl.info("late record after base was known")
        path, text = rl.path, open(rl.path, encoding="utf-8").read()
    finally:
        rl.close()

    assert os.path.basename(path) == "20260720-085055-kidsong-splish-splash.log"
    assert not os.path.exists(startup_path), "startup log should be renamed, not duplicated"
    assert "early record before base existed" in text
    assert "late record after base was known" in text


def test_rebind_onto_an_existing_log_appends_rather_than_clobbering(tmp_path):
    """Resuming a run must not destroy the interrupted attempt's log."""
    existing = tmp_path / "run-base.log"
    existing.write_text("FIRST ATTEMPT RECORD\n", encoding="utf-8")

    rl = RunLog(str(tmp_path), stream=False)
    try:
        rl.info("second attempt record")
        rl.rebind("run-base")
        text = open(rl.path, encoding="utf-8").read()
    finally:
        rl.close()

    assert "FIRST ATTEMPT RECORD" in text
    assert "second attempt record" in text


def test_console_lines_are_unchanged_while_the_file_gets_timestamps(tmp_path, capsys):
    rl = RunLog(str(tmp_path), stream=True)
    try:
        rl.info("  -> %s", "Singing the song (ACE-Step on GPU)…")
        out = capsys.readouterr().out
        text = open(rl.path, encoding="utf-8").read()
    finally:
        rl.close()

    # stdout keeps the bare human-readable progress line, exactly as before.
    assert "  -> Singing the song (ACE-Step on GPU)…" in out
    assert "INFO" not in out
    # the file carries level + timestamp
    assert "INFO" in text and "Singing the song" in text


def test_exception_writes_traceback_to_run_log_and_the_shared_error_log(tmp_path):
    rl = RunLog(str(tmp_path), base="my-run", stream=False)
    try:
        try:
            raise TimeoutError("ComfyUI render exceeded 1800s for prompt 694cd4cb")
        except TimeoutError as e:
            rl.exception("Kidsong run FAILED", e)
        run_text = open(rl.path, encoding="utf-8").read()
    finally:
        rl.close()

    err_text = open(tmp_path / ERROR_LOG_NAME, encoding="utf-8").read()
    for text in (run_text, err_text):
        assert "Kidsong run FAILED" in text
        assert "Traceback (most recent call last)" in text
        assert "ComfyUI render exceeded 1800s" in text
        assert "TimeoutError" in text
    # the shared log identifies which run failed
    assert "base=my-run" in err_text


def test_error_log_is_append_only_across_runs(tmp_path):
    """Failures from several runs must be visible in one place."""
    for base, msg in (("run-a", "boom A"), ("run-b", "boom B")):
        rl = RunLog(str(tmp_path), base=base, stream=False)
        try:
            rl.exception("failed", RuntimeError(msg))
        finally:
            rl.close()

    err_text = open(tmp_path / ERROR_LOG_NAME, encoding="utf-8").read()
    assert "boom A" in err_text and "boom B" in err_text
    assert "base=run-a" in err_text and "base=run-b" in err_text


def test_the_same_exception_is_only_reported_once(tmp_path):
    """Nested guards (render loop, generate_kidsong, CLI) share one exception."""
    rl = RunLog(str(tmp_path), base="dedupe", stream=False)
    exc = RuntimeError("only once please")
    try:
        rl.exception("inner guard", exc)
        rl.exception("outer guard", exc)
    finally:
        rl.close()

    assert open(tmp_path / ERROR_LOG_NAME, encoding="utf-8").read().count("only once please") == 1


def test_install_excepthook_reports_uncaught_exceptions(tmp_path):
    rl = RunLog(str(tmp_path), base="hooked", stream=False)
    original = sys.excepthook
    try:
        install_excepthook(rl)
        try:
            raise ValueError("escaped the guards")
        except ValueError as e:
            sys.excepthook(type(e), e, e.__traceback__)
    finally:
        sys.excepthook = original
        rl.close()

    err_text = open(tmp_path / ERROR_LOG_NAME, encoding="utf-8").read()
    assert "UNCAUGHT ValueError" in err_text
    assert "escaped the guards" in err_text


def test_fingerprint_records_the_environment(tmp_path):
    cfg = {
        "kidsong": {"mode": "director", "ace_python": "D:/nope/python.exe"},
        "comfy": {"url": "http://127.0.0.1:8188", "autostart": True},
    }
    rl = RunLog(str(tmp_path), base="fp", stream=False)
    try:
        rl.fingerprint(cfg, autostarted=False, extra={"topic": "bath time fun"})
        text = open(rl.path, encoding="utf-8").read()
    finally:
        rl.close()

    assert "python" in text and sys.version.split()[0] in text
    assert "http://127.0.0.1:8188" in text
    assert "comfy_started_by_us = False" in text
    assert "D:/nope/python.exe" in text and "exists=False" in text
    assert "bath time fun" in text


# =========================================================== runstate ========
def _take(shots_dir, shot_id, attempt, size=1024):
    os.makedirs(shots_dir, exist_ok=True)
    p = os.path.join(shots_dir, f"{shot_id}_a{attempt}.mp4")
    with open(p, "wb") as f:
        f.write(b"\0" * size)
    return p


def test_existing_takes_and_next_attempt_index(tmp_path):
    sdir = str(tmp_path / "shots")
    _take(sdir, "s00", 0)
    _take(sdir, "s00", 2)
    _take(sdir, "s01", 0)

    takes = runstate.existing_takes(sdir, "s00")
    assert [os.path.basename(p) for p in takes] == ["s00_a0.mp4", "s00_a2.mp4"]
    # New takes never overwrite old media (channel-inventory rule).
    assert runstate.next_attempt_index(sdir, "s00") == 3
    assert runstate.next_attempt_index(sdir, "s01") == 1
    assert runstate.next_attempt_index(sdir, "s99") == 0


def test_a_zero_byte_take_does_not_count_as_done(tmp_path):
    """A render interrupted mid-write is exactly the shot that crashed."""
    sdir = str(tmp_path / "shots")
    empty = _take(sdir, "s08", 0, size=0)
    shot = {"id": "s08", "status": runstate.STATUS_RENDERED, "take": empty}
    assert runstate.shot_is_done(shot, sdir) is False


def test_shot_is_exhausted_bounds_failure(tmp_path):
    assert runstate.shot_is_exhausted({"attempts": 6}, max_attempts=6) is True
    assert runstate.shot_is_exhausted({"attempts": 2}, max_attempts=6) is False
    assert runstate.shot_is_exhausted({"status": runstate.STATUS_FAILED}, max_attempts=6) is True


def test_recover_takes_adopts_orphaned_media_from_a_pre_ledger_run(tmp_path):
    """The bath-time run's 8 takes predate the ledger — adopt, don't re-render."""
    sdir = str(tmp_path / "shots")
    for i in range(3):
        _take(sdir, f"s{i:02d}", 0)
    shotlist = {"shots": [{"id": f"s{i:02d}", "status": "planned"} for i in range(5)]}

    recovered = runstate.recover_takes(shotlist, sdir)

    assert recovered == 3
    assert [s["status"] for s in shotlist["shots"]] == ["rendered"] * 3 + ["planned"] * 2
    assert shotlist["shots"][0]["take"].endswith("s00_a0.mp4")


def test_save_shotlist_is_atomic_and_round_trips(tmp_path):
    shotlist = {"shots": [{"id": "s00", "status": "rendered", "take": "x.mp4"}]}
    runstate.save_shotlist(str(tmp_path), "base1", shotlist)

    assert runstate.load_shotlist(str(tmp_path), "base1") == shotlist
    assert not os.path.exists(runstate.shots_json_path(str(tmp_path), "base1") + ".tmp")


def test_find_resumable_reports_pending_work(tmp_path):
    base = "20260720-085055-kidsong-splish-splash"
    sdir = runstate.shots_dir_path(str(tmp_path), base)
    for i in range(8):
        _take(sdir, f"s{i:02d}", 0)
    runstate.save_shotlist(
        str(tmp_path), base,
        {"shots": [{"id": f"s{i:02d}", "status": "planned"} for i in range(16)]},
    )
    (tmp_path / f"{base}.wav").write_bytes(b"RIFF")
    (tmp_path / f"{base}-song.json").write_text(
        json.dumps({"title": "Splish Splash"}), encoding="utf-8"
    )

    runs = runstate.find_resumable(str(tmp_path))

    assert len(runs) == 1
    r = runs[0]
    assert r["base"] == base and r["title"] == "Splish Splash"
    assert (r["unique_done"], r["unique_pending"], r["unique_total"]) == (8, 8, 16)
    assert r["resumable"] is True and r["has_audio"] is True
    assert "RESUMABLE" in runstate.format_resumable(runs)


def test_find_resumable_excludes_finished_runs(tmp_path):
    base = "done-run"
    sdir = runstate.shots_dir_path(str(tmp_path), base)
    _take(sdir, "s00", 0)
    runstate.save_shotlist(str(tmp_path), base, {"shots": [{"id": "s00"}, {"id": "s01"}]})
    final = tmp_path / "Final"
    final.mkdir()
    (final / f"{base}.mp4").write_bytes(b"\0")

    runs = runstate.find_resumable(str(tmp_path))
    assert runs[0]["finished"] is True and runs[0]["resumable"] is False


def test_format_resumable_handles_an_empty_output_dir(tmp_path):
    assert "No interrupted" in runstate.format_resumable(runstate.find_resumable(str(tmp_path)))


# ================================================ resume, end to end ========
def _write_wav(path, seconds=4.0, rate=8000):
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(rate)
        wf.writeframes(struct.pack("<h", 0) * int(rate * seconds))


@pytest.fixture()
def resume_run(tmp_path, monkeypatch):
    """A half-finished director run on disk: 4 shots, s00/s01 already rendered."""
    out_dir = tmp_path / "output"
    out_dir.mkdir()
    base = "20260720-085055-kidsong-splish-splash"
    shots_dir = runstate.shots_dir_path(str(out_dir), base)

    _write_wav(out_dir / f"{base}.wav")
    song = {
        "title": "Splish Splash",
        "description": "d",
        "tags": ["t"],
        "characters": "Three toddlers: Zuri; Kofi; Nala",
        "verses": [{"scene": "bath", "lines": ["a", "b"]}],
    }
    (out_dir / f"{base}-song.json").write_text(json.dumps(song), encoding="utf-8")

    shots = []
    for i in range(4):
        shot = {
            "id": f"s{i:02d}", "verse": 0, "start": i * 1.0, "end": (i + 1) * 1.0,
            "shot_type": "medium", "characters": ["all"], "action": "splashing",
            "setting": "a bathroom", "camera": "static", "reuse_of": None,
            "seed": 1000 + i, "status": "planned",
        }
        if i < 2:  # already rendered before the crash
            runstate.mark_rendered(shot, _take(shots_dir, shot["id"], 0), score=0.9, attempts=1)
        shots.append(shot)
    runstate.save_shotlist(str(out_dir), base, {"shots": shots})

    cfg = {
        "_root": str(tmp_path),
        "paths": {"output_dir": str(out_dir)},
        "video": {"max_seconds": 60},
        "captions": {"uppercase": True},
        "kidsong": {
            "mode": "director",
            "singer": "ace",
            "seed": 20260717,
            "shot": {"fps": 24, "max_frames": 241, "hires_pass": False},
            # QC gates are exercised by their own suites; here they would only
            # pull in heavy deps and obscure what resume did.
            "review": {"script": False, "cut": False, "max_retries_per_shot": 0},
        },
    }

    rendered = []

    class FakeComfyClient:
        def __init__(self, cfg):
            self.url = "http://fake:8188"
            self._proc = None
            self.ensure_up_calls = 0

        def ensure_up(self):
            self.ensure_up_calls += 1

        def render(self, workflow, patches, out_path):
            rendered.append(os.path.basename(out_path))
            os.makedirs(os.path.dirname(out_path), exist_ok=True)
            with open(out_path, "wb") as f:
                f.write(b"\0" * 2048)

        def free(self):
            pass

    class FakeReviewer:
        def review(self, shot, video_path, out_dir=None):
            return {"accept": True, "score": 1.0, "reasons": [], "retry_hints": {}}

    import pipeline.captions as captions_mod
    import pipeline.kidsong.comfy as comfy_mod
    import pipeline.kidsong.edit as edit_mod
    import pipeline.kidsong.review as review_mod
    import pipeline.kidsong.sing as sing_mod

    monkeypatch.setattr(comfy_mod, "ComfyClient", FakeComfyClient)
    monkeypatch.setattr(review_mod, "get_reviewer", lambda cfg: FakeReviewer())
    monkeypatch.setattr(review_mod, "contact_sheet", lambda *a, **k: None)
    monkeypatch.setattr(review_mod, "review_shots_log", lambda path, entries: path)
    monkeypatch.setattr(captions_mod, "get_word_timestamps", lambda *a, **k: [])
    monkeypatch.setattr(sing_mod, "verse_times_from_words", lambda *a, **k: [(0.0, 4.0)])
    monkeypatch.setattr(edit_mod, "beat_grid", lambda path: [0.0, 1.0, 2.0, 3.0])
    monkeypatch.setattr(edit_mod, "build_cut_list", lambda *a, **k: [])

    def fake_assemble(cfg, cut_list, renders, voice, words, duration, out_path, **kw):
        with open(out_path, "wb") as f:
            f.write(b"\0" * 4096)

    monkeypatch.setattr(edit_mod, "assemble", fake_assemble)

    def _sing_must_not_run(*a, **k):
        raise AssertionError("resume must not re-sing: it would desync every take")

    monkeypatch.setattr(sing_mod, "sing_song", _sing_must_not_run)

    return {
        "cfg": cfg, "base": base, "out_dir": str(out_dir),
        "shots_dir": shots_dir, "rendered": rendered,
    }


def test_resume_renders_only_the_missing_shots(resume_run):
    from pipeline.kidsong.generate import generate_kidsong

    result = generate_kidsong(cfg=resume_run["cfg"], resume_base=resume_run["base"])

    # s00/s01 were already on disk; only s02/s03 hit the (fake) GPU.
    assert sorted(resume_run["rendered"]) == ["s02_a0.mp4", "s03_a0.mp4"]
    assert result["status"] == "approved"
    assert os.path.exists(result["video_path"])


def test_resume_does_not_destroy_or_overwrite_existing_takes(resume_run):
    from pipeline.kidsong.generate import generate_kidsong

    before = {
        name: os.path.getmtime(os.path.join(resume_run["shots_dir"], name))
        for name in os.listdir(resume_run["shots_dir"])
        if name.endswith(".mp4")
    }
    generate_kidsong(cfg=resume_run["cfg"], resume_base=resume_run["base"])

    for name, mtime in before.items():
        path = os.path.join(resume_run["shots_dir"], name)
        assert os.path.exists(path), f"{name} was deleted"
        assert os.path.getmtime(path) == mtime, f"{name} was overwritten"


def test_resume_writes_a_run_log_next_to_the_run_artifacts(resume_run):
    from pipeline.kidsong.generate import generate_kidsong

    generate_kidsong(cfg=resume_run["cfg"], resume_base=resume_run["base"])

    log_path = os.path.join(resume_run["out_dir"], resume_run["base"] + ".log")
    assert os.path.exists(log_path), "the run log must sit beside the wav/shots.json"
    text = open(log_path, encoding="utf-8").read()
    assert "STAGE resume" in text
    assert "reusing existing take" in text
    assert "STAGE render" in text and "pending=2" in text
    assert "Shot s02 DONE" in text
    assert "STAGE promote" in text


def test_ledger_records_each_shot_so_resume_is_deterministic(resume_run):
    from pipeline.kidsong.generate import generate_kidsong

    generate_kidsong(cfg=resume_run["cfg"], resume_base=resume_run["base"])

    shots = runstate.load_shotlist(resume_run["out_dir"], resume_run["base"])["shots"]
    assert [s["status"] for s in shots] == ["rendered"] * 4
    for s in shots:
        assert os.path.exists(s["take"]) and s["take"].endswith(f"{s['id']}_a0.mp4")
        assert s["attempts"] >= 1


def test_a_completed_run_is_not_offered_for_resume(resume_run):
    """Once a run reaches Final/ it is finished, not resumable.

    A completed run deletes its working wav (pre-existing cleanup), so resume
    both *shouldn't* and *can't* re-enter it — --list-resumable must say so
    rather than letting a scheduler pick it up and fail.
    """
    from pipeline.kidsong.generate import generate_kidsong

    generate_kidsong(cfg=resume_run["cfg"], resume_base=resume_run["base"])

    runs = {r["base"]: r for r in runstate.find_resumable(resume_run["out_dir"])}
    assert runs[resume_run["base"]]["finished"] is True
    assert runs[resume_run["base"]]["resumable"] is False
    assert runs[resume_run["base"]]["unique_pending"] == 0

    # And a human who resumes it anyway gets told why, not a bare file error.
    resume_run["rendered"].clear()
    with pytest.raises(FileNotFoundError, match="already completed"):
        generate_kidsong(cfg=resume_run["cfg"], resume_base=resume_run["base"])
    assert resume_run["rendered"] == []


def test_crash_mid_render_persists_the_ledger_and_logs_a_traceback(resume_run, monkeypatch):
    """The bath-time scenario: die on a shot, keep everything before it."""
    import pipeline.kidsong.comfy as comfy_mod
    from pipeline.kidsong.generate import generate_kidsong

    real_render = comfy_mod.ComfyClient.render

    def exploding_render(self, workflow, patches, out_path):
        if "s03" in out_path:
            raise TimeoutError("ComfyUI render exceeded 1800s for prompt deadbeef")
        return real_render(self, workflow, patches, out_path)

    monkeypatch.setattr(comfy_mod.ComfyClient, "render", exploding_render)

    with pytest.raises(TimeoutError):
        generate_kidsong(cfg=resume_run["cfg"], resume_base=resume_run["base"])

    out_dir, base = resume_run["out_dir"], resume_run["base"]

    # s02 finished before the crash and is committed to the ledger.
    shots = {s["id"]: s for s in runstate.load_shotlist(out_dir, base)["shots"]}
    assert shots["s02"]["status"] == "rendered"
    assert shots["s03"]["status"] != "rendered"
    assert shots["s03"]["attempts"] == 1  # the failure was charged to s03

    # The traceback landed in both logs.
    run_log = open(os.path.join(out_dir, base + ".log"), encoding="utf-8").read()
    err_log = open(os.path.join(out_dir, ERROR_LOG_NAME), encoding="utf-8").read()
    assert "ComfyUI render exceeded 1800s" in run_log
    assert "Traceback (most recent call last)" in run_log
    assert "Render loop aborted" in run_log
    assert "ComfyUI render exceeded 1800s" in err_log
    assert f"base={base}" in err_log

    # And the run is resumable: only s03 is still outstanding.
    runs = {r["base"]: r for r in runstate.find_resumable(out_dir)}
    assert runs[base]["unique_done"] == 3 and runs[base]["unique_pending"] == 1


def test_a_repeatedly_failing_shot_is_eventually_marked_failed(resume_run, monkeypatch):
    """Resume must not retry a doomed shot forever."""
    import pipeline.kidsong.comfy as comfy_mod
    from pipeline.kidsong.generate import generate_kidsong

    resume_run["cfg"]["kidsong"]["review"]["max_attempts_per_shot"] = 2
    real_render = comfy_mod.ComfyClient.render

    def explode_on_s02(self, workflow, patches, out_path):
        if "s02" in out_path:
            raise RuntimeError("this shot always dies")
        return real_render(self, workflow, patches, out_path)

    monkeypatch.setattr(comfy_mod.ComfyClient, "render", explode_on_s02)

    for _ in range(2):
        with pytest.raises(RuntimeError):
            generate_kidsong(cfg=resume_run["cfg"], resume_base=resume_run["base"])

    shots = {s["id"]: s for s in
             runstate.load_shotlist(resume_run["out_dir"], resume_run["base"])["shots"]}
    assert shots["s02"]["status"] == runstate.STATUS_FAILED
    assert "this shot always dies" in shots["s02"]["failure_reason"]

    # A further resume skips it instead of looping, and still finishes the run.
    resume_run["rendered"].clear()
    result = generate_kidsong(cfg=resume_run["cfg"], resume_base=resume_run["base"])
    assert "s02_a0.mp4" not in resume_run["rendered"]
    assert result["status"] == "approved"


def test_resume_refuses_when_the_run_state_is_missing(tmp_path):
    from pipeline.kidsong.generate import generate_kidsong

    cfg = {"_root": str(tmp_path), "paths": {"output_dir": str(tmp_path)},
           "video": {"max_seconds": 60}, "captions": {}, "kidsong": {"mode": "director"}}

    with pytest.raises(FileNotFoundError, match="Cannot resume"):
        generate_kidsong(cfg=cfg, resume_base="no-such-run")


# ========================================== QC-honest resume (rejected takes) ===
# A resume must never reuse a take the producer turned down. Before this suite
# existed, generate.py called mark_rendered() unconditionally after the retry
# loop, so an all-rejected shot was recorded `rendered` and every later resume
# logged "reusing existing take" for footage that had failed QC. The second half
# of the same bug: recover_takes() adopted the highest-index take on disk, which
# for a shot that was still retrying is precisely the last rejected attempt.
def _write_review_log(shots_dir, entries):
    """entries: [(shot_id, attempt, accept, score)] -> review/review_log.json"""
    review_dir = os.path.join(shots_dir, "review")
    os.makedirs(review_dir, exist_ok=True)
    payload = [
        {"shot": sid, "attempt": n, "seed": 1,
         "verdict": {"accept": acc, "score": sc, "reasons": [], "retry_hints": {}}}
        for sid, n, acc, sc in entries
    ]
    path = os.path.join(review_dir, "review_log.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f)
    return path


def _write_response(shots_dir, take, accept, score):
    review_dir = os.path.join(shots_dir, "review")
    os.makedirs(review_dir, exist_ok=True)
    path = os.path.join(review_dir, f"{take}.response.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"accept": accept, "score": score, "reasons": []}, f)
    return path


def test_a_rejected_take_never_counts_as_a_finished_shot(tmp_path):
    sdir = str(tmp_path / "shots")
    take = _take(sdir, "s02", 0)
    shot = {"id": "s02"}
    runstate.mark_rendered(shot, take, score=0.2, attempts=1,
                           verdict=runstate.VERDICT_REJECTED)

    assert shot["status"] == runstate.STATUS_REJECTED
    assert shot["take"] == take, "the take stays on record so the cut is never blocked"
    assert runstate.shot_is_done(shot, sdir) is False


def test_a_ledger_written_by_the_buggy_build_is_not_trusted(tmp_path):
    """status=rendered + last_verdict=rejected is what the bug wrote to disk."""
    sdir = str(tmp_path / "shots")
    shot = {"id": "s09", "status": runstate.STATUS_RENDERED,
            "take": _take(sdir, "s09", 1), "last_verdict": "rejected"}
    assert runstate.shot_is_done(shot, sdir) is False


def test_recover_takes_prefers_the_accepted_take_over_the_newest(tmp_path):
    """The highest-index take is typically the LAST REJECTED attempt."""
    sdir = str(tmp_path / "shots")
    _take(sdir, "s05", 0)
    accepted = _take(sdir, "s05", 1)
    _take(sdir, "s05", 2)  # newest, and rejected — and the higher raw score
    _write_review_log(sdir, [("s05", 0, False, 0.1), ("s05", 1, True, 0.8),
                             ("s05", 2, False, 0.9)])

    shotlist = {"shots": [{"id": "s05", "status": "planned"}]}
    assert runstate.recover_takes(shotlist, sdir) == 1

    shot = shotlist["shots"][0]
    assert shot["take"] == accepted, "must adopt the accepted take, not the newest"
    assert shot["verdict"] == runstate.VERDICT_ACCEPTED
    assert runstate.shot_is_done(shot, sdir) is True


def test_best_take_reads_verdicts_from_response_files_too(tmp_path):
    sdir = str(tmp_path / "shots")
    _take(sdir, "s06", 0)
    good = _take(sdir, "s06", 1)
    _write_response(sdir, "s06_a0", False, 0.3)
    _write_response(sdir, "s06_a1", True, 0.6)

    assert runstate.best_take(sdir, "s06") == (good, runstate.VERDICT_ACCEPTED, 0.6)


def test_recover_takes_falls_back_to_the_best_score_when_nothing_was_accepted(tmp_path):
    sdir = str(tmp_path / "shots")
    _take(sdir, "s07", 0)
    best = _take(sdir, "s07", 1)
    _take(sdir, "s07", 2)
    _write_review_log(sdir, [("s07", 0, False, 0.1), ("s07", 1, False, 0.75),
                             ("s07", 2, False, 0.2)])

    shotlist = {"shots": [{"id": "s07", "status": "planned"}]}
    recovered = runstate.recover_takes(shotlist, sdir)

    shot = shotlist["shots"][0]
    assert shot["take"] == best, "best-scoring rejected take, not the newest"
    assert shot["verdict"] == runstate.VERDICT_REJECTED
    assert shot["status"] == runstate.STATUS_REJECTED
    assert runstate.shot_is_done(shot, sdir) is False, "still needs a re-render"
    assert recovered == 0, "adopting a rejected take is not recovered work"


def test_recover_takes_adopts_unverified_media_without_claiming_it_passed(tmp_path):
    """Pre-ledger runs have no verdicts at all — adopt, but say so honestly."""
    sdir = str(tmp_path / "shots")
    _take(sdir, "s00", 0)
    newest = _take(sdir, "s00", 1)

    shotlist = {"shots": [{"id": "s00", "status": "planned"}]}
    assert runstate.recover_takes(shotlist, sdir) == 1

    shot = shotlist["shots"][0]
    assert shot["take"] == newest
    assert shot["verdict"] == runstate.VERDICT_UNKNOWN
    assert runstate.shot_is_done(shot, sdir) is True, "never re-render a whole legacy run"


def test_load_take_verdicts_survives_missing_and_broken_files(tmp_path):
    sdir = str(tmp_path / "shots")
    os.makedirs(os.path.join(sdir, "review"), exist_ok=True)
    assert runstate.load_take_verdicts(str(tmp_path / "nope")) == {}
    with open(os.path.join(sdir, "review", "review_log.json"), "w", encoding="utf-8") as f:
        f.write("{not json")
    assert runstate.load_take_verdicts(sdir) == {}


def test_stage_verdicts_in_the_review_log_are_not_mistaken_for_takes(tmp_path):
    sdir = str(tmp_path / "shots")
    os.makedirs(os.path.join(sdir, "review"), exist_ok=True)
    with open(os.path.join(sdir, "review", "review_log.json"), "w", encoding="utf-8") as f:
        json.dump([{"stage": "cut", "attempt": 0,
                    "verdict": {"accept": False, "score": 0.0}}], f)
    assert runstate.load_take_verdicts(sdir) == {}


def test_ledger_round_trips_the_verdict(tmp_path):
    shot = {"id": "s01"}
    runstate.mark_rendered(shot, "x.mp4", score=0.42, attempts=3,
                           verdict=runstate.VERDICT_REJECTED)
    runstate.save_shotlist(str(tmp_path), "b", {"shots": [shot]})

    loaded = runstate.load_shotlist(str(tmp_path), "b")["shots"][0]
    assert loaded == shot
    assert (loaded["status"], loaded["verdict"], loaded["score"], loaded["attempts"]) == (
        runstate.STATUS_REJECTED, runstate.VERDICT_REJECTED, 0.42, 3
    )


# ----------------------------------------------- end to end through generate ---
@pytest.fixture()
def rejecting_run(resume_run, monkeypatch):
    """The same half-finished run, but the producer rejects s02 every time."""
    import pipeline.kidsong.review as review_mod

    # Real verdict persistence: the next resume must be able to read back what
    # QC said, so this is the path under test, not the fixture's no-op stub.
    monkeypatch.setattr(review_mod, "review_shots_log", _real_review_shots_log)

    class PickyReviewer:
        def review(self, shot, video_path, out_dir=None):
            if shot["id"] == "s02":
                return {"accept": False, "score": 0.25,
                        "reasons": ["duplicate characters"], "retry_hints": {}}
            return {"accept": True, "score": 1.0, "reasons": [], "retry_hints": {}}

    monkeypatch.setattr(review_mod, "get_reviewer", lambda cfg: PickyReviewer())
    return resume_run


def test_a_shot_rejected_on_every_attempt_stays_pending_for_the_next_resume(rejecting_run):
    from pipeline.kidsong.generate import generate_kidsong

    generate_kidsong(cfg=rejecting_run["cfg"], resume_base=rejecting_run["base"])

    shots = {s["id"]: s for s in
             runstate.load_shotlist(rejecting_run["out_dir"], rejecting_run["base"])["shots"]}
    assert shots["s02"]["status"] == runstate.STATUS_REJECTED
    assert shots["s02"]["verdict"] == runstate.VERDICT_REJECTED
    assert shots["s03"]["status"] == runstate.STATUS_RENDERED
    # ...and the ledger no longer carries the dead field the bug used to write.
    assert "last_verdict" not in shots["s02"]
    assert runstate.shot_is_done(shots["s02"], rejecting_run["shots_dir"]) is False


def test_the_rejected_shot_is_re_rendered_and_the_accepted_one_is_not(rejecting_run):
    from pipeline.kidsong.generate import generate_kidsong

    generate_kidsong(cfg=rejecting_run["cfg"], resume_base=rejecting_run["base"])
    rejecting_run["rendered"].clear()

    generate_kidsong(cfg=rejecting_run["cfg"], resume_base=rejecting_run["base"])

    redone = rejecting_run["rendered"]
    assert redone, "the rejected shot must be re-rendered on resume"
    assert all(name.startswith("s02_") for name in redone), redone
    # s03 was accepted first time round: reused, never re-rendered.
    assert not any(name.startswith("s03_") for name in redone), redone
    # And nothing was overwritten — the retry took the next free index.
    assert "s02_a0.mp4" not in redone


def test_the_rejected_shot_still_gets_cut_into_the_episode(rejecting_run):
    """Never block the episode: the best available take covers the slot."""
    from pipeline.kidsong.generate import generate_kidsong

    result = generate_kidsong(cfg=rejecting_run["cfg"], resume_base=rejecting_run["base"])

    assert os.path.exists(result["video_path"])
    log = open(os.path.join(rejecting_run["out_dir"], rejecting_run["base"] + ".log"),
               encoding="utf-8").read()
    assert "Shot s02 REJECTED" in log, log[-2000:]
    assert "s02_a0.mp4" in log


def test_the_attempt_cap_still_terminates_a_hopeless_shot(rejecting_run):
    """A shot nothing can satisfy must end up failed, not retried forever."""
    from pipeline.kidsong.generate import generate_kidsong

    rejecting_run["cfg"]["kidsong"]["review"]["max_attempts_per_shot"] = 2

    # One take per run (max_retries_per_shot=0), so the budget of 2 is spent by
    # the second resume — and s02 is then failed, not silently "rendered".
    generate_kidsong(cfg=rejecting_run["cfg"], resume_base=rejecting_run["base"])
    result = generate_kidsong(cfg=rejecting_run["cfg"], resume_base=rejecting_run["base"])

    shots = {s["id"]: s for s in
             runstate.load_shotlist(rejecting_run["out_dir"], rejecting_run["base"])["shots"]}
    assert shots["s02"]["status"] == runstate.STATUS_FAILED
    assert shots["s02"]["attempts"] == 2
    assert "QC rejected all" in shots["s02"]["failure_reason"]
    # Its (unapproved) take still covered the slot, so the episode was cut.
    assert shots["s02"]["take"].endswith("s02_a1.mp4")
    assert os.path.exists(result["video_path"])

    # The loop terminates: with nothing left that may be retried, the run is
    # over and a further resume is refused instead of burning more GPU time.
    rejecting_run["rendered"].clear()
    with pytest.raises(FileNotFoundError, match="already completed"):
        generate_kidsong(cfg=rejecting_run["cfg"], resume_base=rejecting_run["base"])
    assert rejecting_run["rendered"] == []
    assert runstate.find_resumable(rejecting_run["out_dir"])[0]["resumable"] is False


def test_verdicts_are_flushed_per_shot_not_only_at_the_end_of_the_run(
    resume_run, monkeypatch
):
    """Runs killed mid-render lost their whole review_log.json: verdict history
    was buffered in memory and written only in the render loop's finally block.
    """
    import pipeline.kidsong.comfy as comfy_mod
    import pipeline.kidsong.review as review_mod
    from pipeline.kidsong.generate import generate_kidsong

    # undo the fixture's stub — this test is about what reaches disk
    monkeypatch.setattr(review_mod, "review_shots_log", _real_review_shots_log)

    real_render = comfy_mod.ComfyClient.render

    def die_on_s03(self, workflow, patches, out_path):
        if "s03" in out_path:
            raise KeyboardInterrupt("operator killed the run")
        return real_render(self, workflow, patches, out_path)

    monkeypatch.setattr(comfy_mod.ComfyClient, "render", die_on_s03)

    with pytest.raises(KeyboardInterrupt):
        generate_kidsong(cfg=resume_run["cfg"], resume_base=resume_run["base"])

    log_path = os.path.join(resume_run["shots_dir"], "review", "review_log.json")
    assert os.path.exists(log_path), "s02's verdict must be on disk before s03 ran"
    entries = json.load(open(log_path, encoding="utf-8"))
    assert [e["shot"] for e in entries] == ["s02"]
    assert entries[0]["verdict"]["accept"] is True

    # And that persisted verdict is what the next resume reads back.
    assert runstate.load_take_verdicts(resume_run["shots_dir"])["s02_a0"]["accept"] is True


# ================================= promotion never clobbers channel inventory ===
# output/ is channel inventory: a finished, possibly already-uploaded episode
# must never be replaced by a later re-cut of the same <base>. A resume re-cuts
# the same base, so Final/<base>.mp4 is precisely the name it would land on.
def test_reserve_output_path_uses_the_plain_name_when_it_is_free(tmp_path):
    target = str(tmp_path / "ep.mp4")
    assert runstate.reserve_output_path(target) == target


def test_reserve_output_path_versions_from_v2_upward(tmp_path):
    target = tmp_path / "ep.mp4"
    target.write_bytes(b"first")
    assert runstate.reserve_output_path(str(target)) == str(tmp_path / "ep-v2.mp4")

    (tmp_path / "ep-v2.mp4").write_bytes(b"second")
    assert runstate.reserve_output_path(str(target)) == str(tmp_path / "ep-v3.mp4")


def test_reserve_output_path_also_avoids_an_orphaned_companion_sidecar(tmp_path):
    """A leftover <video>.intro.json must not be adopted by a different cut."""
    target = tmp_path / "ep.mp4"
    (tmp_path / "ep.mp4.intro.json").write_text("{}", encoding="utf-8")
    assert runstate.reserve_output_path(
        str(target), companions=(".intro.json",)
    ) == str(tmp_path / "ep-v2.mp4")


def test_reserve_output_path_refuses_rather_than_overwriting(tmp_path):
    target = tmp_path / "ep.mp4"
    target.write_bytes(b"x")
    (tmp_path / "ep-v2.mp4").write_bytes(b"x")
    with pytest.raises(RuntimeError, match="Refusing to overwrite"):
        runstate.reserve_output_path(str(target), max_versions=2)


# ============================== remove_final_episode (sanctioned removal) =====
# 3 shipped finals were hand-deleted for disk space, stranding orphaned
# <base>.mp4.intro.json sidecars that permanently burn the base name (see the
# _free() trap covered above). remove_final_episode() is the only sanctioned
# way to take a shipped final out of Final/: it removes the mp4 and its
# sidecar(s) together, so no orphan is ever left behind.
def test_remove_final_episode_removes_mp4_and_sidecar_together(tmp_path):
    final_dir = tmp_path / "Final"
    final_dir.mkdir()
    mp4 = final_dir / "ep.mp4"
    sidecar = final_dir / "ep.mp4.intro.json"
    mp4.write_bytes(b"video")
    sidecar.write_text("{}", encoding="utf-8")

    removed = runstate.remove_final_episode(str(mp4))

    assert not mp4.exists()
    assert not sidecar.exists()
    assert str(mp4) in removed and str(sidecar) in removed


def test_remove_final_episode_tolerates_a_missing_sidecar(tmp_path):
    """Some finals never got an intro sidecar — removal must not choke on that."""
    final_dir = tmp_path / "Final"
    final_dir.mkdir()
    mp4 = final_dir / "ep.mp4"
    mp4.write_bytes(b"video")

    removed = runstate.remove_final_episode(str(mp4))

    assert not mp4.exists()
    assert removed == [str(mp4)]


def test_remove_final_episode_refuses_a_non_final_path(tmp_path):
    staging_dir = tmp_path / "staging"
    staging_dir.mkdir()
    mp4 = staging_dir / "ep.mp4"
    mp4.write_bytes(b"video")

    with pytest.raises(ValueError, match="Final"):
        runstate.remove_final_episode(str(mp4))

    assert mp4.exists(), "refused removal must not touch the file"


def test_remove_final_episode_frees_the_name_reserve_output_path_would_otherwise_burn(tmp_path):
    """The exact incident this exists for: an orphaned sidecar with no mp4
    makes _free() report the name taken forever. Removing both together (this
    helper) is what prevents that; hand-deleting only the mp4 would not."""
    final_dir = tmp_path / "Final"
    final_dir.mkdir()
    mp4 = final_dir / "ep.mp4"
    sidecar = final_dir / "ep.mp4.intro.json"
    mp4.write_bytes(b"video")
    sidecar.write_text("{}", encoding="utf-8")

    # Hand-deleting only the mp4 (what actually happened) strands the sidecar.
    os.remove(str(mp4))
    assert runstate.reserve_output_path(
        str(mp4), companions=(".intro.json",)
    ) != str(mp4), "premise: orphaned sidecar burns the base name"

    sidecar.write_text("{}", encoding="utf-8")  # restore to exercise the helper
    mp4.write_bytes(b"video")
    runstate.remove_final_episode(str(mp4))
    assert runstate.reserve_output_path(str(mp4), companions=(".intro.json",)) == str(mp4)


def _final_bytes(out_dir, base):
    p = os.path.join(out_dir, "Final", base + ".mp4")
    return open(p, "rb").read(), os.path.getmtime(p)


def test_a_first_promotion_uses_the_plain_name(resume_run):
    from pipeline.kidsong.generate import generate_kidsong

    result = generate_kidsong(cfg=resume_run["cfg"], resume_base=resume_run["base"])

    assert os.path.basename(result["video_path"]) == resume_run["base"] + ".mp4"
    assert os.path.exists(result["video_path"])


def test_a_resume_never_overwrites_an_already_promoted_episode(rejecting_run):
    """s02 is rejected every time, so the run promotes AND stays resumable."""
    from pipeline.kidsong.generate import generate_kidsong

    out_dir, base = rejecting_run["out_dir"], rejecting_run["base"]
    first = generate_kidsong(cfg=rejecting_run["cfg"], resume_base=base)
    assert os.path.basename(first["video_path"]) == base + ".mp4"
    original_bytes, original_mtime = _final_bytes(out_dir, base)

    second = generate_kidsong(cfg=rejecting_run["cfg"], resume_base=base)

    # The re-cut went to a versioned sibling...
    assert os.path.basename(second["video_path"]) == base + "-v2.mp4"
    assert os.path.exists(second["video_path"])
    # ...and the approved episode is byte-identical and untouched.
    assert _final_bytes(out_dir, base) == (original_bytes, original_mtime)

    # A third promotion versions again rather than reusing -v2.
    third = generate_kidsong(cfg=rejecting_run["cfg"], resume_base=base)
    assert os.path.basename(third["video_path"]) == base + "-v3.mp4"
    assert _final_bytes(out_dir, base) == (original_bytes, original_mtime)
    assert os.path.exists(os.path.join(out_dir, "Final", base + "-v2.mp4"))


def test_the_returned_path_and_the_logged_path_are_the_file_on_disk(rejecting_run):
    from pipeline.kidsong.generate import generate_kidsong

    out_dir, base = rejecting_run["out_dir"], rejecting_run["base"]
    generate_kidsong(cfg=rejecting_run["cfg"], resume_base=base)
    result = generate_kidsong(cfg=rejecting_run["cfg"], resume_base=base)

    log = open(os.path.join(out_dir, base + ".log"), encoding="utf-8").read()
    assert f"promoting to {base}-v2.mp4" in log
    assert f"STAGE promote path={result['video_path']}" in log
    assert os.path.exists(result["video_path"])
    # Downstream consumers (upload, intro sidecar) follow the real name.
    assert result["video_path"].endswith(f"{base}-v2.mp4")


def test_a_rejected_cut_is_not_overwritten_by_the_recut(resume_run, monkeypatch):
    """The staging file is evidence: the recut gets its own name."""
    import pipeline.kidsong.cut_qc as cut_qc_mod
    from pipeline.kidsong.generate import generate_kidsong

    resume_run["cfg"]["kidsong"]["review"]["cut"] = True
    verdicts = iter([
        {"accept": False, "score": 0.1, "reasons": ["off beat"],
         "retry_hints": {"recut": True}},
        {"accept": True, "score": 0.9, "reasons": [], "retry_hints": {}},
    ])
    monkeypatch.setattr(cut_qc_mod, "review_cut",
                        lambda *a, **k: next(verdicts))

    result = generate_kidsong(cfg=resume_run["cfg"], resume_base=resume_run["base"])

    out_dir, base = resume_run["out_dir"], resume_run["base"]
    # The rejected first cut survives; the accepted recut was promoted.
    assert os.path.exists(os.path.join(out_dir, base + ".staging.mp4"))
    assert os.path.basename(result["video_path"]) == base + ".mp4"


def test_backfilled_response_outranks_the_run_log_verdict(tmp_path):
    """The QM-010 recovery path: an unattended run's review_log carries the
    TIMEOUT-HEURISTIC verdict (accept=true) for a take the vision reviewer
    later rejected via a backfilled response.json. The response must WIN —
    before this alignment the log entry laundered the rejection back to
    accepted on resume, and the morphed take was re-adopted instead of
    re-rendered (measured live, storymode render 2026-07-24)."""
    sdir = str(tmp_path / "shots")
    _write_review_log(sdir, [("s09", 0, True, 1.0)])
    _write_response(sdir, "s09_a0", accept=False, score=0.3)

    verdicts = runstate.load_take_verdicts(sdir)
    assert verdicts["s09_a0"]["accept"] is False

    # And the inverse backfill (vision APPROVING a take the heuristic only
    # tolerated) wins too:
    _write_response(sdir, "s09_a0", accept=True, score=0.9)
    verdicts = runstate.load_take_verdicts(sdir)
    assert verdicts["s09_a0"]["accept"] is True
