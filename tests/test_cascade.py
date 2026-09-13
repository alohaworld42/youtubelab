"""Cascade: channel bootstrap -> ideas -> queued kidsong jobs.

No network, no GPU: the LLM completion is monkeypatched everywhere.
"""
import json

import pytest

from studio import cascade, models
from studio import ideas as ideas_mod


CFG = {"llm": {"backend": "groq", "groq_model": "x"}}


def stub_llm(monkeypatch, topics=None, error=None):
    """Replace pipeline.script_gen.complete (imported lazily inside ideas)."""
    import pipeline.script_gen as sg

    def fake_complete(user, cfg, system=None):
        if error:
            raise error
        payload = {"ideas": [{"topic": t, "video_type": "kidsong", "hook": "la"}
                             for t in (topics or [])]}
        return json.dumps(payload)

    monkeypatch.setattr(sg, "complete", fake_complete)


# ------------------------------------------------------------------ channel --
def test_ensure_kidsong_channel_creates_it(tmp_db):
    ch = cascade.ensure_kidsong_channel(CFG)
    assert ch["default_video_type"] == "kidsong"
    assert ch["status"] == "active"
    assert ch["made_for_kids"] == 1
    assert ch["schedule"] == []  # no time-slot cron double-firing
    assert ch["niche"] and ch["description"]


def test_ensure_kidsong_channel_is_idempotent(tmp_db):
    first = cascade.ensure_kidsong_channel(CFG)
    second = cascade.ensure_kidsong_channel(CFG)
    assert first["id"] == second["id"]
    assert len(models.list_channels()) == 1


def test_ensure_kidsong_channel_ignores_other_channels(tmp_db):
    other = models.create_channel("Brainrot")
    ch = cascade.ensure_kidsong_channel(CFG)
    assert ch["id"] != other
    assert ch["default_video_type"] == "kidsong"


# ------------------------------------------------------------------- ideas --
def test_kidsong_prompt_used_for_kidsong_channel(tmp_db, monkeypatch):
    import pipeline.script_gen as sg

    captured = {}

    def fake_complete(user, cfg, system=None):
        captured["user"] = user
        captured["system"] = system
        return json.dumps({"ideas": [{"topic": "brushing teeth"}]})

    monkeypatch.setattr(sg, "complete", fake_complete)
    ch = cascade.ensure_kidsong_channel(CFG)
    inserted = ideas_mod.generate_ideas(ch, CFG, k=3, video_type="kidsong")

    assert inserted == 1
    assert "sing-along song subjects" in captured["user"]
    assert "preschool" in captured["system"]
    # the idea inherits the kidsong type even though the LLM omitted it
    idea = models.next_pending_idea(ch["id"])
    assert idea["video_type"] == "kidsong"


def test_non_kidsong_channel_keeps_old_prompt(tmp_db, monkeypatch):
    import pipeline.script_gen as sg

    captured = {}

    def fake_complete(user, cfg, system=None):
        captured["user"] = user
        return json.dumps({"ideas": [{"topic": "creepy lighthouse", "video_type": "scary"}]})

    monkeypatch.setattr(sg, "complete", fake_complete)
    cid = models.create_channel("Brainrot")
    ch = models.get_channel(cid)
    assert ideas_mod.generate_ideas(ch, CFG, k=2) == 1
    assert "fresh video ideas" in captured["user"]
    assert models.next_pending_idea(cid)["video_type"] == "scary"


def test_fallback_topics_dedupe(tmp_db):
    ch = cascade.ensure_kidsong_channel(CFG)
    assert ideas_mod.add_fallback_kidsong_ideas(ch, 3) == 3
    before = len(models.list_ideas(channel_id=ch["id"]))
    ideas_mod.add_fallback_kidsong_ideas(ch, 3)
    after = len(models.list_ideas(channel_id=ch["id"]))
    assert after == before + 3  # new topics, no repeats of the first three
    topics = [i["topic"] for i in models.list_ideas(channel_id=ch["id"])]
    assert len(set(topics)) == len(topics)


# ----------------------------------------------------------------- cascade --
def test_run_cascade_creates_jobs_linked_to_ideas(tmp_db, monkeypatch):
    stub_llm(monkeypatch, ["brushing teeth", "counting to ten", "colors of the rainbow",
                           "sharing toys", "washing hands"])
    summary = cascade.run_cascade(CFG, count=3)

    assert summary["jobs_created"] == 3
    assert len(summary["job_ids"]) == 3
    assert summary["fallback_used"] is False
    assert "skipped_reason" not in summary

    for job_id in summary["job_ids"]:
        job = models.get_job(job_id)
        assert job["video_type"] == "kidsong"
        assert job["status"] == "queued"
        assert job["channel_id"] == summary["channel_id"]
        assert job["idea_id"] is not None
        assert job["topic"]
        assert job["slot_key"] is None

    # every consumed idea is flipped so it is never picked twice
    used = models.list_ideas(channel_id=summary["channel_id"], status="in_progress")
    assert len(used) == 3
    topics = {models.get_job(j)["topic"] for j in summary["job_ids"]}
    assert topics == {i["topic"] for i in used}


def test_run_cascade_never_auto_approves(tmp_db, monkeypatch):
    stub_llm(monkeypatch, ["brushing teeth", "counting to ten"])
    summary = cascade.run_cascade(CFG, count=2)
    assert summary["jobs_created"] == 2
    for job_id in summary["job_ids"]:
        assert models.get_job(job_id)["auto_approve"] == 0


def test_run_cascade_reuses_existing_pending_ideas(tmp_db, monkeypatch):
    ch = cascade.ensure_kidsong_channel(CFG)
    ideas_mod.add_user_idea(ch["id"], "my own song about socks", video_type="kidsong")
    ideas_mod.add_user_idea(ch["id"], "a song about the moon", video_type="kidsong")

    def boom(*a, **kw):
        raise AssertionError("LLM must not be called when enough ideas are pending")

    monkeypatch.setattr(ideas_mod, "generate_ideas", boom)
    summary = cascade.run_cascade(CFG, count=2)
    assert summary["jobs_created"] == 2
    assert summary["ideas_generated"] == 0
    # user ideas win
    assert {models.get_job(j)["topic"] for j in summary["job_ids"]} == {
        "my own song about socks", "a song about the moon"}


@pytest.mark.parametrize("given,expected", [
    (0, 1), (-5, 1), (1, 1), (7, 7), (11, 10), (999, 10),
    (None, 3), ("abc", 3), ("4", 4),
])
def test_clamp_count(given, expected):
    assert cascade.clamp_count(given) == expected


def test_run_cascade_clamps_count(tmp_db, monkeypatch):
    stub_llm(monkeypatch, [])  # LLM returns nothing -> fallback tops up
    summary = cascade.run_cascade(CFG, count=99)
    assert summary["requested"] == 10
    assert summary["jobs_created"] == 10


def test_run_cascade_falls_back_when_llm_unreachable(tmp_db, monkeypatch):
    stub_llm(monkeypatch, error=RuntimeError("connection refused"))
    summary = cascade.run_cascade(CFG, count=3)

    assert summary["fallback_used"] is True
    assert "connection refused" in summary["llm_error"]
    assert summary["jobs_created"] == 3
    for job_id in summary["job_ids"]:
        job = models.get_job(job_id)
        assert job["topic"] in ideas_mod.KIDSONG_FALLBACK_TOPICS
        assert job["auto_approve"] == 0


def test_run_cascade_stops_when_ideas_run_out(tmp_db, monkeypatch):
    """No LLM ideas and the fallback pool exhausted -> partial run, no crash."""
    stub_llm(monkeypatch, error=RuntimeError("offline"))
    monkeypatch.setattr(ideas_mod, "KIDSONG_FALLBACK_TOPICS", ["only one safe topic"])
    summary = cascade.run_cascade(CFG, count=4)
    assert summary["jobs_created"] == 1
    assert "skipped_reason" in summary


def test_run_cascade_honours_explicit_channel(tmp_db, monkeypatch):
    stub_llm(monkeypatch, ["brushing teeth"])
    cid = models.create_channel("Mein Kanal")
    models.update_channel(cid, default_video_type="kidsong")
    summary = cascade.run_cascade(CFG, count=1, channel_id=cid)
    assert summary["channel_id"] == cid
    assert len(models.list_channels()) == 1  # no extra channel created


# --------------------------------------------------------------- HTTP layer --
@pytest.fixture()
def client(tmp_db):
    from app import create_app

    app = create_app()
    app.config["TESTING"] = True
    with app.test_client() as c:
        yield c


def test_api_cascade_returns_summary(client, monkeypatch):
    stub_llm(monkeypatch, ["brushing teeth", "counting to ten"])
    res = client.post("/api/cascade", json={"count": 2})
    assert res.status_code == 200
    body = res.get_json()
    assert body["jobs_created"] == 2
    assert len(body["job_ids"]) == 2
    assert body["channel_id"]
    for job_id in body["job_ids"]:
        assert models.get_job(job_id)["auto_approve"] == 0


def test_api_cascade_defaults_and_clamps(client, monkeypatch):
    stub_llm(monkeypatch, [])
    res = client.post("/api/cascade", json={"count": 500})
    assert res.status_code == 200
    assert res.get_json()["requested"] == 10

    res = client.post("/api/cascade", json={})
    assert res.status_code == 200
    assert res.get_json()["requested"] == 3


def test_api_cascade_survives_dead_llm(client, monkeypatch):
    stub_llm(monkeypatch, error=RuntimeError("no API key"))
    res = client.post("/api/cascade", json={"count": 1})
    assert res.status_code == 200
    body = res.get_json()
    assert body["fallback_used"] is True
    assert body["jobs_created"] == 1


def test_api_cascade_empty_body_does_not_500(client, monkeypatch):
    stub_llm(monkeypatch, ["brushing teeth"])
    res = client.post("/api/cascade", data="", content_type="application/json")
    assert res.status_code == 200
    assert res.get_json()["requested"] == 3


def test_dashboard_renders_jobs_after_cascade(client, monkeypatch):
    stub_llm(monkeypatch, ["brushing teeth"])
    client.post("/api/cascade", json={"count": 1})
    res = client.get("/")
    assert res.status_code == 200
    html = res.get_data(as_text=True)
    assert "Kaskade starten" in html
    assert "brushing teeth" in html
    assert "queued" in html
    assert "/kidsong-review" in html
    assert "/dashboard" in html


def test_status_endpoint_shape_for_polling(client, monkeypatch):
    stub_llm(monkeypatch, ["brushing teeth"])
    client.post("/api/cascade", json={"count": 1})
    body = client.get("/api/status").get_json()
    assert body["current_job"] is None  # queued, not started (no scheduler in tests)
    assert body["counts"]["queued"] == 1
