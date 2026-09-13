"""Regression suite for the children's-content gate.

Incident: the real channel KinderKrippenLernLieder was repointed from
default_video_type='facts' to 'kidsong'. Its old, pending brainrot ideas stayed
eligible and the cascade queued five of them as toddler sing-alongs, including
"Germany's scariest folkloric creatures come to life".

Everything here is DB-only: temp STUDIO_DB_PATH, LLM stubbed, no network, no GPU.
"""
import json

import pytest

from studio import cascade, models
from studio import ideas as ideas_mod


CFG = {"llm": {"backend": "groq", "groq_model": "x"}}

# The five topics that were actually queued against the toddler channel.
REAL_INCIDENT_TOPICS = [
    "Germany's scariest folkloric creatures come to life",
    "People share the weirdest things they thought were normal growing up in Germany",
    "Bizarre German festivals that will leave you scratching your head",
    "The most obscure German word that has an English equivalent",
    "Germany's scariest folkloric creatures",
]

WHOLESOME_TOPICS = [
    "Washing Little Hands",
    "Counting Clouds at Sunset",
    "Sharing Sunflower Seeds",
    "Baa Baa Black Sheep",
]


def stub_llm(monkeypatch, topics=None, error=None):
    import pipeline.script_gen as sg

    def fake_complete(user, cfg, system=None):
        if error:
            raise error
        payload = {"ideas": [{"topic": t, "video_type": "kidsong", "hook": "la"}
                             for t in (topics or [])]}
        return json.dumps(payload)

    monkeypatch.setattr(sg, "complete", fake_complete)


# ------------------------------------------------------------ the gate itself --
@pytest.mark.parametrize("topic", REAL_INCIDENT_TOPICS)
def test_real_incident_topics_are_rejected(topic):
    reason = ideas_mod.kidsong_topic_rejection(topic)
    assert reason, "UNSAFE topic accepted for a toddler channel: %r" % topic
    assert not ideas_mod.is_kidsong_topic_safe(topic)


@pytest.mark.parametrize("topic", WHOLESOME_TOPICS)
def test_wholesome_topics_are_accepted(topic):
    assert ideas_mod.kidsong_topic_rejection(topic) is None, (
        "safe topic wrongly rejected: %r" % topic
    )


@pytest.mark.parametrize("topic", ideas_mod.KIDSONG_FALLBACK_TOPICS)
def test_every_builtin_fallback_topic_passes_its_own_gate(topic):
    """The offline fallback pool must never be blocked by the gate."""
    assert ideas_mod.is_kidsong_topic_safe(topic)


@pytest.mark.parametrize("topic", [
    "a monster under the bed",
    "ghost stories for the night",
    "creepy dolls that move by themselves",
    "haunted forest legends",
    "true crime for kids",
    "people reveal their darkest secrets",
    "an interview with a stranger",
])
def test_other_unsafe_topics_are_rejected(topic):
    assert not ideas_mod.is_kidsong_topic_safe(topic)


def test_empty_topic_is_allowed():
    """No subject supplied -> the kidsong pipeline invents its own safe one."""
    assert ideas_mod.kidsong_topic_rejection(None) is None
    assert ideas_mod.kidsong_topic_rejection("   ") is None


def test_allow_leaning_rejects_unknown_subject_matter():
    """Not on the deny list, still not a toddler subject -> rejected."""
    reason = ideas_mod.kidsong_topic_rejection("quarterly revenue synergy")
    assert reason == "no toddler subject matched"


def test_video_type_signal_is_independent_of_topic():
    """A facts idea is not a song subject even when its wording is harmless."""
    idea = {"video_type": "facts", "topic": "counting to ten"}
    assert ideas_mod.kidsong_idea_rejection(idea)
    assert ideas_mod.kidsong_idea_rejection(
        {"video_type": "kidsong", "topic": "counting to ten"}) is None


# ------------------------------------------------------------- idea selection --
def test_pending_facts_idea_is_never_handed_to_a_kidsong_channel(tmp_db):
    ch = cascade.ensure_kidsong_channel(CFG)
    models.add_idea(ch["id"], REAL_INCIDENT_TOPICS[0], source="auto",
                    video_type="facts", dedupe_hash="h1")
    assert models.next_pending_idea(ch["id"]) is None


def test_non_kidsong_channel_is_unfiltered(tmp_db):
    cid = models.create_channel("Brainrot")  # default_video_type='facts'
    models.add_idea(cid, REAL_INCIDENT_TOPICS[0], source="auto",
                    video_type="facts", dedupe_hash="h1")
    assert models.next_pending_idea(cid)["topic"] == REAL_INCIDENT_TOPICS[0]


def test_partition_keeps_suitable_ideas_in_pick_order(tmp_db):
    ch = cascade.ensure_kidsong_channel(CFG)
    models.add_idea(ch["id"], REAL_INCIDENT_TOPICS[1], source="auto",
                    video_type="facts", dedupe_hash="h1")
    ideas_mod.add_user_idea(ch["id"], "Counting Clouds at Sunset", video_type="kidsong")
    suitable, rejected = models.partition_pending_ideas(ch["id"], "kidsong")
    assert [i["topic"] for i in suitable] == ["Counting Clouds at Sunset"]
    assert [r["topic"] for r, _reason in rejected] == [REAL_INCIDENT_TOPICS[1]]


# -------------------------------------------------------------------- cascade --
def test_cascade_never_turns_a_facts_idea_into_a_kidsong_job(tmp_db, monkeypatch):
    """The exact incident, end to end."""
    ch = cascade.ensure_kidsong_channel(CFG)
    for n, topic in enumerate(REAL_INCIDENT_TOPICS):
        models.add_idea(ch["id"], topic, source="auto", video_type="facts",
                        dedupe_hash="h%d" % n)
    stub_llm(monkeypatch, ["brushing teeth", "counting to ten", "sharing toys"])

    summary = cascade.run_cascade(CFG, count=3, channel_id=ch["id"])

    queued = {models.get_job(j)["topic"] for j in summary["job_ids"]}
    assert queued.isdisjoint(REAL_INCIDENT_TOPICS)
    for job_id in summary["job_ids"]:
        job = models.get_job(job_id)
        assert job["video_type"] == "kidsong"
        assert ideas_mod.is_kidsong_topic_safe(job["topic"])


def test_cascade_reports_the_skip(tmp_db, monkeypatch):
    ch = cascade.ensure_kidsong_channel(CFG)
    models.add_idea(ch["id"], REAL_INCIDENT_TOPICS[0], source="auto",
                    video_type="facts", dedupe_hash="h1")
    stub_llm(monkeypatch, ["brushing teeth"])

    summary = cascade.run_cascade(CFG, count=1, channel_id=ch["id"])

    assert summary["ideas_rejected"] == 1
    assert REAL_INCIDENT_TOPICS[0] in summary["rejected_topics"]
    assert "content_warning" in summary
    assert REAL_INCIDENT_TOPICS[0] in summary["content_warning"]


def test_cascade_logs_the_skip(tmp_db, monkeypatch, caplog):
    ch = cascade.ensure_kidsong_channel(CFG)
    models.add_idea(ch["id"], REAL_INCIDENT_TOPICS[0], source="auto",
                    video_type="facts", dedupe_hash="h1")
    stub_llm(monkeypatch, ["brushing teeth"])
    with caplog.at_level("WARNING", logger="studio.cascade"):
        cascade.run_cascade(CFG, count=1, channel_id=ch["id"])
    assert any("unsuitable idea" in r.getMessage() for r in caplog.records)


def test_rejected_ideas_keep_their_row_and_status(tmp_db, monkeypatch):
    ch = cascade.ensure_kidsong_channel(CFG)
    idea_id = models.add_idea(ch["id"], REAL_INCIDENT_TOPICS[0], source="auto",
                              video_type="facts", dedupe_hash="h1")
    stub_llm(monkeypatch, ["brushing teeth"])
    cascade.run_cascade(CFG, count=1, channel_id=ch["id"])

    rows = models.list_ideas(channel_id=ch["id"], status="pending")
    survivor = [r for r in rows if r["id"] == idea_id]
    assert survivor, "rejected idea was deleted"
    assert survivor[0]["status"] == "pending"
    assert survivor[0]["video_type"] == "facts"


def test_unsuitable_ideas_do_not_suppress_the_top_up(tmp_db, monkeypatch):
    """Stale facts ideas must not make the cascade think it has material."""
    ch = cascade.ensure_kidsong_channel(CFG)
    for n, topic in enumerate(REAL_INCIDENT_TOPICS):
        models.add_idea(ch["id"], topic, source="auto", video_type="facts",
                        dedupe_hash="h%d" % n)
    stub_llm(monkeypatch, ["brushing teeth", "counting to ten", "sharing toys"])
    summary = cascade.run_cascade(CFG, count=3, channel_id=ch["id"])
    assert summary["jobs_created"] == 3


def test_cascade_does_not_loop_forever_on_a_rejected_idea(tmp_db, monkeypatch):
    """Only unsuitable material available -> partial run, explained, no hang."""
    ch = cascade.ensure_kidsong_channel(CFG)
    models.add_idea(ch["id"], REAL_INCIDENT_TOPICS[0], source="auto",
                    video_type="facts", dedupe_hash="h1")
    stub_llm(monkeypatch, error=RuntimeError("offline"))
    monkeypatch.setattr(ideas_mod, "KIDSONG_FALLBACK_TOPICS", [])

    summary = cascade.run_cascade(CFG, count=3, channel_id=ch["id"])

    assert summary["jobs_created"] == 0
    assert "skipped_reason" in summary
    assert REAL_INCIDENT_TOPICS[0] in summary["skipped_reason"]


def test_channel_repair_warns_about_stale_ideas(tmp_db, caplog):
    """ensure_kidsong_channel repairs default_video_type — say what it orphans."""
    cid = models.create_channel("KinderKrippenLernLieder")
    models.update_channel(cid, default_video_type="facts", made_for_kids=1)
    models.add_idea(cid, REAL_INCIDENT_TOPICS[0], source="auto",
                    video_type="facts", dedupe_hash="h1")

    with caplog.at_level("WARNING", logger="studio.cascade"):
        ch = cascade.ensure_kidsong_channel(CFG)

    assert ch["id"] == cid
    assert ch["default_video_type"] == "kidsong"
    assert any("not toddler-appropriate" in r.getMessage() for r in caplog.records)
    # and the idea itself is untouched
    assert models.list_ideas(channel_id=cid, status="pending")[0]["video_type"] == "facts"


# ---------------------------------------------------------------- HTTP layer --
@pytest.fixture()
def client(tmp_db):
    from app import create_app

    app = create_app()
    app.config["TESTING"] = True
    with app.test_client() as c:
        yield c


@pytest.mark.parametrize("topic", REAL_INCIDENT_TOPICS)
def test_post_generate_rejects_unsafe_kidsong_topic(client, topic):
    r = client.post("/generate", json={"video_type": "kidsong", "topic": topic})
    assert r.status_code == 400
    assert "error" in r.get_json()
    assert models.count_jobs("queued") == 0


@pytest.mark.parametrize("topic", WHOLESOME_TOPICS)
def test_post_generate_accepts_safe_kidsong_topic(client, topic):
    r = client.post("/generate", json={"video_type": "kidsong", "topic": topic})
    assert r.status_code == 200
    assert r.get_json()["job_id"]


def test_post_generate_gates_the_2d_kids_type_too(client):
    r = client.post("/generate",
                    json={"video_type": "kids", "topic": REAL_INCIDENT_TOPICS[0]})
    assert r.status_code == 400


def test_post_generate_leaves_non_kidsong_types_alone(client):
    r = client.post("/generate",
                    json={"video_type": "scary", "topic": REAL_INCIDENT_TOPICS[0]})
    assert r.status_code == 200


@pytest.mark.parametrize("topic", REAL_INCIDENT_TOPICS)
def test_generate_now_rejects_unsafe_kidsong_topic(client, topic):
    cid = models.create_channel("Kidsong")
    models.update_channel(cid, default_video_type="kidsong", made_for_kids=1)
    r = client.post("/channels/%d/generate-now" % cid, json={"topic": topic})
    assert r.status_code == 400
    assert models.count_jobs("queued") == 0


@pytest.mark.parametrize("topic", WHOLESOME_TOPICS)
def test_generate_now_accepts_safe_kidsong_topic(client, topic):
    cid = models.create_channel("Kidsong")
    models.update_channel(cid, default_video_type="kidsong", made_for_kids=1)
    r = client.post("/channels/%d/generate-now" % cid, json={"topic": topic})
    assert r.status_code == 200
    assert r.get_json()["job_id"]


def test_generate_now_without_topic_still_works(client):
    """No topic -> the pipeline invents its own safe subject; not a rejection."""
    cid = models.create_channel("Kidsong")
    models.update_channel(cid, default_video_type="kidsong", made_for_kids=1)
    r = client.post("/channels/%d/generate-now" % cid, json={})
    assert r.status_code == 200


# ------------------------------------------------------------------ scheduler --
def test_scheduler_idea_picker_is_gated_too(tmp_db):
    """The cron path picks ideas via next_pending_idea, so it inherits the gate.

    Asserted on the accessor rather than by running a Scheduler tick: the gate
    lives in the accessor precisely so no caller has to remember it.
    """
    import inspect

    from studio import scheduler as scheduler_mod

    src = inspect.getsource(scheduler_mod.Scheduler._pick_or_generate_idea)
    assert "next_pending_idea" in src

    cid = models.create_channel("Kidsong")
    models.update_channel(cid, default_video_type="kidsong", made_for_kids=1)
    models.add_idea(cid, REAL_INCIDENT_TOPICS[0], source="auto",
                    video_type="facts", dedupe_hash="h1")
    assert models.next_pending_idea(cid) is None
