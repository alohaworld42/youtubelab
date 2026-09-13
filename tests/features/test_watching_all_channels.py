"""FEATURE: an operator opens the front page and sees every channel at once.

A studio that runs one channel needs no overview — the queue IS the channel.
The moment there are three, in two languages, with two content profiles, the
operator has exactly one question every morning: *where do I have to approve
something, and does the rest run?* The page has to answer it without a click.

The failure mode this guards is not a crash. It is a page that renders all
three channels as identical grey rows: the one with a video waiting for
approval, the one mid-render, and the one that has quietly produced nothing
for a week because its `schedule_json` is empty and
`scheduler._enqueue_due_slots` therefore never creates a job for it. That last
one is the expensive case — nothing is broken, nothing is logged, and a healthy
"aktiv" badge sits on a channel that will never publish again.

Everything here goes through the real app over the test client, against the
real SQLite the views read (fixtures in tests/features/conftest.py).
"""
import pytest

from studio import models


# --------------------------------------------------------------------- helpers --
def _page(operator):
    resp = operator.get("/")
    assert resp.status_code == 200, (
        f"the front page did not render: {resp.status_code} "
        f"{resp.headers.get('Location', '')}"
    )
    return resp.get_data(as_text=True)


def _section(page, section_id):
    """The HTML of one channel section (`ch-waiting` / `ch-busy` / `ch-idle`)."""
    marker = f'id="{section_id}"'
    assert marker in page, f"the page has no {section_id} section"
    start = page.index(marker)
    end = page.index("</section>", start)
    return page[start:end]


def _card(page, channel_id):
    """The HTML of one channel's card, wherever on the page it ended up.

    A card runs from its `data-channel` marker to whichever comes first: the
    next card, or the end of its section.
    """
    marker = f'data-channel="{channel_id}"'
    assert marker in page, f"channel {channel_id} does not appear on the page at all"
    start = page.index(marker)
    ends = [page.find(needle, start + len(marker))
            for needle in ("data-channel=", "</section>")]
    end = min(e for e in ends if e != -1)
    return page[start:end]


@pytest.fixture()
def channels(studio):
    """Two real channels in two languages, created the way the app creates them."""
    with studio.app_context():
        de = models.create_channel("Zubi Bop", language="de", content_profile="kidsong_de")
        en = models.create_channel("Tiny Tunes", language="en", content_profile="kidsong_en")
    return {"de": de, "en": en}


def _job(studio, channel_id, status, topic="zaehlen lernen"):
    with studio.app_context():
        job_id = models.create_job(channel_id=channel_id, video_type="kidsong", topic=topic)
        models.set_job_status(job_id, status)
    return job_id


# ============================================== every channel is on the page ==
def test_both_channels_appear_with_their_own_language(operator, channels, studio):
    """One studio, two languages. A page that shows only the studio-wide
    language would have every channel look German."""
    page = _page(operator)

    assert "Zubi Bop" in page
    assert "Tiny Tunes" in page
    assert "DE" in _card(page, channels["de"]), "the German channel does not show its language"
    assert "EN" in _card(page, channels["en"]), "the English channel does not show its language"


def test_a_channel_names_the_profile_it_actually_renders(operator, channels):
    """`content_profile` is what decides the video type and language the
    pipeline uses (studio/profiles.resolve_profile); the raw
    `default_video_type` column is overridden by it and would be a lie."""
    page = _page(operator)
    assert "kidsong_de" in _card(page, channels["de"])
    assert "kidsong_en" in _card(page, channels["en"])


def test_a_channel_shows_its_status(operator, channels, studio):
    with studio.app_context():
        models.update_channel(channels["en"], status="needs_reauth")

    page = _page(operator)
    assert "aktiv" in _card(page, channels["de"])
    assert "Re-Auth nötig" in _card(page, channels["en"]), (
        "a channel whose uploads are blocked on a re-auth says nothing about it"
    )


# ================================================== the work waiting on a human ==
def test_a_channel_with_an_open_approval_stands_in_the_waiting_section(operator,
                                                                       channels, studio):
    """The whole point of the page: the channel the operator can unblock is
    named as such, and the one with nothing to do is not standing next to it."""
    _job(studio, channels["de"], "ready")

    page = _page(operator)
    waiting = _section(page, "ch-waiting")

    assert "Wartet auf dich (1)" in page
    assert "Zubi Bop" in waiting, "the channel with an open approval is not in the waiting section"
    assert "Tiny Tunes" not in waiting, (
        "a channel with nothing to approve was listed as waiting on the operator"
    )
    assert "Tiny Tunes" in _section(page, "ch-idle")


def test_the_waiting_section_comes_before_the_rest(operator, channels, studio):
    """DOM order is reading order. Work the operator can clear must not sit
    below work they cannot — that exact defect was fixed on /dashboard."""
    _job(studio, channels["de"], "ready")

    page = _page(operator)
    assert page.index('id="ch-waiting"') < page.index('id="ch-busy"') < page.index('id="ch-idle"')


def test_a_waiting_channel_links_straight_into_the_approval(operator, channels, studio):
    _job(studio, channels["de"], "ready")

    card = _card(_page(operator), channels["de"])
    assert 'href="/review"' in card, "the card says something is waiting but not where"
    assert "freigeben" in card


def test_a_running_channel_is_not_confused_with_a_waiting_one(operator, channels, studio):
    """"rendering" is the machine's problem. It must not share a section with
    "somebody has to click approve"."""
    _job(studio, channels["de"], "ready")
    _job(studio, channels["en"], "rendering")

    page = _page(operator)
    assert "Tiny Tunes" in _section(page, "ch-busy")
    assert "Tiny Tunes" not in _section(page, "ch-waiting")
    assert "1</b> laufen" in _card(page, channels["en"])


def test_an_approved_job_stuck_behind_a_re_auth_is_visible_on_its_channel(operator,
                                                                          channels, studio):
    """'approved' means "waiting for the uploader thread", and the uploader
    refuses a `needs_reauth` channel — so this is exactly where uploads pile up
    invisibly."""
    _job(studio, channels["en"], "approved")
    with studio.app_context():
        models.update_channel(channels["en"], status="needs_reauth")

    card = _card(_page(operator), channels["en"])
    assert "warten auf Upload" in card


# ======================================================= the silent channel ==
def test_an_active_channel_without_a_schedule_is_flagged(operator, channels):
    """`scheduler._enqueue_due_slots` iterates `channel["schedule"]`. An empty
    schedule means the worker never creates a job — the channel is `active`,
    looks healthy, and produces nothing, forever."""
    card = _card(_page(operator), channels["de"])
    assert "kein Zeitplan hinterlegt" in card, (
        "an active channel that can never produce anything looks healthy"
    )
    assert "Zeitplan eintragen" in card, "the warning does not say what to do about it"


def test_a_scheduled_channel_is_not_flagged(operator, channels, studio):
    with studio.app_context():
        models.update_channel(channels["de"], schedule=["09:00"])

    page = _page(operator)
    assert "kein Zeitplan hinterlegt" not in _card(page, channels["de"])
    assert "kein Zeitplan hinterlegt" in _card(page, channels["en"]), (
        "the other channel lost its warning too — the flag is not per channel"
    )


def test_a_channel_with_a_job_in_flight_is_not_called_silent(operator, channels, studio):
    """It has no schedule, but something IS running: calling that "produces
    nothing" would be the page contradicting itself."""
    _job(studio, channels["de"], "rendering")

    assert "kein Zeitplan hinterlegt" not in _card(_page(operator), channels["de"])


def test_a_paused_channel_is_not_called_silent(operator, channels, studio):
    """Paused is a decision, not a failure. Warning about it would train the
    operator to ignore the warning."""
    with studio.app_context():
        models.update_channel(channels["de"], status="paused")

    card = _card(_page(operator), channels["de"])
    assert "pausiert" in card
    assert "kein Zeitplan hinterlegt" not in card


# ================================================================ last upload ==
def test_a_channel_that_never_uploaded_says_so(operator, channels):
    assert "noch keiner" in _card(_page(operator), channels["de"])


def test_the_last_upload_is_a_readable_age_not_raw_seconds(operator, channels, studio):
    """The take queue once rendered an age as `wartet 18472s`. No number of
    seconds belongs on a page somebody reads at breakfast."""
    import time

    from studio.db import get_db

    old = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(time.time() - 18472))
    with studio.app_context():
        job_id = models.create_job(channel_id=channels["de"], video_type="kidsong", topic="t")
        video_id = models.create_video(job_id, channels["de"], "/tmp/x.mp4", "T", "d", [], 30.0)
        models.record_upload(video_id, channels["de"], "yt123", "https://y.t/1", "private")
        # `record_upload` stamps "now"; back-date it to the age the queue once
        # printed as `18472s` so the assertion names a real historical value.
        db = get_db()
        db.execute("UPDATE uploads SET uploaded_at=? WHERE video_id=?", (old, video_id))
        db.commit()

    card = _card(_page(operator), channels["de"])
    assert "5 h 7 min" in card, "the upload age is not human-readable"
    assert "18472" not in card, "raw seconds reached the page"


# ================================================================ no channels ==
def test_a_studio_without_channels_says_what_to_do(operator, studio, monkeypatch):
    """A configured studio with zero channels must still render — and an empty
    page that just says "Kanäle (0)" tells a new operator nothing."""
    # Make the studio "set up" so `/` renders its own empty state instead of
    # sending a brand-new install to the setup wizard. Config only, no mocking:
    # this is the state the setup page itself writes.
    studio.config["STUDIO_CFG"]["llm"]["backend"] = "groq"
    monkeypatch.setenv("GROQ_API_KEY", "gsk_feature_suite")

    page = _page(operator)
    assert "Noch kein Kanal" in page
    assert 'href="/channels/"' in page, "the empty state does not link anywhere"
    assert "Kaskade" in page


def test_the_page_still_renders_when_a_channel_has_no_jobs_at_all(operator, channels):
    """Two fresh channels, zero jobs, zero uploads: the commonest state right
    after setup, and the one where every count is a division by nothing."""
    page = _page(operator)
    assert "Wartet auf dich (0)" in page
    assert "Keine offene Freigabe" in page
    assert "Ohne offene Arbeit (2)" in page
