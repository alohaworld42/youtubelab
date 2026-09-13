"""Invariants of the rendered studio pages.

These are not "does the template compile" tests — every assertion here is a
defect that shipped:

  * API keys were `type=text`, legible on a page the app serves on 0.0.0.0 by
    default (phone on the WLAN, screen share, tunnel).
  * Every external link opened with `target=_blank` and no `rel=noopener`.
  * The failed-jobs table truncated the error at 120 characters with no way to
    reach the rest — on a failed render that column IS the diagnosis.
  * `approved`/`uploading` jobs were counted in no view at all, so an upload
    blocked on a channel re-auth vanished from the operator's screen.
  * Action buttons fired their request without disabling themselves, so a
    double click sent two POSTs and the second came back 409 right after the
    first had succeeded.
"""
import re

import pytest

from studio import models


@pytest.fixture()
def client(tmp_db):
    from app import create_app

    app = create_app()
    app.config["TESTING"] = True
    with app.test_client() as c:
        yield c


def _html(client, url):
    res = client.get(url)
    assert res.status_code == 200, f"{url} -> {res.status_code}"
    return res.get_data(as_text=True)


# ------------------------------------------------------------------ secrets --
def test_api_key_fields_are_not_plain_text(client):
    html = _html(client, "/setup")
    for field_id in ("llm-key", "k-pexels", "k-pixabay"):
        match = re.search(r'<input[^>]*id="%s"[^>]*>' % field_id, html)
        assert match, f"{field_id} not found on the setup page"
        assert 'type="password"' in match.group(0), (
            f"{field_id} renders the secret in clear text: {match.group(0)}"
        )


# -------------------------------------------------------------- link safety --
@pytest.mark.parametrize("url", ["/setup", "/history", "/dashboard"])
def test_external_links_carry_rel_noopener(client, tmp_db, url):
    html = _html(client, url)
    for tag in re.findall(r"<a[^>]*target=\"_blank\"[^>]*>", html):
        assert "noopener" in tag, f"{url}: {tag}"


# ------------------------------------------------------------------ history --
def test_a_long_job_error_is_reachable_in_full(client):
    ch = models.create_channel("Test")
    job_id = models.create_job(channel_id=ch, video_type="kidsong", topic="counting")
    tail = "THE-ACTUAL-CAUSE-IS-DOWN-HERE"
    models.update_job(job_id, status="failed", error="x" * 200 + tail)

    html = _html(client, "/history")
    assert tail in html, "the failed-job error is truncated with no way to see the rest"
    assert "<details" in html


def test_a_short_job_error_is_not_wrapped_in_details(client):
    ch = models.create_channel("Test")
    job_id = models.create_job(channel_id=ch, video_type="kidsong", topic="counting")
    models.update_job(job_id, status="failed", error="ComfyUI timed out")

    html = _html(client, "/history")
    assert "ComfyUI timed out" in html
    assert "<details" not in html


# ----------------------------------------------------- the invisible backlog --
def test_an_approved_job_is_visible_somewhere(client):
    """'approved' means "waiting for the uploader". It used to appear in no
    view: not the review queue, not the failed list, not a counter."""
    ch = models.create_channel("Test")
    job_id = models.create_job(channel_id=ch, video_type="kidsong", topic="counting")
    models.set_job_status(job_id, "approved")

    counts = client.get("/api/status").get_json()["counts"]
    assert counts["waiting_upload"] == 1

    assert "Wartet auf Upload" in _html(client, "/")


def test_waiting_upload_counts_uploading_too(client):
    ch = models.create_channel("Test")
    a = models.create_job(channel_id=ch, video_type="kidsong", topic="a")
    b = models.create_job(channel_id=ch, video_type="kidsong", topic="b")
    models.set_job_status(a, "approved")
    models.set_job_status(b, "uploading")

    assert client.get("/api/status").get_json()["counts"]["waiting_upload"] == 2


# ------------------------------------------------------------ double-submit --
@pytest.mark.parametrize("url,fn", [
    ("/history", "retry"),
    ("/ideas", "addIdea"),
    ("/ideas", "genIdeas"),
])
def test_action_buttons_go_through_the_disable_helper(client, url, fn):
    """`withButton` is what stops a double click from firing a second POST that
    comes back 409 right after the first one succeeded — and what guarantees the
    button comes back if the request throws."""
    ch = models.create_channel("Test")
    models.create_job(channel_id=ch, video_type="kidsong", topic="t")
    html = _html(client, url)
    body = html.split(f"function {fn}", 1)
    assert len(body) == 2, f"{fn} not found on {url}"
    assert "withButton(" in body[1][:400], f"{fn} on {url} does not disable its button"


def test_the_review_page_confirms_before_a_re_render(client):
    """"Verwerfen & neu generieren" was the one button here with no
    confirmation — the cheap discard asked, the expensive full re-render did
    not."""
    html = _html(client, "/review")
    assert "async function reject" in html
    reject_fn = html.split("async function reject", 1)[1][:700]
    assert "confirm(" in reject_fn
    assert "regenerate" in reject_fn


# ------------------------------------------------------------ shared chrome --
@pytest.mark.parametrize("url", ["/", "/dashboard", "/kidsong-review", "/ideas",
                                 "/review", "/history", "/quick", "/setup", "/channels/"])
def test_every_page_carries_the_nav_and_the_shared_helpers(client, url):
    ch = models.create_channel("Test")  # so "/" renders the dashboard, not a redirect
    html = _html(client, url)
    assert 'href="/kidsong-review"' in html, "page dropped the nav"
    assert "function withButton" in html
    assert "function pollJson" in html


# ------------------------------------------------------ schedule validation --
@pytest.mark.parametrize("raw,expected", [
    ("09:00, 18:30", ["09:00", "18:30"]),
    ("9:00", ["09:00"]),               # a leading zero is not required of the user
    ("09.00", ["09:00"]),              # a dot is the other way people write it
    ("18:00, 09:00", ["09:00", "18:00"]),
    ("09:00, 09:00", ["09:00"]),       # duplicates would fire once anyway
    ("", []),
    ("  ", []),
])
def test_schedule_normalisation_accepts_what_people_actually_type(raw, expected):
    from studio.views.channels import normalize_schedule

    accepted, rejected = normalize_schedule(raw)
    assert accepted == expected
    assert rejected == []


@pytest.mark.parametrize("raw", ["9am", "25:00", "09:70", "morgens", "9", "09:00:00"])
def test_schedule_normalisation_reports_what_it_cannot_parse(raw):
    from studio.views.channels import normalize_schedule

    accepted, rejected = normalize_schedule(raw)
    assert accepted == []
    assert rejected == [raw]


def test_an_unparseable_schedule_is_refused_instead_of_silently_dropped(client):
    """`scheduler.due_slots` skips what it cannot parse, so saving "9am" left a
    channel that looked scheduled and never produced anything, forever."""
    ch = models.create_channel("Test")
    res = client.post(f"/channels/{ch}", data={
        "name": "Test", "schedule": "09:00, 9am", "privacy": "private",
    })
    assert res.status_code == 400
    assert "9am" in res.get_data(as_text=True)
    assert models.get_channel(ch)["schedule"] == []  # unchanged, not half-saved


def test_a_valid_schedule_saves_normalised(client):
    ch = models.create_channel("Test")
    res = client.post(f"/channels/{ch}", data={
        "name": "Test", "schedule": "18:00, 9.30", "privacy": "private",
    })
    assert res.status_code == 302
    assert models.get_channel(ch)["schedule"] == ["09:30", "18:00"]


def test_the_schedule_slots_a_channel_saves_are_ones_the_scheduler_understands(client):
    """The two halves have to agree, or the form accepts what the worker drops."""
    from studio.scheduler import due_slots
    from studio.views.channels import normalize_schedule

    accepted, _ = normalize_schedule("00:00, 9:05, 23:59")
    # Every accepted slot must be recognised at a time late enough for it to be due.
    import time as _t
    end_of_day = _t.struct_time((2026, 1, 1, 23, 59, 0, 0, 1, 0))
    assert due_slots(accepted, end_of_day) == accepted


# ------------------------------------------------------- originality guard --
def test_no_product_surface_points_the_user_at_a_protected_brand(client):
    """The channel treats look-alike drift toward an existing kids' brand as a
    ship-blocking finding (AGENTS.md, docs/quality/SOURCES.md), so the UI must
    not hand the operator that target — the style picker used to label the kids
    preset "CoComelon-artig"."""
    ch = models.create_channel("Test")
    for url in ("/quick", f"/channels/{ch}"):
        html = _html(client, url).lower()
        assert "cocomelon" not in html, url


def test_the_kids_script_prompt_does_not_ask_for_a_brand_imitation():
    import os

    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    with open(os.path.join(root, "prompts", "kids.txt"), encoding="utf-8") as f:
        prompt = f.read()
    instruction = prompt.split("Rules:", 1)[0].lower()
    assert "cocomelon" not in instruction, (
        "the prompt sent to the LLM asks it to imitate a protected brand"
    )
    assert "original only" in prompt.lower()


def test_an_unlabelled_style_is_still_labelled_in_the_channel_editor(client, monkeypatch):
    """The label lookup was `{...}[s]`. Jinja resolves a missing key to Undefined
    rather than raising, so a style with no hardcoded German label rendered as a
    blank, unpickable <option> — the silent version of a crash. `.get(s, s)`
    falls back to the style's own name."""
    import studio.views.channels as channels_view

    ch = models.create_channel("Test")
    monkeypatch.setattr(channels_view, "STYLES", ["", "kids", "claymation"])
    html = _html(client, f"/channels/{ch}")

    option = re.search(r'<option value="claymation"[^>]*>(.*?)</option>', html, re.S)
    assert option, "the style is missing from the picker entirely"
    assert option.group(1).strip(), "the style renders as a blank, unpickable option"


def test_a_rejected_schedule_keeps_the_rest_of_the_edits_on_screen(client):
    """Re-rendering from the stored row would throw away every other change the
    user made in the same visit — a worse bug than the typo that triggered it."""
    ch = models.create_channel("Old name")
    res = client.post(f"/channels/{ch}", data={
        "name": "New name",
        "niche": "Gruselgeschichten",
        "description": "Eine neue Beschreibung",
        "privacy": "unlisted",
        "made_for_kids": "1",
        "voice_narrator": "de-DE-ConradNeural",
        "gameplay_urls": "https://example.com/clip",
        "schedule": "09:00, halb zehn",
    })
    assert res.status_code == 400
    html = res.get_data(as_text=True)
    for typed in ("New name", "Gruselgeschichten", "Eine neue Beschreibung",
                  "de-DE-ConradNeural", "https://example.com/clip", "halb zehn"):
        assert typed in html, f"the form lost {typed!r}"
    assert 'value="unlisted" selected' in html or 'value="unlisted"  selected' in html
    # ...and nothing was written.
    assert models.get_channel(ch)["name"] == "Old name"
