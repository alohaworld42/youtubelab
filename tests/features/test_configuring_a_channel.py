"""FEATURE: an operator configures a channel and the scheduler honours it.

A channel's settings are only worth anything if the worker that reads them
agrees with the form that wrote them. That agreement spans three modules —
the edit form, `models.update_channel`, and `scheduler.due_slots` — so it is
precisely what a unit test of any one of them cannot check.

The signature failure this guards: `due_slots` silently skips a slot it cannot
parse, and the editor used to save anything. Typing "9am" gave you a channel
that *looked* scheduled, showed its schedule back to you, and never produced a
single video.
"""
import re
import time

import pytest

from studio import models
from studio.scheduler import due_slots

# `due_slots` only returns slots whose time has already passed *today*, so a
# test that just called it would pass in the evening and fail before 09:00.
# Pin "now" to one minute before midnight: every valid slot is due.
_END_OF_DAY = time.struct_time((2026, 7, 31, 23, 59, 0, 4, 212, 0))


@pytest.fixture()
def channel(studio):
    """A real row in the real DB, created through the model layer the app uses."""
    with studio.app_context():
        cid = models.create_channel(name="ZubiBop")
    return cid


def _save(operator, channel_id, **overrides):
    form = {
        "name": "ZubiBop", "niche": "kidsong", "description": "toddler songs",
        "default_video_type": "kidsong", "privacy": "private",
        "schedule": "09:00, 18:30", "source_order": "gameplay",
    }
    form.update(overrides)
    return operator.post(f"/channels/{channel_id}", data=form)


def _configured_profiles(studio):
    """The profile names the running app itself offers — read from its own
    config, not hardcoded, so this suite tracks config.json instead of pinning
    the five names that happen to ship today."""
    from studio import profiles

    return profiles.profile_names(studio.config["STUDIO_CFG"])


def _selected(html, value):
    """Whether `<option value="...">` is the selected one."""
    match = re.search(r'<option value="%s"([^>]*)>' % re.escape(value), html)
    return bool(match) and "selected" in match.group(1)


# ================================================ a saved schedule really runs ==
def test_a_saved_schedule_is_one_the_scheduler_can_read(operator, channel, studio):
    """The contract that matters: what the form stores, the worker must parse."""
    resp = _save(operator, channel, schedule="09:00, 18:30")
    assert resp.status_code in (302, 200)

    with studio.app_context():
        stored = models.get_channel(channel)["schedule"]
    assert stored, "nothing was saved"
    assert due_slots(stored, _END_OF_DAY), (
        "the scheduler cannot parse the schedule the editor just saved — "
        "this channel would look scheduled and never produce a video"
    )


@pytest.mark.parametrize("typed,expected_slot", [
    ("9:00", "09:00"),
    ("09.00", "09:00"),
    ("9:00, 18:30", "09:00"),
])
def test_the_editor_accepts_what_people_actually_type(operator, channel, studio,
                                                      typed, expected_slot):
    _save(operator, channel, schedule=typed)
    with studio.app_context():
        stored = models.get_channel(channel)["schedule"]
    assert expected_slot in stored
    assert due_slots(stored, _END_OF_DAY)


def test_a_time_the_scheduler_cannot_parse_is_refused_not_swallowed(operator, channel,
                                                                     studio):
    resp = _save(operator, channel, schedule="9am")
    assert resp.status_code == 400, "a schedule the worker ignores was accepted"
    body = resp.get_data(as_text=True)
    assert "9am" in body, "the error does not say which entry was wrong"

    with studio.app_context():
        assert not (models.get_channel(channel)["schedule"] or ""), (
            "an unparseable schedule was stored anyway"
        )


def test_a_rejected_save_keeps_the_rest_of_the_form(operator, channel):
    """Refusing the typo must not also throw away the four other fields the
    operator edited in the same visit — that is worse than the typo."""
    resp = _save(operator, channel, schedule="9am", niche="lullabies",
                 description="a brand new description")
    body = resp.get_data(as_text=True)

    assert "lullabies" in body, "the niche the operator typed was discarded"
    assert "a brand new description" in body


# ================================================ the kids declaration sticks ===
def test_the_made_for_kids_flag_round_trips(operator, channel, studio):
    _save(operator, channel, made_for_kids="on")
    with studio.app_context():
        assert models.get_channel(channel)["made_for_kids"] == 1


def test_leaving_the_kids_box_unticked_stores_zero(operator, channel, studio):
    _save(operator, channel)
    with studio.app_context():
        assert models.get_channel(channel)["made_for_kids"] == 0


# ============================== language, profile and branding are reachable ==
# The columns exist (`models.create_channel(language=…, content_profile=…,
# branding=…)`) and the workers read them (`profiles.resolve_profile`,
# `branding.render_title`), but until the editor writes them there is no way
# for an operator to set them at all — the whole multi-channel feature would
# be dead code behind a form that cannot reach it.
def test_language_profile_and_branding_reach_the_database(operator, channel, studio):
    profile = _configured_profiles(studio)[0]
    resp = _save(
        operator, channel,
        language="de",
        content_profile=profile,
        branding_title_template="{topic} | {channel}",
        branding_description_template="Heute geht es um {topic}.",
        branding_tags="kinderlieder, lernen",
        branding_use_colors="1",
        branding_primary_color="#123456",
        branding_secondary_color="#abcdef",
        branding_font="Baloo2-Bold.ttf",
        branding_intro_path="assets/branding/intro.mp4",
        branding_outro_path="assets/branding/outro.mp4",
        branding_logo_path="assets/branding/logo.png",
    )
    assert resp.status_code in (302, 200), resp.get_data(as_text=True)[:2000]

    with studio.app_context():
        stored = models.get_channel(channel)
    assert stored["language"] == "de"
    assert stored["content_profile"] == profile

    branding = stored["branding"]
    assert branding["title_template"] == "{topic} | {channel}"
    assert branding["description_template"] == "Heute geht es um {topic}."
    assert branding["tags"] == ["kinderlieder", "lernen"]
    assert branding["primary_color"] == "#123456"
    assert branding["secondary_color"] == "#abcdef"
    assert branding["font"] == "Baloo2-Bold.ttf"
    assert branding["intro_path"] == "assets/branding/intro.mp4"
    assert branding["outro_path"] == "assets/branding/outro.mp4"
    assert branding["logo_path"] == "assets/branding/logo.png"


def test_the_saved_title_template_is_one_the_uploader_can_fill(operator, channel, studio):
    """Same contract as the schedule test above: what the form stores, the
    worker must be able to use — here `studio.branding`, which drops any
    placeholder it does not know rather than raising."""
    from studio import branding as branding_mod

    _save(operator, channel, language="de",
          branding_title_template="{topic} | {channel}")
    with studio.app_context():
        stored = models.get_channel(channel)
    title = branding_mod.render_title(stored, studio.config["STUDIO_CFG"], "Zähne putzen")
    assert title == "Zähne putzen | ZubiBop", title


def test_what_was_saved_is_shown_again_when_the_page_is_reopened(operator, channel, studio):
    profile = _configured_profiles(studio)[0]
    _save(operator, channel, language="de", content_profile=profile,
          branding_title_template="{topic} — ZubiBop",
          branding_tags="kinderlieder, lernen",
          branding_use_colors="1", branding_primary_color="#123456",
          branding_secondary_color="#abcdef",
          branding_logo_path="assets/branding/logo.png")

    page = operator.get(f"/channels/{channel}").get_data(as_text=True)
    assert _selected(page, "de"), "the language select forgot the saved language"
    assert _selected(page, profile), "the profile select forgot the saved profile"
    assert "{topic} — ZubiBop" in page
    assert "kinderlieder, lernen" in page
    assert "#123456" in page and "#abcdef" in page
    assert "assets/branding/logo.png" in page


def test_the_language_field_offers_exactly_the_two_supported_codes(operator, channel):
    """Free text here reaches the column `branding._disclosure_note` matches
    on: "Deutsch" or "de_DE" would silently upload the English AI notice under
    a German video."""
    page = operator.get(f"/channels/{channel}").get_data(as_text=True)
    select = re.search(r'<select name="language".*?</select>', page, re.S)
    assert select, "there is no language picker on the page at all"
    assert sorted(re.findall(r'value="([^"]*)"', select.group(0))) == ["de", "en"]


def test_the_profile_picker_offers_the_configured_profiles_and_an_empty_choice(
        operator, channel, studio):
    page = operator.get(f"/channels/{channel}").get_data(as_text=True)
    select = re.search(r'<select name="content_profile".*?</select>', page, re.S)
    assert select, "there is no content-profile picker on the page at all"
    offered = re.findall(r'value="([^"]*)"', select.group(0))
    assert "" in offered, "no way to run a channel without a profile"
    for name in _configured_profiles(studio):
        assert name in offered, f"config defines {name} but the editor hides it"


@pytest.mark.parametrize("typed,expected", [
    ("a, b ,, c", ["a", "b", "c"]),
    ("  ", []),
    ("eins,, zwei ", ["eins", "zwei"]),
])
def test_the_tag_field_normalises_what_people_actually_type(operator, channel, studio,
                                                            typed, expected):
    _save(operator, channel, branding_tags=typed)
    with studio.app_context():
        assert models.get_channel(channel)["branding"]["tags"] == expected


def test_an_untouched_colour_picker_does_not_invent_a_brand_colour(operator, channel,
                                                                    studio):
    """`type=color` has no empty state — a browser posts #000000 for a picker
    nobody touched. Without the "use colours" checkbox every save would hand
    the channel a black brand colour it never chose."""
    _save(operator, channel, branding_primary_color="#000000",
          branding_secondary_color="#000000")
    with studio.app_context():
        branding = models.get_channel(channel)["branding"]
    assert branding["primary_color"] == ""
    assert branding["secondary_color"] == ""


# ==================================== nothing the operator typed is lost ======
def test_a_profile_the_config_no_longer_defines_is_not_dropped(operator, channel, studio):
    """A channel set to a profile that was later removed from config.json.
    If the editor omitted that name from the picker, the select would submit
    something else — opening the page and pressing Speichern would silently
    re-point the channel."""
    with studio.app_context():
        models.update_channel(channel, content_profile="retired_profile")

    page = operator.get(f"/channels/{channel}").get_data(as_text=True)
    assert _selected(page, "retired_profile"), (
        "the channel's own profile is missing from the picker — saving would "
        "overwrite it with whatever option happens to be selected instead"
    )

    resp = _save(operator, channel, content_profile="retired_profile")
    assert resp.status_code in (302, 200), resp.get_data(as_text=True)[:2000]
    with studio.app_context():
        assert models.get_channel(channel)["content_profile"] == "retired_profile"


def test_a_rejected_save_keeps_the_language_profile_and_branding_too(operator, channel,
                                                                     studio):
    """The schedule typo already kept name/niche/description on screen; the new
    fields have to survive the same 400, or an operator loses the branding they
    just typed because of a mistake three fields away."""
    profile = _configured_profiles(studio)[0]
    resp = _save(operator, channel, schedule="9am",
                 language="de", content_profile=profile,
                 branding_title_template="{topic} — ZubiBop",
                 branding_tags="kinderlieder, lernen",
                 branding_use_colors="1", branding_primary_color="#123456",
                 branding_logo_path="assets/branding/logo.png")
    assert resp.status_code == 400
    body = resp.get_data(as_text=True)

    assert _selected(body, "de"), "the language the operator picked was discarded"
    assert _selected(body, profile), "the profile the operator picked was discarded"
    for typed in ("{topic} — ZubiBop", "kinderlieder, lernen", "#123456",
                  "assets/branding/logo.png"):
        assert typed in body, f"the form lost {typed!r}"


def test_a_placeholder_branding_cannot_fill_is_refused_not_swallowed(operator, channel,
                                                                     studio):
    """`branding._fill_template` deletes an unknown placeholder, so `{thema}`
    used to mean every upload carried a title with no topic in it and nothing
    ever said so."""
    resp = _save(operator, channel, schedule="18:30",
                 branding_title_template="{thema} | ZubiBop",
                 branding_tags="kinderlieder")
    assert resp.status_code == 400, "a template that silently loses the topic was accepted"
    body = resp.get_data(as_text=True)
    assert "{thema}" in body, "the error does not say which placeholder was wrong"
    # ...and the rest of the visit survives the refusal, schedule included.
    assert "18:30" in body
    assert "kinderlieder" in body

    with studio.app_context():
        assert models.get_channel(channel)["branding"]["title_template"] == "", (
            "the unfillable template was stored anyway"
        )


# ======================================================== pause / resume works ==
def test_pausing_a_channel_stops_it_being_scheduled(operator, channel, studio):
    """Asserted through the query the scheduler itself runs
    (`list_channels(status="active")`), not through a column name — pausing has
    to remove the channel from the worker's view, whatever the storage is."""
    _save(operator, channel, schedule="09:00")
    with studio.app_context():
        assert channel in [c["id"] for c in models.list_channels(status="active")]

    operator.post(f"/channels/{channel}/pause")
    with studio.app_context():
        assert channel not in [c["id"] for c in models.list_channels(status="active")], (
            "a paused channel is still picked up by the scheduler"
        )

    operator.post(f"/channels/{channel}/resume")
    with studio.app_context():
        assert channel in [c["id"] for c in models.list_channels(status="active")]


# ================================================== the list page shows it all ==
def test_the_channel_appears_on_the_channels_page(operator, channel):
    page = operator.get("/channels/").get_data(as_text=True)
    assert "ZubiBop" in page


def test_the_editor_renders_the_stored_values_back(operator, channel):
    _save(operator, channel, niche="lullabies", schedule="09:00")
    page = operator.get(f"/channels/{channel}").get_data(as_text=True)
    assert "lullabies" in page
    assert "09:00" in page


def test_deleting_a_channel_removes_it(operator, channel, studio):
    operator.post(f"/channels/{channel}/delete")
    with studio.app_context():
        assert models.get_channel(channel) is None
