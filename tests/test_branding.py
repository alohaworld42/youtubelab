"""Title, description and tags must come out of channel branding alone — the
operator never hand-types them for an upload.

That only holds if a hand-edited branding record can never sink an upload:
an unknown `{placeholder}`, a template that is the wrong type, a duplicate
tag with different casing, an intro that points at a file that no longer
exists. Those hardening cases are the point of this file, not padding around
it — each has its own test below, independent of the other two API-level
tests.

`studio.branding` is written to prefer `models.channel_language` /
`models.channel_branding` when they exist, but at the time this file was
written the other worker's models.py change had not landed yet, so every
test here builds its own plain channel dict (`language`, `branding_json`)
instead of depending on it. That is also, deliberately, the fallback path
`branding.py` falls back to — these tests exercise it directly rather than
mocking it out.
"""
import json
import os

from studio import branding


def _channel(branding_dict=None, language=None, **extra):
    """A plain channel dict, the shape `studio.branding` must work against
    even without `models.channel_language` / `models.channel_branding`."""
    ch = dict(extra)
    if branding_dict is not None:
        ch["branding_json"] = json.dumps(branding_dict)
    if language is not None:
        ch["language"] = language
    return ch


# --------------------------------------------------------- render_title ----
def test_a_template_is_filled_from_topic_channel_and_language():
    ch = _channel({"title_template": "{topic} | {channel} ({language})"}, language="de",
                   name="Kinderlieder")
    out = branding.render_title(ch, {}, "Bad Time", fallback=None)
    assert out == "Bad Time | Kinderlieder (de)"


def test_no_template_falls_back_to_fallback_or_topic():
    ch = _channel({})
    assert branding.render_title(ch, {}, "The Topic", fallback="The Fallback") == "The Fallback"
    assert branding.render_title(ch, {}, "The Topic", fallback=None) == "The Topic"
    assert branding.render_title(ch, {}, "The Topic", fallback="") == "The Topic"


# ------------------------------------------------ hardness 1: unknown key --
def test_an_unknown_placeholder_is_removed_not_a_key_error():
    """A hand-edited branding record with a typo'd or stale placeholder must
    not be able to crash an upload."""
    ch = _channel({"title_template": "Hello {foo} World"})
    out = branding.render_title(ch, {}, "topic")
    assert out == "Hello World"  # placeholder gone, double space collapsed


def test_several_unknown_placeholders_in_a_description_template_are_all_removed():
    ch = _channel({"description_template": "{topic} brought to you by {sponsor} in {year}"})
    out = branding.render_description(ch, {"ai_disclosure": {"enabled": False}}, "Song Time")
    assert out == "Song Time brought to you by in"


# --------------------------------------------- hardness 2: wrong template type --
def test_a_non_string_title_template_falls_back():
    for bad in (123, ["a", "b"], {"x": 1}):
        ch = _channel({"title_template": bad})
        assert branding.render_title(ch, {}, "topic", fallback="Fallback Title") == "Fallback Title"


def test_a_non_string_description_template_falls_back_to_base_description():
    ch = _channel({"description_template": 42})
    out = branding.render_description(ch, {"ai_disclosure": {"enabled": False}}, "topic",
                                       base_description="Hand-written body.")
    assert out == "Hand-written body."


def test_a_none_title_template_falls_back_same_as_missing():
    ch = _channel({"title_template": None})
    assert branding.render_title(ch, {}, "topic", fallback="F") == "F"


# ------------------------------------------------- hardness 3: hard caps --
def test_render_title_never_exceeds_the_youtube_100_char_cap():
    ch = _channel({"title_template": "{topic}"})
    out = branding.render_title(ch, {}, "x" * 500)
    assert len(out) == 100


def test_render_title_cap_also_applies_to_the_fallback_path():
    ch = _channel({})
    out = branding.render_title(ch, {}, "y" * 400, fallback=None)
    assert len(out) <= 100


def test_render_tags_never_exceeds_the_youtube_15_tag_cap():
    ch = _channel({"tags": [f"chan{i}" for i in range(10)]})
    video_tags = [f"vid{i}" for i in range(20)]
    out = branding.render_tags(ch, video_tags)
    assert len(out) == 15


# ------------------------------------------------------ render_description --
def test_order_is_template_then_hashtags_then_disclosure_note():
    ch = _channel({
        "description_template": "A song about {topic}.",
        "tags": ["kids", "song"],
    }, language="en")
    cfg = {"ai_disclosure": {"description_note_by_language": {"en": "Made with AI."}}}
    out = branding.render_description(ch, cfg, "brushing teeth")

    lines = out.split("\n\n")
    assert lines[0] == "A song about brushing teeth."
    assert lines[1] == "#kids #song"
    assert lines[2] == "Made with AI."


def test_no_template_uses_base_description():
    ch = _channel({})
    cfg = {"ai_disclosure": {"enabled": False}}
    out = branding.render_description(ch, cfg, "topic", base_description="Hand body.")
    assert out == "Hand body."


# ----------------------------------------- disclosure language selection --
def test_disclosure_note_language_selection_de_vs_en():
    cfg = {"ai_disclosure": {"description_note_by_language": {
        "de": "Mit KI erstellt.",
        "en": "Made with AI.",
    }}}
    de_channel = _channel({}, language="de")
    en_channel = _channel({}, language="en")

    de_out = branding.render_description(de_channel, cfg, "topic")
    en_out = branding.render_description(en_channel, cfg, "topic")

    assert de_out.endswith("Mit KI erstellt.")
    assert en_out.endswith("Made with AI.")
    assert "Made with AI." not in de_out
    assert "Mit KI erstellt." not in en_out


def test_a_language_with_no_matching_variant_falls_back_to_the_plain_note():
    from pipeline import ai_disclosure
    cfg = {"ai_disclosure": {"description_note_by_language": {"de": "Mit KI erstellt."}}}
    ch = _channel({}, language="fr")
    out = branding.render_description(ch, cfg, "topic")
    assert out.endswith(ai_disclosure.DEFAULT_NOTE)


def test_no_by_language_dict_uses_ai_disclosure_note_unchanged():
    from pipeline import ai_disclosure
    ch = _channel({}, language="de")
    out = branding.render_description(ch, {}, "topic")
    assert out.endswith(ai_disclosure.DEFAULT_NOTE)


# ------------------------------------------------- note appears exactly once --
def test_the_note_appears_exactly_once_even_if_base_description_already_has_it():
    from pipeline import ai_disclosure
    ch = _channel({})
    cfg = {}
    base = "Hand-written body.\n\n" + ai_disclosure.DEFAULT_NOTE
    out = branding.render_description(ch, cfg, "topic", base_description=base)
    assert out.count(ai_disclosure.DEFAULT_NOTE) == 1
    assert out.endswith(ai_disclosure.DEFAULT_NOTE)


def test_rendering_twice_does_not_duplicate_the_note():
    ch = _channel({})
    cfg = {}
    once = branding.render_description(ch, cfg, "topic", base_description="Body.")
    twice = branding.render_description(ch, cfg, "topic", base_description=once)
    assert twice == once


# -------------------------------------------------------------- render_tags --
def test_tags_are_deduped_case_insensitively_first_spelling_wins():
    ch = _channel({"tags": ["Kids", "Fun"]})
    out = branding.render_tags(ch, ["KIDS", "New", "fun", "  ", "", None, 7])
    assert out == ["Kids", "Fun", "New"]


def test_tags_come_from_channel_json_fallback_without_models_helper():
    ch = _channel({"tags": ["evergreen"]})
    assert "branding_json" in ch  # sanity: this really goes through the fallback path
    out = branding.render_tags(ch, ["fresh"])
    assert out == ["evergreen", "fresh"]


def test_empty_or_missing_channel_tags_still_returns_video_tags():
    assert branding.render_tags(_channel({}), ["a", "b"]) == ["a", "b"]
    assert branding.render_tags(None, ["a"]) == ["a"]


# --------------------------------------------------------- intro / outro ---
def test_intro_path_that_does_not_exist_is_none(tmp_path):
    ch = _channel({"intro_path": "no/such/file.mp4", "outro_path": "also/missing.mp4"})
    intro, outro = branding.channel_intro_outro(ch, str(tmp_path))
    assert intro is None
    assert outro is None


def test_intro_path_that_exists_resolves_to_an_absolute_path(tmp_path):
    intro_file = tmp_path / "assets" / "intro.mp4"
    intro_file.parent.mkdir(parents=True)
    intro_file.write_bytes(b"fake")

    ch = _channel({"intro_path": "assets/intro.mp4", "outro_path": "assets/outro.mp4"})
    intro, outro = branding.channel_intro_outro(ch, str(tmp_path))

    assert intro == os.path.abspath(str(intro_file))
    assert os.path.isabs(intro)
    assert outro is None


def test_no_branding_gives_no_intro_or_outro(tmp_path):
    assert branding.channel_intro_outro(_channel({}), str(tmp_path)) == (None, None)
    assert branding.channel_intro_outro(None, str(tmp_path)) == (None, None)
