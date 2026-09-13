"""Title, description and tags come from channel branding, not from a human.

The operator configures a channel once — a title/description template, a set
of channel tags, an intro/outro, a language — and every upload after that is
derived, never hand-typed. That only holds if a hand-edited branding record
can never break an upload: an unknown `{placeholder}` in a template, a
template that is a number or a list because someone fat-fingered `config.json`,
a tag list with duplicate casing, an intro path that no longer exists on disk
— none of these may raise, and none may produce a title over YouTube's 100
character cap or a tag list over its 15-tag cap. That defensiveness is the
point of this module, not incidental to it.

`models.channel_language(channel)` and `models.channel_branding(channel)` are
the intended source of a channel's language and branding dict, but this module
never assumes they exist: it looks them up with `getattr` and falls back to
reading `channel["language"]` / `channel["branding_json"]` directly, so it
works against a plain channel dict (as the studio DB layer hands back today,
or as a test builds one) either way.
"""
import json
import os
import re

TITLE_LIMIT = 100
TAGS_LIMIT = 15

_PLACEHOLDER_RE = re.compile(r"\{(\w+)\}")


# ------------------------------------------------------------ channel reads --
def _channel_language(channel):
    """The channel's language code (e.g. "de", "en"), or None if unset.

    Prefers `models.channel_language` when the other worker's version exists;
    never lets its absence, or an exception from a half-finished version of
    it, take down an upload.
    """
    channel = channel or {}
    try:
        from studio import models
    except ImportError:
        models = None
    fn = getattr(models, "channel_language", None) if models else None
    if callable(fn):
        try:
            lang = fn(channel)
        except Exception:
            lang = None
        if isinstance(lang, str) and lang.strip():
            return lang.strip().lower()

    lang = channel.get("language")
    if isinstance(lang, str) and lang.strip():
        return lang.strip().lower()
    return None


def _channel_branding(channel):
    """The channel's branding dict, defaulting to {} for anything malformed."""
    channel = channel or {}
    try:
        from studio import models
    except ImportError:
        models = None
    fn = getattr(models, "channel_branding", None) if models else None
    if callable(fn):
        try:
            branding = fn(channel)
        except Exception:
            branding = None
        if isinstance(branding, dict):
            return branding

    raw = channel.get("branding_json")
    if isinstance(raw, dict):
        return raw
    try:
        branding = json.loads(raw or "{}")
    except (ValueError, TypeError):
        branding = {}
    return branding if isinstance(branding, dict) else {}


def _channel_name(channel):
    channel = channel or {}
    return channel.get("name") or channel.get("yt_channel_title") or ""


# --------------------------------------------------------------- templating --
def _fill_template(template, values):
    """Fill `{topic}`/`{channel}`/`{language}` (or whatever is in `values`).

    Any placeholder not in `values` — a typo, an old key from a renamed
    field — is removed rather than raising KeyError: a hand-edited branding
    record must not be able to sink an upload. Whitespace left behind by a
    removed placeholder is collapsed afterwards.
    """
    def repl(match):
        value = values.get(match.group(1))
        return "" if value is None else str(value)

    filled = _PLACEHOLDER_RE.sub(repl, template)
    return " ".join(filled.split())


def _template_result(template, values):
    """Fill `template` if it is a usable string; None otherwise (bad type,
    empty, or it filled out to nothing)."""
    if not isinstance(template, str) or not template.strip():
        return None
    filled = _fill_template(template, values)
    return filled if filled.strip() else None


# -------------------------------------------------------------------- title --
def render_title(channel, cfg, topic, fallback=None):
    """Fill the channel's `title_template`, or fall back to `fallback or topic`.

    Always <= 100 chars (YouTube's title cap), whatever the template does.
    """
    branding = _channel_branding(channel)
    language = _channel_language(channel)
    values = {"topic": topic, "channel": _channel_name(channel), "language": language}

    result = _template_result(branding.get("title_template"), values)
    if result is None:
        result = fallback or topic or ""

    result = " ".join(str(result).split())
    if len(result) > TITLE_LIMIT:
        result = result[:TITLE_LIMIT].rstrip()
    return result


# ------------------------------------------------------------- description --
def _hashtag_line(tags):
    if not isinstance(tags, (list, tuple)):
        return ""
    out = []
    for t in tags:
        if not isinstance(t, str):
            continue
        t = t.strip()
        if not t:
            continue
        out.append(t if t.startswith("#") else "#" + t.replace(" ", ""))
    return " ".join(out)


def _disclosure_note(cfg, language):
    """The language-right AI-disclosure sentence, or "" when disclosure is off.

    `ai_disclosure.settings()` only keeps keys it already knows about, so a
    config-only extension like `description_note_by_language` has to be read
    straight off the raw config block here rather than through `settings()`.
    """
    from pipeline import ai_disclosure  # lazy: heavy-ish module, per repo convention

    if not ai_disclosure.is_enabled(cfg):
        return ""

    raw_block = (cfg or {}).get("ai_disclosure")
    by_language = raw_block.get("description_note_by_language") if isinstance(raw_block, dict) else None
    if isinstance(by_language, dict) and language:
        for key in (language, language.lower(), language.upper()):
            candidate = by_language.get(key)
            if isinstance(candidate, str) and candidate.strip():
                return " ".join(candidate.split())

    return ai_disclosure.note(cfg)


def _has_note(text, cfg, note_text):
    """Whether `note_text` is already present in `text`.

    Uses `ai_disclosure.has_note` when it is checking the same note text that
    plain `ai_disclosure.note(cfg)` would produce (the common case, with no
    per-language override in play); otherwise — and if `has_note` is not
    available at all — falls back to a whitespace-normalised substring check,
    per the same contract `has_note` itself documents.
    """
    if not note_text:
        return False

    from pipeline import ai_disclosure

    has_note_fn = getattr(ai_disclosure, "has_note", None)
    if callable(has_note_fn) and note_text == ai_disclosure.note(cfg):
        try:
            return bool(has_note_fn(text, cfg))
        except Exception:
            pass

    haystack = " ".join((text or "").split()).lower()
    needle = " ".join(note_text.split()).lower()
    return needle in haystack


def render_description(channel, cfg, topic, base_description=""):
    """Filled `description_template` (or `base_description`), then the
    channel's tags as a hashtag line, then the language-right AI-disclosure
    note — always last, never duplicated.
    """
    branding = _channel_branding(channel)
    language = _channel_language(channel)
    values = {"topic": topic, "channel": _channel_name(channel), "language": language}

    body = _template_result(branding.get("description_template"), values)
    if body is None:
        body = base_description if isinstance(base_description, str) else ""
    body = body.strip()

    hashtag_line = _hashtag_line(branding.get("tags"))

    text = "\n\n".join(p for p in (body, hashtag_line) if p)

    note_text = _disclosure_note(cfg, language)
    if note_text and not _has_note(text, cfg, note_text):
        joiner = "\n\n" if text.strip() else ""
        text = f"{text.rstrip()}{joiner}{note_text}"
    return text


# -------------------------------------------------------------------- tags --
def render_tags(channel, video_tags):
    """Channel tags first, then video tags; case-insensitive de-duped (first
    spelling wins), blanks dropped, capped at YouTube's 15-tag limit."""
    branding = _channel_branding(channel)
    channel_tags = branding.get("tags")
    channel_tags = list(channel_tags) if isinstance(channel_tags, (list, tuple)) else []
    video_tags = list(video_tags) if isinstance(video_tags, (list, tuple)) else []

    seen = set()
    out = []
    for tag in channel_tags + video_tags:
        if not isinstance(tag, str):
            continue
        tag = tag.strip()
        if not tag:
            continue
        key = tag.lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(tag)
        if len(out) >= TAGS_LIMIT:
            break
    return out


# ------------------------------------------------------------ intro / outro --
def _resolve_existing_path(rel_or_abs, root):
    if not isinstance(rel_or_abs, str) or not rel_or_abs.strip():
        return None
    path = rel_or_abs if os.path.isabs(rel_or_abs) else os.path.join(root or "", rel_or_abs)
    path = os.path.abspath(path)
    return path if os.path.exists(path) else None


def channel_intro_outro(channel, root):
    """Absolute (intro_path, outro_path) from the channel's branding, each
    None unless the file actually exists — a branding record that points at
    a deleted or never-uploaded asset must not stall or crash a render."""
    branding = _channel_branding(channel)
    intro = _resolve_existing_path(branding.get("intro_path"), root)
    outro = _resolve_existing_path(branding.get("outro_path"), root)
    return intro, outro
