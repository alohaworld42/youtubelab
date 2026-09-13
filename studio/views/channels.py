import re

from flask import Blueprint, current_app, jsonify, redirect, render_template, request, url_for

from studio import ideas as ideas_mod
from studio import models, oauth, profiles, scheduler
from studio.views import STYLES, VIDEO_TYPES, VIDEO_TYPE_IDS

bp = Blueprint("channels", __name__, url_prefix="/channels")

# The languages the editor offers. `models._normalize_language` accepts any
# tag, but a free-text field here would let "Deutsch" or "de_DE" reach the
# column that `profiles.resolve_profile` and `branding._disclosure_note` match
# on — a channel that silently uploads with the wrong disclosure sentence.
LANGUAGES = (("de", "Deutsch"), ("en", "English"))
LANGUAGE_CODES = tuple(code for code, _ in LANGUAGES)

# What `studio.branding.render_title`/`render_description` can actually fill.
TEMPLATE_PLACEHOLDERS = ("topic", "channel", "language")

# Mirrors `studio.branding._PLACEHOLDER_RE` (private there on purpose). The
# two must agree: this decides what we accept, that one decides what gets
# filled in — see `unknown_placeholders`.
_PLACEHOLDER_RE = re.compile(r"\{(\w+)\}")

# Every branding key the form edits, and how it is read out of the submission.
# `models.channel_branding` also carries "tags" (its own field, comma-separated)
# and the two colours (behind the "use colours" checkbox), handled separately.
_BRANDING_TEXT_FIELDS = ("title_template", "description_template",
                         "intro_path", "outro_path", "logo_path", "font")
_BRANDING_COLOR_FIELDS = ("primary_color", "secondary_color")
_BRANDING_FIELDS = _BRANDING_TEXT_FIELDS + _BRANDING_COLOR_FIELDS + ("tags",)


def normalize_schedule(raw):
    """`(["09:00", "18:30"], ["9am", "25:00"])` — accepted slots, rejected ones.

    `scheduler.due_slots` skips anything it cannot parse, so an unparseable
    entry is not a small mistake: the channel keeps its schedule, shows it back
    in the form, and never fires. Accepts "9:00" and "09.00" (normalising both
    to "09:00"); rejects anything that is not a real 24h time.
    """
    accepted, rejected = [], []
    for chunk in (raw or "").replace(";", ",").split(","):
        slot = chunk.strip()
        if not slot:
            continue
        parts = slot.replace(".", ":").split(":")
        if len(parts) != 2:
            rejected.append(slot)
            continue
        try:
            hour, minute = int(parts[0]), int(parts[1])
        except ValueError:
            rejected.append(slot)
            continue
        if not (0 <= hour <= 23 and 0 <= minute <= 59):
            rejected.append(slot)
            continue
        normalized = f"{hour:02d}:{minute:02d}"
        if normalized not in accepted:
            accepted.append(normalized)
    return sorted(accepted), rejected


def normalize_tags(raw):
    """`"a, b ,, c"` -> `["a", "b", "c"]`.

    Trimmed, blanks dropped, order kept, case-insensitive duplicates dropped
    (first spelling wins — the same rule `studio.branding.render_tags` applies
    downstream, so what the operator sees stored is what YouTube gets).
    Newlines count as separators too: a tag list is usually pasted.
    """
    tags, seen = [], set()
    for chunk in (raw or "").replace("\n", ",").split(","):
        tag = chunk.strip()
        if not tag or tag.lower() in seen:
            continue
        seen.add(tag.lower())
        tags.append(tag)
    return tags


def unknown_placeholders(*templates):
    """The `{placeholder}` names in `templates` that branding cannot fill.

    `studio.branding._fill_template` *deletes* a placeholder it does not know,
    so `{thema}` instead of `{topic}` produced a title with no topic in it —
    on every upload, with nothing anywhere saying so. Reported in the order
    typed, each name once.
    """
    unknown = []
    for template in templates:
        for name in _PLACEHOLDER_RE.findall(template or ""):
            if name not in TEMPLATE_PLACEHOLDERS and name not in unknown:
                unknown.append(name)
    return unknown


def profile_choices(cfg, channel):
    """The pickable content profiles: everything in `cfg["content_profiles"]`,
    plus the channel's own profile when the config no longer defines it.

    Dropping that stale name from the list would make the select submit
    something else, so merely *opening* the page and saving would rewrite a
    channel's profile — the config was edited, the channel was not. It is
    shown, flagged as unknown (`known=False`), and kept on save.
    """
    known = [str(name) for name in profiles.profile_names(cfg)]
    choices = [{"name": name, "known": True} for name in known]
    current = str((channel or {}).get("content_profile") or "")
    if current and current not in known:
        choices.append({"name": current, "known": False})
    return choices


def _language_from_form(channel, form):
    """The submitted language if the editor offers it, else the channel's
    current one. The field is a two-option select: anything else is a stale
    form or a crafted POST, not a value to store."""
    if "language" not in form:
        return models.channel_language(channel)
    lang = (form.get("language") or "").strip().lower()
    return lang if lang in LANGUAGE_CODES else models.channel_language(channel)


def _has_branding_fields(form):
    return any(f"branding_{key}" in form for key in _BRANDING_FIELDS)


def _branding_from_form(channel, form):
    """The channel's stored branding with this submission laid over it.

    Only keys whose field is actually in the submission are touched, so a POST
    that carries no branding fields at all (the API-shaped posts the older
    tests make) cannot blank a channel's branding as a side effect.
    """
    branding = models.channel_branding(channel)
    for key in _BRANDING_TEXT_FIELDS:
        field = f"branding_{key}"
        if field in form:
            branding[key] = (form.get(field) or "").strip()
    if "branding_tags" in form:
        branding["tags"] = normalize_tags(form.get("branding_tags"))
    # `type=color` has no empty state: a browser submits #000000 for a picker
    # the operator never touched, which would give every saved channel a black
    # brand colour it never asked for. The checkbox is that empty state.
    use_colors = bool(form.get("branding_use_colors"))
    for key in _BRANDING_COLOR_FIELDS:
        field = f"branding_{key}"
        if field in form:
            branding[key] = (form.get(field) or "").strip() if use_colors else ""
    return branding


def _channel_with_form(channel, form):
    """The stored channel row with this submission's values laid over it, so a
    rejected save re-renders the form the user is looking at rather than the
    state they were editing away from."""
    merged = dict(channel)
    for field in ("name", "niche", "description", "default_video_type", "privacy", "style"):
        if field in form:
            merged[field] = form.get(field) or ""
    merged["made_for_kids"] = 1 if form.get("made_for_kids") else 0
    merged["language"] = _language_from_form(channel, form)
    if "content_profile" in form:
        # Kept verbatim, even when it is the value that caused the rejection:
        # the operator has to see what they picked to be able to fix it.
        merged["content_profile"] = (form.get("content_profile") or "").strip()
    if _has_branding_fields(form):
        merged["branding"] = _branding_from_form(channel, form)
    merged["voices"] = {
        spk: (form.get(f"voice_{spk}") or "").strip()
        for spk in ("narrator", "speaker_a", "speaker_b")
        if (form.get(f"voice_{spk}") or "").strip()
    }
    merged["asset_prefs"] = dict(channel.get("asset_prefs") or {})
    merged["asset_prefs"]["gameplay_urls"] = [
        u.strip() for u in (form.get("gameplay_urls") or "").splitlines() if u.strip()
    ]
    merged["asset_prefs"]["source_order"] = (
        ["gameplay", "stock"] if form.get("source_order", "gameplay") == "gameplay"
        else ["stock", "gameplay"]
    )
    return merged


@bp.route("/")
def index():
    return render_template("channels.html", channels=models.list_channels())


@bp.route("/connect", methods=["POST"])
def connect():
    cfg = current_app.config["STUDIO_CFG"]
    started = oauth.start_connect(cfg)
    return jsonify({"started": started, **oauth.connect_status()})


@bp.route("/connect/status")
def connect_status():
    return jsonify(oauth.connect_status())


def _render_edit(channel, cfg, **extra):
    """One place that knows what `channel_edit.html` needs — the GET and the
    rejected POST used to build that argument list separately, which is how a
    field ends up rendering on one path and blank on the other."""
    return render_template(
        "channel_edit.html",
        channel=channel,
        types=VIDEO_TYPES,
        styles=[s for s in STYLES if s],
        languages=LANGUAGES,
        profile_options=profile_choices(cfg, channel),
        placeholders=TEMPLATE_PLACEHOLDERS,
        **extra,
    )


@bp.route("/<int:channel_id>", methods=["GET", "POST"])
def edit(channel_id):
    channel = models.get_channel(channel_id)
    if not channel:
        return redirect(url_for("channels.index"))
    cfg = current_app.config["STUDIO_CFG"]

    if request.method == "POST":
        form = request.form
        errors = {}
        # A slot the scheduler cannot parse is silently ignored by
        # `scheduler.due_slots`, so a typo like "9am" used to save happily and
        # the channel then produced nothing, forever, with no error anywhere.
        # Normalise what we can ("9:00", "09.00") and refuse the rest.
        schedule, bad = normalize_schedule(form.get("schedule"))
        if bad:
            errors["schedule_error"] = (
                "Diese Zeiten versteht der Zeitplan nicht: %s. "
                "Format ist HH:MM (24 Stunden), z.B. 09:00, 18:30."
                % ", ".join(bad)
            )

        branding = _branding_from_form(channel, form)
        # Only what was typed in this submission is judged: a template already
        # in the DB is not something the operator can fix from a 400 they did
        # not cause.
        bad_placeholders = unknown_placeholders(*[
            form.get(f"branding_{key}")
            for key in ("title_template", "description_template")
            if f"branding_{key}" in form
        ])
        if bad_placeholders:
            errors["branding_error"] = (
                "Diese Platzhalter kann die Vorlage nicht füllen: %s. "
                "Erlaubt sind nur %s — alles andere wird beim Upload "
                "ersatzlos gestrichen."
                % (", ".join("{%s}" % p for p in bad_placeholders),
                   ", ".join("{%s}" % p for p in TEMPLATE_PLACEHOLDERS))
            )

        content_profile = (form.get("content_profile") or "").strip()
        pickable = {c["name"] for c in profile_choices(cfg, channel)}
        if content_profile and content_profile not in pickable:
            errors["profile_error"] = (
                "Unbekanntes Content-Profil: %s. Wähle eines aus der Liste "
                "oder „aus Video-Typ ableiten“." % content_profile
            )

        if errors:
            # Re-render with everything the user typed still in place. Handing
            # back the stored channel row instead would throw away the other
            # edits they made in the same visit — a worse bug than the typo.
            return _render_edit(
                _channel_with_form(channel, form), cfg,
                schedule_raw=form.get("schedule") or "",
                **errors,
            ), 400
        voices = {}
        for spk in ("narrator", "speaker_a", "speaker_b"):
            v = (form.get(f"voice_{spk}") or "").strip()
            if v:
                voices[spk] = v
        gameplay_urls = [
            u.strip() for u in (form.get("gameplay_urls") or "").splitlines() if u.strip()
        ]
        vt = form.get("default_video_type")
        style = form.get("style", "")
        # Fields the form may legitimately not carry at all are only written
        # when they were submitted — writing a default for an absent field is
        # how a partial POST wipes a column nobody was editing.
        extra = {}
        if "language" in form:
            extra["language"] = _language_from_form(channel, form)
        if "content_profile" in form:
            extra["content_profile"] = content_profile
        if _has_branding_fields(form):
            extra["branding"] = branding
        models.update_channel(
            channel_id,
            name=(form.get("name") or channel["name"]).strip(),
            niche=(form.get("niche") or "").strip(),
            description=(form.get("description") or "").strip(),
            default_video_type=vt if vt in VIDEO_TYPE_IDS else channel["default_video_type"],
            privacy=form.get("privacy") or "private",
            style=style if style in STYLES else "",
            made_for_kids=1 if form.get("made_for_kids") else 0,
            schedule=schedule,
            voices=voices,
            asset_prefs={
                "gameplay_urls": gameplay_urls,
                "source_order": ["gameplay", "stock"]
                if form.get("source_order", "gameplay") == "gameplay"
                else ["stock", "gameplay"],
            },
            **extra,
        )
        return redirect(url_for("channels.index"))

    return _render_edit(channel, cfg)


@bp.route("/<int:channel_id>/pause", methods=["POST"])
def pause(channel_id):
    models.update_channel(channel_id, status="paused")
    return redirect(url_for("channels.index"))


@bp.route("/<int:channel_id>/resume", methods=["POST"])
def resume(channel_id):
    models.update_channel(channel_id, status="active")
    return redirect(url_for("channels.index"))


@bp.route("/<int:channel_id>/delete", methods=["POST"])
def delete(channel_id):
    models.delete_channel(channel_id)
    return redirect(url_for("channels.index"))


@bp.route("/<int:channel_id>/generate-now", methods=["POST"])
def generate_now(channel_id):
    channel = models.get_channel(channel_id)
    if not channel:
        return jsonify({"error": "unbekannter Kanal"}), 404
    data = request.get_json(silent=True) or {}
    topic = (data.get("topic") or "").strip() or None
    vt = data.get("video_type")
    if vt not in VIDEO_TYPE_IDS:
        vt = channel["default_video_type"]
    # Children's-content gate (same policy as the cascade and /generate).
    if ideas_mod.is_kidsong_type(vt):
        reason = ideas_mod.kidsong_topic_rejection(topic)
        if reason:
            return jsonify({
                "error": "Thema ist für ein Kinderlied nicht geeignet (%s). "
                         "Bitte ein einfaches Alltagsthema wählen "
                         "(Tiere, Zählen, Farben, Wetter, Teilen, Familie)." % reason
            }), 400

    job_id = models.create_job(channel_id=channel_id, video_type=vt, topic=topic)
    scheduler.wake()
    return jsonify({"job_id": job_id})
