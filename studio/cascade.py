"""One-click cascade: idea generation -> queued kidsong jobs.

The scheduler already knows how to render a job whose video_type is 'kidsong'
(pipeline.generate dispatches to pipeline.kidsong.generate). What was missing is
the front of the funnel: somebody had to create the channel, feed it ideas and
turn those ideas into jobs. This module does exactly that, so the "Kaskade
starten" button on the dashboard is the only thing a non-developer needs.

Deliberate safety rules:
  * jobs are created with auto_approve=False — a kids' video NEVER reaches
    YouTube without a human pressing approve in /review.
  * the cascade channel keeps an EMPTY schedule, so the time-slot cron in
    studio.scheduler does not enqueue a second copy of the same work.
  * if the LLM is unreachable we fall back to built-in, original,
    child-appropriate topics (mirrors pipeline.kidsong.lyrics.FALLBACK_SONG)
    and say so in the summary instead of failing.
"""
import logging
import os

from studio import ideas as ideas_mod
from studio import models, scheduler

log = logging.getLogger("studio.cascade")

KIDSONG_VIDEO_TYPE = "kidsong"
KIDSONG_CHANNEL_NAME = "Kidsong"
KIDSONG_NICHE = "Original sing-along songs for toddlers"
KIDSONG_DESCRIPTION = (
    "A preschool sing-along channel starring original, adorable Black toddler "
    "characters. Every video is a short, repetitive nursery rhyme about everyday "
    "toddler life — routines, feelings, numbers, colors, animals and kindness — "
    "with GPU-rendered 3D scenes. All songs and characters are original; the "
    "channel never references real brands, shows or characters."
)

MIN_IDEA_BATCH = 5
MAX_COUNT = 10


def clamp_count(count, default=3):
    """1..10, tolerant of None/strings/junk from the JSON body."""
    try:
        count = int(count)
    except (TypeError, ValueError):
        count = default
    return max(1, min(count, MAX_COUNT))


def channel_can_upload(channel, cfg=None):
    """True if this channel row could actually reach YouTube.

    Needs a yt_channel_id AND a token file at the path oauth resolves for it
    (studio.oauth.channel_upload_paths, which falls back to <root>/token.json
    when the row has no token_path of its own).
    """
    if not channel or not (channel.get("yt_channel_id") or "").strip():
        return False
    try:
        from studio.oauth import channel_upload_paths

        _secret, token = channel_upload_paths(channel, cfg or {"_root": "."})
    except Exception:
        return False
    return bool(token) and os.path.exists(token)


# words that mark a channel as "the kids' sing-along one" even when its
# default_video_type was never switched to kidsong (the exact bug: the real
# OAuth-connected channel sat on default_video_type='facts').
_KIDS_WORDS = ("kid", "kinder", "krippe", "lieder", "toddler", "preschool",
               "nursery", "sing", "song", "lern", "baby")


def _kidsong_affinity(channel):
    """How strongly this row looks like the kidsong channel. 0 == not a candidate."""
    score = 0
    if channel.get("default_video_type") == KIDSONG_VIDEO_TYPE:
        score += 8
    if channel.get("made_for_kids"):
        score += 4
    if ideas_mod.is_kids_style(channel.get("style")):
        score += 3
    haystack = " ".join(str(channel.get(k) or "") for k in
                        ("name", "yt_channel_title", "niche", "description")).lower()
    if any(w in haystack for w in _KIDS_WORDS):
        score += 2
    if (channel.get("name") or "").strip().lower() == KIDSONG_CHANNEL_NAME.lower():
        score += 1
    return score


def ensure_kidsong_channel(cfg=None):
    """Return the kidsong channel, strongly preferring one that can upload.

    History: this used to match only on default_video_type=='kidsong' (or the
    literal name "Kidsong") and otherwise CREATE a channel. The user's real,
    OAuth-connected channel had default_video_type='facts', so every cascade
    run queued jobs against a freshly created channel with no yt_channel_id
    and no token — work that could never upload. Creating a disconnected
    channel is now the last resort, and callers can see it via
    channel_can_upload().
    """
    candidates = []
    for channel in models.list_channels():
        if (channel.get("status") or "").lower() in ("paused", "disabled", "archived"):
            continue
        score = _kidsong_affinity(channel)
        if score:
            candidates.append((channel_can_upload(channel, cfg), score, -channel["id"], channel))

    if candidates:
        # connected first, then kidsong affinity, then the oldest row
        candidates.sort(key=lambda t: (t[0], t[1], t[2]), reverse=True)
        channel = candidates[0][3]
        if channel.get("default_video_type") != KIDSONG_VIDEO_TYPE:
            # durable repair: stop this row from being missed on the next run
            previous = channel.get("default_video_type")
            models.update_channel(channel["id"], default_video_type=KIDSONG_VIDEO_TYPE)
            channel = models.get_channel(channel["id"])
            log.info("adopted existing channel %s as the kidsong channel", channel["id"])
            # The repair is right, but it silently re-labels a channel whose
            # pending ideas were written for the OLD type. Those ideas do not
            # become kids' songs just because the channel changed — say so
            # loudly here instead of letting the picker discover it later.
            _, stale = models.partition_pending_ideas(channel["id"], KIDSONG_VIDEO_TYPE)
            if stale:
                log.warning(
                    "channel %s switched %s -> %s; %d pending idea(s) are not "
                    "toddler-appropriate and stay pending, unused: %s",
                    channel["id"], previous, KIDSONG_VIDEO_TYPE, len(stale),
                    "; ".join("%r (%s)" % (row["topic"], reason)
                              for row, reason in stale[:5]),
                )
        return channel

    channel_id = models.create_channel(KIDSONG_CHANNEL_NAME)
    models.update_channel(
        channel_id,
        niche=KIDSONG_NICHE,
        description=KIDSONG_DESCRIPTION,
        default_video_type=KIDSONG_VIDEO_TYPE,
        status="active",
        made_for_kids=1,
        # empty schedule on purpose: the cascade button is the only trigger
        schedule=[],
    )
    log.info("created kidsong channel %s", channel_id)
    return models.get_channel(channel_id)


def _pending_count(channel_id, video_type=KIDSONG_VIDEO_TYPE):
    """Pending ideas this run could actually USE.

    Unsuitable leftovers (e.g. facts ideas on a repointed channel) must not
    count, otherwise the cascade thinks it has enough material and skips the
    top-up, then queues nothing.
    """
    suitable, _rejected = models.partition_pending_ideas(channel_id, video_type)
    return len(suitable)


def _top_up_ideas(channel, cfg, needed, summary):
    """Make sure at least `needed` more pending ideas exist. Never raises."""
    if needed <= 0:
        return

    try:
        summary["ideas_generated"] = ideas_mod.generate_ideas(
            channel, cfg, k=max(needed, MIN_IDEA_BATCH),
            video_type=KIDSONG_VIDEO_TYPE,
        )
    except Exception as e:  # LLM down, bad JSON, no API key, ...
        log.warning("kidsong idea generation failed, using fallback topics: %s", e)
        summary["llm_error"] = f"{type(e).__name__}: {e}"

    shortfall = needed - summary["ideas_generated"]
    if shortfall > 0:
        added = ideas_mod.add_fallback_kidsong_ideas(
            channel, shortfall, video_type=KIDSONG_VIDEO_TYPE
        )
        if added:
            summary["fallback_used"] = True
            summary["ideas_generated"] += added


def run_cascade(cfg, count=3, channel_id=None):
    """Ideas -> queued kidsong jobs. Returns a summary dict for the UI."""
    count = clamp_count(count)

    channel = models.get_channel(channel_id) if channel_id else None
    if not channel:
        channel = ensure_kidsong_channel(cfg)

    summary = {
        "channel_id": channel["id"],
        "channel_name": channel["name"],
        "requested": count,
        "ideas_generated": 0,
        "jobs_created": 0,
        "job_ids": [],
        "fallback_used": False,
        "channel_connected": channel_can_upload(channel, cfg),
    }
    if not summary["channel_connected"]:
        summary["channel_warning"] = (
            "Kanal „%s“ ist NICHT mit YouTube verbunden (kein OAuth-Token). "
            "Die Jobs werden gerendert, aber es kann nichts hochgeladen werden, "
            "bis der Kanal unter /channels verbunden ist."
            % channel["name"]
        )
        log.warning(
            "cascade queueing against DISCONNECTED channel %s (%s) — "
            "no yt_channel_id/token, uploads impossible until connected",
            channel["id"], channel["name"],
        )

    default_type = channel.get("default_video_type") or KIDSONG_VIDEO_TYPE
    _top_up_ideas(channel, cfg, count - _pending_count(channel["id"], default_type), summary)

    style = channel.get("style") or None

    suitable, rejected = models.partition_pending_ideas(channel["id"], default_type)
    summary["ideas_rejected"] = len(rejected)
    if rejected:
        # Fail loudly: a cascade that quietly queues fewer videos than asked is
        # exactly how the "scariest folkloric creatures" batch went unnoticed.
        summary["rejected_topics"] = [row["topic"] for row, _r in rejected][:10]
        summary["content_warning"] = (
            "%d vorhandene Idee(n) sind für einen Kinderlied-Kanal NICHT geeignet "
            "und wurden übersprungen (bleiben unangetastet offen): %s"
            % (len(rejected),
               "; ".join("„%s“" % row["topic"] for row, _r in rejected[:5]))
        )
        log.warning(
            "cascade channel=%s skipped %d unsuitable idea(s) for video_type=%s: %s",
            channel["id"], len(rejected), default_type,
            "; ".join("%r (%s)" % (row["topic"], reason) for row, reason in rejected[:10]),
        )

    for idea in suitable[:count]:
        job_id = models.create_job(
            channel_id=channel["id"],
            idea_id=idea["id"],
            video_type=idea.get("video_type") or default_type,
            topic=idea["topic"],
            slot_key=None,
            auto_approve=False,  # kids' videos always need human approval
            style=style,
        )
        if not job_id:
            summary["skipped_reason"] = "Job konnte nicht angelegt werden."
            break
        models.set_idea_status(idea["id"], "in_progress")
        summary["job_ids"].append(job_id)
        summary["jobs_created"] += 1

    if summary["jobs_created"] < count and "skipped_reason" not in summary:
        summary["skipped_reason"] = (
            "Keine geeigneten offenen Ideen mehr — nur %d von %d Jobs erstellt."
            % (summary["jobs_created"], count)
        )
        if rejected:
            summary["skipped_reason"] += " " + summary["content_warning"]

    if summary["jobs_created"]:
        scheduler.wake()
    log.info(
        "cascade: channel=%s ideas=%s jobs=%s",
        channel["id"], summary["ideas_generated"], summary["jobs_created"],
    )
    return summary
