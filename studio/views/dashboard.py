"""The studio's front page (`/`): the job queue, and the per-channel overview.

`/dashboard` (templates/production.html) reads the FILESYSTEM — it infers what
the detached render processes are doing from the artifacts in `output/`. This
page reads the DATABASE: channels, jobs, uploads. That split is deliberate and
stays that way; a directory walk of every episode ever rendered does not belong
in front of the page the operator opens first.

The channel overview answers the one question a multi-channel operator has
every morning: *where do I have to approve something, and is the rest running?*
It used to be a four-column table (name, type, status, open ideas) in which a
channel with a video waiting for approval, a channel mid-render, and a channel
that had produced nothing for a week all rendered as the same grey row — the
same defect that was found and fixed on the production page, where sixteen
episodes blocked on a human wore the same badge as sixteen with nothing to do.
"""
import time

from flask import Blueprint, current_app, redirect, render_template

from studio import models, profiles
from studio.setup import setup_status
from studio.views import VIDEO_TYPES

bp = Blueprint("dashboard", __name__)

# A machine is working on it right now (scripting/tts/captions/rendering, and
# uploading — the uploader thread counts as work in flight).
RUNNING_STATES = tuple(models.WORKING_JOB_STATES)
# The one job state that means a PERSON has to act: the video is rendered and
# sits on /review until somebody approves or rejects it.
AWAITING_HUMAN = "ready"
# Approved but not uploaded yet. Not human work — but it is where a
# `needs_reauth` channel silently piles up (studio/scheduler.py refuses to
# upload for such a channel), so it belongs on the channel's own card.
AWAITING_UPLOAD = "approved"

# How long an active channel may go without a single job before this page calls
# it silent. Two days: longer than any legitimate render, short enough that a
# channel which fell out of production is noticed the next morning.
SILENT_AFTER_SECONDS = 2 * 24 * 3600

# Dating every channel's last job needs the newest jobs, not all of them. A
# channel that appears in none of the newest N has produced nothing in a very
# long time — which is exactly the state the silence warning is looking for, so
# the window truncating is a correct answer rather than a missing one.
RECENT_JOB_WINDOW = 500


def _age_seconds(stamp, now=None):
    """Seconds since a ``models.now()`` stamp (``"%Y-%m-%d %H:%M:%S"``).

    Returns ``None`` for a missing or unparseable value so the template can
    branch on it — a single malformed timestamp in one row must never be what
    takes down the front page.
    """
    if not stamp:
        return None
    try:
        parsed = time.mktime(time.strptime(str(stamp), "%Y-%m-%d %H:%M:%S"))
    except (TypeError, ValueError, OverflowError):
        return None
    return max(0.0, (time.time() if now is None else now) - parsed)


def _is_silent(row):
    """Is this channel an active-but-dead channel?

    The quiet failure this page exists to catch: a channel is `active`, so
    every status badge in the app calls it healthy, but its `schedule_json` is
    empty. `scheduler._enqueue_due_slots` iterates `channel["schedule"]` — an
    empty schedule means the worker never creates a single job for it. Nothing
    breaks, nothing is logged, and the channel just never produces again.

    Only claimed when there is genuinely nothing in flight: a channel with a
    queued or running job is producing right now, whatever its schedule says.
    """
    if row["status"] != "active":
        return False
    if row["running"] or row["queued"] or row["waiting_review"] or row["waiting_upload"]:
        return False
    if row["schedule"]:
        return False
    age = row["last_job_age"]
    return age is None or age >= SILENT_AFTER_SECONDS


def channel_overview(cfg, channels=None):
    """One row per channel, ordered so the work a human must do comes first.

    Every row carries what the operator scans for: language and content
    profile (what this channel actually renders — `profiles.resolve_profile`,
    not the raw `default_video_type` column, because a profile overrides it),
    the counts in flight, when it last uploaded, and whether it is silently
    doing nothing.

    Ages are seconds; the template runs them through the `duration` filter.
    Nothing here formats a number for display.
    """
    channels = models.list_channels() if channels is None else channels
    if not channels:
        return []

    # Three targeted queries for the whole page, not three per channel.
    running, queued, waiting_review, waiting_upload = {}, {}, {}, {}
    for job in models.jobs_in_states(
        RUNNING_STATES + ("queued", AWAITING_HUMAN, AWAITING_UPLOAD)
    ):
        bucket = {
            "queued": queued,
            AWAITING_HUMAN: waiting_review,
            AWAITING_UPLOAD: waiting_upload,
        }.get(job["status"], running)
        cid = job["channel_id"]
        bucket[cid] = bucket.get(cid, 0) + 1

    last_job = {}
    for job in models.list_jobs(limit=RECENT_JOB_WINDOW):  # newest id first
        last_job.setdefault(
            job["channel_id"],
            job["finished_at"] or job["started_at"] or job["created_at"],
        )

    last_upload = {}
    for upload in models.list_uploads(limit=RECENT_JOB_WINDOW):  # newest id first
        last_upload.setdefault(upload["channel_id"], upload["uploaded_at"])

    now = time.time()
    rows = []
    for channel in channels:
        cid = channel["id"]
        profile = profiles.resolve_profile(cfg, channel)
        row = dict(channel)
        row["profile"] = (
            profile["name"] or profile["video_type"] or channel["default_video_type"] or "—"
        )
        row["language"] = profile["language"] or models.channel_language(channel)
        row["running"] = running.get(cid, 0)
        row["queued"] = queued.get(cid, 0)
        row["waiting_review"] = waiting_review.get(cid, 0)
        row["waiting_upload"] = waiting_upload.get(cid, 0)
        row["last_job_age"] = _age_seconds(last_job.get(cid), now)
        row["last_upload_at"] = last_upload.get(cid)
        row["last_upload_age"] = _age_seconds(row["last_upload_at"], now)
        row["silent"] = _is_silent(row)
        # Used by the template to split "produces something right now" from
        # "has nothing to do" — those two must not render alike.
        row["busy"] = bool(row["running"] or row["queued"] or row["waiting_upload"])
        rows.append(row)

    # Waiting on a human first, then the ones that are silently doing nothing,
    # then the rest. The template splits on the same fields, so the order
    # inside each of its sections is this one.
    rows.sort(key=lambda r: (
        0 if r["waiting_review"] else 1,
        0 if r["silent"] else 1,
        (r["name"] or "").lower(),
        r["id"],
    ))
    return rows


@bp.route("/")
def index():
    cfg = current_app.config["STUDIO_CFG"]
    channels = models.list_channels()
    if not channels:
        status = setup_status(cfg)
        if not status["llm"]["ok"]:
            return redirect("/setup")
    counts = {
        "queued": models.count_jobs("queued"),
        "ready": models.count_jobs("ready"),
        "failed": models.count_jobs("failed"),
        "uploads_today": models.uploads_today(),
        # See studio/views/api.py: 'approved'/'uploading' were counted nowhere,
        # so an upload blocked on a channel re-auth was invisible in every view.
        "waiting_upload": models.count_jobs("approved") + models.count_jobs("uploading"),
    }
    pending_ideas = {
        c["id"]: len(models.list_ideas(channel_id=c["id"], status="pending"))
        for c in channels
    }
    # The filesystem/production view (scan_output) lives at /dashboard —
    # this page is the job-queue view and deliberately does not scan output/.
    # The other human gate, the per-take kidsong queue, is filesystem state
    # with no channel behind it (an episode base is a timestamp plus the song
    # title), so it cannot be attributed to a channel card and is reached
    # through the counter base.html keeps live in the nav instead.
    return render_template(
        "dashboard.html",
        channel_rows=channel_overview(cfg, channels),
        counts=counts,
        pending_ideas=pending_ideas,
        recent_jobs=models.list_jobs(limit=12),
    )


@bp.route("/quick")
def quick():
    cfg = current_app.config["STUDIO_CFG"]
    return render_template(
        "index.html",
        types=VIDEO_TYPES,
        channels=models.list_channels(status="active"),
        backend=cfg["llm"]["backend"],
        privacy=cfg["youtube"]["privacy"],
    )
