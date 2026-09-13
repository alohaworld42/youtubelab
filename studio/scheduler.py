"""Background worker: enqueues jobs from channel schedules, renders one video
at a time (CPU-bound: Whisper + x264), uploads approved videos.

One daemon thread, tick every studio.tick_seconds (default 15s). `wake()` skips
the wait — called whenever the UI creates work.
"""
import logging
import os
import threading
import time
import traceback

from pipeline import gpu_lock
from pipeline.config import abspath
from studio import assets, branding, ideas, models, profiles
from studio.db import get_db

log = logging.getLogger("studio.scheduler")

_instance = None


def _studio_cfg(cfg, key, default):
    return (cfg.get("studio") or {}).get(key, default)


def due_slots(schedule, now_struct=None):
    """Slots (\"HH:MM\") from a channel schedule whose time has passed today.

    Pure function for testability. Invalid entries are ignored.
    """
    now_struct = now_struct or time.localtime()
    now_minutes = now_struct.tm_hour * 60 + now_struct.tm_min
    due = []
    for slot in schedule or []:
        try:
            h, m = str(slot).strip().split(":")
            slot_minutes = int(h) * 60 + int(m)
        except ValueError:
            continue
        if 0 <= slot_minutes <= now_minutes:
            due.append(f"{int(h):02d}:{int(m):02d}")
    return due


def slot_key_for(channel_id, date_str, slot):
    return f"{channel_id}:{date_str}:{slot}"


def _made_for_kids(channel, job):
    """Whether this upload must be declared "Made for Kids" to YouTube.

    Every signal here is an OPT-IN, never an opt-out: a video that is children's
    content stays children's content whatever any single column says. This used
    to pass the channel flag alone, so a kidsong job queued against a channel
    whose box was unchecked — /quick with a channel picked, or a channel someone
    edited — uploaded a toddler sing-along declared as not-for-kids. The kidsong
    CLI path (`pipeline/kidsong/generate.py`) already forces True; the studio's
    own uploader was the one place that did not, against this repo's own rule
    ("Kidsong uploads: always declare made_for_kids=True", AGENTS.md).

    The **style** is the third signal, and it closes a gap the first two leave
    open. YouTube's classification rules do not ask what a database column is
    called: content counts as made for kids when children are the primary
    audience, and also when its characters, songs, activities or language appeal
    to children as part of a mixed audience. A `style="kids"` video is exactly
    that — `studio/svm.py` gives it a kids voice, happy music at high volume and
    a pink caption background, and `studio/views/api.py` forces the style on for
    the kids video type — yet a channel carrying that style with an unticked box
    and a non-kids `video_type` used to upload declared not-for-kids. That is
    not hypothetical: `studio/ideas.py` records that the real channel
    KinderKrippenLernLieder "used to be a 'facts' channel", which is precisely
    this shape. `studio/cascade.py` already scores the same style as a kids
    signal; the uploader was the one place that ignored it.

    Deliberately NOT inferred: channel name or description wording. `cascade`
    may guess from those because guessing wrong there only mis-ranks a
    candidate row, whereas a wrong COPPA declaration is a legal filing — and it
    is not free in the other direction either (a made-for-kids video loses
    personalised ads, comments and end screens). Style and video type are
    explicit operator choices about the output; prose is not.
    """
    from studio.ideas import is_kids_style, is_kidsong_type

    if channel and channel.get("made_for_kids"):
        return True
    if is_kidsong_type((job or {}).get("video_type")):
        return True
    # The job's own style wins when set; otherwise the channel's default.
    return is_kids_style((job or {}).get("style") or (channel or {}).get("style"))


class Scheduler:
    def __init__(self, cfg):
        self.cfg = cfg
        self.tick_seconds = int(_studio_cfg(cfg, "tick_seconds", 15))
        self.max_attempts = int(_studio_cfg(cfg, "max_attempts", 3))
        self.daily_upload_cap = int(_studio_cfg(cfg, "daily_upload_cap", 6))
        self.upload_tick_seconds = int(
            _studio_cfg(cfg, "upload_tick_seconds", self.tick_seconds)
        )
        self._wake = threading.Event()
        self._upload_wake = threading.Event()
        self._stop = threading.Event()
        self._thread = None
        self._upload_thread = None
        # serializes _process_uploads: the uploader thread owns it, but tests
        # and any future caller must never run a second pass concurrently.
        self._upload_lock = threading.Lock()
        # Date the "daily cap reached" warning was last logged, and the set of
        # jobs already told they are blocked on a channel re-auth — both exist
        # only to keep a per-tick condition from writing a per-tick log line.
        self._cap_warned_for = None
        self._reauth_notified = set()

    # ------------------------------------------------------------- lifecycle --
    def start(self):
        if self._thread and self._thread.is_alive():
            return
        self._recover()
        self._thread = threading.Thread(target=self._loop, daemon=True, name="studio-scheduler")
        self._thread.start()
        # Uploads live on their OWN thread on purpose. _process_one_render()
        # runs the pipeline INLINE and a kidsong render takes ~1 hour, so an
        # approved video used to wait for the current render to finish before
        # anyone even looked at it — wake() could not help, the scheduler
        # thread was blocked inside generate(). Renders stay single-threaded
        # and GPU-locked; uploading is network I/O and needs no GPU.
        self._upload_thread = threading.Thread(
            target=self._upload_loop, daemon=True, name="studio-uploader")
        self._upload_thread.start()
        log.info("scheduler started (tick=%ss, upload tick=%ss)",
                 self.tick_seconds, self.upload_tick_seconds)

    def stop(self):
        self._stop.set()
        self._wake.set()
        self._upload_wake.set()

    def wake(self):
        self._wake.set()
        self._upload_wake.set()

    def _loop(self):
        while not self._stop.is_set():
            try:
                self.tick()
            except Exception:
                log.error("tick crashed:\n%s", traceback.format_exc())
            self._wake.wait(self.tick_seconds)
            self._wake.clear()

    def _upload_loop(self):
        while not self._stop.is_set():
            try:
                self._process_uploads()
            except Exception:
                log.error("upload tick crashed:\n%s", traceback.format_exc())
            self._upload_wake.wait(self.upload_tick_seconds)
            self._upload_wake.clear()

    # -------------------------------------------------------------- recovery --
    def _recover(self):
        """Jobs stuck in transient states after a crash/restart."""
        for job in models.jobs_in_states(models.TRANSIENT_JOB_STATES):
            attempts = job["attempts"] + 1
            if attempts >= self.max_attempts:
                models.update_job(job["id"], status="failed", attempts=attempts,
                                  error="abgebrochen (App-Neustart), max. Versuche erreicht")
                if job["idea_id"]:
                    models.set_idea_status(job["idea_id"], "pending")
            else:
                models.update_job(job["id"], status="queued", attempts=attempts,
                                  error="requeued nach App-Neustart")
        for job in models.jobs_in_states(("uploading",)):
            models.update_job(job["id"], status="approved",
                              error="Upload nach Neustart erneut versuchen")

    # ------------------------------------------------------------------ tick --
    def tick(self):
        # NOTE: uploads are NOT here — they run on self._upload_thread so an
        # approved video never waits behind an hour-long render.
        self._enqueue_due_slots()
        self._process_one_render()

    def _enqueue_due_slots(self):
        date_str = models.today()
        for channel in models.list_channels(status="active"):
            for slot in due_slots(channel["schedule"]):
                key = slot_key_for(channel["id"], date_str, slot)
                if models.slot_key_exists(key):
                    continue
                idea = self._pick_or_generate_idea(channel)
                if not idea:
                    # reserve the slot anyway? no — retry next tick, maybe LLM was down
                    continue
                # The profile decides what this channel makes; it outranks both
                # the idea's own suggestion and the channel's default column.
                # An idea row outlives config changes — repointing a channel
                # from kidsong to hyperframes used to leave its already-queued
                # ideas producing the old type for as long as the backlog
                # lasted, which reads as "the setting did nothing". Channels
                # without a profile keep the previous precedence exactly.
                # Only a REAL profile outranks the idea. `profile_video_type`
                # falls back to the channel's default column when no profile is
                # set, so using it unconditionally would silently demote the
                # idea's own suggestion for every channel in the studio.
                _profile = profiles.resolve_profile(self.cfg, channel)
                profiled_type = _profile["video_type"] if _profile.get("name") else None
                job_id = models.create_job(
                    channel_id=channel["id"],
                    idea_id=idea["id"],
                    video_type=(profiled_type
                                or idea.get("video_type")
                                or channel["default_video_type"]),
                    topic=idea["topic"],
                    scheduled_for=f"{date_str} {slot}:00",
                    slot_key=key,
                )
                if job_id:
                    models.set_idea_status(idea["id"], "in_progress")
                    log.info("enqueued job %s for channel %s slot %s", job_id, channel["name"], slot)

    def _pick_or_generate_idea(self, channel):
        idea = models.next_pending_idea(channel["id"])
        if idea:
            return idea
        try:
            n = ideas.generate_ideas(channel, self.cfg,
                                     k=int(_studio_cfg(self.cfg, "auto_ideas_batch", 5)))
            log.info("auto-generated %s ideas for channel %s", n, channel["name"])
        except Exception as e:
            log.warning("auto idea generation failed for %s: %s", channel["name"], e)
            return None
        return models.next_pending_idea(channel["id"])

    # ---------------------------------------------------------------- render --
    def _process_one_render(self):
        # Single render at a time. Checked against the RENDER states only —
        # models.current_job() also counts "uploading", which since uploads
        # moved to their own thread would let a 2-minute upload stall the
        # render queue for no reason.
        if models.jobs_in_states(models.TRANSIENT_JOB_STATES):
            return
        job = models.next_due_job()
        if not job:
            return
        # Single-GPU mutual exclusion: a CLI batch (output/batch_queue.ps1 ->
        # python -m pipeline.kidsong.generate) can be rendering in its own
        # process right now, and the cascade UI queues kidsong jobs that
        # render right here on this thread. Both hit the same GPU (ComfyUI /
        # ACE-Step / Whisper-cuda) — colliding has already killed a render
        # (ComfyUI timed out mid-job). If another live process holds the
        # lock, skip this tick quietly: the job stays 'queued' untouched (no
        # failure, no burned retry attempt) and the next tick tries again.
        lock = gpu_lock.get_lock(self.cfg)
        if not lock.try_acquire():
            log.debug(
                "GPU lock busy (another render in progress elsewhere) — "
                "skipping this tick, job %s stays queued", job["id"],
            )
            return
        try:
            self._run_pipeline(job)
        finally:
            lock.release()

    def _job_log_path(self, job_id):
        d = os.path.join(self.cfg["_root"], "logs")
        os.makedirs(d, exist_ok=True)
        return os.path.join(d, f"job_{job_id}.log")

    def _run_pipeline(self, job):
        from pipeline.generate import generate

        job_id = job["id"]
        channel = models.get_channel(job["channel_id"]) if job["channel_id"] else None
        log_path = self._job_log_path(job_id)

        stage_map = (
            ("Writing script", "scripting"),
            ("Generating voices", "tts"),
            ("Aligning word-level", "captions"),
            ("Loading background", "rendering"),
        )
        state = {"stage": "scripting"}

        def progress(msg):
            for needle, stage in stage_map:
                if needle in msg and state["stage"] != stage:
                    state["stage"] = stage
                    models.set_job_status(job_id, stage)
            models.add_event(job_id, state["stage"], msg)
            try:
                with open(log_path, "a", encoding="utf-8") as f:
                    f.write(f"{models.now()}  [{state['stage']}] {msg}\n")
            except OSError:
                pass

        models.set_job_status(job_id, "scripting")
        models.update_job(job_id, started_at=models.now())

        style = job.get("style") or (channel.get("style") if channel else None)
        overrides = {}
        extra_context = None
        if channel:
            if channel.get("voices"):
                overrides["voices"] = channel["voices"]
            niche = (channel.get("niche") or "").strip()
            desc = (channel.get("description") or "").strip()
            if niche or desc:
                extra_context = f"{niche}. {desc}".strip(". ")

        # The channel's content profile decides language and look — which
        # product this channel makes, as opposed to `overrides`, which carries
        # per-channel deviations and is handed to the generator separately.
        #
        # Deliberately a LOCAL binding, never `self.cfg`: one Scheduler instance
        # serves every channel for the whole life of the process, so assigning
        # the resolved config back to the instance would leave the studio stuck
        # in the last channel's language — a German episode would be followed by
        # a German "English" one, and only on the second channel would anyone
        # notice. `apply_profile` returns a new dict and never mutates its
        # input, and it never raises: a typo in one channel's profile column
        # must not take down the worker tick for every other channel.
        job_cfg = profiles.apply_profile(self.cfg, channel) if channel else self.cfg
        profile = profiles.resolve_profile(self.cfg, channel) if channel else None
        if profile and profile.get("name"):
            progress(f"Profil {profile['name']} ({profile['language']})")

        # merged view for asset sourcing so style presets steer the tier order
        from pipeline.config import apply_style, merge_overrides

        styled_cfg = apply_style(job_cfg, style)
        asset_cfg = merge_overrides(styled_cfg, overrides or {})
        engine = (styled_cfg.get("render_engine") or "moviepy").lower()

        # short-video-maker sources its own scene B-roll, and kidsong renders
        # its own footage via ComfyUI inside generate_kidsong — both ignore
        # a sourced background asset entirely, so skip our background tier
        # for them (would needlessly download stock / hit the network).
        bg_path, bg_asset_id = None, None
        if engine != "svm" and job["video_type"] != "kidsong":
            try:
                bg_path, bg_asset_id = assets.get_background(
                    channel, job["topic"], None, asset_cfg
                )
                if bg_path:
                    progress(f"Background asset: {os.path.basename(bg_path)}")
            except Exception as e:
                log.warning("asset sourcing failed, falling back to local: %s", e)

        # Contract 3: resume an interrupted kidsong run instead of always
        # starting a fresh episode. resume_base is nullable on the jobs row
        # (studio/db.py migration v3) and gets set below when a kidsong run
        # dies mid-render; a plain "queued -> retry" job (never rendered
        # anything yet) has it as None, so this is a no-op for every other
        # case.
        resume_base = job.get("resume_base")
        try:
            result = generate(
                job["video_type"] or "facts",
                topic=job["topic"],
                do_upload=False,
                cfg=job_cfg,
                on_progress=progress,
                channel_overrides=overrides or None,
                extra_context=extra_context,
                background_path=bg_path,
                style=style,
                resume_base=resume_base,
            )
            # Branding is applied HERE, on the way into the videos row, not at
            # upload time. The operator's whole job is to approve a video, and
            # they can only approve what they can see: metadata stamped on
            # during the upload tick would mean the review page shows one title
            # and YouTube receives another. Storing the final text also makes it
            # editable — a per-channel template is a default, not a cage.
            branded_title = branding.render_title(
                channel, job_cfg, job["topic"], fallback=result["title"]
            )
            branded_description = branding.render_description(
                channel, job_cfg, job["topic"],
                base_description=result["description"],
            )
            branded_tags = branding.render_tags(channel, result["tags"])

            models.create_video(
                job_id=job_id,
                channel_id=job["channel_id"],
                path=result["video_path"],
                title=branded_title,
                description=branded_description,
                tags=branded_tags,
                duration=result["duration"],
                script=result.get("script"),
                bg_asset_id=bg_asset_id,
                thumb_path=result.get("thumb_path"),
            )
            if resume_base:
                # The run finished (whether QC-approved or held for manual
                # review) — nothing left to resume, so clear the pointer
                # rather than leave a stale base a future retry would (if
                # this job were ever requeued) try to pick back up.
                models.update_job(job_id, resume_base=None)
            # A kidsong cut that didn't pass its own QC gate (director mode's
            # cut_qc review) comes back with status != "approved" (currently
            # "needs_review") instead of raising — it's not an error, but it
            # must never auto-upload. Force human review regardless of
            # auto_approve. Other video types never set "status", so this is
            # a no-op for them (unchanged behaviour: approved/ready as before).
            review_status = result.get("status")
            if review_status and review_status != "approved":
                models.set_job_status(job_id, "ready")
                models.add_event(
                    job_id, "review",
                    f"QC gate did not approve the cut (status={review_status}) — "
                    "held for manual review, not auto-approved/uploaded.",
                )
                log.warning(
                    "job %s needs manual review before upload (status=%s)",
                    job_id, review_status,
                )
            else:
                models.set_job_status(job_id, "approved" if job.get("auto_approve") else "ready")
            log.info("job %s ready: %s", job_id, result["title"])
        except Exception as e:
            attempts = job["attempts"] + 1
            err = f"{type(e).__name__}: {e}"
            models.add_event(job_id, "error", err)
            try:
                with open(log_path, "a", encoding="utf-8") as f:
                    f.write(traceback.format_exc())
            except OSError:
                pass
            # Contract 3: a kidsong run that died mid-render tags the
            # exception with the base it was rendering (pipeline/kidsong/
            # generate.py::_run). Only overwrite the job's resume_base when we
            # actually got a fresh one -- an unrelated/earlier failure (e.g.
            # before any base existed) must not clobber a good pointer a
            # previous attempt already recorded.
            new_resume_base = getattr(e, "kidsong_resume_base", None)
            update_fields = {}
            if new_resume_base:
                update_fields["resume_base"] = new_resume_base
            if attempts >= self.max_attempts:
                models.update_job(job_id, status="failed", attempts=attempts, error=err,
                                  finished_at=models.now(), **update_fields)
                if job["idea_id"]:
                    models.set_idea_status(job["idea_id"], "pending")
            else:
                models.update_job(job_id, status="queued", attempts=attempts, error=err,
                                  **update_fields)
            log.error("job %s failed (attempt %s): %s", job_id, attempts, err)
            return

        # Post-success bookkeeping, deliberately OUTSIDE the try above. It used
        # to sit inside it, between set_job_status(ready) and the log line — so
        # a failure in idea bookkeeping, asset touching or cache pruning was
        # caught by the render's own `except`, which then flipped a finished job
        # back to 'queued' and re-rendered it. An hour of GPU thrown away
        # because a cache file could not be deleted. None of this is worth
        # failing a rendered video over.
        for label, work in (
            ("idea status", lambda: job["idea_id"] and models.set_idea_status(job["idea_id"], "used")),
            ("asset touch", lambda: bg_asset_id and models.touch_asset(bg_asset_id)),
            ("cache prune", lambda: assets.prune_cache(self.cfg)),
        ):
            try:
                work()
            except Exception as e:
                log.warning("job %s: %s failed after a successful render (%s) — "
                            "the video is fine, this is bookkeeping only",
                            job_id, label, e)

    # ---------------------------------------------------------------- upload --
    def _claim_for_upload(self, job_id):
        """Atomically flip approved -> uploading. False if somebody beat us.

        A plain read-then-write would race the render thread (which also
        rewrites job rows) and any second upload pass; the WHERE status clause
        makes the claim the single point of truth for "this job is mine".
        """
        db = get_db()
        cur = db.execute(
            "UPDATE jobs SET status='uploading' WHERE id=? AND status='approved'",
            (job_id,),
        )
        db.commit()
        return cur.rowcount == 1

    def _already_uploaded(self, video_id):
        row = get_db().execute(
            "SELECT 1 FROM uploads WHERE video_id=? LIMIT 1", (video_id,)
        ).fetchone()
        return row is not None

    def _process_uploads(self):
        if not self._upload_lock.acquire(blocking=False):
            return  # another pass is already running
        try:
            self._process_uploads_locked()
        finally:
            self._upload_lock.release()

    def _process_uploads_locked(self):
        approved = models.jobs_in_states(("approved",))
        if not approved:
            return
        if models.uploads_today() >= self.daily_upload_cap:
            # Once per day, not once per tick: at the default 15s upload tick
            # this line used to be written ~5700 times between hitting the cap
            # and midnight, burying every other warning in logs/studio.log.
            today = models.today()
            if self._cap_warned_for != today:
                self._cap_warned_for = today
                log.warning(
                    "daily upload cap (%s) reached — %s approved job(s) wait for tomorrow",
                    self.daily_upload_cap, len(approved),
                )
            return

        from pipeline.youtube_upload import upload
        from studio.oauth import channel_upload_paths

        for job in approved:
            if models.uploads_today() >= self.daily_upload_cap:
                break
            channel = models.get_channel(job["channel_id"]) if job["channel_id"] else None
            video = models.get_video_by_job(job["id"])
            if not video or not os.path.exists(video["path"]):
                models.update_job(job["id"], status="failed",
                                  error="Videodatei fehlt", finished_at=models.now())
                continue
            if channel and channel["status"] == "needs_reauth":
                # This job is now stuck indefinitely, and 'approved' shows up in
                # NO view: not the review queue (it is past it), not the failed
                # list, not a dashboard counter. It used to sit there silently
                # forever. Record the reason once on the job so the event trail
                # explains it, and let /api/status count it (see
                # models.count_jobs("approved") in the dashboard's
                # "Wartet auf Upload" tile).
                if job["id"] not in self._reauth_notified:
                    self._reauth_notified.add(job["id"])
                    models.add_event(
                        job["id"], "upload",
                        "Upload blockiert: Kanal „%s“ braucht eine neue Anmeldung "
                        "(unter /channels neu verbinden). Der Job wartet, bis das "
                        "erledigt ist." % channel["name"],
                    )
                    log.warning(
                        "job %s cannot upload: channel %s needs re-auth",
                        job["id"], channel["name"],
                    )
                continue
            if self._already_uploaded(video["id"]):
                # belt and braces: never send the same file to YouTube twice
                models.set_job_status(job["id"], "uploaded")
                log.warning("job %s already has an upload record, not re-uploading", job["id"])
                continue

            if not self._claim_for_upload(job["id"]):
                continue
            try:
                thumb = video.get("thumb_path")
                if channel:
                    secret, token = channel_upload_paths(channel, self.cfg)
                    up = upload(
                        video["path"], video["title"], video["description"],
                        video["tags"], self.cfg,
                        token_path=token, secret_path=secret,
                        privacy=channel.get("privacy") or None,
                        allow_interactive=False,
                        made_for_kids=_made_for_kids(channel, job),
                        thumbnail_path=thumb,
                        video_type=job.get("video_type"),
                    )
                else:
                    up = upload(video["path"], video["title"], video["description"],
                                video["tags"], self.cfg, allow_interactive=False,
                                thumbnail_path=thumb,
                                video_type=job.get("video_type"))
                models.record_upload(video["id"], job["channel_id"],
                                     up["video_id"], up["url"], up["privacy"])
                models.set_job_status(job["id"], "uploaded")
                models.add_event(job["id"], "upload", f"Hochgeladen: {up['url']}")
                log.info("job %s uploaded: %s", job["id"], up["url"])
            except Exception as e:
                err = f"{type(e).__name__}: {e}"
                models.add_event(job["id"], "error", err)
                auth_issue = "invalid_grant" in err or "re-auth" in err.lower() or "Reconnect" in err
                if channel and auth_issue:
                    models.update_channel(channel["id"], status="needs_reauth")
                    models.update_job(job["id"], status="approved", error=err)
                else:
                    attempts = job["attempts"] + 1
                    if attempts >= self.max_attempts:
                        models.update_job(job["id"], status="failed", attempts=attempts,
                                          error=err, finished_at=models.now())
                    else:
                        models.update_job(job["id"], status="approved",
                                          attempts=attempts, error=err)
                log.error("upload for job %s failed: %s", job["id"], err)


# ------------------------------------------------------------------ singleton --
def start_scheduler(cfg):
    global _instance
    if _instance is None:
        _instance = Scheduler(cfg)
        _instance.start()
    return _instance


def get_scheduler():
    return _instance


def wake():
    if _instance:
        _instance.wake()
