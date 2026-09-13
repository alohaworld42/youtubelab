"""kidsong.seedance — Offline-safe client for ByteDance's Seedance 2.5 cloud
video model: the paid-API alternative to the local ComfyUI/LTX render path
for the kidsong director loop (pipeline/kidsong/generate.py).

Structure mirrors studio/higgsfield.py: ``_conf``/``_keys``/``is_enabled``/
``is_available`` decide availability from config alone (no network probe —
this is a paid queue, so real errors surface on the first render call), and
the flow is submit -> poll -> download, published atomically via
``pipeline.atomicio``.

UNLIKE studio/higgsfield.py, failures do NOT collapse to ``return None``.
higgsfield.py's contract lets a bad model id, an expired key, or a budget
event look identical to "the network hiccuped" — all three retry the same
way. That is fine for a few free-tier credits; it is not fine for a metered
API where the caller's own infra-retry loop
(``pipeline/kidsong/generate.py`` line ~2170) catches exactly
``(TimeoutError, RuntimeError, requests.RequestException)`` and retries with
backoff. See ``SeedanceError`` below for the exception taxonomy this module
uses instead, and why it is deliberately NOT a ``RuntimeError``.

**Nothing in this module ever reaches the network unless
``seedance.budget.dry_run`` is explicitly set to ``false``.** ``dry_run``
defaults to ``true`` even when ``seedance.enabled`` is ``true`` — flipping
"enabled" alone still logs the estimated spend and raises before any HTTP
call. See ``render()`` and ``stage_input_image()``.

Auth: ``SEEDANCE_API_KEY`` from the environment (.env or
``keys/seedance_api_key``); ``SEEDANCE_BASE_URL`` optionally overrides
``seedance.base_url``.

Duck-typed interface the kidsong render loop uses (matches
``kidsong.comfy.ComfyClient``'s shape so the two backends are
interchangeable): ``render(workflow_name, patches, out_path) -> out_path``,
``stage_input_image(src) -> handle``, ``free()``, ``restart_if_hung()``, and
a ``url`` attribute. Also ``accepts_reference_images = True``.
"""
import json
import logging
import math
import os
import time
from urllib.parse import urlparse

import requests

from pipeline.atomicio import atomic_write_bytes, atomic_write_json
from pipeline.kidsong.render_style import resolve_style

log = logging.getLogger("kidsong.seedance")

# process-wide "log this once, not once per shot" keys — see _warn_once.
_WARNED_ONCE = set()


def _warn_once(key, msg, *args):
    if key in _WARNED_ONCE:
        return
    _WARNED_ONCE.add(key)
    log.warning(msg, *args)


# --------------------------------------------------------------- exceptions --
class SeedanceError(Exception):
    """Base for TERMINAL Seedance failures: never retry these.

    Deliberately a direct subclass of ``Exception`` — NOT of ``RuntimeError``,
    ``TimeoutError``, or ``requests.RequestException``. The kidsong render
    loop's infra-retry wrapper catches exactly
    ``(TimeoutError, RuntimeError, requests.RequestException)``
    (``pipeline/kidsong/generate.py`` around line 2170) and retries with its
    own backoff, up to ``comfy_infra_retries`` times. That is the right
    behavior for a transient 5xx/timeout/queue-full — but a moderation
    reject, an expired key, an unknown model id, or a budget cap that has
    already been hit will fail EXACTLY the same way on retry #2, #3 and #4 —
    except each retry against a metered API also re-attempts the paid submit,
    so a bug that let this inherit from RuntimeError would silently turn one
    rejected/over-budget clip into four billed (or four times as
    over-budget) ones. Subclassing ``Exception`` directly means
    ``except (TimeoutError, RuntimeError, requests.RequestException)`` does
    NOT catch a ``SeedanceError``, so it propagates straight out of the
    render loop and aborts the episode loudly — which is exactly what should
    happen to the single most expensive class of bug available in this
    module. See ``tests/test_seedance.py::test_budget_exceeded_is_not_a_runtime_error``.
    """


class SeedanceBudgetExceeded(SeedanceError):
    """Submitting this clip would exceed ``seedance.budget.max_usd_per_episode``
    or ``max_clips_per_episode`` for the running episode ledger, OR the
    Seedance API itself reported an account-balance/credit failure
    (HTTP 402) once submitted."""


class SeedanceRejected(SeedanceError):
    """A terminal, non-retryable rejection: moderation, bad/expired auth, an
    unknown model id, a job that ended in a rejected/failed/cancelled
    status, or (raised at CONSTRUCTION time, before any of that) a
    ``base_url`` whose host is not in ``seedance.allowed_hosts``."""


class SeedanceDryRun(SeedanceError):
    """Raised by ``render()``/``stage_input_image()`` when
    ``seedance.budget.dry_run`` is true, immediately after logging the
    estimated spend and BEFORE any HTTP call is made. Not a failure in the
    usual sense — but it still must not be retried (retrying makes the exact
    same no-op four times), and the caller has no ``out_path`` to return, so
    raising is the only honest option for a duck-typed ``render()`` that
    otherwise always returns a path or raises."""


# --------------------------------------------------------------- config ------
def _conf(cfg):
    return (cfg or {}).get("seedance") or {}


def _keys():
    return os.environ.get("SEEDANCE_API_KEY"), os.environ.get("SEEDANCE_BASE_URL")


def is_enabled(cfg):
    return bool(_conf(cfg).get("enabled"))


def is_available(cfg):
    """True if enabled AND an API key is set. No network probe — see the
    module docstring; real errors surface (and abort loudly, per
    SeedanceError) on the first render call."""
    if not is_enabled(cfg):
        return False
    key, _base = _keys()
    return bool(key)


def pick_duration(seconds, choices):
    """Smallest allowed Seedance duration bucket that covers ``seconds``,
    else the largest available.

    Deliberately mirrors ``studio/higgsfield.py``'s ``pick_duration`` (same
    "assemble trims a too-long bucket, loops/holds a too-short one" billing
    pattern) rather than importing it: dependencies run studio/ -> pipeline/
    in this repo, never the other way, so this ~6-line pure function is
    duplicated here instead of reaching backward across that boundary.
    """
    opts = sorted(int(c) for c in (choices or [4]))
    for c in opts:
        if c >= seconds:
            return c
    return opts[-1]


def _check_host_allowed(base_url, allowed_hosts):
    """Host allowlist for ``base_url``, enforced at ``SeedanceBackend``
    construction.

    This is a ``made_for_kids`` channel. Third-party reseller/aggregator
    endpoints for a paid generation API are out of scope BY POLICY, not by
    accident — an aggregator sits between us and ByteDance's own moderation
    and ToS, and this channel does not get to find out the hard way what
    that means for kids' content. A ``base_url`` whose host is not explicitly
    listed in ``seedance.allowed_hosts`` is therefore a hard, terminal,
    construction-time failure, never a silent fallback to "try it anyway".
    """
    host = (urlparse(base_url).hostname or "").lower()
    allowed = {str(h).lower() for h in (allowed_hosts or [])}
    if not host or host not in allowed:
        raise SeedanceRejected(
            f"Seedance base_url {base_url!r} (host {host!r}) is not in "
            f"seedance.allowed_hosts ({sorted(allowed) or 'none configured'}); "
            "refusing to construct a client that could submit to an "
            "unapproved endpoint."
        )


# ------------------------------------------------------------- translate -----
def _scalar(value, key):
    """A patch value may be a plain scalar or a ``{input_name: value}`` dict
    — the same duck-typed shape ``kidsong.comfy._apply_patches`` accepts
    (see pipeline/kidsong/comfy.py's module docstring). Extract the
    meaningful value either way; ``generate.py`` always sends the dict form
    (e.g. ``{"PROMPT": {"text": ...}}``) but the scalar form is accepted too
    so a hand-built patches dict (as in this module's own tests) doesn't
    need to wrap every value."""
    if isinstance(value, dict):
        return value.get(key)
    return value


_KNOWN_ASPECTS = {
    (16, 9): "16:9", (9, 16): "9:16", (1, 1): "1:1",
    (4, 3): "4:3", (3, 4): "3:4", (21, 9): "21:9",
}


def _aspect_from_dims(width, height):
    """Map a WIDTH/HEIGHT patch pair to a known aspect-ratio string, or
    ``None`` if absent/unrecognized.

    kidsong shots render at 896x512, which reduces to 7:4 — not a standard
    bucket — so that case (the common one, today) falls through to ``None``
    and the caller uses the configured ``aspect_ratio`` instead (1.75 vs
    16:9's 1.778 is close enough that nothing is cropped meaningfully
    differently). This still does real work for any style/graph that
    renders a standard ratio.
    """
    try:
        w, h = int(width), int(height)
    except (TypeError, ValueError):
        return None
    if w <= 0 or h <= 0:
        return None
    g = math.gcd(w, h)
    return _KNOWN_ASPECTS.get((w // g, h // g))


_RESOLUTION_HEIGHT_PX = {
    "480p": 480, "540p": 540, "576p": 576, "720p": 720,
    "1080p": 1080, "1440p": 1440, "2k": 1440, "4k": 2160,
}
_MAX_RESOLUTION = "720p"


# ------------------------------------------------------------------ backend --
class SeedanceBackend:
    """Client for ByteDance Seedance 2.5, duck-typed to swap in for
    ``kidsong.comfy.ComfyClient`` in the kidsong render loop.

    Construction raises ``SeedanceRejected`` immediately if ``base_url``'s
    host is not in ``seedance.allowed_hosts`` — see ``_check_host_allowed``.
    """

    accepts_reference_images = True

    def __init__(self, cfg):
        self.cfg = cfg or {}
        self.conf = _conf(self.cfg)

        api_key, base_override = _keys()
        self.api_key = api_key
        self.url = str(base_override or self.conf.get("base_url") or "").rstrip("/")
        if self.url:
            _check_host_allowed(self.url, self.conf.get("allowed_hosts"))

        self.model = self.conf.get("model", "")
        self.resolution = self.conf.get("resolution", "720p")
        self.aspect_ratio = self.conf.get("aspect_ratio", "16:9")
        self.durations = sorted(int(d) for d in (self.conf.get("durations") or [4]))
        self.max_reference_images = int(self.conf.get("max_reference_images", 1))
        self.poll_seconds = float(self.conf.get("poll_seconds", 5))
        self.timeout_seconds = float(self.conf.get("timeout_seconds", 600))
        self.params = dict(self.conf.get("params") or {})

        budget = self.conf.get("budget") or {}
        # Defaults TRUE even if the caller forgot to set it explicitly — the
        # unsafe state (actually spending money) must always be opt-in.
        self.dry_run = bool(budget.get("dry_run", True))
        self._price_per_second = float(budget.get("price_per_second_usd") or 0.0)
        self.max_usd_per_episode = budget.get("max_usd_per_episode")
        self.max_clips_per_episode = budget.get("max_clips_per_episode")

        # (realpath, size, mtime) -> uploaded handle. Survives free() by
        # design — see free()'s docstring.
        self._image_cache = {}

    # ------------------------------------------------------------ headers ---
    def _headers(self, json_content=True):
        h = {"Authorization": f"Bearer {self.api_key}", "Accept": "application/json"}
        if json_content:
            h["Content-Type"] = "application/json"
        return h

    def _capped_resolution(self):
        req = str(self.resolution or _MAX_RESOLUTION).strip().lower()
        req_px = _RESOLUTION_HEIGHT_PX.get(req)
        cap_px = _RESOLUTION_HEIGHT_PX[_MAX_RESOLUTION]
        if req_px is None or req_px > cap_px:
            _warn_once(
                "seedance_resolution_capped",
                "seedance: configured resolution %r exceeds the 720p cap "
                "(output is 1080p and shots render at 896x512 -- anything "
                "above 720p is pure waste); capping to 720p.",
                self.resolution,
            )
            return _MAX_RESOLUTION
        return req

    # ---------------------------------------------------------- translate ---
    def _translate(self, patches):
        """Translate ComfyUI-graph-vocabulary ``patches`` (see
        ``kidsong.comfy``'s module docstring and ``_PRIMARY_INPUT``) into a
        Seedance API payload. Returns ``(payload, duration_seconds)`` — the
        caller needs the snapped duration separately to price the clip
        before submitting.
        """
        patches = patches or {}

        prompt = str(_scalar(patches.get("PROMPT"), "text") or "")
        style = resolve_style(self.cfg)
        trigger = style.get("style_trigger") or ""
        # generate.py composes f"{trigger}, {skeleton}" whenever trigger is
        # set (pipeline/kidsong/refs.py's _reference_prompt documents the
        # same reasoning for its own Z-Image path): the trigger activates an
        # LTX-specific LoRA and is meaningless -- worse, pure noise -- to a
        # different hosted model's text encoder.
        prefix = f"{trigger}, "
        if trigger and prompt.startswith(prefix):
            prompt = prompt[len(prefix):]
        payload = {"prompt": prompt}

        negative = _scalar(patches.get("NEGATIVE"), "text")
        if negative:
            neg_field = self.conf.get("negative_prompt_field", "negative_prompt")
            if neg_field:
                payload[neg_field] = str(negative)
            else:
                # Operator has confirmed (by blanking the field name) that
                # this deployment has no dedicated negative-prompt field.
                # NEVER silently drop the constraint -- fold it into the
                # prompt instead.
                payload["prompt"] = f"{payload['prompt']} avoid: {negative}".strip()
                _warn_once(
                    "seedance_negative_prompt_fallback",
                    "seedance: negative_prompt_field is empty -- folding "
                    "NEGATIVE into the prompt as an 'avoid: ...' clause "
                    "instead of a dedicated field.",
                )

        seed = _scalar(patches.get("SEED"), "noise_seed")
        if seed is not None:
            payload["seed"] = seed

        if "LORA_STYLE" in patches:
            _warn_once(
                "seedance_lora_style_ignored",
                "seedance: LORA_STYLE patch ignored -- Seedance is a hosted "
                "model with no LoRA loading; the look is carried entirely "
                "by the prompt text.",
            )
        # SIGMAS (the i2v refine sampling schedule) and FILENAME_PREFIX are
        # pure LTX/ComfyUI plumbing with no Seedance equivalent. Dropped
        # without a log, unlike LORA_STYLE: losing them changes nothing about
        # the requested OUTPUT, only how a local graph would have sampled it.

        width = _scalar(patches.get("WIDTH"), "value")
        height = _scalar(patches.get("HEIGHT"), "value")
        payload["aspect_ratio"] = _aspect_from_dims(width, height) or self.aspect_ratio
        # Hard-capped at 720p regardless of config -- see _capped_resolution.
        payload["resolution"] = self._capped_resolution()

        frames = _scalar(patches.get("FRAMES"), "value")
        fps = float(((self.cfg.get("kidsong") or {}).get("shot") or {}).get("fps", 24))
        seconds = float(frames) / fps if frames else 0.0
        duration = pick_duration(seconds, self.durations)
        payload["duration"] = duration

        img = _scalar(patches.get("INPUT_IMAGE"), "image")
        if img:
            payload["image"] = img

        refs = _scalar(patches.get("REFERENCE_IMAGES"), "images")
        if refs:
            cap = max(0, int(self.max_reference_images))
            payload["reference_images"] = list(refs)[:cap]

        # ACE-Step sings the lyrics; any co-generated Seedance audio track is
        # discarded downstream. Hardcoded, not config-driven, on purpose --
        # there is no legitimate reason to ever pay for it.
        payload["generate_audio"] = False

        if self.model:
            payload["model"] = self.model
        payload.update(self.params)

        return payload, duration

    # -------------------------------------------------------------- budget ---
    def _ledger_path(self, out_path):
        """``spend.json`` lives beside the takes it pays for.

        ``SeedanceBackend`` only receives ``cfg`` at construction, not an
        episode/base id -- by the time ``render()`` is called it already has
        ``out_path`` in hand, and every kidsong ``out_path`` lives under
        ``output/<base>-shots/<shot_id>_a<attempt>.mp4``
        (``pipeline/kidsong/generate.py``), so the ledger sits in that same
        directory: one ``spend.json`` per episode's shots dir.
        """
        directory = os.path.dirname(os.path.abspath(out_path)) or "."
        return os.path.join(directory, "spend.json")

    @staticmethod
    def _read_ledger(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except (OSError, ValueError):
            return {"total_usd": 0.0, "clips": 0}
        if not isinstance(data, dict):
            return {"total_usd": 0.0, "clips": 0}
        return data

    def _check_budget(self, out_path, duration):
        """Check the running spend BEFORE every submit -- in both dry_run
        and live mode, so a config that would blow the budget is caught even
        while the operator is still previewing with dry_run on."""
        ledger = self._read_ledger(self._ledger_path(out_path))
        est_cost = self._price_per_second * duration
        total_after = float(ledger.get("total_usd", 0.0)) + est_cost
        clips_after = int(ledger.get("clips", 0)) + 1

        if self.max_clips_per_episode is not None and clips_after > self.max_clips_per_episode:
            raise SeedanceBudgetExceeded(
                f"Seedance: this clip would be #{clips_after} against "
                f"seedance.budget.max_clips_per_episode="
                f"{self.max_clips_per_episode}."
            )
        if self.max_usd_per_episode is not None and total_after > self.max_usd_per_episode:
            raise SeedanceBudgetExceeded(
                f"Seedance: this clip's ~${est_cost:.4f} would bring the "
                f"episode total to ~${total_after:.4f}, over "
                f"seedance.budget.max_usd_per_episode="
                f"${self.max_usd_per_episode}."
            )

    def _record_spend(self, out_path, duration):
        path = self._ledger_path(out_path)
        ledger = self._read_ledger(path)
        est_cost = self._price_per_second * duration
        payload = {
            "total_usd": round(float(ledger.get("total_usd", 0.0)) + est_cost, 6),
            "clips": int(ledger.get("clips", 0)) + 1,
            "updated_at": time.time(),
        }
        atomic_write_json(path, payload, indent=2)

    # --------------------------------------------------------------- http ---
    def _submit(self, payload):
        url = f"{self.url}/v1/video/generations"
        r = requests.post(
            url, json=payload, headers=self._headers(), timeout=self.timeout_seconds,
        )
        if r.status_code == 402:
            raise SeedanceBudgetExceeded(
                f"Seedance reported insufficient account balance/credits "
                f"(HTTP 402): {r.text[:300]}"
            )
        if r.status_code in (401, 403):
            raise SeedanceRejected(f"Seedance auth failed (HTTP {r.status_code}).")
        if r.status_code in (400, 404, 422):
            raise SeedanceRejected(
                f"Seedance rejected the request (HTTP {r.status_code}): "
                f"{r.text[:300]}"
            )
        # Everything else (429 queue full, 5xx) becomes requests.HTTPError --
        # a requests.RequestException subclass -- so the caller's infra-retry
        # loop retries it with backoff, exactly the three exceptions it
        # catches.
        r.raise_for_status()
        return r.json()

    _SUCCESS_STATUSES = {"completed", "succeeded", "success", "done"}
    _BUDGET_STATUSES = {"budget_exceeded", "insufficient_credits", "quota_exceeded"}
    _TERMINAL_STATUSES = {
        "failed", "rejected", "moderation_rejected", "nsfw", "cancelled",
        "content_flagged",
    }

    def _poll(self, job):
        request_id = job.get("id") or job.get("request_id")
        if not request_id:
            raise SeedanceRejected(f"Seedance submit returned no request id: {job}")
        url = f"{self.url}/v1/video/generations/{request_id}"
        deadline = time.time() + self.timeout_seconds
        while True:
            if time.time() > deadline:
                raise TimeoutError(
                    f"Seedance render exceeded {self.timeout_seconds}s for "
                    f"request {request_id}."
                )
            r = requests.get(url, headers=self._headers(), timeout=30)
            if r.status_code in (401, 403):
                raise SeedanceRejected(
                    f"Seedance auth failed while polling (HTTP {r.status_code})."
                )
            r.raise_for_status()
            data = r.json()
            status = str(data.get("status") or "").lower()
            if status in self._SUCCESS_STATUSES:
                return data
            if status in self._BUDGET_STATUSES:
                raise SeedanceBudgetExceeded(
                    f"Seedance reported {status!r} for request {request_id}: {data}"
                )
            if status in self._TERMINAL_STATUSES:
                raise SeedanceRejected(
                    f"Seedance job {request_id} ended as {status!r}: {data}"
                )
            time.sleep(self.poll_seconds)

    def _download(self, video_url, out_path):
        # No try/except here on purpose (unlike higgsfield.py's old
        # `return None` contract): a failed download must propagate so the
        # caller's retry loop sees it, and atomic_write_bytes already
        # guarantees a truncated response never lands at out_path.
        r = requests.get(
            video_url, headers=self._headers(json_content=False),
            stream=True, timeout=self.timeout_seconds,
        )
        r.raise_for_status()
        atomic_write_bytes(out_path, r.iter_content(chunk_size=1 << 16))
        return out_path

    # ------------------------------------------------------------- render ---
    def render(self, workflow_name, patches, out_path):
        """Render one Seedance clip and write the MP4 to ``out_path``.

        Signature/return mirror ``kidsong.comfy.ComfyClient.render`` so the
        kidsong render loop can swap backends. See the module docstring for
        why failures raise instead of returning ``None``.
        """
        payload, duration = self._translate(patches)
        self._check_budget(out_path, duration)

        if self.dry_run:
            est = self._price_per_second * duration
            log.warning(
                "seedance dry_run: would submit %r (duration=%ss, ~$%.4f) "
                "-- no HTTP call made. Set seedance.budget.dry_run=false to "
                "actually submit.", workflow_name, duration, est,
            )
            raise SeedanceDryRun(
                f"seedance.budget.dry_run is true; refusing to submit "
                f"{workflow_name!r} (est. ${est:.4f})."
            )

        job = self._submit(payload)
        result = self._poll(job)
        video = result.get("video")
        video_url = (video or {}).get("url") if isinstance(video, dict) \
            else result.get("video_url")
        if not video_url:
            raise SeedanceRejected(
                f"Seedance job completed but returned no video url: {result}"
            )
        self._download(video_url, out_path)
        self._record_spend(out_path, duration)
        return out_path

    # -------------------------------------------------------------- input ---
    def stage_input_image(self, src_path):
        """Return an opaque handle Seedance can use as an INPUT_IMAGE,
        uploading only on a cache miss.

        Cached by ``(realpath, size, mtime)`` so a render RETRY on the same
        source file (identical bytes, same disk location -- the common case
        when the caller's infra-retry loop re-renders a shot) is a cache
        hit, not a second upload. See ``free()`` for why the cache survives
        between attempts.
        """
        real = os.path.realpath(src_path)
        st = os.stat(real)  # missing file -> OSError propagates; a missing
        # reference image is a bug at the call site, not something to hide.
        key = (real, st.st_size, st.st_mtime)
        handle = self._image_cache.get(key)
        if handle is not None:
            return handle
        handle = self._upload_image(real)
        self._image_cache[key] = handle
        return handle

    def _upload_image(self, path):
        if self.dry_run:
            raise SeedanceDryRun(
                f"seedance.budget.dry_run is true; refusing to upload "
                f"{path!r}."
            )
        url = f"{self.url}/v1/images"
        with open(path, "rb") as f:
            r = requests.post(
                url, headers=self._headers(json_content=False),
                files={"file": (os.path.basename(path), f)},
                timeout=self.timeout_seconds,
            )
        if r.status_code in (401, 403):
            raise SeedanceRejected(
                f"Seedance auth failed uploading {path!r} (HTTP {r.status_code})."
            )
        if r.status_code in (400, 404, 422):
            raise SeedanceRejected(
                f"Seedance rejected image upload {path!r} "
                f"(HTTP {r.status_code}): {r.text[:300]}"
            )
        r.raise_for_status()
        data = r.json()
        handle = data.get("id") or data.get("handle") or data.get("url")
        if not handle:
            raise SeedanceRejected(f"Seedance image upload returned no handle: {data}")
        return handle

    # ------------------------------------------------------------- cleanup ---
    def free(self):
        """No-op: the image-upload cache (and the spend ledger, which is
        read/written straight from disk) are episode-scoped and must survive
        a ``free()`` call -- the render loop calls this between retry
        attempts on the SAME shot, and a cache that didn't survive would
        turn every retry back into a fresh upload."""
        pass

    def restart_if_hung(self):
        """No local server process to restart -- Seedance is a stateless
        hosted API client, unlike ``ComfyClient`` (which owns a ComfyUI
        subprocess it can kill and relaunch). Always returns ``False``
        (never claims to have healed anything); the caller's infra-retry
        loop simply retries the render itself after its own backoff."""
        return False
