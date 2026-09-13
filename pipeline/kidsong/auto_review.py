"""
kidsong.auto_review — headless answerer for the vision QC gate (QM-010,
docs/quality/DEFECT_BACKLOG.md).

`review.ExternalReviewer` (and `cut_qc`/`script_qc`'s external gates) write a
`<name>.request.json` + contact-sheet PNG and poll for a matching
`<name>.response.json`, so a human or a Claude session can look at the take
before it ships. In unattended production nobody is watching the browser
queue, so those requests just pile up unanswered and every gate falls back to
its weak heuristic verdict — recent episodes shipped with 0-6% real vision
coverage even though the requests were faithfully written the whole time.

This module is the missing answerer: it scans for pending requests (reusing
`review_queue.list_pending` for discovery), re-reads each request's own raw
JSON for the full contract (`list_pending` intentionally returns a stripped,
display-only projection for the web UI — see its module docstring), builds a
review prompt (the `scene-plausibility` rubric for shot takes, a matching
cut/script/shotlist prompt for the other three gate kinds), shells out to the
locally installed `claude` CLI in headless print mode, and writes the parsed
verdict back to the exact response path the gate is polling — atomically, so
a reader never observes a half-written file.

Run it once (drains the current backlog) or as a standing watcher alongside
the scheduler:

    python -m pipeline.kidsong.auto_review --once
    python -m pipeline.kidsong.auto_review --watch

Nothing here changes any QC *criteria* — it only makes sure the vision gate
that already exists actually gets exercised in unattended runs.
"""
import argparse
import json
import logging
import os
import shutil
import subprocess
import sys
import tempfile
import time
from logging.handlers import RotatingFileHandler

from pipeline.config import abspath, load_config
from pipeline.kidsong.review_queue import list_pending

log = logging.getLogger("kidsong.auto_review")

_RUBRIC_RELPATH = os.path.join(".claude", "agents", "scene-plausibility.md")

_DEFAULTS = {
    "enabled": True,
    "poll_seconds": 10,
    "model": "claude-sonnet-5",
    "claude_cmd": "claude",
    "max_age_hours": 48,
}

_CALL_TIMEOUT_SECONDS = 180
_MAX_ATTEMPTS = 2

_LOG_HANDLER_TAG = "_kidsong_auto_review_handler"


# ------------------------------------------------------------------ config ---
def _auto_cfg(cfg):
    review_cfg = (cfg.get("kidsong", {}) or {}).get("review", {}) or {}
    merged = dict(_DEFAULTS)
    merged.update(review_cfg.get("auto", {}) or {})
    return merged


def _setup_logging(root):
    """One file (+ console) handler on this module's own logger, mirroring
    app.py's `_setup_logging` — but scoped to `log` rather than the root
    logger so running this alongside app.py in the same process (tests,
    an all-in-one launcher) can never duplicate app.py's own handlers.
    Idempotent: safe to call more than once (e.g. every `--watch` tick if a
    caller re-invokes `main`).
    """
    if any(getattr(h, _LOG_HANDLER_TAG, None) for h in log.handlers):
        return
    logs_dir = os.path.join(root, "logs")
    os.makedirs(logs_dir, exist_ok=True)
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s")

    log.setLevel(logging.INFO)
    log.propagate = False

    file_handler = RotatingFileHandler(
        os.path.join(logs_dir, "auto_review.log"),
        maxBytes=2 * 1024 * 1024, backupCount=3, encoding="utf-8",
    )
    file_handler.setFormatter(fmt)
    setattr(file_handler, _LOG_HANDLER_TAG, True)
    log.addHandler(file_handler)

    console = logging.StreamHandler(stream=sys.stdout)
    console.setFormatter(logging.Formatter("%(message)s"))
    setattr(console, _LOG_HANDLER_TAG, True)
    log.addHandler(console)


# -------------------------------------------------------------------- json ---
def extract_json(text):
    """Return the LAST well-formed top-level JSON object embedded in `text`,
    or `None`. Tolerant of prose before/after and markdown code fences —
    braces inside a JSON string value are correctly ignored so a reasons
    string like "the '{' looked odd" cannot desync the scan.
    """
    if not text:
        return None

    candidates = []
    depth = 0
    start = None
    in_string = False
    escape = False
    for i, ch in enumerate(text):
        if in_string:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
        elif ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}":
            if depth > 0:
                depth -= 1
                if depth == 0 and start is not None:
                    candidates.append(text[start : i + 1])

    for candidate in reversed(candidates):
        try:
            obj = json.loads(candidate)
        except (json.JSONDecodeError, ValueError):
            continue
        if isinstance(obj, dict):
            return obj
    return None


def validate_verdict(obj):
    """The verdict contract every gate's response file must satisfy:
    `accept` bool, `score` a number in [0, 1], `reasons` a list, `retry_hints`
    a dict. Key content of `retry_hints` is deliberately NOT checked — each
    gate (shot/cut/script/shotlist) has its own hint vocabulary and every
    downstream consumer (`review.py`/`cut_qc.py`/`script_qc.py`) already
    `setdefault`s sane fallbacks for anything it doesn't recognize.
    """
    if not isinstance(obj, dict):
        return False
    if not isinstance(obj.get("accept"), bool):
        return False
    score = obj.get("score")
    if not isinstance(score, (int, float)) or isinstance(score, bool):
        return False
    if not (0.0 <= float(score) <= 1.0):
        return False
    if not isinstance(obj.get("reasons"), list):
        return False
    if not isinstance(obj.get("retry_hints"), dict):
        return False
    return True


def _write_response_atomic(response_path, payload):
    """tmp-file + os.replace so a reader (poll_response) never observes a
    partially written response — the same discipline every gate's own
    request-writer already follows."""
    out_dir = os.path.dirname(response_path) or "."
    os.makedirs(out_dir, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(prefix=".auto_review_", suffix=".tmp", dir=out_dir)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)
        os.replace(tmp_path, response_path)
    except BaseException:
        try:
            os.remove(tmp_path)
        except OSError:
            pass
        raise


# --------------------------------------------------------------- claude CLI ---
def _resolve_claude_cmd(claude_cmd):
    """Resolve the claude CLI to a launchable path. A bare name like "claude"
    is not enough on Windows: the CLI installs to per-user locations that are
    on the interactive shell's PATH but not on a service/pipeline process's
    PATH, and subprocess does not consult PATHEXT for bare names. Returns the
    resolved path (cached), or the input unchanged if nothing better is found
    (the launch error then names what was tried)."""
    if os.path.sep in claude_cmd or (os.path.altsep and os.path.altsep in claude_cmd):
        return claude_cmd  # explicit path configured — use as-is
    cached = _resolve_claude_cmd._cache.get(claude_cmd)
    if cached:
        return cached
    found = shutil.which(claude_cmd)
    if not found:
        home = os.path.expanduser("~")
        for cand in (
            os.path.join(home, ".local", "bin", claude_cmd + ".exe"),
            os.path.join(home, ".local", "bin", claude_cmd),
            os.path.join(os.environ.get("APPDATA", ""), "npm", claude_cmd + ".cmd"),
        ):
            if cand and os.path.isfile(cand):
                found = cand
                break
    if found:
        log.info("claude CLI resolved: %s", found)
        _resolve_claude_cmd._cache[claude_cmd] = found
        return found
    return claude_cmd


_resolve_claude_cmd._cache = {}


def _call_claude(claude_cmd, prompt, model, timeout=_CALL_TIMEOUT_SECONDS):
    """One headless invocation of the claude CLI. Returns the model's final
    message text (the envelope's `result` field — verified against a live
    `claude -p ... --output-format json` call; NOT `text`), or `None` on any
    failure (never raises — the caller treats `None` exactly like a bad
    parse and retries/gives up the same way)."""
    cmd = [
        _resolve_claude_cmd(claude_cmd), "-p", prompt,
        "--model", model,
        "--allowedTools", "Read",
        "--output-format", "json",
    ]
    try:
        proc = subprocess.run(
            cmd, capture_output=True, text=True, timeout=timeout,
            encoding="utf-8", errors="replace",
        )
    except subprocess.TimeoutExpired:
        log.warning("claude CLI timed out after %ss", timeout)
        return None
    except OSError as e:
        log.warning("claude CLI failed to launch (%r): %s", claude_cmd, e)
        return None

    stdout = proc.stdout or ""
    try:
        envelope = json.loads(stdout)
    except (json.JSONDecodeError, ValueError):
        log.warning(
            "claude CLI produced non-JSON stdout (exit %s): %.500s",
            proc.returncode, stdout or (proc.stderr or ""),
        )
        return None

    if envelope.get("is_error"):
        log.warning("claude CLI reported an error: %.500s", str(envelope.get("result")))

    text = envelope.get("result")
    return text if isinstance(text, str) else None


def _get_verdict(claude_cmd, prompt, model, label):
    """Up to `_MAX_ATTEMPTS` tries to get a schema-valid verdict out of the
    CLI. Returns the parsed dict, or `None` if every attempt failed — the
    caller must leave the request pending (never write a malformed
    response) in that case."""
    for attempt in range(1, _MAX_ATTEMPTS + 1):
        text = _call_claude(claude_cmd, prompt, model)
        if text is None:
            log.warning("%s: claude call failed (attempt %d/%d)", label, attempt, _MAX_ATTEMPTS)
            continue
        obj = extract_json(text)
        if obj is not None and validate_verdict(obj):
            return obj
        log.warning(
            "%s: no valid verdict JSON in reply (attempt %d/%d): %.300s",
            label, attempt, _MAX_ATTEMPTS, text,
        )
    return None


# ------------------------------------------------------------------ rubric ---
def _rubric_text(root):
    path = os.path.join(root, _RUBRIC_RELPATH)
    with open(path, "r", encoding="utf-8") as f:
        text = f.read()
    # Strip the agent's YAML frontmatter (---\nname: ...\n---) -- it's Claude
    # Code agent metadata, not part of the rubric prose meant for the prompt.
    if text.startswith("---"):
        end = text.find("\n---", 3)
        if end != -1:
            text = text[end + 4 :].lstrip("\n")
    return text


_CUT_RUBRIC = """You are the cut-quality reviewer for an AI-generated preschool kids' song \
video. You are shown a contact sheet of the assembled, beat-cut edit -- one row per cut, \
three frames each (start, middle, end of that cut) -- plus a programmatic verdict already \
computed by automated checks (container integrity, cut-length/beat alignment, audio \
musicality, black-seam detection at cut boundaries).

Judge whether the EDIT AS A WHOLE is fit to ship: cuts land cleanly with no black or broken \
frames at the seams, the footage order makes visual sense, nothing is jarring or glitched. \
You are judging the ASSEMBLY, not re-litigating any one shot's content -- the per-shot vision \
gate already reviewed each take individually before it reached the edit.

Your final message must be ONLY this JSON (no prose before or after):
{
  "accept": true|false,
  "score": 0.0-1.0,
  "reasons": ["specific, frame-referenced findings"],
  "retry_hints": {"recut": true|false, "reshoot": ["shot_id", ...]}
}
"""

_SCRIPT_RUBRIC = """You are reviewing an original nursery-rhyme song generated for a toddler \
YouTube channel, before any rendering happens. You are given the song (title, verses, the \
characters sentence) and the automated QC gate's own programmatic verdict (script_qc.py, \
which already checks verse/line counts, vocabulary, rhyme/meter, brand-word safety, and \
lyric craft mechanically).

Judge whether the song is genuinely fit to build an episode from: wholesome and age-appropriate \
for toddlers, nothing unsettling or off-brand, and does it actually read as a singable song \
rather than disconnected sentences. Defer to the programmatic verdict on anything it already \
checks mechanically (line length, rhyme, brand words) unless you spot something it clearly \
missed.

Your final message must be ONLY this JSON (no prose before or after):
{
  "accept": true|false,
  "score": 0.0-1.0,
  "reasons": ["specific findings"],
  "retry_hints": {"regenerate": true|false}
}
"""

_SHOTLIST_RUBRIC = """You are reviewing a shot list planned for an AI-rendered toddler kids' \
song video, before any GPU rendering happens. You are given the song, the full shot list \
(each shot's verse, timing, shot_type, camera, characters, action, setting), and the \
automated QC gate's own programmatic verdict (script_qc.py, which already checks tiling, \
durations, shot-type variety, staging/location rules, and head-count consistency \
mechanically).

Judge whether the PLAN reads as a coherent, filmable preschool-TV episode: the shot sequence \
makes visual sense, settings and actions agree with each other, staging is calm and informal \
(never martial or lockstep), and a toddler viewer could follow it. Defer to the programmatic \
verdict on anything it already checks mechanically unless you spot something it clearly \
missed.

Your final message must be ONLY this JSON (no prose before or after):
{
  "accept": true|false,
  "score": 0.0-1.0,
  "reasons": ["specific findings"],
  "retry_hints": {"replan": true|false}
}
"""


# ------------------------------------------------------------------ prompts ---
def _sibling_path(request_path_abs, stored_path):
    """Resolve a path (contact sheet / cut sheet) recorded inside a request
    payload. It always lives alongside its request.json, so the request's
    own directory + the stored path's basename is trusted over a possibly
    stale/foreign absolute path (the request could have been written on a
    different machine, or a test fixture may only supply a relative name).
    Falls back to the stored path verbatim if a sibling file isn't found.
    """
    if not stored_path:
        return None
    sibling = os.path.join(os.path.dirname(request_path_abs), os.path.basename(stored_path))
    if os.path.exists(sibling):
        return sibling
    if os.path.isabs(stored_path) and os.path.exists(stored_path):
        return stored_path
    return sibling


def _shot_prompt(rubric, raw, sheet_path):
    payload = {
        "shot_id": raw.get("shot_id"),
        "take": raw.get("take"),
        "shot": raw.get("shot"),
        "cast_text": raw.get("cast_text"),
        "cast_version": raw.get("cast_version"),
        "expected_children": raw.get("expected_children"),
        "heuristic": raw.get("heuristic"),
    }
    return "\n".join([
        rubric,
        "--- REQUEST ---",
        json.dumps(payload, indent=2, ensure_ascii=False, default=str),
        "",
        f"The contact sheet is at: {sheet_path}",
        "Read it with your vision (the Read tool) before judging.",
        "",
        "Your final output must be ONLY the verdict JSON described above -- no prose before or after.",
    ])


def _cut_prompt(raw, sheet_path):
    payload = {
        "video_path": raw.get("video_path"),
        "cut_list_summary": raw.get("cut_list_summary"),
        "programmatic_verdict": raw.get("programmatic_verdict"),
    }
    return "\n".join([
        _CUT_RUBRIC,
        "--- REQUEST ---",
        json.dumps(payload, indent=2, ensure_ascii=False, default=str),
        "",
        f"The cut contact sheet is at: {sheet_path}",
        "Read it with your vision (the Read tool) before judging.",
        "",
        "Your final output must be ONLY the verdict JSON described above -- no prose before or after.",
    ])


def _text_prompt(kind, raw):
    rubric = _SCRIPT_RUBRIC if kind == "script" else _SHOTLIST_RUBRIC
    return "\n".join([
        rubric,
        "--- REQUEST ---",
        json.dumps(raw, indent=2, ensure_ascii=False, default=str),
        "",
        "Your final output must be ONLY the verdict JSON described above -- no prose before or after.",
    ])


# --------------------------------------------------------------- discovery ---
def _iter_pending(cfg, out_dir, max_age_hours, include_stale):
    max_age_seconds = float(max_age_hours or 0) * 3600.0
    for item in list_pending(cfg, out_dir=out_dir):
        age = float(item.get("age_seconds", 0.0))
        if not include_stale and max_age_seconds > 0 and age > max_age_seconds:
            log.info(
                "skipping stale %s request (base=%s, age=%.1fh > max_age_hours=%.1f)",
                item.get("kind"), item.get("video_base"), age / 3600.0, max_age_hours,
            )
            continue
        yield item


# ----------------------------------------------------------------- worker ---
def _process_item(item, out_dir, auto_cfg, root):
    """Answer one pending request. Returns the written verdict dict, or
    `None` if nothing was written (already answered, unreadable request,
    unknown kind, or the claude CLI never produced a valid verdict)."""
    request_path = os.path.normpath(os.path.join(out_dir, item["request_path"]))
    response_path = os.path.normpath(os.path.join(out_dir, item["response_path"]))

    if os.path.exists(response_path):
        return None  # answered by someone else since the scan

    try:
        with open(request_path, "r", encoding="utf-8") as f:
            raw = json.load(f)
    except (OSError, json.JSONDecodeError) as e:
        log.warning("unreadable request %s: %s", request_path, e)
        return None

    kind = item.get("kind")
    base = item.get("video_base") or "?"
    if kind == "shot":
        label = f"{base}/{raw.get('take') or raw.get('shot_id') or '?'}"
        sheet = _sibling_path(request_path, raw.get("contact_sheet"))
        prompt = _shot_prompt(_rubric_text(root), raw, sheet)
    elif kind == "cut":
        label = f"{base}/cut"
        sheet = _sibling_path(request_path, raw.get("contact_sheet"))
        prompt = _cut_prompt(raw, sheet)
    elif kind in ("script", "shotlist"):
        label = f"{base}/{kind}"
        prompt = _text_prompt(kind, raw)
    else:
        log.warning("unknown review kind %r for %s -- skipping", kind, request_path)
        return None

    verdict = _get_verdict(auto_cfg["claude_cmd"], prompt, auto_cfg["model"], label)
    if verdict is None:
        log.warning("%s: giving up after %d failed attempt(s) -- left pending", label, _MAX_ATTEMPTS)
        return None

    if os.path.exists(response_path):
        # Somebody else answered while we were talking to the model --
        # last-writer-wins is fine per contract, but don't clobber a fresh
        # answer with a call that may have started against stale context.
        log.info("%s: answered elsewhere while we were reviewing -- discarding our verdict", label)
        return None

    _write_response_atomic(response_path, verdict)
    log.info(
        "%s: accept=%s score=%.2f",
        label, verdict.get("accept"), float(verdict.get("score", 0.0)),
    )
    return verdict


# -------------------------------------------------------------------- runs ---
def run_once(cfg=None, include_stale=False):
    """Answer everything currently pending. Returns the count answered."""
    cfg = cfg or load_config()
    auto_cfg = _auto_cfg(cfg)
    out_dir = abspath(cfg, cfg["paths"]["output_dir"])
    root = cfg.get("_root") or os.getcwd()

    answered = 0
    for item in _iter_pending(cfg, out_dir, auto_cfg["max_age_hours"], include_stale):
        if _process_item(item, out_dir, auto_cfg, root) is not None:
            answered += 1
    return answered


def run_watch(cfg=None, include_stale=False, poll_seconds=None, stop_after=None):
    """Poll forever (or `stop_after` iterations, for tests), answering new
    requests as they appear."""
    cfg = cfg or load_config()
    auto_cfg = _auto_cfg(cfg)
    interval = float(poll_seconds if poll_seconds is not None else auto_cfg["poll_seconds"])
    log.info("kidsong auto_review: watching every %ss (Ctrl+C to stop)", interval)

    iterations = 0
    while True:
        answered = run_once(cfg, include_stale=include_stale)
        if answered:
            log.info("kidsong auto_review: answered %d request(s) this pass", answered)
        iterations += 1
        if stop_after is not None and iterations >= stop_after:
            return
        time.sleep(interval)


# --------------------------------------------------------------------- CLI ---
def main(argv=None):
    parser = argparse.ArgumentParser(
        description=(
            "Headlessly answer pending kidsong review requests (shot/cut/script/shotlist) "
            "via the claude CLI, so the vision QC gate does not go dark on unattended runs "
            "(QM-010)."
        )
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--once", action="store_true", help="Answer everything currently pending, then exit (default).")
    mode.add_argument("--watch", action="store_true", help="Poll forever, answering new requests as they appear.")
    parser.add_argument("--poll-seconds", type=float, default=None, help="Override kidsong.review.auto.poll_seconds for --watch.")
    parser.add_argument("--include-stale", action="store_true", help="Also answer requests older than max_age_hours.")
    parser.add_argument("--model", default=None, help="Override kidsong.review.auto.model.")
    parser.add_argument("--claude-cmd", default=None, help="Override kidsong.review.auto.claude_cmd.")
    args = parser.parse_args(argv)

    cfg = load_config()
    _setup_logging(cfg.get("_root") or os.getcwd())

    auto_cfg = _auto_cfg(cfg)
    if args.model:
        auto_cfg["model"] = args.model
    if args.claude_cmd:
        auto_cfg["claude_cmd"] = args.claude_cmd
    cfg = dict(cfg)
    cfg["kidsong"] = dict(cfg.get("kidsong", {}))
    cfg["kidsong"]["review"] = dict(cfg["kidsong"].get("review", {}))
    cfg["kidsong"]["review"]["auto"] = auto_cfg

    if not auto_cfg.get("enabled", True):
        log.info("kidsong auto_review: disabled via kidsong.review.auto.enabled=false -- exiting")
        return

    if args.watch:
        run_watch(cfg, include_stale=args.include_stale, poll_seconds=args.poll_seconds)
    else:
        answered = run_once(cfg, include_stale=args.include_stale)
        log.info("kidsong auto_review: answered %d request(s)", answered)


if __name__ == "__main__":
    main()
