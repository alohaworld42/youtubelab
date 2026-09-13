"""
kidsong.review_queue — Filesystem scanner for the LOCAL human review queue.

Mirrors `pipeline.kidsong.review.poll_response`'s contract from the read
side: every external review gate in this package (per-shot takes in
`review.ExternalReviewer`, script/shotlist in `script_qc`, cut in `cut_qc`)
writes a `<name>.request.json` and then blocks polling for a matching
`<name>.response.json`. This module finds every request that's still
waiting on a response so a human can look at it in the web UI and write
that response file directly — zero LLM tokens per review.

Two on-disk shapes exist (confirmed against live `output/` state, not just
the module docstrings that describe them):

  - Fixed-path gates: `<output_dir>/_kidsong_review/{script,shotlist}.request.json`
    (script_qc.review_script / review_shotlist — no per-video subfolder, so
    there's no video_base to recover; these apply to whatever run is
    currently gating).
  - Per-video gates: `<output_dir>/<base>-shots/review/*.request.json` —
    both the per-take shot requests (`s03_a0.request.json`, written by
    review.ExternalReviewer.review) AND the cut gate (`cut.request.json`,
    written by cut_qc.review_cut with review_dir=<base>-shots/review). Both
    live in the same per-video review/ directory, so video_base is always
    recoverable by stripping the "-shots" suffix off the parent-of-parent
    directory name.

Like status.py, every helper here is best-effort and never raises: a
missing/partial/corrupt request file just drops that one entry rather than
blowing up the whole scan. No Flask imports — pure filesystem logic.
"""
import glob
import json
import os
import re
import time

from pipeline.config import abspath

_SHOT_REQUEST_RE = re.compile(r"^(?P<take>s\d+_a\d+)\.request\.json$")
_REQUEST_SUFFIX = ".request.json"
_RESPONSE_SUFFIX = ".response.json"


# ------------------------------------------------------------------ small utils ---
def _safe_read_json(path):
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError):
        return None


def _safe_mtime(path):
    try:
        return os.stat(path).st_mtime
    except OSError:
        return time.time()


def _relpath(out_dir, path):
    if not path:
        return None
    try:
        return os.path.relpath(path, out_dir).replace(os.sep, "/")
    except (OSError, ValueError):
        return None


def _response_path_for(request_path):
    if request_path.endswith(_REQUEST_SUFFIX):
        return request_path[: -len(_REQUEST_SUFFIX)] + _RESPONSE_SUFFIX
    # Defensive fallback — shouldn't happen given how callers glob for
    # _REQUEST_SUFFIX, but never raise over a naming edge case.
    root, _ = os.path.splitext(request_path)
    return root + _RESPONSE_SUFFIX


def _video_base_from_shots_dir(shots_dir_name):
    if shots_dir_name.endswith("-shots"):
        return shots_dir_name[: -len("-shots")]
    return shots_dir_name


# --------------------------------------------------------------------- items ---
def _base_item(out_dir, request_path, response_path, kind, video_base):
    return {
        "kind": kind,
        "request_path": _relpath(out_dir, request_path),
        "response_path": _relpath(out_dir, response_path),
        "video_base": video_base,
        "age_seconds": max(0.0, time.time() - _safe_mtime(request_path)),
    }


def _build_shot_item(out_dir, request_path, response_path, video_base):
    body = _safe_read_json(request_path)
    if body is None:
        return None
    shot = body.get("shot") or {}
    item = _base_item(out_dir, request_path, response_path, "shot", video_base)
    item.update(
        {
            "shot_id": body.get("shot_id"),
            "take": body.get("take"),
            "contact_sheet": _relpath(out_dir, body.get("contact_sheet")),
            "shot": {
                "action": shot.get("action"),
                "characters": shot.get("characters"),
                "shot_type": shot.get("shot_type"),
                "camera": shot.get("camera"),
                "setting": shot.get("setting"),
            },
            "heuristic": body.get("heuristic"),
            # Cast ground-truth (pipeline.kidsong.review._resolve_cast_enrichment)
            # -- optional, so a request written before this enrichment existed
            # (or one where cast.py lookup degraded) just yields None here.
            "cast_text": body.get("cast_text"),
            "cast_version": body.get("cast_version"),
            "expected_children": body.get("expected_children"),
        }
    )
    return item


def _build_cut_item(out_dir, request_path, response_path, video_base):
    body = _safe_read_json(request_path)
    if body is None:
        return None
    item = _base_item(out_dir, request_path, response_path, "cut", video_base)
    item.update(
        {
            "video_path": _relpath(out_dir, body.get("video_path")),
            "cut_list_summary": body.get("cut_list_summary") or [],
            "programmatic_verdict": body.get("programmatic_verdict"),
        }
    )
    return item


def _build_script_item(out_dir, request_path, response_path):
    body = _safe_read_json(request_path)
    if body is None:
        return None
    item = _base_item(out_dir, request_path, response_path, "script", "current run")
    item.update(
        {
            "song": body.get("song") or {},
            "programmatic_verdict": body.get("programmatic_verdict"),
        }
    )
    return item


def _build_shotlist_item(out_dir, request_path, response_path):
    body = _safe_read_json(request_path)
    if body is None:
        return None
    shotlist = body.get("shotlist") or {}
    shots = shotlist.get("shots") if isinstance(shotlist, dict) else shotlist
    shots = list(shots or [])
    item = _base_item(out_dir, request_path, response_path, "shotlist", "current run")
    item.update(
        {
            "song": body.get("song") or {},
            "shots": shots,
            "shot_count": len(shots),
            "programmatic_verdict": body.get("programmatic_verdict"),
        }
    )
    return item


# ------------------------------------------------------------------ scanning ---
def _pending_requests(pattern):
    """Every `*.request.json` matching glob `pattern` without a sibling response."""
    for request_path in sorted(glob.glob(pattern)):
        response_path = _response_path_for(request_path)
        if not os.path.exists(response_path):
            yield request_path, response_path


def list_pending(cfg, out_dir=None):
    """Scan `output/` for every pending (response-less) review request.

    Returns a list of display-ready dicts (see module docstring for the
    on-disk shapes), sorted oldest-request-first. `out_dir` may be passed
    explicitly to point the scan at an isolated test directory instead of
    the configured output_dir.
    """
    out_dir = out_dir if out_dir is not None else abspath(cfg, cfg["paths"]["output_dir"])
    items = []

    # -- fixed-path gates: script + shotlist (no per-video subfolder) --
    review_dir = os.path.join(out_dir, "_kidsong_review")
    builders = {"script": _build_script_item, "shotlist": _build_shotlist_item}
    for stage, builder in builders.items():
        request_path = os.path.join(review_dir, f"{stage}{_REQUEST_SUFFIX}")
        response_path = _response_path_for(request_path)
        if os.path.exists(request_path) and not os.path.exists(response_path):
            try:
                item = builder(out_dir, request_path, response_path)
            except Exception:
                item = None
            if item is not None:
                items.append(item)

    # -- per-video gates: shot takes + cut, both under <base>-shots/review/ --
    pattern = os.path.join(out_dir, "*-shots", "review", f"*{_REQUEST_SUFFIX}")
    for request_path, response_path in _pending_requests(pattern):
        review_subdir = os.path.dirname(request_path)  # .../<base>-shots/review
        shots_dir = os.path.dirname(review_subdir)  # .../<base>-shots
        video_base = _video_base_from_shots_dir(os.path.basename(shots_dir))
        name = os.path.basename(request_path)

        try:
            if name == f"cut{_REQUEST_SUFFIX}":
                item = _build_cut_item(out_dir, request_path, response_path, video_base)
            elif _SHOT_REQUEST_RE.match(name):
                item = _build_shot_item(out_dir, request_path, response_path, video_base)
            else:
                item = None  # unrecognized request file — skip rather than guess
        except Exception:
            item = None

        if item is not None:
            items.append(item)

    items.sort(key=lambda it: it.get("age_seconds", 0), reverse=True)  # oldest (largest age) first
    return items


_EMPTY_RETRY_HINTS = {"seed_bump": False, "simplify_action": False, "force_i2v": False}


def _auto_accept_policy(cfg):
    """The per-shot prefilter policy, from `review._shot_auto_accept_policy`.

    Imported lazily so this scanner stays free of `review`'s heavy cv2/numpy
    import (the studio browser imports this module just to LIST the queue). If
    `review` can't be imported for any reason, degrade to the safe default —
    enabled, surface nothing — so a drain still works headless.
    """
    try:
        from pipeline.kidsong.review import _shot_auto_accept_policy

        return _shot_auto_accept_policy(cfg)
    except Exception:
        return True, set(), 1


def _stored_shot_exceeds_child_cap(body, max_children):
    """True when the stored request's shot stages more expected children than
    `max_children`. Prefers the request's own `expected_children` enrichment
    (written by the live reviewer), falls back to resolving the stored shot
    dict via cast. Best-effort: anything unresolvable returns False (fail-open,
    matching the live prefilter's guard)."""
    try:
        expected = body.get("expected_children")
        if expected is None:
            shot = body.get("shot") or {}
            from pipeline.kidsong import cast

            expected = cast.expected_child_count(
                shot.get("characters"), shot.get("shot_type")
            )
        return expected is not None and int(expected) > int(max_children)
    except Exception:
        return False


def drain_auto_acceptable(cfg, out_dir=None, dry_run=False):
    """Retroactively clear the backlog: write an auto-accept response for every
    pending SHOT request whose stored heuristic verdict the per-shot prefilter
    would clear (heuristic accepted, no soft reason in `surface_reasons`).

    This is the one-time complement to the live prefilter in
    `review.ExternalReviewer.review`: episodes rendered before the prefilter (or
    while unattended mode wrote requests it never answered) left hundreds of
    per-shot requests with no response, clogging the human queue with shots that
    need no eyes. Cut / script / shotlist requests are NEVER touched — those
    still surface. Idempotent and best-effort: a request already answered, or one
    the policy would surface, is left alone. Returns a summary dict.
    """
    out_dir = out_dir if out_dir is not None else abspath(cfg, cfg["paths"]["output_dir"])
    enabled, surface_reasons, max_children = _auto_accept_policy(cfg)
    summary = {"scanned": 0, "drained": 0, "surfaced": 0, "skipped_non_shot": 0, "errors": 0, "drained_paths": []}
    if not enabled:
        return summary

    pattern = os.path.join(out_dir, "*-shots", "review", f"*{_REQUEST_SUFFIX}")
    for request_path, response_path in _pending_requests(pattern):
        name = os.path.basename(request_path)
        if not _SHOT_REQUEST_RE.match(name):  # cut.request.json etc. — keep for the human
            summary["skipped_non_shot"] += 1
            continue
        summary["scanned"] += 1
        body = _safe_read_json(request_path)
        if body is None:
            summary["errors"] += 1
            continue
        heuristic = body.get("heuristic") or {}
        reasons = set(heuristic.get("reasons") or [])
        if not heuristic.get("accept") or (surface_reasons & reasons):
            # The live reviewer would have surfaced this one too — leave it.
            summary["surfaced"] += 1
            continue
        if max_children and _stored_shot_exceeds_child_cap(body, max_children):
            # A stored multi-child request: the live prefilter no longer blind-
            # auto-accepts these (clone-collapse territory the heuristic can't
            # see), so the drain must not either — keep it for real eyes.
            summary["surfaced"] += 1
            continue
        response = {
            "accept": True,
            "score": heuristic.get("score", 1.0),
            "reasons": list(heuristic.get("reasons") or []) + [
                "auto-accepted (backlog drain: heuristic clear, no surfaced concerns)"
            ],
            "retry_hints": dict(_EMPTY_RETRY_HINTS),
            "auto_accepted": True,
        }
        if dry_run:
            summary["drained"] += 1
            summary["drained_paths"].append(_relpath(out_dir, request_path))
            continue
        try:
            with open(response_path, "w", encoding="utf-8") as f:
                json.dump(response, f, indent=2)
            summary["drained"] += 1
            summary["drained_paths"].append(_relpath(out_dir, request_path))
        except OSError:
            summary["errors"] += 1
    return summary


if __name__ == "__main__":
    import argparse

    from pipeline.config import load_config

    parser = argparse.ArgumentParser(description="Kidsong review-queue scanner / backlog drain.")
    parser.add_argument("--drain", action="store_true",
                        help="Write auto-accept responses for pending shot requests the prefilter clears.")
    parser.add_argument("--dry-run", action="store_true",
                        help="With --drain, report what WOULD be drained without writing anything.")
    args = parser.parse_args()

    _cfg = load_config()
    if args.drain:
        _summary = drain_auto_acceptable(_cfg, dry_run=args.dry_run)
        _paths = _summary.pop("drained_paths", [])
        print(("DRY-RUN " if args.dry_run else "") + "drain summary: " + json.dumps(_summary))
        for _p in _paths:
            print(("would drain: " if args.dry_run else "drained: ") + str(_p))
    else:
        for _item in list_pending(_cfg):
            print(json.dumps(_item, indent=2, default=str))
