"""sync_comfy_workflows — install the repo's API-format workflows into the
ComfyUI web UI as openable, editable graphs.

The repo's ``workflows/*.json`` stay the single source of truth (the pipeline
POSTs them verbatim to ``/prompt``). This script converts each one to UI format
and writes it under ComfyUI's user workflow directory, where the workflow
sidebar picks it up. It is idempotent: re-run it after any workflow change.

    venv\\Scripts\\python.exe tools/sync_comfy_workflows.py
    venv\\Scripts\\python.exe tools/sync_comfy_workflows.py --dry-run
    venv\\Scripts\\python.exe tools/sync_comfy_workflows.py --object-info cached.json

WARNING: edits made in the ComfyUI UI change only the synced copy. They do not
affect what the pipeline renders. To make a UI tweak stick, export the graph in
**API format** and save it back over the matching file in ``workflows/``.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from comfy_workflow_to_ui import (  # noqa: E402
    DEFAULT_OBJECT_INFO_URL,
    ConversionError,
    convert_workflow,
    load_object_info,
    verify_conversion,
)

_ROOT = Path(__file__).resolve().parents[1]
_REPO_WORKFLOWS = _ROOT / "workflows"

# Where ComfyUI keeps user workflows. Overridable via --comfy-path / config.
_DEFAULT_COMFY_PATH = Path(r"C:\Users\Aloha\Desktop\projects\ComfyUI")
_USER_WORKFLOW_SUBDIR = Path("user") / "default" / "workflows"

# Grouping subfolder so the graphs cluster in the sidebar.
_GROUP = "kidsong"
_PREFIX = "kidsong_"


def _comfy_path(explicit=None):
    """Resolve the ComfyUI install dir: flag > config.json > default."""
    if explicit:
        return Path(explicit)
    try:
        sys.path.insert(0, str(_ROOT))
        from pipeline.config import load_config

        configured = (load_config() or {}).get("comfy", {}).get("path")
        if configured:
            return Path(configured)
    except Exception:
        pass
    return _DEFAULT_COMFY_PATH


def sync(comfy_path=None, object_info_source=None, dry_run=False,
         workflows_dir=None):
    """Convert every repo workflow and write it into ComfyUI's user dir.

    Returns a list of ``(source_path, dest_path, node_count)`` tuples.
    """
    workflows_dir = Path(workflows_dir or _REPO_WORKFLOWS)
    sources = sorted(workflows_dir.glob("*.json"))
    if not sources:
        raise SystemExit(f"no workflows found in {workflows_dir}")

    object_info = load_object_info(object_info_source)

    dest_dir = _comfy_path(comfy_path) / _USER_WORKFLOW_SUBDIR / _GROUP
    if not dry_run:
        dest_dir.mkdir(parents=True, exist_ok=True)

    written = []
    for source in sources:
        api_workflow = json.loads(source.read_text(encoding="utf-8"))
        try:
            ui_workflow = convert_workflow(api_workflow, object_info)
        except ConversionError as exc:
            raise SystemExit(f"{source.name}: {exc}")

        problems = verify_conversion(api_workflow, ui_workflow, object_info)
        if problems:
            raise SystemExit(
                f"{source.name}: refusing to write a graph that failed "
                f"verification:\n  " + "\n  ".join(problems)
            )

        dest = dest_dir / (_PREFIX + source.name)
        text = json.dumps(ui_workflow, indent=2)
        if not dry_run:
            dest.write_text(text, encoding="utf-8")
        written.append((source, dest, len(ui_workflow["nodes"])))
    return written


def main(argv=None):
    import argparse

    parser = argparse.ArgumentParser(
        description="Sync repo workflows into the ComfyUI web UI (UI format)."
    )
    parser.add_argument("--comfy-path", default=None,
                        help="ComfyUI install dir (default: config.json, then "
                             "the standard local path)")
    parser.add_argument("--object-info", default=None,
                        help=f"URL or cached JSON path "
                             f"(default: {DEFAULT_OBJECT_INFO_URL})")
    parser.add_argument("--dry-run", action="store_true",
                        help="convert and verify but write nothing")
    args = parser.parse_args(argv)

    written = sync(
        comfy_path=args.comfy_path,
        object_info_source=args.object_info,
        dry_run=args.dry_run,
    )

    verb = "would write" if args.dry_run else "wrote"
    for source, dest, node_count in written:
        print(f"{verb} {dest}  ({node_count} nodes, from {source.name})")
    print(f"\n{len(written)} workflow(s) synced.")
    if not args.dry_run:
        print("Open them in ComfyUI: Workflows sidebar -> "
              f"{_GROUP}/ (refresh the browser tab if it was already open).")
    print("NOTE: the repo's workflows/ copies remain authoritative; UI edits "
          "do not change what the pipeline renders.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
