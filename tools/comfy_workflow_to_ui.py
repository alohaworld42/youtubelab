"""comfy_workflow_to_ui — convert an API-format ComfyUI workflow into the
UI ("workflow") format that the ComfyUI web editor can open and edit.

Why this exists
---------------
The repo's ``workflows/*.json`` are **API format** — ``{node_id: {class_type,
inputs, _meta}}`` — because that is the only thing ``POST /prompt`` accepts, and
``pipeline/kidsong/comfy.py`` patches them by ``_meta.title``. The ComfyUI
workflow sidebar, however, loads files through ``validateComfyWorkflow()``,
which rejects anything without a ``version`` field and expects the litegraph
serialisation (``{nodes: [...], links: [...]}``). API JSON therefore never shows
up as an openable graph. This module bridges the two.

The conversion is driven entirely by ``/object_info``, which is authoritative
for input *ordering*. Getting that ordering wrong is the failure mode that
silently corrupts a graph: ``widgets_values`` is a **positional** array, so a
single missing or extra entry shifts every later value onto the wrong widget.

Rules (each verified against ComfyUI's own 1513 template workflows — see
``tests/test_comfy_workflow_sync.py`` and the module docstring notes below):

1. An input is a *widget* if its type is INT / FLOAT / STRING / BOOLEAN / COMBO
   or an inline option list (``[["a.safetensors", ...], {...}]``); anything else
   (MODEL, CLIP, LATENT, IMAGE, ...) is a *link* input.
2. ``widgets_values`` is ordered by ``input_order`` (required then optional) and
   contains an entry for **every** widget input — including widget inputs that
   are driven by a link (verified: 115 real nodes keep the slot vs 7 that drop
   it). Linked widgets store the class default as a placeholder.
3. An INT/FLOAT widget whose spec carries ``control_after_generate`` contributes
   an **extra** trailing value (``"fixed"``/``"randomize"``). This is why
   ``RandomNoise`` serialises as ``[seed, "fixed"]`` and ``PrimitiveInt`` as
   ``[value, "fixed"]``.
4. A combo widget whose spec carries ``image_upload`` / ``video_upload`` /
   ``audio_upload`` contributes an extra trailing value naming the upload kind
   (``LoadImage`` -> ``[filename, "image"]``; observed 442x in templates).
5. Link inputs always get an input slot, connected or not. A *widget* input that
   is connected additionally gets a slot carrying ``shape: 7`` and
   ``widget: {"name": ...}``.

Node titles from ``_meta.title`` are preserved verbatim — they are the
pipeline's patch contract (PROMPT, SEED, WIDTH, ...) and must stay visible and
editable in the UI.

CLI::

    python tools/comfy_workflow_to_ui.py workflows/ltx23_t2v_toon.json -o out.json
"""
from __future__ import annotations

import json
from pathlib import Path

# Input types that render as an on-node widget rather than a link socket.
WIDGET_TYPES = {"INT", "FLOAT", "STRING", "BOOLEAN", "COMBO"}

# Spec options that append an extra trailing widget value.
_UPLOAD_OPTS = {
    "image_upload": "image",
    "video_upload": "video",
    "audio_upload": "audio",
}

# Layout geometry for the generated grid.
_COL_WIDTH = 400
_ROW_HEIGHT = 190
_ORIGIN = (40, 40)
_DEFAULT_SIZE = [270, 100]

DEFAULT_OBJECT_INFO_URL = "http://127.0.0.1:8188/object_info"


class ConversionError(RuntimeError):
    """Raised when a workflow cannot be faithfully converted."""


# ------------------------------------------------------------------ specs ---
def is_widget_spec(spec):
    """True if an ``/object_info`` input spec renders as a widget."""
    if not isinstance(spec, (list, tuple)) or not spec:
        return False
    type_ = spec[0]
    if isinstance(type_, list):
        # Inline option list, e.g. [["model_a.safetensors", ...], {...}]
        return True
    return type_ in WIDGET_TYPES


def _spec_opts(spec):
    if isinstance(spec, (list, tuple)) and len(spec) > 1 and isinstance(spec[1], dict):
        return spec[1]
    return {}


def _spec_default(spec):
    """Best-effort default value for a widget spec."""
    opts = _spec_opts(spec)
    if "default" in opts:
        return opts["default"]
    type_ = spec[0]
    if isinstance(type_, list):
        return type_[0] if type_ else ""
    if type_ == "COMBO":
        options = opts.get("options") or []
        return options[0] if options else ""
    if type_ == "INT":
        return int(opts.get("min", 0) or 0)
    if type_ == "FLOAT":
        return float(opts.get("min", 0.0) or 0.0)
    if type_ == "BOOLEAN":
        return False
    return ""


def _ordered_inputs(class_def):
    """Yield ``(name, spec, section)`` in ComfyUI's authoritative input order.

    ``hidden`` inputs (``prompt``, ``extra_pnginfo``) are intentionally skipped:
    they are server-injected and never appear in the UI graph.
    """
    inputs = class_def.get("input") or {}
    order = class_def.get("input_order") or {}
    for section in ("required", "optional"):
        specs = inputs.get(section) or {}
        names = order.get(section) or list(specs.keys())
        for name in names:
            if name in specs:
                yield name, specs[name], section


def widget_plan(class_def):
    """Ordered widget slots for a class.

    Returns a list of ``(input_name, kind, default)`` where ``kind`` is
    ``"value"`` for the real input, or ``"control"`` / ``"upload"`` for the
    synthetic trailing entries the frontend appends.
    """
    plan = []
    for name, spec, _section in _ordered_inputs(class_def):
        if not is_widget_spec(spec):
            continue
        opts = _spec_opts(spec)
        plan.append((name, "value", _spec_default(spec)))
        control = opts.get("control_after_generate")
        if control:
            # A string value is itself the default mode; True means "fixed"
            # is a safe choice for a pipeline graph (the caller sets the seed).
            plan.append((name, "control",
                         control if isinstance(control, str) else "fixed"))
        for opt_key, kind in _UPLOAD_OPTS.items():
            if opts.get(opt_key):
                plan.append((name, "upload", kind))
    return plan


def link_plan(class_def):
    """Ordered link-type inputs for a class as ``(input_name, type_name)``."""
    out = []
    for name, spec, _section in _ordered_inputs(class_def):
        if is_widget_spec(spec):
            continue
        type_ = spec[0] if isinstance(spec, (list, tuple)) and spec else "*"
        out.append((name, type_ if isinstance(type_, str) else "*"))
    return out


def _widget_type_name(spec):
    type_ = spec[0]
    if isinstance(type_, list):
        return "COMBO"
    return type_


def _spec_for(class_def, input_name):
    for name, spec, _section in _ordered_inputs(class_def):
        if name == input_name:
            return spec
    return None


# ----------------------------------------------------------------- layout ---
def _node_depth(api_workflow):
    """Longest-path depth per node id, for a left-to-right layout."""
    deps = {}
    for node_id, node in api_workflow.items():
        upstream = []
        for value in (node.get("inputs") or {}).values():
            if isinstance(value, list) and len(value) == 2:
                upstream.append(str(value[0]))
        deps[str(node_id)] = upstream

    depth = {}

    def resolve(node_id, seen):
        if node_id in depth:
            return depth[node_id]
        if node_id in seen:      # cycle guard; ComfyUI graphs are DAGs
            return 0
        seen = seen | {node_id}
        parents = [p for p in deps.get(node_id, []) if p in deps]
        d = 0 if not parents else 1 + max(resolve(p, seen) for p in parents)
        depth[node_id] = d
        return d

    for node_id in deps:
        resolve(node_id, frozenset())
    return depth


# ------------------------------------------------------------- conversion ---
def convert_workflow(api_workflow, object_info):
    """Convert an API-format workflow dict into a UI-format workflow dict.

    Raises ``ConversionError`` if a class is unknown to ``object_info`` or an
    input cannot be placed, rather than emitting a silently corrupt graph.
    """
    if not isinstance(api_workflow, dict) or not api_workflow:
        raise ConversionError("API workflow must be a non-empty object")

    missing = sorted({
        node.get("class_type") for node in api_workflow.values()
        if node.get("class_type") not in object_info
    })
    if missing:
        raise ConversionError(
            "object_info has no definition for: " + ", ".join(map(str, missing))
        )

    depth = _node_depth(api_workflow)
    # Stable ordering: by depth, then by numeric node id.
    def sort_key(node_id):
        try:
            numeric = int(node_id)
        except (TypeError, ValueError):
            numeric = 0
        return (depth.get(str(node_id), 0), numeric, str(node_id))

    ordered_ids = sorted(api_workflow.keys(), key=sort_key)

    # ---- pass 1: build nodes (no links yet) --------------------------------
    nodes = {}
    column_counts = {}
    for order_index, node_id in enumerate(ordered_ids):
        api_node = api_workflow[node_id]
        class_type = api_node["class_type"]
        class_def = object_info[class_type]
        api_inputs = api_node.get("inputs") or {}

        col = depth.get(str(node_id), 0)
        row = column_counts.get(col, 0)
        column_counts[col] = row + 1

        # widgets_values: one entry per widget slot, positional.
        widgets_values = []
        for input_name, kind, default in widget_plan(class_def):
            if kind != "value":
                widgets_values.append(default)
                continue
            value = api_inputs.get(input_name, default)
            if isinstance(value, list) and len(value) == 2:
                # Driven by a link: keep the positional slot, store the default.
                value = default
            widgets_values.append(value)

        # input slots: all link inputs, plus connected widget inputs.
        input_slots = []
        link_types = dict(link_plan(class_def))
        for input_name, _spec, _section in _ordered_inputs(class_def):
            spec = _spec_for(class_def, input_name)
            connected = isinstance(api_inputs.get(input_name), list) and \
                len(api_inputs.get(input_name)) == 2
            if not is_widget_spec(spec):
                input_slots.append({
                    "name": input_name,
                    "type": link_types.get(input_name, "*"),
                    "link": None,
                })
            elif connected:
                input_slots.append({
                    "name": input_name,
                    "shape": 7,
                    "type": _widget_type_name(spec),
                    "widget": {"name": input_name},
                    "link": None,
                })

        output_types = class_def.get("output") or []
        output_names = class_def.get("output_name") or output_types
        output_slots = []
        for slot, out_type in enumerate(output_types):
            name = output_names[slot] if slot < len(output_names) else out_type
            output_slots.append({
                "name": name if isinstance(name, str) else str(out_type),
                "type": out_type if isinstance(out_type, str) else "*",
                "slot_index": slot,
                "links": [],
            })

        try:
            numeric_id = int(node_id)
        except (TypeError, ValueError):
            raise ConversionError(
                f"Node id {node_id!r} is not an integer; the UI format requires "
                f"numeric node ids."
            )

        node = {
            "id": numeric_id,
            "type": class_type,
            "pos": [_ORIGIN[0] + col * _COL_WIDTH, _ORIGIN[1] + row * _ROW_HEIGHT],
            "size": list(_DEFAULT_SIZE),
            "flags": {},
            "order": order_index,
            "mode": 0,
            "inputs": input_slots,
            "outputs": output_slots,
            "properties": {"Node name for S&R": class_type},
            "widgets_values": widgets_values,
        }
        title = (api_node.get("_meta") or {}).get("title")
        if title:
            # The pipeline patches by title; it must survive into the UI.
            node["title"] = title
        nodes[str(node_id)] = node

    # ---- pass 2: wire links ------------------------------------------------
    links = []
    next_link_id = 1
    for node_id in ordered_ids:
        api_node = api_workflow[node_id]
        target = nodes[str(node_id)]
        for input_name, value in (api_node.get("inputs") or {}).items():
            if not (isinstance(value, list) and len(value) == 2):
                continue
            origin_id, origin_slot = str(value[0]), value[1]
            origin = nodes.get(origin_id)
            if origin is None:
                raise ConversionError(
                    f"Node {node_id} input {input_name!r} references unknown "
                    f"node {origin_id!r}"
                )
            slot_index = next(
                (i for i, s in enumerate(target["inputs"]) if s["name"] == input_name),
                None,
            )
            if slot_index is None:
                raise ConversionError(
                    f"Node {node_id} ({target['type']}) has no input slot for "
                    f"{input_name!r}; /object_info may be out of date"
                )
            try:
                origin_slot = int(origin_slot)
            except (TypeError, ValueError):
                raise ConversionError(
                    f"Node {node_id} input {input_name!r} has non-integer slot "
                    f"{origin_slot!r}"
                )
            if origin_slot >= len(origin["outputs"]):
                raise ConversionError(
                    f"Node {node_id} input {input_name!r} references output slot "
                    f"{origin_slot} of node {origin_id} "
                    f"({origin['type']}), which has {len(origin['outputs'])} outputs"
                )

            link_id = next_link_id
            next_link_id += 1
            link_type = origin["outputs"][origin_slot]["type"]
            target["inputs"][slot_index]["link"] = link_id
            origin["outputs"][origin_slot]["links"].append(link_id)
            links.append([
                link_id, origin["id"], origin_slot,
                target["id"], slot_index, link_type,
            ])

    node_list = [nodes[str(n)] for n in ordered_ids]
    for node in node_list:
        for slot in node["outputs"]:
            if not slot["links"]:
                slot["links"] = None

    last_node_id = max((n["id"] for n in node_list), default=0)
    return {
        "id": "",
        "revision": 0,
        "last_node_id": last_node_id,
        "last_link_id": next_link_id - 1,
        "nodes": node_list,
        "links": links,
        "groups": [],
        "config": {},
        "extra": {
            "ds": {"scale": 0.7, "offset": [0, 0]},
            # Provenance: this graph is generated; the repo copy is authoritative.
            "brainrot_generated_from": "workflows/ (API format, source of truth)",
        },
        "version": 0.4,
    }


# ------------------------------------------------------------- object_info ---
def load_object_info(source=None):
    """Load ``/object_info`` from a URL or a cached JSON file path."""
    source = source or DEFAULT_OBJECT_INFO_URL
    text = str(source)
    if text.startswith("http://") or text.startswith("https://"):
        import requests  # lazy: keeps this importable without the dependency
        response = requests.get(text, timeout=120)
        response.raise_for_status()
        return response.json()
    with open(text, "r", encoding="utf-8") as handle:
        return json.load(handle)


def convert_file(api_path, object_info):
    """Read an API-format workflow file and return the UI-format dict."""
    with open(api_path, "r", encoding="utf-8") as handle:
        return convert_workflow(json.load(handle), object_info)


# ------------------------------------------------------------------ verify ---
def verify_conversion(api_workflow, ui_workflow, object_info):
    """Check a converted graph against its source. Returns a list of problems.

    An empty list means: every node survived, every widget array has the exact
    length ``/object_info`` dictates, every link endpoint resolves to a real
    node and slot, every scalar input value round-trips, and every
    ``_meta.title`` is still present as a node title.
    """
    problems = []
    nodes_by_id = {n["id"]: n for n in ui_workflow["nodes"]}

    if len(ui_workflow["nodes"]) != len(api_workflow):
        problems.append(
            f"node count {len(ui_workflow['nodes'])} != source {len(api_workflow)}"
        )

    for node_id, api_node in api_workflow.items():
        try:
            key = int(node_id)
        except (TypeError, ValueError):
            problems.append(f"source node id {node_id!r} is not numeric")
            continue
        node = nodes_by_id.get(key)
        if node is None:
            problems.append(f"node {node_id} missing from converted graph")
            continue
        if node["type"] != api_node["class_type"]:
            problems.append(
                f"node {node_id} type {node['type']} != {api_node['class_type']}"
            )
        title = (api_node.get("_meta") or {}).get("title")
        if title and node.get("title") != title:
            problems.append(
                f"node {node_id} title {node.get('title')!r} != {title!r}"
            )

        class_def = object_info[api_node["class_type"]]
        plan = widget_plan(class_def)
        if len(node["widgets_values"]) != len(plan):
            problems.append(
                f"node {node_id} ({node['type']}) widgets_values has "
                f"{len(node['widgets_values'])} entries, expected {len(plan)} "
                f"({[p[0] for p in plan]})"
            )
        else:
            # Every scalar input value must round-trip into its widget slot.
            positions = {}
            for index, (name, kind, _default) in enumerate(plan):
                if kind == "value":
                    positions[name] = index
            for input_name, value in (api_node.get("inputs") or {}).items():
                if isinstance(value, list) and len(value) == 2:
                    continue   # link, checked below
                index = positions.get(input_name)
                if index is None:
                    problems.append(
                        f"node {node_id} ({node['type']}) input {input_name!r}="
                        f"{value!r} has no widget slot in the converted graph"
                    )
                elif node["widgets_values"][index] != value:
                    problems.append(
                        f"node {node_id} ({node['type']}) widget {input_name!r} "
                        f"is {node['widgets_values'][index]!r}, expected {value!r}"
                    )

    # Links: every endpoint must resolve.
    seen_link_ids = set()
    for link in ui_workflow["links"]:
        link_id, origin_id, origin_slot, target_id, target_slot, _type = link
        if link_id in seen_link_ids:
            problems.append(f"duplicate link id {link_id}")
        seen_link_ids.add(link_id)
        origin, target = nodes_by_id.get(origin_id), nodes_by_id.get(target_id)
        if origin is None:
            problems.append(f"link {link_id} origin node {origin_id} does not exist")
            continue
        if target is None:
            problems.append(f"link {link_id} target node {target_id} does not exist")
            continue
        if not 0 <= origin_slot < len(origin["outputs"]):
            problems.append(
                f"link {link_id} origin slot {origin_slot} out of range for "
                f"node {origin_id} ({origin['type']})"
            )
        elif link_id not in (origin["outputs"][origin_slot]["links"] or []):
            problems.append(
                f"link {link_id} not listed on origin node {origin_id} output "
                f"slot {origin_slot}"
            )
        if not 0 <= target_slot < len(target["inputs"]):
            problems.append(
                f"link {link_id} target slot {target_slot} out of range for "
                f"node {target_id} ({target['type']})"
            )
        elif target["inputs"][target_slot]["link"] != link_id:
            problems.append(
                f"link {link_id} not attached to target node {target_id} input "
                f"slot {target_slot}"
            )

    # Every link in the source must exist in the converted graph.
    expected_links = sum(
        1
        for api_node in api_workflow.values()
        for value in (api_node.get("inputs") or {}).values()
        if isinstance(value, list) and len(value) == 2
    )
    if len(ui_workflow["links"]) != expected_links:
        problems.append(
            f"link count {len(ui_workflow['links'])} != source {expected_links}"
        )

    if ui_workflow.get("version") is None:
        problems.append("missing 'version' — ComfyUI will reject the file")
    return problems


# -------------------------------------------------------------------- main ---
def main(argv=None):
    import argparse

    parser = argparse.ArgumentParser(
        description="Convert an API-format ComfyUI workflow to UI format."
    )
    parser.add_argument("workflow", help="path to an API-format workflow JSON")
    parser.add_argument("-o", "--out", help="output path (default: stdout)")
    parser.add_argument(
        "--object-info", default=None,
        help=f"URL or cached JSON path (default: {DEFAULT_OBJECT_INFO_URL})",
    )
    args = parser.parse_args(argv)

    object_info = load_object_info(args.object_info)
    with open(args.workflow, "r", encoding="utf-8") as handle:
        api_workflow = json.load(handle)
    ui_workflow = convert_workflow(api_workflow, object_info)

    problems = verify_conversion(api_workflow, ui_workflow, object_info)
    if problems:
        raise SystemExit(
            "conversion produced problems:\n  " + "\n  ".join(problems)
        )

    text = json.dumps(ui_workflow, indent=2)
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(text, encoding="utf-8")
        print(f"wrote {args.out}")
    else:
        print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
