"""Structural parity tests for `workflows/ltx23_i2v_toon_hires.json`.

This graph is not hand-authored from scratch: it is `ltx23_i2v_toon.json`
(the i2v graph, 26 nodes) with node 16's SIGMAS text swapped for the 9-step
hires schedule, node 21's DECODE rewired to read the refine pass's output
instead of the first pass's, and node 23's FILENAME_PREFIX pointed at the
hires output path -- plus a refine subgraph, nodes 27-36, copied verbatim
from `ltx23_t2v_toon_hires.json`'s own refine nodes (24-33) with every
internal link shifted by the same +3 id offset the copy applied.

One node is NOT a copy: 37 (IMG_TO_VIDEO_REFINE) re-runs the base graph's
LTXVImgToVideoInplace on the UPSAMPLED latent before the refine concat, and
CONCAT_AV_REFINE (34) reads it instead of the raw upsampler. A naive graft
without it rendered BLACK frames (measured 2026-07-24 on the GPU proof): the
upsampler pops the video half of the i2v noise_mask while the base-pass audio
latent keeps its half, and the refine sampler blacks out on the resulting
synthesized nested mask. Re-injecting restores the exact mask shape the
(working) base pass samples with, and re-anchors frame 0 at 2x -- the
official LTX multiscale i2v pattern.

These tests re-derive that relationship generically (a recursive dict-diff
and a link-remapping helper) rather than asserting a handful of spot values,
so ANY unintended drift between the three files -- a changed cfg value, a
rewired link, a dropped input -- fails loudly instead of slipping through a
narrower check.

CPU-only: no GPU, no network, no real ComfyUI. Reads the real repo JSON
straight off disk (same trick as tests/test_workflow_negatives.py).
"""
import json
import os

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_WORKFLOWS_DIR = os.path.join(_ROOT, "workflows")

_MISSING = object()

# The t2v-hires refine node ids that were copied into the i2v-hires graph,
# and the offset applied to their own ids and to any link pointing at them.
_REMAPPED_ID_RANGE = range(24, 34)
_ID_OFFSET = 3


def _load(filename):
    with open(os.path.join(_WORKFLOWS_DIR, filename), encoding="utf-8") as fh:
        return json.load(fh)


def _i2v():
    return _load("ltx23_i2v_toon.json")


def _i2v_hires():
    return _load("ltx23_i2v_toon_hires.json")


def _t2v_hires():
    return _load("ltx23_t2v_toon_hires.json")


def _diff_paths(a, b, prefix=""):
    """Dotted key-paths where `a` and `b` differ, recursing into dicts.

    A key present on only one side counts as a diff too (via the `_MISSING`
    sentinel), so a dropped or added input is never silently missed by a
    same-keys-only comparison.
    """
    if isinstance(a, dict) and isinstance(b, dict):
        paths = set()
        for key in set(a) | set(b):
            sub_prefix = f"{prefix}.{key}" if prefix else key
            paths |= _diff_paths(a.get(key, _MISSING), b.get(key, _MISSING), sub_prefix)
        return paths
    return set() if a == b else {prefix}


def _remap_links(node):
    """A deep copy of `node` with every `["<id>", idx]` link array whose id
    falls in the refine-node range (24..33) shifted by `_ID_OFFSET` -- the
    same shift applied when the t2v-hires refine subgraph was copied into
    the i2v-hires graph. Links to the shared base graph (ids 1..23, which
    mean the same thing in both graphs) are left untouched."""
    remapped = json.loads(json.dumps(node))
    for value in remapped.get("inputs", {}).values():
        if (
            isinstance(value, list) and len(value) == 2
            and isinstance(value[0], str) and value[0].isdigit()
        ):
            nid = int(value[0])
            if nid in _REMAPPED_ID_RANGE:
                value[0] = str(nid + _ID_OFFSET)
    return remapped


# ------------------------------------------------------------------- (a) ids ---
def test_node_ids_are_exactly_1_through_37_contiguous():
    graph = _i2v_hires()
    assert len(graph) == 37
    assert {int(nid) for nid in graph} == set(range(1, 38))


# ---------------------------------------------------- (b) shared base 1..26 ---
def test_shared_base_nodes_1_to_26_match_i2v_toon_except_three_patched_nodes():
    """Every node id 1..26 must be byte-identical to the same id in
    `ltx23_i2v_toon.json` EXCEPT the three deliberately patched nodes: SIGMAS
    (16, hires 9-step schedule), DECODE (21, rewired to the refine pass's
    output) and FILENAME_PREFIX (23, hires-specific output path)."""
    i2v = _i2v()
    i2v_hires = _i2v_hires()
    t2v_hires = _t2v_hires()

    for nid in (str(n) for n in range(1, 27)):
        diff = _diff_paths(i2v[nid], i2v_hires[nid])
        if nid == "16":
            assert diff == {"inputs.sigmas"}, f"node 16: unexpected diff {diff}"
            assert (
                i2v_hires["16"]["inputs"]["sigmas"] == t2v_hires["16"]["inputs"]["sigmas"]
            ), "node 16 sigmas must be byte-identical to the t2v-hires 9-step schedule"
        elif nid == "21":
            assert diff == {"inputs.samples"}, f"node 21: unexpected diff {diff}"
            assert i2v_hires["21"]["inputs"]["samples"] == ["36", 0]
        elif nid == "23":
            assert diff == {"inputs.filename_prefix"}, f"node 23: unexpected diff {diff}"
            assert i2v_hires["23"]["inputs"]["filename_prefix"] == "kidsong/ltx_i2v_hires"
        else:
            assert diff == set(), (
                f"node {nid}: unexpected drift from ltx23_i2v_toon.json: {diff}"
            )


# -------------------------------------------------- (c) copied refine 27..36 ---
def test_refine_nodes_27_to_36_are_t2v_hires_24_to_33_with_links_remapped():
    """The refine subgraph (upscale model, upsampler, crop-guides, refine
    guider/seed/sampler-select/sigmas/concat/sampler/separate) is the
    t2v-hires graph's nodes 24..33 copied verbatim at ids 27..36, with every
    internal link that points at another refine node shifted by the same
    +3 offset applied to the node ids themselves.

    The ONE deliberate wiring change: CONCAT_AV_REFINE (34) reads its video
    latent from the image re-injection node 37 instead of the upsampler
    directly (see the module docstring for the measured black-frame failure
    this prevents)."""
    i2v_hires = _i2v_hires()
    t2v_hires = _t2v_hires()

    for new_id in range(27, 37):
        old_id = new_id - _ID_OFFSET
        expected = _remap_links(t2v_hires[str(old_id)])
        actual = i2v_hires[str(new_id)]
        if new_id == 34:
            diff = _diff_paths(actual, expected)
            assert diff == {"inputs.video_latent"}, (
                f"node 34: only video_latent may differ from the t2v-hires "
                f"copy, got diff {diff}"
            )
            assert actual["inputs"]["video_latent"] == ["37", 0]
            continue
        assert actual == expected, (
            f"node {new_id}: expected t2v-hires node {old_id} (link-remapped), "
            f"got a drifted copy: {actual!r} != {expected!r}"
        )


# ---------------------------------------------------- (d) refine schedule -----
def test_sigmas_refine_schedule_is_byte_identical_to_t2v_hires():
    i2v_hires = _i2v_hires()
    t2v_hires = _t2v_hires()
    assert i2v_hires["33"]["_meta"]["title"] == "SIGMAS_REFINE"
    assert t2v_hires["30"]["_meta"]["title"] == "SIGMAS_REFINE"
    assert i2v_hires["33"]["inputs"]["sigmas"] == t2v_hires["30"]["inputs"]["sigmas"]
    assert i2v_hires["33"]["inputs"]["sigmas"] == "0.85, 0.7250, 0.4219, 0.0"


def test_concat_av_refine_reads_the_reinjected_video_and_first_pass_audio():
    node = _i2v_hires()["34"]
    assert node["_meta"]["title"] == "CONCAT_AV_REFINE"
    assert node["inputs"]["video_latent"] == ["37", 0]
    assert node["inputs"]["audio_latent"] == ["20", 1]


def test_img_to_video_refine_reinjects_the_preprocessed_image_at_2x():
    """Node 37 must mirror the base graph's image conditioning (same
    class_type, same preprocessed image, full strength, not bypassed) but
    applied to the UPSAMPLED latent -- the refine-stage anchor whose absence
    rendered black frames."""
    graph = _i2v_hires()
    node = graph["37"]
    base = graph["26"]  # the base pass's LTXVImgToVideoInplace
    assert node["_meta"]["title"] == "IMG_TO_VIDEO_REFINE"
    assert node["class_type"] == "LTXVImgToVideoInplace"
    assert node["class_type"] == base["class_type"]
    assert node["inputs"]["image"] == base["inputs"]["image"] == ["25", 0]
    assert node["inputs"]["vae"] == base["inputs"]["vae"] == ["1", 2]
    assert node["inputs"]["latent"] == ["28", 0]  # the upsampled latent
    assert node["inputs"]["strength"] == base["inputs"]["strength"]
    assert node["inputs"]["bypass"] is False
