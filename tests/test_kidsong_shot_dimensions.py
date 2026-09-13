"""The kidsong shot frame is LANDSCAPE, and `kidsong.shot` actually governs it.

The defect this file locks down: `config.json` declared
`kidsong.shot = {width: 896, height: 512}` (landscape) while all four
`workflows/*.json` carried a hardcoded `WIDTH=512, HEIGHT=896` (the old
vertical Shorts format) and `generate.py` NEVER patched WIDTH or HEIGHT. The
config keys were therefore dead and the portrait graph values won: real takes
on disk are 1024x1792 while `output/Final/*.mp4` is 1920x1080, so
`edit.assemble`'s cover-scale threw away ~69% of the picture height and cut
heads and feet off every shot.

`kidsong.shot.style_lora` / `.style_strength` are the same anti-pattern still
sitting in config unread, which is exactly why the central test here asserts a
NON-DEFAULT configured size reaches the render call: a `_shot_dims` that
ignored config and returned a hardcoded 896x512 would satisfy every "is it
landscape" assertion while leaving the keys just as dead as before.

CPU-only: no GPU, no network, no real ComfyUI. The workflow assertions read
the real repo JSON off local disk.
"""
import json
import os
import struct
import wave

import pytest

from pipeline.kidsong import runstate
from pipeline.kidsong.comfy import ComfyClient as _RealComfyClient
from pipeline.kidsong.generate import _shot_dims

_REAL_LOAD_WORKFLOW = _RealComfyClient.load_workflow
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_WORKFLOWS_DIR = os.path.join(_REPO_ROOT, "workflows")

# Every graph the kidsong pipeline can POST. The t2v pair renders shots; i2v
# (and its hires refine variant) and flf2v render conditioned shots and
# transitions between them -- a graph left portrait here would put a
# differently-shaped clip into the same cut.
_ALL_WORKFLOWS = [
    "ltx23_t2v_toon.json",
    "ltx23_t2v_toon_hires.json",
    "ltx23_i2v_toon.json",
    "ltx23_i2v_toon_hires.json",
    "ltx23_flf2v_toon.json",
]


def _write_wav(path, seconds=2.0, rate=8000):
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(rate)
        wf.writeframes(struct.pack("<h", 0) * int(rate * seconds))


def _titled(workflow, title):
    matches = [
        node for node in workflow.values()
        if node.get("_meta", {}).get("title") == title
    ]
    assert len(matches) == 1, f"expected exactly one {title} node, got {len(matches)}"
    return matches[0]


# --------------------------------------------------------------- _shot_dims ---
def test_shot_dims_reads_configured_values():
    """Not a constant: an unusual configured size must come straight back."""
    cfg = {"kidsong": {"shot": {"width": 1024, "height": 576}}}
    assert _shot_dims(cfg) == (1024, 576)


def test_shot_dims_defaults_are_landscape():
    """A config missing the keys entirely must NOT resurrect the portrait
    frame -- the failure mode that produced the cropped episodes."""
    for cfg in ({}, {"kidsong": {}}, {"kidsong": {"shot": {}}}):
        width, height = _shot_dims(cfg)
        assert (width, height) == (896, 512)
        assert width > height, f"default must be landscape, got {width}x{height}"


def test_shot_dims_snaps_to_multiple_of_32():
    """LTX's EmptyLTXVLatentVideo declares step=32 and builds its latent as
    `height // 32, width // 32`, so a non-multiple silently decodes at the
    wrong size. Round DOWN rather than raise: a config typo must not block a
    render."""
    assert _shot_dims({"kidsong": {"shot": {"width": 900, "height": 530}}}) == (896, 512)
    # Never below the node's own minimum of 64.
    assert _shot_dims({"kidsong": {"shot": {"width": 10, "height": 10}}}) == (64, 64)


def test_shipped_config_shot_size_is_landscape_and_valid():
    from pipeline.config import load_config

    width, height = _shot_dims(load_config())
    assert width > height, f"shipped kidsong.shot is not landscape: {width}x{height}"
    assert width % 32 == 0 and height % 32 == 0


def test_shipped_shot_aspect_is_close_to_the_output_aspect():
    """The whole point of the fix: the rendered frame and the delivered frame
    must be nearly the same shape, so the cut is a small uniform scale instead
    of a catastrophic crop."""
    from pipeline.config import load_config
    from pipeline.kidsong.edit import _output_dims

    cfg = load_config()
    shot_w, shot_h = _shot_dims(cfg)
    out_w, out_h = _output_dims(cfg)
    assert abs((shot_w / shot_h) - (out_w / out_h)) < 0.05


# ----------------------------------------------------------------- workflows ---
@pytest.mark.parametrize("filename", _ALL_WORKFLOWS)
def test_workflow_baked_defaults_are_landscape(filename):
    """An UNPATCHED render must also be right. These defaults are the values
    that actually shipped the broken episodes."""
    with open(os.path.join(_WORKFLOWS_DIR, filename), encoding="utf-8") as fh:
        workflow = json.load(fh)

    width = _titled(workflow, "WIDTH")["inputs"]["value"]
    height = _titled(workflow, "HEIGHT")["inputs"]["value"]

    assert (width, height) == (896, 512), (
        f"{filename} bakes in {width}x{height}; expected landscape 896x512"
    )
    assert width % 32 == 0 and height % 32 == 0


@pytest.mark.parametrize("filename", _ALL_WORKFLOWS)
def test_latent_video_node_is_driven_by_the_width_height_primitives(filename):
    """The WIDTH/HEIGHT primitives are only load-bearing if the latent node
    actually reads from them -- patching a node nothing consumes would be a
    silent no-op indistinguishable from the original bug."""
    with open(os.path.join(_WORKFLOWS_DIR, filename), encoding="utf-8") as fh:
        workflow = json.load(fh)

    width_id = next(
        k for k, v in workflow.items() if v.get("_meta", {}).get("title") == "WIDTH"
    )
    height_id = next(
        k for k, v in workflow.items() if v.get("_meta", {}).get("title") == "HEIGHT"
    )
    latent = _titled(workflow, "LATENT_VIDEO")
    assert latent["inputs"]["width"][0] == width_id
    assert latent["inputs"]["height"][0] == height_id


# ------------------------------------------------------- render-call wiring ---
@pytest.fixture
def pending_shot_run(tmp_path, monkeypatch):
    """A director run with exactly ONE PENDING shot so the render loop really
    executes, against a fake ComfyClient that captures the patch dict.

    Same fixture shape as tests/test_kidsong_negative_wiring.py -- see its
    docstring for why a resumed single-pending-shot run is the setup that
    actually reaches `client.render`.
    """
    out_dir = tmp_path / "output"
    out_dir.mkdir()
    base = "20260720-093000-kidsong-shot-dimensions-test"
    os.makedirs(runstate.shots_dir_path(str(out_dir), base), exist_ok=True)

    _write_wav(out_dir / f"{base}.wav")
    song = {
        "title": "Shot Dimensions Test Song",
        "description": "d",
        "tags": ["t"],
        "characters": "Three toddlers: Zuri; Kofi; Nala",
        "verses": [{"scene": "bathroom", "lines": ["a", "b"]}],
    }
    (out_dir / f"{base}-song.json").write_text(json.dumps(song), encoding="utf-8")

    shot = {
        "id": "s00", "verse": 0, "start": 0.0, "end": 2.0,
        "shot_type": "closeup", "characters": ["Kofi"],
        "action": "Kofi brushes his teeth with a big smile",
        "setting": "a bright cheerful bathroom", "camera": "static",
        "reuse_of": None, "seed": 1000, "status": "planned",
    }
    runstate.save_shotlist(str(out_dir), base, {"shots": [shot]})

    render_calls = []

    import pipeline.captions as captions_mod
    import pipeline.kidsong.comfy as comfy_mod
    import pipeline.kidsong.cut_qc as cut_qc_mod
    import pipeline.kidsong.edit as edit_mod
    import pipeline.kidsong.review as review_mod
    import pipeline.kidsong.sing as sing_mod

    class FakeComfyClient:
        def __init__(self, cfg):
            self.url = "http://fake:8188"
            self._proc = None

        def load_workflow(self, workflow_name):
            return _REAL_LOAD_WORKFLOW(self, workflow_name)

        def ensure_up(self):
            pass

        def render(self, workflow, patches, out_path):
            render_calls.append((workflow, patches))
            with open(out_path, "wb") as f:
                f.write(b"\0" * 1024)

        def free(self):
            pass

    class FakeReviewer:
        def review(self, shot, video_path, out_dir=None):
            return {"accept": True, "score": 1.0, "reasons": [], "retry_hints": {}}

    monkeypatch.setattr(comfy_mod, "ComfyClient", FakeComfyClient)
    monkeypatch.setattr(review_mod, "get_reviewer", lambda cfg: FakeReviewer())
    monkeypatch.setattr(review_mod, "contact_sheet", lambda *a, **k: None)
    monkeypatch.setattr(review_mod, "review_shots_log", lambda path, entries: path)
    monkeypatch.setattr(captions_mod, "get_word_timestamps", lambda *a, **k: [])
    monkeypatch.setattr(sing_mod, "verse_times_from_words", lambda *a, **k: [(0.0, 2.0)])
    monkeypatch.setattr(edit_mod, "beat_grid", lambda path: [0.0, 1.0, 2.0])
    monkeypatch.setattr(edit_mod, "build_cut_list", lambda *a, **k: [])
    monkeypatch.setattr(
        edit_mod, "assemble",
        lambda cfg, cl, r, v, w, d, out_path, **kw: open(out_path, "wb").write(b"\0" * 4096),
    )
    monkeypatch.setattr(
        cut_qc_mod, "review_cut",
        lambda *a, **k: {"accept": True, "score": 1.0, "reasons": [], "retry_hints": {}},
    )
    monkeypatch.setattr(
        edit_mod, "prepend_intro",
        lambda video_path, cfg, on_progress=None: {
            "applied": False, "intro_seconds": 0.0, "reason": "disabled for test"},
    )
    monkeypatch.setattr(
        sing_mod, "sing_song",
        lambda *a, **k: (_ for _ in ()).throw(AssertionError("resume must not re-sing")),
    )

    def make_cfg(shot_extra):
        return {
            "_root": _REPO_ROOT,
            "paths": {"output_dir": str(out_dir)},
            "video": {"max_seconds": 60},
            "captions": {"uppercase": True},
            "kidsong": {
                "mode": "director",
                "singer": "ace",
                "seed": 20260717,
                "shot": {
                    "fps": 24, "max_frames": 241, "hires_pass": False,
                    "style_trigger": "P1x4r", **shot_extra,
                },
                "review": {"script": False, "cut": True, "max_retries_per_shot": 0},
                "intro": {"enabled": False},
            },
        }

    return {
        "make_cfg": make_cfg, "base": base,
        "out_dir": str(out_dir), "render_calls": render_calls,
    }


def test_configured_shot_size_reaches_the_render_patch_dict(pending_shot_run):
    """THE regression test. Deliberately configures 1024x576 -- neither the
    old portrait 512x896 nor the new 896x512 default -- so this fails both if
    WIDTH/HEIGHT stop being patched at all AND if they get hardcoded to the
    new default while `kidsong.shot` goes back to being a dead config key.
    """
    from pipeline.kidsong.generate import generate_kidsong

    run = pending_shot_run
    cfg = run["make_cfg"]({"width": 1024, "height": 576})
    result = generate_kidsong(cfg=cfg, resume_base=run["base"])

    assert result["status"] == "approved"
    assert len(run["render_calls"]) == 1
    _workflow, patches = run["render_calls"][0]

    assert "WIDTH" in patches, "WIDTH is never patched -- the graph's own value wins"
    assert "HEIGHT" in patches, "HEIGHT is never patched -- the graph's own value wins"
    assert patches["WIDTH"] == {"value": 1024}
    assert patches["HEIGHT"] == {"value": 576}


def test_patched_dimensions_actually_land_on_the_latent_node(pending_shot_run):
    """A patch dict is only worth anything if `_apply_patches` writes it into
    the graph the server receives -- close the loop rather than trusting the
    title mapping."""
    from pipeline.kidsong.generate import generate_kidsong

    run = pending_shot_run
    generate_kidsong(cfg=run["make_cfg"]({"width": 1024, "height": 576}),
                     resume_base=run["base"])
    workflow_name, patches = run["render_calls"][0]

    graph = _RealComfyClient.load_workflow(_RealComfyClient, workflow_name)
    _RealComfyClient._apply_patches(graph, patches)

    assert _titled(graph, "WIDTH")["inputs"]["value"] == 1024
    assert _titled(graph, "HEIGHT")["inputs"]["value"] == 576


def test_render_uses_landscape_when_config_omits_the_size(pending_shot_run):
    from pipeline.kidsong.generate import generate_kidsong

    run = pending_shot_run
    generate_kidsong(cfg=run["make_cfg"]({}), resume_base=run["base"])
    _workflow, patches = run["render_calls"][0]

    assert patches["WIDTH"]["value"] > patches["HEIGHT"]["value"]


# ---------------------------------------------------------------- transitions ---
def test_transitions_use_the_same_shot_dims_as_the_renders():
    """A transition is cut in among the shots it bridges, so a differently
    shaped one would be visibly rescaled. `render_transition` used to read
    kidsong.width/kidsong.height -- keys no shipped config defines -- and so
    always fell through to a hardcoded portrait 512x896."""
    from pipeline.kidsong import transitions

    captured = {}

    class FakeClient:
        def stage_input_image(self, path):
            return os.path.basename(path)

        def render(self, workflow_name, patches, out_path):
            captured.update(patches)
            return out_path

    cfg = {"kidsong": {"shot": {"width": 1024, "height": 576}}}
    transitions.render_transition(
        FakeClient(), "first.png", "last.png", "a prompt", 7, cfg, "out.mp4"
    )

    assert captured["WIDTH"] == 1024
    assert captured["HEIGHT"] == 576
