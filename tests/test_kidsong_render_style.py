"""Tests for the render-style registry (Phase 1).

The kidsong pipeline used to hardcode a single visual look in two places:
the workflow name (generate.py picked "ltx23_t2v_toon[_hires]" with a literal)
and the shot prompt (a fixed trigger + "3D CGI toon" skeleton + identity clamp).
That look is now one named entry (`pixar_toon`) in a config registry
(`kidsong.render_styles`), resolved by `pipeline.kidsong.render_style`, so a
channel can select a different look — e.g. `flat_storybook` — with no code change.

The load-bearing guarantee is REGRESSION SAFETY: the `pixar_toon` path must be
byte-identical to the pre-registry pipeline — same workflow name chosen, same
`_shot_prompt` output, same patches. These baselines are the exact strings the
pre-change implementation produced (also pinned in tests/test_kidsong_shot_prompt.py,
captured from the real cast bible before this change).

CPU-only: no GPU, no network, no ffmpeg, no real ComfyUI. `load_workflow` reads
the real repo workflow JSON off local disk (same trick as
tests/test_kidsong_negative_wiring.py).
"""
import json
import os
import struct
import wave

import pytest

from pipeline.kidsong import runstate
from pipeline.kidsong.comfy import ComfyClient as _RealComfyClient
from pipeline.kidsong.generate import _shot_prompt
from pipeline.kidsong.render_style import (
    DEFAULT_STYLE_NAME,
    _PIXAR_TOON_DEFAULTS,
    default_style,
    i2v_workflow_name,
    resolve_style,
    workflow_name,
)

_REAL_LOAD_WORKFLOW = _RealComfyClient.load_workflow
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

_IDENTITY_CLAUSE = (
    "Every child on screen is one of these Black toddlers — no other children appear."
)


# --------------------------------------------------------------- sample data ---
def _song():
    return {
        "characters": (
            "Three adorable Black toddlers: Zuri, a girl with afro puffs and a yellow "
            "shirt; Kofi, a boy with curly hair and a blue shirt; Nala, a girl with "
            "braids and a yellow dress."
        )
    }


def _kofi_closeup():
    return {
        "id": "s01", "shot_type": "closeup", "characters": ["Kofi"],
        "action": "The child brushes their teeth with a big smile",
        "setting": "a bright cheerful bathroom", "camera": "static",
    }


def _all_wide():
    return {
        "id": "s00", "shot_type": "wide", "characters": ["all"],
        "action": "The kids clap their hands together",
        "setting": "a sunny backyard", "camera": "static",
    }


def _registry():
    """A self-contained two-style registry (does not depend on config.json,
    which the user edits) mirroring the shipped defaults."""
    return {
        "pixar_toon": dict(_PIXAR_TOON_DEFAULTS),
        "flat_storybook": {
            "workflow_base": "ltx23_t2v_toon",
            "style_lora": "ltx23_pixar_toon.safetensors",
            "style_strength": 0.0,
            "style_trigger": "",
            "prompt_skeleton": (
                "a soft flat 2D storybook illustration with bold simple shapes, "
                "flat matte colors and gentle rounded outlines, in a warm "
                "picture-book style."
            ),
            "identity_clause": _IDENTITY_CLAUSE,
        },
    }


def _cfg(render_style="pixar_toon", hires_pass=True):
    return {
        "kidsong": {
            "render_style": render_style,
            "render_styles": _registry(),
            "shot": {"hires_pass": hires_pass},
        }
    }


# ==================================================== pixar_toon REGRESSION ===
# NOTE: these baselines changed once, on 2026-07-31, and only here: the cast
# bible's `build` values were adjective phrases spliced after "with", so two of
# three characters described themselves in broken English ("a 4-year-old boy
# with noticeably tall and sturdy for a preschooler, with slightly broader
# shoulders"). QM-029/IMP-036 made them noun phrases. No descriptor was dropped
# and no prompt-composition code moved — if a future diff touches these strings
# for any other reason, that is a regression, not a rebaseline.
# The exact pre-registry _shot_prompt output for the sample shots, captured
# from the real cast bible before this change (and pinned identically in
# tests/test_kidsong_shot_prompt.py). Byte-for-byte equality here is the
# regression guarantee: pixar_toon must render precisely as the old code did.
_PIXAR_CLOSEUP = (
    "P1x4r, a high-quality 3D CGI toon animation in a preschool show "
    "style. A closeup shot of exactly one child: Kofi, a 4-year-old boy "
    "with a tall, sturdy build for a preschooler and slightly broader "
    "shoulders, medium-deep brown skin, short dark natural curly hair, "
    "wearing a blue t-shirt, grey jogger shorts and red sneakers. The "
    "child brushes their teeth with a big smile, in a bright cheerful "
    "bathroom. No other children are visible in frame. Every child on "
    "screen is one of these Black toddlers — no other children appear. The "
    "camera holds steady at the children's eye level. The children have "
    "soft rounded shapes and big expressive eyes and face the camera as "
    "they move. A warm key light from one side with a soft cool rim light "
    "along hair and shoulders, gentle soft-edged shadows. The set is "
    "lovingly dressed with two or three simple props that fit the "
    "location, kept clear around the children — bold happy colors, a "
    "joyful everyday toddler moment."
)

_PIXAR_WIDE = (
    "P1x4r, a high-quality 3D CGI toon animation in a preschool show "
    "style. Exactly three children are on screen: Zuri, a 3-year-old girl "
    "with an average toddler height and a soft, rounded little build, deep "
    "warm brown skin, round afro puffs tied with soft yellow fabric "
    "scrunchies, wearing a sunny yellow t-shirt, denim dungaree shorts and "
    "white sneakers; Kofi, a 4-year-old boy with a tall, sturdy build for "
    "a preschooler and slightly broader shoulders, medium-deep brown skin, "
    "short dark natural curly hair, wearing a blue t-shirt, grey jogger "
    "shorts and red sneakers; and Nala, a 3-year-old girl with a petite, "
    "delicate build, a little shorter and slighter, deep brown skin, neat "
    "cornrow braids, wearing a yellow pinafore dress over a white t-shirt, "
    "white tights and pink sneakers. A wide shot: The kids clap their "
    "hands together, in a sunny backyard. No other children appear "
    "anywhere in the frame, foreground or background. Every child on "
    "screen is one of these Black toddlers — no other children appear. The "
    "camera holds steady at the children's eye level. The children have "
    "soft rounded shapes and big expressive eyes and face the camera as "
    "they move. A warm key light from one side with a soft cool rim light "
    "along hair and shoulders, gentle soft-edged shadows. The set is "
    "lovingly dressed with two or three simple props that fit the "
    "location, kept clear around the children — bold happy colors, a "
    "joyful everyday toddler moment."
)


def test_pixar_toon_prompt_is_byte_identical_to_pre_registry_baseline():
    assert _shot_prompt(_kofi_closeup(), _song(), _cfg()) == _PIXAR_CLOSEUP
    assert _shot_prompt(_all_wide(), _song(), _cfg()) == _PIXAR_WIDE


def test_pixar_toon_prompt_identical_with_a_minimal_config():
    """A config carrying NO render_styles at all (the old shape) must resolve
    to the built-in pixar_toon defaults and produce the same prompt — so
    existing configs keep working unchanged."""
    minimal = {"kidsong": {"shot": {"style_trigger": "P1x4r"}}}
    assert _shot_prompt(_kofi_closeup(), _song(), minimal) == _PIXAR_CLOSEUP
    assert _shot_prompt(_all_wide(), _song(), minimal) == _PIXAR_WIDE


@pytest.mark.parametrize("hires,expected", [
    (True, "ltx23_t2v_toon_hires"),
    (False, "ltx23_t2v_toon"),
])
def test_pixar_toon_workflow_name_matches_pre_registry(hires, expected):
    assert workflow_name(_cfg(hires_pass=hires)) == expected


def test_pixar_toon_resolves_to_the_builtin_defaults():
    style = resolve_style(_cfg())
    assert style["name"] == "pixar_toon"
    for key, value in _PIXAR_TOON_DEFAULTS.items():
        assert style[key] == value


# ======================================================= flat_storybook LOOK ===
def test_flat_storybook_reuses_the_same_workflow_base():
    style = resolve_style(_cfg("flat_storybook"))
    assert style["workflow_base"] == "ltx23_t2v_toon"
    # Same base means the same graph files as pixar_toon — no new workflow file.
    assert workflow_name(_cfg("flat_storybook", hires_pass=True)) == "ltx23_t2v_toon_hires"
    assert workflow_name(_cfg("flat_storybook", hires_pass=False)) == "ltx23_t2v_toon"


def test_flat_storybook_neutralizes_the_lora_and_drops_the_trigger():
    style = resolve_style(_cfg("flat_storybook"))
    assert style["style_strength"] == 0.0
    assert style["style_trigger"] == ""
    # Same LoRA FILE as pixar_toon — nothing new to download, just neutralized.
    assert style["style_lora"] == "ltx23_pixar_toon.safetensors"


def test_flat_storybook_prompt_uses_the_flat_skeleton_not_the_toon_one():
    prompt = _shot_prompt(_kofi_closeup(), _song(), _cfg("flat_storybook"))
    assert "storybook illustration" in prompt
    assert "flat matte colors" in prompt
    # The 3D CGI toon skeleton must be gone…
    assert "3D CGI toon animation" not in prompt
    # …and with an empty trigger there is no "P1x4r, " prefix or stray comma.
    assert not prompt.startswith("P1x4r")
    assert not prompt.startswith(", ")
    assert prompt.startswith("a soft flat 2D storybook illustration")


def test_flat_storybook_keeps_the_identity_clause_and_head_count():
    prompt = _shot_prompt(_kofi_closeup(), _song(), _cfg("flat_storybook"))
    assert _IDENTITY_CLAUSE in prompt
    # The cast/head-count machinery is style-independent: still names only Kofi.
    assert "exactly one child: Kofi" in prompt
    assert "Zuri" not in prompt and "Nala" not in prompt


# ============================================================ selector logic ===
def test_workflow_name_hires_toggle():
    assert workflow_name(_cfg(hires_pass=True)) == "ltx23_t2v_toon_hires"
    assert workflow_name(_cfg(hires_pass=False)) == "ltx23_t2v_toon"


def test_hires_pass_defaults_to_true_when_unset():
    cfg = {"kidsong": {"render_styles": _registry(), "render_style": "pixar_toon"}}
    assert workflow_name(cfg) == "ltx23_t2v_toon_hires"


@pytest.mark.parametrize("hires,expected", [
    (True, "ltx23_i2v_toon_hires"),
    (False, "ltx23_i2v_toon"),
])
def test_i2v_workflow_name_hires_toggle(hires, expected):
    assert i2v_workflow_name(_cfg(hires_pass=hires)) == expected


def test_i2v_hires_pass_defaults_to_true_when_unset():
    cfg = {"kidsong": {"render_styles": _registry(), "render_style": "pixar_toon"}}
    assert i2v_workflow_name(cfg) == "ltx23_i2v_toon_hires"


@pytest.mark.parametrize("hires_pass", [True, False])
def test_i2v_and_t2v_workflow_names_agree_on_the_hires_toggle(hires_pass):
    """i2v_workflow_name mirrors workflow_name's hires decision: both must
    flip to their `_hires` graph together, whichever style is active -- the
    keyframe-first/reference-anchored path must get the same 2x refine the
    t2v path does, never silently fall behind it."""
    cfg = _cfg(hires_pass=hires_pass)
    assert i2v_workflow_name(cfg).endswith("_hires") == workflow_name(cfg).endswith("_hires")


def test_unknown_style_falls_back_to_pixar_toon_and_logs(caplog):
    cfg = _cfg("does_not_exist")
    with caplog.at_level("WARNING", logger="kidsong.render_style"):
        style = resolve_style(cfg)
    assert style["name"] == DEFAULT_STYLE_NAME
    for key, value in _PIXAR_TOON_DEFAULTS.items():
        assert style[key] == value
    assert any("does_not_exist" in r.message for r in caplog.records)
    assert any(r.levelname == "WARNING" for r in caplog.records)


def test_unknown_style_never_crashes_prompt_or_workflow_selection():
    cfg = _cfg("nope_not_here", hires_pass=True)
    # Falls back to pixar_toon: byte-identical prompt + workflow, no exception.
    assert _shot_prompt(_kofi_closeup(), _song(), cfg) == _PIXAR_CLOSEUP
    assert workflow_name(cfg) == "ltx23_t2v_toon_hires"


# ====================================================== resolver defaults ====
def test_resolver_fills_all_defaults_for_empty_config():
    style = resolve_style({})
    assert style["name"] == DEFAULT_STYLE_NAME
    for key, value in _PIXAR_TOON_DEFAULTS.items():
        assert style[key] == value


def test_resolver_fills_missing_keys_from_pixar_defaults():
    """A partial style entry keeps its own keys and inherits the rest."""
    cfg = {"kidsong": {
        "render_style": "partial",
        "render_styles": {"partial": {"style_trigger": "ZZZ"}},
    }}
    style = resolve_style(cfg)
    assert style["name"] == "partial"
    assert style["style_trigger"] == "ZZZ"          # kept
    assert style["workflow_base"] == "ltx23_t2v_toon"  # inherited
    assert style["style_lora"] == _PIXAR_TOON_DEFAULTS["style_lora"]  # inherited
    assert style["style_strength"] == 1.0            # inherited
    assert style["prompt_skeleton"] == _PIXAR_TOON_DEFAULTS["prompt_skeleton"]
    assert style["identity_clause"] == _IDENTITY_CLAUSE


def test_resolver_preserves_meaningful_zero_and_empty_values():
    """style_strength 0.0 and style_trigger '' are real values, not 'unset' —
    they must NOT be overwritten by the (truthy) pixar defaults."""
    cfg = {"kidsong": {
        "render_style": "x",
        "render_styles": {"x": {"style_strength": 0.0, "style_trigger": ""}},
    }}
    style = resolve_style(cfg)
    assert style["style_strength"] == 0.0
    assert style["style_trigger"] == ""


def test_default_style_helper_returns_a_fresh_copy():
    a = default_style()
    a["style_trigger"] = "MUTATED"
    assert default_style()["style_trigger"] == "P1x4r"
    assert _PIXAR_TOON_DEFAULTS["style_trigger"] == "P1x4r"


# ================================================= config.json round-trips ====
def test_shipped_config_registers_both_styles():
    from pipeline.config import load_config

    cfg = load_config()
    styles = cfg["kidsong"]["render_styles"]
    assert set(styles) >= {"pixar_toon", "flat_storybook"}
    # The shipped default must be one of the registered styles (which one is a
    # channel/brand decision — claymation since 2026-07-21, QM-011).
    assert cfg["kidsong"]["render_style"] in styles


def test_shipped_pixar_toon_matches_the_builtin_defaults():
    """The pixar_toon entry the user ships must equal the code's fallback
    defaults, so config-driven and default-driven renders agree."""
    from pipeline.config import load_config

    cfg = load_config()
    cfg["kidsong"]["render_style"] = "pixar_toon"
    style = resolve_style(cfg)
    for key, value in _PIXAR_TOON_DEFAULTS.items():
        assert style[key] == value


def test_shipped_flat_storybook_is_download_free():
    from pipeline.config import load_config

    cfg = load_config()
    cfg["kidsong"]["render_style"] = "flat_storybook"
    style = resolve_style(cfg)
    assert style["workflow_base"] == "ltx23_t2v_toon"   # same graph
    assert style["style_lora"] == "ltx23_pixar_toon.safetensors"  # same file
    assert style["style_strength"] == 0.0               # neutralized


# ============================= LORA_STYLE patch dict lands on the graph node ===
def test_lora_style_dict_patch_writes_both_inputs_on_the_real_graph():
    """Verify against comfy.ComfyClient._apply_patches (contract item 3): the
    {'lora_name', 'strength_model'} dict form writes BOTH inputs on the
    LoraLoaderModelOnly node titled LORA_STYLE. Reads the real repo workflow
    off local disk — no network, no GPU."""
    client = _RealComfyClient({})
    workflow = client.load_workflow("ltx23_t2v_toon")
    client._apply_patches(
        workflow, {"LORA_STYLE": {"lora_name": "some_other.safetensors", "strength_model": 0.0}}
    )
    lora_nodes = [
        n for n in workflow.values() if n.get("_meta", {}).get("title") == "LORA_STYLE"
    ]
    assert len(lora_nodes) == 1
    inputs = lora_nodes[0]["inputs"]
    assert inputs["lora_name"] == "some_other.safetensors"
    assert inputs["strength_model"] == 0.0


# ============================ LORA_STYLE reaches the actual render() call ======
def _write_wav(path, seconds=2.0, rate=8000):
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(rate)
        wf.writeframes(struct.pack("<h", 0) * int(rate * seconds))


@pytest.fixture
def pending_shot_run(tmp_path, monkeypatch):
    """A director run with exactly ONE PENDING shot so the render loop actually
    runs; returns a `make(render_style)` factory that builds the cfg for a given
    active style. Modeled on tests/test_kidsong_negative_wiring.py."""
    out_dir = tmp_path / "output"
    out_dir.mkdir()
    base = "20260720-093000-kidsong-render-style-test"
    shots_dir = runstate.shots_dir_path(str(out_dir), base)
    os.makedirs(shots_dir, exist_ok=True)

    _write_wav(out_dir / f"{base}.wav")
    song = {
        "title": "Render Style Test Song", "description": "d", "tags": ["t"],
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

    def fake_assemble(cfg, cut_list, renders, voice, words, duration, out_path, **kw):
        with open(out_path, "wb") as f:
            f.write(b"\0" * 4096)

    def fake_review_cut(staging_path, cut_list, beats, words, duration, cfg, review_dir):
        return {"accept": True, "score": 1.0, "reasons": [], "retry_hints": {}}

    def fake_prepend_intro(video_path, cfg, on_progress=None):
        return {"applied": False, "intro_seconds": 0.0, "reason": "disabled for test"}

    monkeypatch.setattr(comfy_mod, "ComfyClient", FakeComfyClient)
    monkeypatch.setattr(review_mod, "get_reviewer", lambda cfg: FakeReviewer())
    monkeypatch.setattr(review_mod, "contact_sheet", lambda *a, **k: None)
    monkeypatch.setattr(review_mod, "review_shots_log", lambda path, entries: path)
    monkeypatch.setattr(captions_mod, "get_word_timestamps", lambda *a, **k: [])
    monkeypatch.setattr(sing_mod, "verse_times_from_words", lambda *a, **k: [(0.0, 2.0)])
    monkeypatch.setattr(edit_mod, "beat_grid", lambda path: [0.0, 1.0, 2.0])
    monkeypatch.setattr(edit_mod, "build_cut_list", lambda *a, **k: [])
    monkeypatch.setattr(edit_mod, "assemble", fake_assemble)
    monkeypatch.setattr(cut_qc_mod, "review_cut", fake_review_cut)
    monkeypatch.setattr(edit_mod, "prepend_intro", fake_prepend_intro)
    monkeypatch.setattr(
        sing_mod, "sing_song",
        lambda *a, **k: (_ for _ in ()).throw(AssertionError("resume must not re-sing")),
    )

    def make(render_style):
        cfg = {
            "_root": _REPO_ROOT,
            "paths": {"output_dir": str(out_dir)},
            "video": {"max_seconds": 60},
            "captions": {"uppercase": True},
            "kidsong": {
                "mode": "director",
                "singer": "ace",
                "seed": 20260717,
                "render_style": render_style,
                "render_styles": _registry(),
                "shot": {"fps": 24, "max_frames": 241, "hires_pass": False},
                "review": {"script": False, "cut": True, "max_retries_per_shot": 0},
                "intro": {"enabled": False},
            },
        }
        return cfg

    return {"make": make, "base": base, "render_calls": render_calls}


def test_render_patch_carries_pixar_toon_lora_file_and_full_strength(pending_shot_run):
    from pipeline.kidsong.generate import generate_kidsong

    run = pending_shot_run
    result = generate_kidsong(cfg=run["make"]("pixar_toon"), resume_base=run["base"])
    assert result["status"] == "approved"
    assert len(run["render_calls"]) == 1
    workflow, patches = run["render_calls"][0]

    assert workflow == "ltx23_t2v_toon"  # non-hires in this fixture
    assert "LORA_STYLE" in patches
    assert patches["LORA_STYLE"] == {
        "lora_name": "ltx23_pixar_toon.safetensors",
        "strength_model": 1.0,
    }
    # And the prompt carried the Pixar trigger + skeleton.
    assert patches["PROMPT"]["text"].startswith(
        "P1x4r, a high-quality 3D CGI toon animation"
    )


def test_render_patch_carries_flat_storybook_zero_strength(pending_shot_run):
    from pipeline.kidsong.generate import generate_kidsong

    run = pending_shot_run
    generate_kidsong(cfg=run["make"]("flat_storybook"), resume_base=run["base"])
    assert len(run["render_calls"]) == 1
    workflow, patches = run["render_calls"][0]

    # Same graph file as pixar_toon — the look changed with zero new workflows.
    assert workflow == "ltx23_t2v_toon"
    assert patches["LORA_STYLE"] == {
        "lora_name": "ltx23_pixar_toon.safetensors",
        "strength_model": 0.0,
    }
    # The flat skeleton reached the render, and no P1x4r trigger.
    prompt = patches["PROMPT"]["text"]
    assert prompt.startswith("a soft flat 2D storybook illustration")
    assert "3D CGI toon animation" not in prompt
