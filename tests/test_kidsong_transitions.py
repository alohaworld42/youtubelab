"""Tests for kidsong.transitions — generated shot-to-shot transitions (FLF2V)
and free shot chaining (i2v conditioned on the previous shot's last frame).

CPU-only, no GPU, no network, no ComfyUI, no ffmpeg: every test that would
otherwise touch a real video file monkeypatches `extract_frame`, and every
render goes through a fake in-process client that never makes an HTTP call.
"""
import pytest

from pipeline.kidsong import transitions


# ------------------------------------------------------------------ options ---
def test_options_returns_defaults_when_cfg_has_no_kidsong_section():
    opts = transitions.options({})
    assert opts == transitions.DEFAULTS
    # Must be a fresh copy, not the module dict itself.
    assert opts is not transitions.DEFAULTS


def test_options_merges_overrides_and_ignores_unrelated_keys():
    cfg = {
        "kidsong": {
            "transitions_enabled": True,
            "transitions_max_per_episode": 2,
            "some_unrelated_kidsong_key": "ignored",
        },
        "video": {"width": 1080},
    }
    opts = transitions.options(cfg)
    assert opts["transitions_enabled"] is True
    assert opts["transitions_max_per_episode"] == 2
    # Untouched defaults survive the merge.
    assert opts["transitions_at"] == "verse"
    assert opts["shot_chaining_enabled"] is False
    assert "some_unrelated_kidsong_key" not in opts


def test_options_tolerates_missing_or_non_dict_cfg():
    assert transitions.options(None) == transitions.DEFAULTS
    assert transitions.options({"kidsong": None}) == transitions.DEFAULTS


# --------------------------------------------------------------- snap_frames ---
@pytest.mark.parametrize("n, expected", [
    (0, 9),
    (1, 9),
    (9, 9),
    (25, 25),
    (26, 25),
    (97, 97),
])
def test_snap_frames_edge_cases(n, expected):
    assert transitions.snap_frames(n) == expected


def test_snap_frames_always_returns_valid_ltx_length():
    for n in range(0, 130):
        snapped = transitions.snap_frames(n)
        assert snapped >= 9
        assert (snapped - 1) % 8 == 0


# ------------------------------------------------------------------- fixture ---
def _shots():
    return [
        {"id": "s0", "verse": 0},
        {"id": "s1", "verse": 0},
        {"id": "s2", "verse": 1},
        {"id": "s3", "verse": 1},
        {"id": "s4", "verse": 2},
    ]


def _cut_list():
    return [
        {"shot_id": "s0", "src": "s0", "start": 0.0, "end": 1.0},
        {"shot_id": "s1", "src": "s1", "start": 1.0, "end": 2.0},
        {"shot_id": "s2", "src": "s2", "start": 2.0, "end": 3.0},
        {"shot_id": "s3", "src": "s3", "start": 3.0, "end": 4.0},
        {"shot_id": "s4", "src": "s4", "start": 4.0, "end": 5.0},
    ]


_ENABLED_CFG = {"kidsong": {"transitions_enabled": True, "transitions_at": "verse"}}


# ---------------------------------------------------------- plan_transitions ---
def test_plan_transitions_empty_when_disabled():
    cfg = {"kidsong": {"transitions_enabled": False}}
    plan = transitions.plan_transitions(_cut_list(), {"shots": _shots()}, cfg)
    assert plan == []


def test_plan_transitions_empty_when_transitions_at_none():
    cfg = {"kidsong": {"transitions_enabled": True, "transitions_at": "none"}}
    plan = transitions.plan_transitions(_cut_list(), {"shots": _shots()}, cfg)
    assert plan == []


def test_plan_transitions_finds_exactly_the_verse_boundaries():
    plan = transitions.plan_transitions(_cut_list(), {"shots": _shots()}, _ENABLED_CFG)
    # verse changes at cut index 2 (s1 verse0 -> s2 verse1) and index 4
    # (s3 verse1 -> s4 verse2); indices 1 and 3 stay within the same verse.
    assert [c["index"] for c in plan] == [2, 4]
    first = plan[0]
    assert first["prev_shot_id"] == "s1"
    assert first["next_shot_id"] == "s2"
    assert first["prev_src"] == "s1"
    assert first["next_src"] == "s2"
    assert first["frames"] == transitions.snap_frames(transitions.DEFAULTS["transition_frames"])


def test_plan_transitions_accepts_bare_list_shotlist():
    plan_dict_form = transitions.plan_transitions(_cut_list(), {"shots": _shots()}, _ENABLED_CFG)
    plan_bare_form = transitions.plan_transitions(_cut_list(), _shots(), _ENABLED_CFG)
    assert plan_dict_form == plan_bare_form


def test_plan_transitions_skips_unknown_shot_ids_without_crashing():
    shots = _shots()
    cuts = _cut_list()
    # Point cut index 1 at a shot id absent from the shotlist; that boundary
    # (unknown verse) must not be treated as a transition, and nothing raises.
    cuts[1] = {"shot_id": "sUNKNOWN", "src": "sUNKNOWN", "start": 1.0, "end": 2.0}
    plan = transitions.plan_transitions(cuts, {"shots": shots}, _ENABLED_CFG)
    # index 1 (s0 -> sUNKNOWN) and index 2 (sUNKNOWN -> s2) are both skipped
    # since one side's verse is unknown; index 4 (s3 -> s4) still fires.
    assert [c["index"] for c in plan] == [4]


def test_plan_transitions_respects_max_per_episode_and_spreads():
    # Six shots, each its own verse -> five candidate boundaries (index 1..5).
    shots = [{"id": f"s{i}", "verse": i} for i in range(6)]
    cuts = [
        {"shot_id": f"s{i}", "src": f"s{i}", "start": float(i), "end": float(i + 1)}
        for i in range(6)
    ]
    cfg = {
        "kidsong": {
            "transitions_enabled": True,
            "transitions_at": "verse",
            "transitions_max_per_episode": 3,
        }
    }
    plan = transitions.plan_transitions(cuts, {"shots": shots}, cfg)
    assert len(plan) == 3
    indices = [c["index"] for c in plan]
    assert len(set(indices)) == 3
    # Spread, not just "first 3": the earliest (1) and latest (5) candidate
    # boundaries are both kept rather than truncated off the end.
    assert indices[0] == 1
    assert indices[-1] == 5


def test_plan_transitions_max_zero_returns_empty():
    cfg = {
        "kidsong": {
            "transitions_enabled": True,
            "transitions_at": "verse",
            "transitions_max_per_episode": 0,
        }
    }
    plan = transitions.plan_transitions(_cut_list(), {"shots": _shots()}, cfg)
    assert plan == []


def test_plan_transitions_empty_cut_list_or_shotlist():
    assert transitions.plan_transitions([], {"shots": _shots()}, _ENABLED_CFG) == []
    assert transitions.plan_transitions(_cut_list(), {"shots": []}, _ENABLED_CFG) == []


# --------------------------------------------------------------- should_chain ---
_CHAIN_CFG = {"kidsong": {"shot_chaining_enabled": True, "shot_chain_max_len": 3}}


def test_should_chain_false_when_disabled():
    cfg = {"kidsong": {"shot_chaining_enabled": False}}
    prev = {"location": "backyard", "verse": 0}
    shot = {"location": "backyard", "verse": 0}
    assert transitions.should_chain(prev, shot, cfg, chain_len=0) is False


def test_should_chain_false_when_reuse_of_set():
    prev = {"location": "backyard"}
    shot = {"location": "backyard", "reuse_of": "s0"}
    assert transitions.should_chain(prev, shot, _CHAIN_CFG, chain_len=0) is False


def test_should_chain_false_when_no_prev_shot():
    shot = {"location": "backyard"}
    assert transitions.should_chain(None, shot, _CHAIN_CFG, chain_len=0) is False
    assert transitions.should_chain({}, shot, _CHAIN_CFG, chain_len=0) is False


def test_should_chain_false_at_chain_length_cap():
    prev = {"location": "backyard"}
    shot = {"location": "backyard"}
    # shot_chain_max_len is 3: chain_len 0,1,2 are still allowed to extend;
    # chain_len == max_len must not extend further.
    assert transitions.should_chain(prev, shot, _CHAIN_CFG, chain_len=2) is True
    assert transitions.should_chain(prev, shot, _CHAIN_CFG, chain_len=3) is False
    assert transitions.should_chain(prev, shot, _CHAIN_CFG, chain_len=4) is False


def test_should_chain_true_on_same_location_case_insensitive():
    prev = {"location": "Sunny Backyard"}
    shot = {"location": "sunny backyard"}
    assert transitions.should_chain(prev, shot, _CHAIN_CFG, chain_len=0) is True


def test_should_chain_false_on_location_change():
    prev = {"location": "backyard"}
    shot = {"location": "kitchen"}
    assert transitions.should_chain(prev, shot, _CHAIN_CFG, chain_len=0) is False


def test_should_chain_uses_scene_key_as_location_fallback():
    prev = {"scene": "Park"}
    shot = {"scene": "park"}
    assert transitions.should_chain(prev, shot, _CHAIN_CFG, chain_len=0) is True


def test_should_chain_one_sided_location_is_treated_as_a_change():
    prev = {"location": "backyard"}
    shot = {}  # no location/scene at all -> cannot verify sameness
    assert transitions.should_chain(prev, shot, _CHAIN_CFG, chain_len=0) is False
    prev2 = {}
    shot2 = {"location": "backyard"}
    assert transitions.should_chain(prev2, shot2, _CHAIN_CFG, chain_len=0) is False


def test_should_chain_falls_back_to_verse_when_no_location_anywhere():
    prev = {"verse": 1}
    shot = {"verse": 1}
    assert transitions.should_chain(prev, shot, _CHAIN_CFG, chain_len=0) is True

    shot_diff_verse = {"verse": 2}
    assert transitions.should_chain(prev, shot_diff_verse, _CHAIN_CFG, chain_len=0) is False


def test_should_chain_verse_fallback_false_when_verse_missing_on_both():
    prev = {}
    shot = {}
    assert transitions.should_chain(prev, shot, _CHAIN_CFG, chain_len=0) is False


# ---------------------------------------------------------------- extract_frame ---
def test_extract_frame_rejects_invalid_position():
    with pytest.raises(ValueError):
        transitions.extract_frame("video.mp4", "middle", "out.png")


# ------------------------------------------------------------- render_transition ---
class _FakeClient:
    """Records stage_input_image/render calls; never touches the network."""

    def __init__(self, fail_on_seed=None):
        self.staged = []
        self.render_calls = []
        self.fail_on_seed = fail_on_seed

    def stage_input_image(self, src_path):
        name = f"staged_{len(self.staged)}.png"
        self.staged.append((src_path, name))
        return name

    def render(self, workflow_name, patches, out_path):
        if self.fail_on_seed is not None and patches.get("SEED") == self.fail_on_seed:
            raise RuntimeError(f"simulated ComfyUI failure for seed {self.fail_on_seed}")
        self.render_calls.append((workflow_name, dict(patches), out_path))
        with open(out_path, "wb") as f:
            f.write(b"fake-transition-video")
        return out_path


def test_render_transition_builds_expected_patch_dict(tmp_path):
    client = _FakeClient()
    first_png = str(tmp_path / "first.png")
    last_png = str(tmp_path / "last.png")
    out_path = str(tmp_path / "out.mp4")
    cfg = {
        "kidsong": {
            # Shot size comes from kidsong.shot — the SAME key the t2v renders
            # read — so a transition is always the shape of the shots it
            # bridges. This used to read kidsong.width/kidsong.height, keys no
            # shipped config defines, so it silently rendered portrait.
            "shot": {"width": 640, "height": 384},  # both multiples of 32, as LTX requires
            "transition_frames": 33,
            "transition_guide_strength": 0.55,
        }
    }

    result = transitions.render_transition(
        client, first_png, last_png, "a transition prompt", 777, cfg, out_path,
    )

    assert result == out_path
    assert len(client.render_calls) == 1
    workflow_name, patches, patched_out = client.render_calls[0]
    assert workflow_name == "ltx23_flf2v_toon"
    assert patched_out == out_path
    assert patches["PROMPT"] == "a transition prompt"
    assert patches["SEED"] == 777
    assert patches["WIDTH"] == 640
    assert patches["HEIGHT"] == 384
    assert patches["FRAMES"] == transitions.snap_frames(33)
    assert patches["FIRST_IMAGE"] == client.staged[0][1]
    assert patches["LAST_IMAGE"] == client.staged[1][1]
    assert client.staged[0][0] == first_png
    assert client.staged[1][0] == last_png
    assert patches["GUIDE_FIRST"] == 0.55
    assert patches["GUIDE_LAST"] == 0.55
    assert patches["FILENAME_PREFIX"] == "kidsong_transition"
    assert "NEGATIVE" not in patches


def test_render_transition_default_width_height(tmp_path):
    """The fallback is LANDSCAPE. It used to be a hardcoded portrait 512x896,
    which is the format the 16:9 cut then cropped to pieces — a config with no
    explicit size must never bring that back."""
    client = _FakeClient()
    out_path = str(tmp_path / "out.mp4")
    transitions.render_transition(
        client, str(tmp_path / "a.png"), str(tmp_path / "b.png"),
        "prompt", 1, {}, out_path,
    )
    _, patches, _ = client.render_calls[0]
    assert patches["WIDTH"] == 896
    assert patches["HEIGHT"] == 512
    assert patches["WIDTH"] > patches["HEIGHT"]


def test_render_transition_includes_negative_when_configured(tmp_path):
    client = _FakeClient()
    out_path = str(tmp_path / "out.mp4")
    cfg = {"kidsong": {"negative": "blurry, deformed"}}
    transitions.render_transition(
        client, str(tmp_path / "a.png"), str(tmp_path / "b.png"),
        "prompt", 1, cfg, out_path,
    )
    _, patches, _ = client.render_calls[0]
    assert patches["NEGATIVE"] == "blurry, deformed"


# ------------------------------------------------------------------- sigmas ---
def _render_and_get_patches(tmp_path, client, cfg):
    out_path = str(tmp_path / "out.mp4")
    transitions.render_transition(
        client, str(tmp_path / "a.png"), str(tmp_path / "b.png"),
        "prompt", 1, cfg, out_path,
    )
    _, patches, _ = client.render_calls[0]
    return patches


def test_render_transition_defaults_to_the_official_nine_step_schedule(tmp_path):
    """FLF2V must NOT inherit the channel's 3-step distilled schedule: at 3
    steps the guides land but the interior of the clip collapses into
    unrelated content (measured — see the module docstring)."""
    patches = _render_and_get_patches(tmp_path, _FakeClient(), {})

    assert patches["SIGMAS"] == {"sigmas": transitions.SIGMAS_OFFICIAL_9STEP}
    assert patches["SIGMAS"]["sigmas"] != transitions.SIGMAS_DISTILLED_3STEP
    # Patched in dict form: "SIGMAS" has no scalar _PRIMARY_INPUT mapping.
    assert isinstance(patches["SIGMAS"], dict)


def test_render_transition_sigmas_override(tmp_path):
    cfg = {"kidsong": {"transition_sigmas": "1.0, 0.5, 0.0"}}
    patches = _render_and_get_patches(tmp_path, _FakeClient(), cfg)
    assert patches["SIGMAS"] == {"sigmas": "1.0, 0.5, 0.0"}


def test_render_transition_sigmas_none_leaves_workflow_schedule_alone(tmp_path):
    cfg = {"kidsong": {"transition_sigmas": None}}
    patches = _render_and_get_patches(tmp_path, _FakeClient(), cfg)
    assert "SIGMAS" not in patches


def test_official_nine_step_schedule_starts_at_one_and_ends_at_zero():
    steps = [float(s) for s in transitions.SIGMAS_OFFICIAL_9STEP.split(",")]
    assert steps[0] == 1.0
    assert steps[-1] == 0.0
    # Monotonically decreasing, and genuinely more steps than the distilled one.
    assert steps == sorted(steps, reverse=True)
    assert len(steps) > len(transitions.SIGMAS_DISTILLED_3STEP.split(","))


# ------------------------------------------------------------ generate_transitions ---
def _fake_extract_frame(monkeypatch):
    def _fake(video_path, position, out_png):
        with open(out_png, "wb") as f:
            f.write(b"fake-frame")
        return out_png

    monkeypatch.setattr(transitions, "extract_frame", _fake)


def test_generate_transitions_empty_plan_returns_empty_dict():
    assert transitions.generate_transitions([], {}, {}) == {}


def test_generate_transitions_skips_failed_index_without_raising(tmp_path, monkeypatch):
    _fake_extract_frame(monkeypatch)

    plan = [
        {"index": 1, "prev_shot_id": "s0", "next_shot_id": "s1",
         "prev_src": "s0", "next_src": "s1", "frames": 25},
        {"index": 3, "prev_shot_id": "s2", "next_shot_id": "s3",
         "prev_src": "s2", "next_src": "s3", "frames": 25},
        {"index": 5, "prev_shot_id": "s4", "next_shot_id": "s5",
         "prev_src": "s4", "next_src": "s5", "frames": 25},
    ]
    renders = {
        sid: str(tmp_path / f"{sid}.mp4")
        for sid in ("s0", "s1", "s2", "s3", "s4", "s5")
    }
    cfg = {"kidsong": {"seed": 100}}
    # seed = seed_base(100) + index; fail exactly the index-3 transition
    # (seed 103) so we can assert it — and only it — is dropped.
    client = _FakeClient(fail_on_seed=103)
    messages = []

    result = transitions.generate_transitions(
        plan, renders, cfg, client=client,
        workdir=str(tmp_path / "work"), on_progress=messages.append,
    )

    assert set(result.keys()) == {1, 5}
    assert 3 not in result
    assert any("3" in m and "fail" in m.lower() for m in messages)


def test_generate_transitions_all_fail_returns_empty_dict(tmp_path, monkeypatch):
    _fake_extract_frame(monkeypatch)

    class _AlwaysFailClient:
        def stage_input_image(self, src_path):
            return "staged.png"

        def render(self, workflow_name, patches, out_path):
            raise RuntimeError("ComfyUI is on fire")

    plan = [
        {"index": 0, "prev_shot_id": "s0", "next_shot_id": "s1",
         "prev_src": "s0", "next_src": "s1", "frames": 25},
        {"index": 2, "prev_shot_id": "s1", "next_shot_id": "s2",
         "prev_src": "s1", "next_src": "s2", "frames": 25},
    ]
    renders = {sid: str(tmp_path / f"{sid}.mp4") for sid in ("s0", "s1", "s2")}

    result = transitions.generate_transitions(
        plan, renders, {}, client=_AlwaysFailClient(), workdir=str(tmp_path / "work"),
    )
    assert result == {}


def test_generate_transitions_missing_render_is_a_soft_failure(tmp_path, monkeypatch):
    _fake_extract_frame(monkeypatch)
    plan = [
        {"index": 0, "prev_shot_id": "s0", "next_shot_id": "s1",
         "prev_src": "s0", "next_src": "s1", "frames": 25},
    ]
    # s1's render is missing entirely.
    renders = {"s0": str(tmp_path / "s0.mp4")}
    result = transitions.generate_transitions(
        plan, renders, {}, client=_FakeClient(), workdir=str(tmp_path / "work"),
    )
    assert result == {}


# --------------------------------------------------------------- chain_guide_image ---
def test_chain_guide_image_extracts_last_frame(tmp_path, monkeypatch):
    calls = []

    def _fake(video_path, position, out_png):
        calls.append((video_path, position))
        with open(out_png, "wb") as f:
            f.write(b"fake-frame")
        return out_png

    monkeypatch.setattr(transitions, "extract_frame", _fake)

    prev_render = str(tmp_path / "scene_00.mp4")
    workdir = str(tmp_path / "chain_work")
    out_png = transitions.chain_guide_image(prev_render, workdir)

    assert out_png.endswith("scene_00_chainguide.png")
    assert calls == [(prev_render, "last")]
    import os
    assert os.path.exists(out_png)
