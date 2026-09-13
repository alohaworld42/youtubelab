"""kidsong.identity_check + the keyframe identity-retry loop in generate.

The klein reference-anchored keyframe is stochastic — most renders land the
right cast child, a minority drift into a castmate. `identity_check` scores a
keyframe against every cast reference (CLIP), and `_render_keyframe_on_model`
re-renders a flagged single-subject keyframe until the intended child wins,
keeping the best attempt. These tests pin the retry control flow and the
detector's best-effort contract WITHOUT loading CLIP or touching a GPU.
"""
import os

import pytest

from pipeline.kidsong import generate, identity_check


# ------------------------------------------------------- identity_check ------
def test_identity_margin_is_none_when_clip_is_unavailable(monkeypatch):
    monkeypatch.setattr(identity_check, "_load", lambda cfg=None: None)
    assert identity_check.identity_margin("whatever.png", "kofi", {}) is None


def test_identity_margin_is_own_minus_best_other(monkeypatch):
    # Fake "embeddings": 2-D unit-ish vectors keyed by path, so cosine is exact.
    vecs = {
        "img": (1.0, 0.0),
        "kofi_ref": (0.9, 0.1),   # closest to img
        "zuri_ref": (0.2, 0.9),
        "nala_ref": (0.0, 1.0),
    }

    class _T:  # a tiny stand-in tensor supporting @ and .item via a helper
        pass

    monkeypatch.setattr(identity_check, "_load", lambda cfg=None: ("torch", "model", "proc"))
    monkeypatch.setattr(identity_check, "_embed", lambda loaded, path: vecs[path])

    def fake_reference_for(cfg, cid):
        return {"kofi": "kofi_ref", "zuri": "zuri_ref", "nala": "nala_ref"}[cid]

    import pipeline.kidsong.refs as refs_mod
    import pipeline.kidsong.cast as cast_mod
    monkeypatch.setattr(refs_mod, "reference_for", fake_reference_for)
    monkeypatch.setattr(cast_mod, "load_bible",
                        lambda: {"characters": [{"id": "zuri"}, {"id": "kofi"}, {"id": "nala"}]})

    # Patch the cosine used inside identity_margin by giving _embed vectors and
    # overriding the local sim via monkeypatching os.path.exists + a real dot.
    monkeypatch.setattr(os.path, "exists", lambda p: True)

    # identity_margin builds sim from `img @ ref.T`; our _embed returns plain
    # tuples, so patch the module's arithmetic by supplying a dot through a
    # wrapper: redefine _embed to return objects implementing @.
    class V:
        def __init__(self, t): self.t = t
        def __matmul__(self, other):
            return V((sum(a * b for a, b in zip(self.t, other.t)),))
        @property
        def T(self): return self
        def item(self): return self.t[0]

    monkeypatch.setattr(identity_check, "_embed", lambda loaded, path: V(vecs[path]))

    m = identity_check.identity_margin("img", "kofi", {})
    # own (kofi) = 0.9 ; best other = max(zuri 0.2, nala 0.0) = 0.2 -> margin 0.7
    assert m == pytest.approx(0.9 - 0.2, abs=1e-6)


# ------------------------------------------------------- prop_presence -------
def test_prop_presence_is_none_when_clip_is_unavailable(monkeypatch):
    monkeypatch.setattr(identity_check, "_load", lambda cfg=None: None)
    assert identity_check.prop_presence("whatever.png", "an umbrella", {}) is None


def test_prop_presence_is_none_for_empty_prop_text_without_loading_clip(monkeypatch):
    calls = []
    monkeypatch.setattr(identity_check, "_load",
                        lambda cfg=None: calls.append(1) or ("torch", "model", "proc"))
    assert identity_check.prop_presence("whatever.png", "", {}) is None
    assert identity_check.prop_presence("whatever.png", None, {}) is None
    assert calls == []   # _load never touched — no CLIP work for an empty prop


def test_prop_presence_never_raises_on_a_nonexistent_image_path(monkeypatch):
    # _load "succeeds" with a bogus stand-in; the real _embed then fails trying
    # to open the missing file — prop_presence must swallow that, not raise.
    monkeypatch.setattr(identity_check, "_load", lambda cfg=None: ("torch", "model", "proc"))
    assert identity_check.prop_presence("does/not/exist.png", "an umbrella", {}) is None


def test_prop_presence_orders_with_closer_than_without(monkeypatch):
    # Tiny stand-in tensor supporting @ and .item(), same shape as the
    # identity_margin test above, so this runs with no real torch/CLIP.
    class V:
        def __init__(self, t): self.t = t
        def __matmul__(self, other):
            return V((sum(a * b for a, b in zip(self.t, other.t)),))
        @property
        def T(self): return self
        def item(self): return self.t[0]

    class _Scale:
        def exp(self): return self
        def item(self): return 10.0

    class _Model:
        logit_scale = _Scale()

    monkeypatch.setattr(identity_check, "_load", lambda cfg=None: ("torch", _Model(), "proc"))
    monkeypatch.setattr(identity_check, "_embed", lambda loaded, path: V((1.0, 0.0)))

    def fake_embed_text(loaded, text):
        # image embedding (1, 0) is much closer to "with" (0.9, 0.1) than to
        # "without" (0.0, 1.0) -> P(with) should land well above 0.5. (Check
        # "without" first — "a picture without X" also startswith "a picture
        # with", since "without" itself starts with "with".)
        return V((0.0, 1.0)) if "without" in text else V((0.9, 0.1))

    monkeypatch.setattr(identity_check, "_embed_text", fake_embed_text)

    p = identity_check.prop_presence("img.png", "an umbrella", {})
    assert p is not None
    assert 0.0 <= p <= 1.0
    assert p > 0.5


def test_prop_presence_orders_without_closer_than_with(monkeypatch):
    class V:
        def __init__(self, t): self.t = t
        def __matmul__(self, other):
            return V((sum(a * b for a, b in zip(self.t, other.t)),))
        @property
        def T(self): return self
        def item(self): return self.t[0]

    class _Scale:
        def exp(self): return self
        def item(self): return 10.0

    class _Model:
        logit_scale = _Scale()

    monkeypatch.setattr(identity_check, "_load", lambda cfg=None: ("torch", _Model(), "proc"))
    # image embedding is now closer to "without" than "with".
    monkeypatch.setattr(identity_check, "_embed", lambda loaded, path: V((0.0, 1.0)))

    def fake_embed_text(loaded, text):
        return V((0.0, 1.0)) if "without" in text else V((0.9, 0.1))

    monkeypatch.setattr(identity_check, "_embed_text", fake_embed_text)

    p = identity_check.prop_presence("img.png", "an umbrella", {})
    assert p is not None
    assert 0.0 <= p <= 1.0
    assert p < 0.5


# ------------------------------------------- _render_keyframe_on_model -------
class _Runlog:
    def __init__(self):
        self.infos, self.warnings = [], []

    def info(self, *a, **k):
        self.infos.append(a)

    def warning(self, *a, **k):
        self.warnings.append(a)


@pytest.fixture()
def wired(monkeypatch):
    """Stub the two GPU/model touchpoints and the prompt builder; record every
    render's dest+seed so a test can assert the retry control flow."""
    renders = []
    prompts = []

    def fake_generate_still(client, cfg, prompt, w, h, dest, seed, ref_images=None):
        with open(dest, "wb") as f:
            f.write(b"\0" * 16)
        renders.append((os.path.basename(dest), seed))
        prompts.append(prompt)
        return dest

    import pipeline.kidsong.refs as refs_mod
    monkeypatch.setattr(refs_mod, "generate_still", fake_generate_still)
    monkeypatch.setattr(generate, "_keyframe_prompt", lambda shot, song, cfg: "PROMPT")
    return {"renders": renders, "prompts": prompts, "monkeypatch": monkeypatch}


def _margins(monkeypatch, values):
    seq = {"i": 0}

    def fake_margin(path, char_id, cfg):
        v = values[min(seq["i"], len(values) - 1)]
        seq["i"] += 1
        return v

    monkeypatch.setattr(identity_check, "identity_margin", fake_margin)


_KF_CFG = {"kidsong": {"keyframe_first": {"identity_retries": 3, "identity_min_margin": 0.0}}}
_SOLO = {"id": "s01", "characters": ["Kofi"], "shot_type": "closeup", "seed": 100}
_GROUP = {"id": "s02", "characters": ["all"], "shot_type": "wide", "seed": 200}


def test_solo_keyframe_retries_until_on_model(wired, tmp_path):
    _margins(wired["monkeypatch"], [-0.05, 0.02])   # base drifts, retry #1 clean
    kf = str(tmp_path / "s01.png")
    log = _Runlog()
    generate._render_keyframe_on_model("client", _KF_CFG, _SOLO, {}, kf, 896, 512,
                                       ["kofi_ref.png"], log)
    # one base render + exactly one retry (bumped seed), then accept
    assert wired["renders"] == [("s01.png", 100), ("s01.png.try1.png", 101)]
    assert os.path.exists(kf) and not os.path.exists(kf + ".try1.png")
    assert any("on-model after" in a[0] for a in log.infos)


def test_solo_keyframe_no_retry_when_first_render_is_on_model(wired, tmp_path):
    _margins(wired["monkeypatch"], [0.03])          # clean on the first render
    kf = str(tmp_path / "s01.png")
    generate._render_keyframe_on_model("client", _KF_CFG, _SOLO, {}, kf, 896, 512,
                                       ["kofi_ref.png"], _Runlog())
    assert wired["renders"] == [("s01.png", 100)]   # no retry


def test_group_keyframe_never_enters_the_identity_loop(wired, tmp_path):
    _margins(wired["monkeypatch"], [-0.9, -0.9, -0.9, -0.9])  # would retry if checked
    kf = str(tmp_path / "s02.png")
    generate._render_keyframe_on_model("client", _KF_CFG, _GROUP, {}, kf, 896, 512,
                                       ["zuri.png", "kofi.png", "nala.png"], _Runlog())
    assert wired["renders"] == [("s02.png", 200)]   # single render, no identity check


def test_no_detector_means_a_single_render(wired, tmp_path):
    _margins(wired["monkeypatch"], [None])          # detector unavailable
    kf = str(tmp_path / "s01.png")
    generate._render_keyframe_on_model("client", _KF_CFG, _SOLO, {}, kf, 896, 512,
                                       ["kofi_ref.png"], _Runlog())
    assert wired["renders"] == [("s01.png", 100)]


def test_keeps_the_best_attempt_when_margin_never_reached(wired, tmp_path):
    # base -0.05, retries -0.04, -0.02, -0.03 -> best is the -0.02 (attempt 2)
    _margins(wired["monkeypatch"], [-0.05, -0.04, -0.02, -0.03])
    kf = str(tmp_path / "s01.png")
    log = _Runlog()
    generate._render_keyframe_on_model("client", _KF_CFG, _SOLO, {}, kf, 896, 512,
                                       ["kofi_ref.png"], log)
    # 1 base + 3 retries, all attempted; the best (-0.02) is what remains on disk
    seeds = [s for _, s in wired["renders"]]
    assert seeds == [100, 101, 102, 103]
    assert os.path.exists(kf)
    assert not any(os.path.exists(kf + f".try{i}.png") for i in (1, 2, 3))
    assert any("still failing gates" in a[0] for a in log.warnings)


def test_retries_disabled_by_config_is_a_single_render(wired, tmp_path):
    _margins(wired["monkeypatch"], [-0.9, 0.5])     # would retry if enabled
    cfg = {"kidsong": {"keyframe_first": {"identity_retries": 0}}}
    kf = str(tmp_path / "s01.png")
    generate._render_keyframe_on_model("client", cfg, _SOLO, {}, kf, 896, 512,
                                       ["kofi_ref.png"], _Runlog())
    assert wired["renders"] == [("s01.png", 100)]


def test_base_render_is_plain_but_retries_carry_the_identity_emphasis(wired, tmp_path):
    _margins(wired["monkeypatch"], [-0.05, 0.02])   # base drifts -> one retry
    kf = str(tmp_path / "s01.png")
    generate._render_keyframe_on_model("client", _KF_CFG, _SOLO, {}, kf, 896, 512,
                                       ["kofi_ref.png"], _Runlog())
    prompts = wired["prompts"]
    assert prompts[0] == "PROMPT"                    # base render: plain
    assert prompts[1].startswith("PROMPT")           # retry: plain + emphasis
    assert "short dark natural curly hair" in prompts[1]   # Kofi's own hair
    assert "afro puffs" in prompts[1]                # …and the castmate hair to avoid


# ---------------------------------------------------------- _identity_emphasis ---
def test_identity_emphasis_names_own_hair_and_excludes_the_others():
    clause = generate._identity_emphasis("kofi")
    assert "short dark natural curly hair" in clause          # Kofi's own
    assert "afro puffs" in clause and "cornrow braids" in clause  # the girls' — to avoid
    assert clause.startswith(" ") and clause.endswith(".")


def test_identity_emphasis_is_empty_for_unknown_character():
    assert generate._identity_emphasis("nobody-xyz") == ""
