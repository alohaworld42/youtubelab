"""Tests for pipeline.kidsong.refs.generate_card — the Z-Image "learning card"
generator for a taught letter/number/shape (video diffusion renders on-screen
text/shapes poorly; a clean Z-Image still is the reliable legible asset).

CPU-only, no GPU, no network: mirrors tests/test_kidsong_refs.py's
`generate_reference` tests exactly — the ComfyUI client is a fake object that
just records its call and writes a stub file, so nothing here renders or
reaches a server.
"""
import os

import pytest

from pipeline.kidsong import refs


def _cfg(tmp_path):
    return {
        "_root": str(tmp_path),
        "paths": {"output_dir": str(tmp_path / "output")},
        "kidsong": {"seed": 20260717},
    }


class _FakeClient:
    def __init__(self):
        self.calls = []

    def render(self, workflow, patches, out_path):
        self.calls.append((workflow, patches, out_path))
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        with open(out_path, "wb") as f:
            f.write(b"PNGDATA")
        return out_path


# ------------------------------------------------------------------ card_path
def test_card_path_is_under_learning_cards(tmp_path):
    p = refs.card_path(_cfg(tmp_path), "A", "letter")
    assert p.replace("\\", "/").endswith("output/_learning_cards/letter_a.png")


def test_card_path_slugifies_label_and_kind(tmp_path):
    p = refs.card_path(_cfg(tmp_path), "red circle", "SHAPE")
    assert p.replace("\\", "/").endswith("output/_learning_cards/shape_red-circle.png")


def test_card_path_never_blank_for_empty_label(tmp_path):
    p = refs.card_path(_cfg(tmp_path), "", "")
    assert p.replace("\\", "/").endswith("output/_learning_cards/item_item.png")


# ------------------------------------------------------------- generate_card
def test_generate_card_renders_zimage_graph_to_card_path(tmp_path):
    cfg = _cfg(tmp_path)
    client = _FakeClient()
    out = refs.generate_card(client, cfg, "A", "letter")

    assert out == refs.card_path(cfg, "A", "letter")
    assert os.path.isfile(out) and os.path.getsize(out) > 0
    assert len(client.calls) == 1
    workflow, patches, out_path = client.calls[0]
    assert workflow == "zimage_ref"
    assert out_path == out
    # patched via the exact input names the Z-Image graph exposes
    assert set(patches) == {"PROMPT", "SEED", "WIDTH", "HEIGHT", "FILENAME_PREFIX"}
    assert "prompt" in patches["PROMPT"]
    assert "value" in patches["SEED"] and "value" in patches["WIDTH"] and "value" in patches["HEIGHT"]
    assert patches["FILENAME_PREFIX"]["filename_prefix"] == "_learning_cards/letter_a"


def test_generate_card_prompt_names_the_letter_clearly(tmp_path):
    client = _FakeClient()
    refs.generate_card(client, _cfg(tmp_path), "A", "letter")
    prompt = client.calls[0][1]["PROMPT"]["prompt"]
    assert "letter A" in prompt
    assert "capital" in prompt.lower()
    assert "centered" in prompt.lower()
    assert "plain solid background" in prompt.lower()
    assert "no other text" in prompt.lower()
    # single-subject, no people
    assert "child" not in prompt.lower() and "kid" not in prompt.lower()


def test_generate_card_prompt_names_the_number_clearly(tmp_path):
    client = _FakeClient()
    refs.generate_card(client, _cfg(tmp_path), "3", "number")
    prompt = client.calls[0][1]["PROMPT"]["prompt"]
    assert "number 3" in prompt


def test_generate_card_prompt_names_the_shape_clearly(tmp_path):
    client = _FakeClient()
    refs.generate_card(client, _cfg(tmp_path), "circle", "shape")
    prompt = client.calls[0][1]["PROMPT"]["prompt"]
    assert "circle shape" in prompt


def test_generate_card_unknown_kind_still_renders_generically(tmp_path):
    client = _FakeClient()
    out = refs.generate_card(client, _cfg(tmp_path), "star", "sticker")
    prompt = client.calls[0][1]["PROMPT"]["prompt"]
    assert "star" in prompt
    assert os.path.isfile(out)


def test_generate_card_uses_the_active_render_style_skeleton(tmp_path):
    cfg = _cfg(tmp_path)
    cfg["kidsong"]["render_style"] = "flat_storybook"
    cfg["kidsong"]["render_styles"] = {
        "flat_storybook": {"prompt_skeleton": "a flat storybook illustration."},
    }
    client = _FakeClient()
    refs.generate_card(client, cfg, "B", "letter")
    prompt = client.calls[0][1]["PROMPT"]["prompt"]
    assert prompt.startswith("a flat storybook illustration,")


def test_generate_card_respects_explicit_dest(tmp_path):
    cfg = _cfg(tmp_path)
    dest = str(tmp_path / "somewhere" / "custom.png")
    client = _FakeClient()
    out = refs.generate_card(client, cfg, "5", "number", dest=dest)
    assert out == dest
    assert os.path.isfile(dest)


def test_generate_card_seed_is_deterministic_per_kind_and_label(tmp_path):
    cfg = _cfg(tmp_path)
    c1, c2 = _FakeClient(), _FakeClient()
    refs.generate_card(c1, cfg, "A", "letter")
    refs.generate_card(c2, cfg, "A", "letter")
    assert c1.calls[0][1]["SEED"]["value"] == c2.calls[0][1]["SEED"]["value"]

    c3 = _FakeClient()
    refs.generate_card(c3, cfg, "B", "letter")
    assert c3.calls[0][1]["SEED"]["value"] != c1.calls[0][1]["SEED"]["value"]

    # Same label, different kind -> different seed too (a "circle" shape card
    # and a hypothetical "circle" letter card must not collide).
    c4 = _FakeClient()
    refs.generate_card(c4, cfg, "circle", "shape")
    c5 = _FakeClient()
    refs.generate_card(c5, cfg, "circle", "letter")
    assert c4.calls[0][1]["SEED"]["value"] != c5.calls[0][1]["SEED"]["value"]


def test_card_dims_default_square_and_snap_to_multiple_of_16(tmp_path):
    cfg = _cfg(tmp_path)
    w, h = refs._card_dims(cfg)
    assert w == 1024 and h == 1024

    cfg["kidsong"]["learning_card"] = {"width": 700, "height": 1001}
    w2, h2 = refs._card_dims(cfg)
    assert w2 % 16 == 0 and h2 % 16 == 0
    assert w2 == 688 and h2 == 992


def test_generate_card_never_touches_the_gpu_except_via_client(tmp_path):
    """The only thing capable of a real render is `client.render` — assert a
    client that raises on render is the only way this could ever fail, i.e.
    nothing else in the call path reaches out to hardware or the network."""
    class _BoomClient:
        def render(self, *a, **k):
            raise AssertionError("gpu touched")

    with pytest.raises(AssertionError):
        refs.generate_card(_BoomClient(), _cfg(tmp_path), "A", "letter")
