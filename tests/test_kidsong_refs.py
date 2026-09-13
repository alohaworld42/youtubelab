"""Tests for pipeline.kidsong.refs (canonical per-character reference images),
generate_still (the raw-prompt still primitive), the cast.reference_for
accessor, the load_bible mtime cache invalidation, and the ComfyClient
staged-input cleanup.

CPU-only, no GPU, no network: the ComfyUI client is faked, transitions'
frame-grab is stubbed, and the cast bible is a temp file. Nothing here renders
or reaches a server.
"""
import json
import os

import pytest

from pipeline.kidsong import cast, refs


# ------------------------------------------------------------------ helpers ---
def _cfg(tmp_path):
    return {
        "_root": str(tmp_path),
        "paths": {"output_dir": str(tmp_path / "output")},
        "kidsong": {"seed": 20260717},
    }


_VALID_CHARS = [
    {
        "id": "zuri", "name": "Zuri", "age": "3-year-old girl",
        "skin": "deep warm brown skin", "hair": "afro puffs",
        "top": "a yellow t-shirt", "bottom": "shorts", "shoes": "sneakers",
        "prop": "a toy", "seed_offset": 1013, "must_not": [],
    },
    {
        "id": "kofi", "name": "Kofi", "age": "4-year-old boy",
        "skin": "medium-deep brown skin", "hair": "short curly hair",
        "top": "a blue t-shirt", "bottom": "shorts", "shoes": "sneakers",
        "prop": "a drum", "seed_offset": 2027, "must_not": [],
    },
]


def _write_bible(path, characters, version="1.0.0"):
    with open(path, "w", encoding="utf-8") as f:
        json.dump({
            "version": version, "channel": "ZubiBop",
            "ensemble_label": "Adorable Black toddlers",
            "characters": characters,
        }, f)
    return str(path)


def _use_bible(monkeypatch, path):
    """Point the module-level cast bible at `path` and clear its cache."""
    monkeypatch.setattr(cast, "BIBLE_PATH", str(path))
    monkeypatch.setattr(cast, "_bible_cache", None)
    monkeypatch.setattr(cast, "_bible_cache_path", None)
    monkeypatch.setattr(cast, "_bible_cache_mtime", None)


# ============================================================ reference_path ===
def test_reference_path_is_under_cast_refs(tmp_path):
    p = refs.reference_path(_cfg(tmp_path), "zuri")
    assert p.replace("\\", "/").endswith("output/_cast_refs/zuri/canonical.png")


def test_reference_path_lowercases_id(tmp_path):
    p = refs.reference_path(_cfg(tmp_path), "ZURI")
    assert "/zuri/" in p.replace("\\", "/")


# ============================================================ reference_for ====
def test_reference_for_none_when_no_file(tmp_path):
    assert refs.reference_for(_cfg(tmp_path), "zuri") is None


def test_reference_for_default_location_when_present(tmp_path):
    cfg = _cfg(tmp_path)
    dest = refs.reference_path(cfg, "zuri")
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    with open(dest, "wb") as f:
        f.write(b"PNGDATA")
    assert refs.reference_for(cfg, "zuri") == dest


def test_reference_for_empty_file_is_ignored(tmp_path):
    cfg = _cfg(tmp_path)
    dest = refs.reference_path(cfg, "zuri")
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    open(dest, "wb").close()  # zero-byte
    assert refs.reference_for(cfg, "zuri") is None


def test_reference_for_prefers_bible_recorded_path(tmp_path, monkeypatch):
    recorded = tmp_path / "custom_ref.png"
    with open(recorded, "wb") as f:
        f.write(b"PNGDATA")
    chars = [dict(_VALID_CHARS[0], reference_image=str(recorded)), _VALID_CHARS[1]]
    _use_bible(monkeypatch, _write_bible(tmp_path / "bible.json", chars))
    assert refs.reference_for(_cfg(tmp_path), "zuri") == str(recorded)


# ============================================================ generate_reference
class _FakeClient:
    def __init__(self):
        self.calls = []

    def render(self, workflow, patches, out_path):
        self.calls.append((workflow, patches, out_path))
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        with open(out_path, "wb") as f:
            f.write(b"PNGDATA")
        return out_path


def test_generate_reference_renders_zimage_graph_to_reference_path(tmp_path):
    cfg = _cfg(tmp_path)
    client = _FakeClient()
    out = refs.generate_reference(client, cfg, "kofi")

    assert out == refs.reference_path(cfg, "kofi")
    assert os.path.isfile(out) and os.path.getsize(out) > 0
    assert len(client.calls) == 1
    workflow, patches, _ = client.calls[0]
    assert workflow == "zimage_ref"
    # patched via the exact input names the Z-Image graph exposes
    assert set(patches) == {"PROMPT", "SEED", "WIDTH", "HEIGHT", "FILENAME_PREFIX"}
    assert "prompt" in patches["PROMPT"]
    assert "value" in patches["SEED"] and "value" in patches["WIDTH"]
    assert patches["FILENAME_PREFIX"]["filename_prefix"].endswith("kofi")


def test_generate_reference_prompt_describes_the_child_without_ltx_trigger(tmp_path):
    client = _FakeClient()
    refs.generate_reference(client, _cfg(tmp_path), "kofi")
    prompt = client.calls[0][1]["PROMPT"]["prompt"]
    assert "Kofi" in prompt
    assert cast.character("kofi")["skin"] in prompt
    # the LTX LoRA trigger token is meaningless to Z-Image and must be absent
    assert "P1x4r" not in prompt
    # single-subject framing is stated
    assert "one child" in prompt.lower()


def test_generate_reference_seed_is_deterministic_per_character(tmp_path):
    c1, c2 = _FakeClient(), _FakeClient()
    refs.generate_reference(c1, _cfg(tmp_path), "kofi")
    refs.generate_reference(c2, _cfg(tmp_path), "kofi")
    assert c1.calls[0][1]["SEED"]["value"] == c2.calls[0][1]["SEED"]["value"]
    # different child -> different seed family
    c3 = _FakeClient()
    refs.generate_reference(c3, _cfg(tmp_path), "zuri")
    assert c3.calls[0][1]["SEED"]["value"] != c1.calls[0][1]["SEED"]["value"]


def test_reference_dims_snap_to_multiple_of_16(tmp_path):
    cfg = _cfg(tmp_path)
    cfg["kidsong"]["reference"] = {"width": 700, "height": 1001}
    w, h = refs._reference_dims(cfg)
    assert w % 16 == 0 and h % 16 == 0
    assert w == 688 and h == 992


# ================================================================ generate_still
def test_generate_still_renders_zimage_graph_with_raw_prompt(tmp_path):
    cfg = _cfg(tmp_path)
    client = _FakeClient()
    dest = str(tmp_path / "output" / "_stills" / "s06_a0.png")

    out = refs.generate_still(client, cfg, "RAW PROMPT, exactly as given", 512, 768, dest, 4242)

    assert out == dest
    assert os.path.isfile(out) and os.path.getsize(out) > 0
    assert len(client.calls) == 1
    workflow, patches, out_path = client.calls[0]
    assert workflow == "zimage_ref"
    assert out_path == dest
    # patched via the exact input names the Z-Image graph exposes
    assert set(patches) == {"PROMPT", "SEED", "WIDTH", "HEIGHT", "FILENAME_PREFIX"}
    assert patches["PROMPT"] == {"prompt": "RAW PROMPT, exactly as given"}
    assert patches["SEED"] == {"value": 4242}
    assert patches["WIDTH"] == {"value": 512}
    assert patches["HEIGHT"] == {"value": 768}
    assert patches["FILENAME_PREFIX"]["filename_prefix"] == "stills/s06_a0"


def test_generate_still_prompt_is_used_verbatim_no_templating(tmp_path):
    client = _FakeClient()
    dest = str(tmp_path / "still.png")
    refs.generate_still(client, _cfg(tmp_path), "a single sentence, nothing added", 512, 512, dest, 1)
    assert client.calls[0][1]["PROMPT"]["prompt"] == "a single sentence, nothing added"


def test_generate_still_does_not_snap_odd_dims(tmp_path):
    client = _FakeClient()
    dest = str(tmp_path / "still.png")
    refs.generate_still(client, _cfg(tmp_path), "p", 513, 767, dest, 1)
    assert client.calls[0][1]["WIDTH"]["value"] == 513
    assert client.calls[0][1]["HEIGHT"]["value"] == 767


def test_generate_still_seed_is_coerced_to_int(tmp_path):
    client = _FakeClient()
    dest = str(tmp_path / "still.png")
    refs.generate_still(client, _cfg(tmp_path), "p", 512, 512, dest, "777")
    assert client.calls[0][1]["SEED"]["value"] == 777


def test_generate_still_creates_dest_dir(tmp_path):
    client = _FakeClient()
    dest = str(tmp_path / "nested" / "deep" / "still.png")
    assert not os.path.isdir(os.path.dirname(dest))

    out = refs.generate_still(client, _cfg(tmp_path), "p", 512, 512, dest, 1)

    assert out == dest
    assert os.path.isdir(os.path.dirname(dest))
    assert os.path.isfile(dest)


def test_generate_still_filename_prefix_derived_from_dest_basename(tmp_path):
    client = _FakeClient()
    dest = str(tmp_path / "somewhere" / "s12_a3.png")
    refs.generate_still(client, _cfg(tmp_path), "p", 512, 512, dest, 1)
    assert client.calls[0][1]["FILENAME_PREFIX"]["filename_prefix"] == "stills/s12_a3"


def test_generate_still_never_touches_the_gpu_except_via_client(tmp_path):
    """The only thing capable of a real render is `client.render` — a client
    that raises on render is the only way this could ever fail, i.e. nothing
    else in the call path reaches out to hardware or the network."""
    class _BoomClient:
        def render(self, *a, **k):
            raise AssertionError("gpu touched")

    with pytest.raises(AssertionError):
        refs.generate_still(_BoomClient(), _cfg(tmp_path), "p", 512, 512,
                            str(tmp_path / "still.png"), 1)


# ============================================================ harvest_reference
def test_harvest_reference_none_when_no_accepted_take(tmp_path, monkeypatch):
    monkeypatch.setattr(refs, "_find_accepted_single_subject_take", lambda cfg, cid: None)
    assert refs.harvest_reference(_cfg(tmp_path), "kofi") is None


def test_harvest_reference_extracts_frame_from_accepted_take(tmp_path, monkeypatch):
    cfg = _cfg(tmp_path)
    fake_src = str(tmp_path / "s00_a0.mp4")
    monkeypatch.setattr(refs, "_find_accepted_single_subject_take", lambda c, cid: fake_src)

    from pipeline.kidsong import transitions

    def fake_extract(video, position, out_png):
        assert position in ("first", "last")
        os.makedirs(os.path.dirname(out_png), exist_ok=True)
        with open(out_png, "wb") as f:
            f.write(b"PNGFRAME")
        return out_png

    monkeypatch.setattr(transitions, "extract_frame", fake_extract)
    out = refs.harvest_reference(cfg, "kofi")
    assert out == refs.reference_path(cfg, "kofi")
    assert os.path.isfile(out) and os.path.getsize(out) > 0


def test_harvest_reference_returns_none_if_extract_fails(tmp_path, monkeypatch):
    monkeypatch.setattr(refs, "_find_accepted_single_subject_take",
                        lambda c, cid: str(tmp_path / "x.mp4"))
    from pipeline.kidsong import transitions

    def boom(*a, **k):
        raise RuntimeError("no ffmpeg")

    monkeypatch.setattr(transitions, "extract_frame", boom)
    assert refs.harvest_reference(_cfg(tmp_path), "kofi") is None


# ============================================================ ensure_reference =
def test_ensure_reference_returns_existing_without_regenerating(tmp_path):
    """An existing reference whose style sidecar MATCHES the active style is
    returned untouched. (A sidecar-less reference is deliberately treated as
    stale since the style-stamp change — covered by the staleness tests below.)"""
    cfg = _cfg(tmp_path)
    dest = refs.reference_path(cfg, "kofi")
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    with open(dest, "wb") as f:
        f.write(b"EXISTING")
    refs._write_style_sidecar(cfg, dest)

    class _NoRender:
        def render(self, *a, **k):
            raise AssertionError("must not regenerate an existing reference")

    assert refs.ensure_reference(_NoRender(), cfg, "kofi") == dest


def test_ensure_reference_harvests_before_generating(tmp_path, monkeypatch):
    cfg = _cfg(tmp_path)
    harvested = refs.reference_path(cfg, "kofi")
    os.makedirs(os.path.dirname(harvested), exist_ok=True)

    def fake_harvest(c, cid):
        with open(harvested, "wb") as f:
            f.write(b"HARVEST")
        return harvested

    monkeypatch.setattr(refs, "harvest_reference", fake_harvest)

    class _NoRender:
        def render(self, *a, **k):
            raise AssertionError("must harvest before generating")

    assert refs.ensure_reference(_NoRender(), cfg, "kofi") == harvested


def test_ensure_reference_generates_when_nothing_to_harvest(tmp_path, monkeypatch):
    cfg = _cfg(tmp_path)
    monkeypatch.setattr(refs, "harvest_reference", lambda c, cid: None)
    client = _FakeClient()
    out = refs.ensure_reference(client, cfg, "kofi")
    assert out == refs.reference_path(cfg, "kofi")
    assert client.calls and client.calls[0][0] == "zimage_ref"


def test_ensure_reference_none_when_no_client_and_nothing_to_harvest(tmp_path, monkeypatch):
    monkeypatch.setattr(refs, "harvest_reference", lambda c, cid: None)
    assert refs.ensure_reference(None, _cfg(tmp_path), "kofi") is None


# ============================================================ cast.reference_for
def test_cast_reference_for_none_without_field():
    # the shipped bible records no reference_image, so this is None
    assert cast.reference_for("kofi") is None


def test_cast_reference_for_reads_bible_field(tmp_path, monkeypatch):
    chars = [dict(_VALID_CHARS[0], reference_image="refs/zuri.png"), _VALID_CHARS[1]]
    _use_bible(monkeypatch, _write_bible(tmp_path / "bible.json", chars))
    assert cast.reference_for("zuri") == "refs/zuri.png"
    assert cast.reference_for("kofi") is None


def test_cast_reference_for_resolves_aliases(tmp_path, monkeypatch):
    # an unresolvable name must return the SUBSTITUTED cast member's reference,
    # via the same _resolve_with_fallback the renderer uses.
    chars = [dict(_VALID_CHARS[0], reference_image="refs/zuri.png"), _VALID_CHARS[1]]
    _use_bible(monkeypatch, _write_bible(tmp_path / "bible.json", chars))
    got = cast.reference_for("Zephyr")  # unresolvable -> stable substitution
    assert got in ("refs/zuri.png", None)  # whichever it stably maps to
    # deterministic: same answer every call
    assert cast.reference_for("Zephyr") == got


# ======================================================= load_bible mtime cache
def test_load_bible_invalidates_cache_on_mtime(tmp_path, monkeypatch):
    path = _write_bible(tmp_path / "bible.json", _VALID_CHARS)
    os.utime(path, (1_000_000, 1_000_000))
    _use_bible(monkeypatch, path)  # start from a clean cache

    b1 = cast.load_bible()
    assert b1["characters"][0].get("reference_image") is None

    # edit the bible on disk and give it a distinct mtime
    chars2 = [dict(_VALID_CHARS[0], reference_image="refs/zuri.png"), _VALID_CHARS[1]]
    _write_bible(path, chars2)
    os.utime(path, (2_000_000, 2_000_000))

    b2 = cast.load_bible()
    assert b2["characters"][0].get("reference_image") == "refs/zuri.png", (
        "a long-lived process must pick up an edited bible via mtime"
    )


def test_load_bible_same_mtime_hits_cache(tmp_path, monkeypatch):
    path = _write_bible(tmp_path / "bible.json", _VALID_CHARS)
    os.utime(path, (1_500_000, 1_500_000))
    _use_bible(monkeypatch, path)
    first = cast.load_bible()
    second = cast.load_bible()
    assert first is second  # same object -> served from cache, no reparse


# ======================================================= comfy staged cleanup ==
def _comfy_cfg(comfy_path):
    return {"comfy": {"url": "http://127.0.0.1:8188", "path": str(comfy_path),
                      "autostart": False}}


def test_stage_input_image_tracks_and_cleanup_removes(tmp_path):
    from pipeline.kidsong.comfy import ComfyClient

    comfy_dir = tmp_path / "ComfyUI"
    (comfy_dir / "input").mkdir(parents=True)
    src = tmp_path / "ref.png"
    with open(src, "wb") as f:
        f.write(b"PNGDATA")

    client = ComfyClient(_comfy_cfg(comfy_dir))
    name = client.stage_input_image(str(src))
    staged_path = comfy_dir / "input" / name
    assert staged_path.exists()
    assert name in client._staged_inputs

    client.cleanup_staged()
    assert not staged_path.exists()
    assert client._staged_inputs == []
    # idempotent
    client.cleanup_staged()


def test_free_triggers_staged_cleanup(tmp_path, monkeypatch):
    import pipeline.kidsong.comfy as comfy_mod
    from pipeline.kidsong.comfy import ComfyClient

    comfy_dir = tmp_path / "ComfyUI"
    (comfy_dir / "input").mkdir(parents=True)
    src = tmp_path / "ref.png"
    with open(src, "wb") as f:
        f.write(b"PNGDATA")

    # keep free() offline: no real POST to a server
    monkeypatch.setattr(comfy_mod.requests, "post", lambda *a, **k: None)

    client = ComfyClient(_comfy_cfg(comfy_dir))
    name = client.stage_input_image(str(src))
    assert (comfy_dir / "input" / name).exists()

    client.free()
    assert not (comfy_dir / "input" / name).exists()
    assert client._staged_inputs == []


# ===================================================== generate_still + refs ==
class _StagingFakeClient(_FakeClient):
    def __init__(self):
        super().__init__()
        self.staged = []

    def stage_input_image(self, src_path):
        self.staged.append(src_path)
        return f"staged_{len(self.staged)}.png"


def _real_png(path, w, h):
    from PIL import Image
    os.makedirs(os.path.dirname(path), exist_ok=True)
    Image.new("RGB", (w, h), (120, 60, 30)).save(path)
    return path


def test_generate_still_with_ref_images_prefits_and_switches_workflow(tmp_path):
    """Identity refs are PRE-FIT to 384x512 before staging (the graph's own
    auto-resize lands portrait refs on odd, patchify-incompatible latent dims —
    measured as 16x '[KSampler] shape ... is invalid' on job 23), and the
    render uses the zimage_still_ref graph with IMAGE1/2/3 patched."""
    from PIL import Image

    client = _StagingFakeClient()
    r1 = _real_png(str(tmp_path / "refs" / "zuri.png"), 768, 1024)
    r2 = _real_png(str(tmp_path / "refs" / "kofi.png"), 768, 1024)
    dest = str(tmp_path / "keyframes" / "s00.png")
    refs.generate_still(client, {}, "a prompt", 896, 512, dest, seed=7,
                        ref_images=[r1, r2])
    wf, patches, _out = client.calls[-1]
    # identity work rides the WORKING klein graph, not the retired Omni one
    assert wf == "flux2_klein_ref"
    # klein dialect: CLIPTextEncode wants "text", with the reference preamble
    assert "text" in patches["PROMPT"] and "prompt" not in patches["PROMPT"]
    # style-neutral identity clause (85dd58d) — no hardcoded 'clay children'
    assert patches["PROMPT"]["text"].startswith(
        "Using the children from the reference images"
    )
    assert patches["PROMPT"]["text"].endswith("a prompt")
    assert patches["SEED_VALUE"] == {"value": 7}
    # 2 refs -> third slot repeats the first staged name
    assert patches["IMAGE1"] == {"image": "staged_1.png"}
    assert patches["IMAGE2"] == {"image": "staged_2.png"}
    assert patches["IMAGE3"] == {"image": "staged_1.png"}
    # staged paths are the PRE-FIT _omni_ variants at exactly 384x512
    assert all("_omni_" in os.path.basename(p) for p in client.staged)
    for p in client.staged:
        with Image.open(p) as im:
            assert im.size == (384, 512)


def test_generate_still_without_refs_keeps_plain_graph(tmp_path):
    client = _StagingFakeClient()
    dest = str(tmp_path / "keyframes" / "s01.png")
    refs.generate_still(client, {}, "a prompt", 896, 512, dest, seed=7)
    wf, patches, _out = client.calls[-1]
    assert wf == "zimage_ref"
    assert "IMAGE1" not in patches and client.staged == []


# ============================================================ style sidecar
def _write_ref(cfg, cid, payload=b"REF"):
    dest = refs.reference_path(cfg, cid)
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    with open(dest, "wb") as f:
        f.write(payload)
    return dest


class _CountingClient:
    def __init__(self):
        self.calls = []
        self.staged = []

    def render(self, wf, patches, dest):
        self.calls.append((wf, patches, dest))
        os.makedirs(os.path.dirname(dest), exist_ok=True)
        with open(dest, "wb") as f:
            f.write(b"FRESH")

    def stage_input_image(self, path):
        self.staged.append(path)
        return os.path.basename(str(path))


def test_generate_reference_writes_a_style_sidecar(tmp_path):
    cfg = _cfg(tmp_path)
    client = _CountingClient()
    dest = refs.generate_reference(client, cfg, "kofi")
    sidecar = refs._style_sidecar_path(dest)
    assert os.path.isfile(sidecar)
    with open(sidecar, encoding="utf-8") as f:
        data = json.load(f)
    assert data["render_style"] == refs._active_style_name(cfg)


def test_ensure_reference_missing_sidecar_regenerates_and_skips_harvest(tmp_path, monkeypatch):
    cfg = _cfg(tmp_path)
    _write_ref(cfg, "kofi")  # no sidecar -> stale by definition

    harvest_calls = []
    monkeypatch.setattr(refs, "harvest_reference",
                        lambda c, cid: harvest_calls.append(cid) or None)
    client = _CountingClient()
    out = refs.ensure_reference(client, cfg, "kofi")
    assert client.calls, "stale (sidecar-less) reference must be regenerated"
    assert harvest_calls == [], "harvest must be skipped for a stale reference"
    assert out == refs.reference_path(cfg, "kofi")


def test_ensure_reference_style_mismatch_regenerates(tmp_path, monkeypatch):
    cfg = _cfg(tmp_path)
    dest = _write_ref(cfg, "kofi")
    with open(refs._style_sidecar_path(dest), "w", encoding="utf-8") as f:
        json.dump({"render_style": "some-retired-style"}, f)

    monkeypatch.setattr(refs, "harvest_reference", lambda c, cid: None)
    client = _CountingClient()
    refs.ensure_reference(client, cfg, "kofi")
    assert client.calls


def test_ensure_reference_corrupt_sidecar_regenerates_never_raises(tmp_path, monkeypatch):
    cfg = _cfg(tmp_path)
    dest = _write_ref(cfg, "kofi")
    with open(refs._style_sidecar_path(dest), "w", encoding="utf-8") as f:
        f.write("{not valid json")

    monkeypatch.setattr(refs, "harvest_reference", lambda c, cid: None)
    client = _CountingClient()
    out = refs.ensure_reference(client, cfg, "kofi")
    assert client.calls and out


def test_ensure_reference_bible_pinned_is_never_regenerated(tmp_path, monkeypatch):
    cfg = _cfg(tmp_path)
    pinned = tmp_path / "hand-chosen.png"
    pinned.write_bytes(b"PINNED")
    monkeypatch.setattr(refs, "_pinned_reference_for", lambda c, cid: str(pinned))

    class _NoRender:
        def render(self, *a, **k):
            raise AssertionError("bible-pinned reference must never be regenerated")

    # no sidecar next to the pinned file -> stale, but pinned wins anyway
    assert refs.ensure_reference(_NoRender(), cfg, "kofi") == str(pinned)


def test_klein_still_prompt_carries_the_caller_prompt_verbatim(tmp_path):
    """4c pin: the klein ref branch must pass the caller's FULL prompt
    (which generate._keyframe_prompt builds starting with the style
    prompt_skeleton) through to the graph, only prefixed by the reference
    preamble — never truncated or replaced."""
    cfg = _cfg(tmp_path)
    r1 = tmp_path / "a.png"; r1.write_bytes(b"A")
    caller_prompt = "a testable skeleton sentence. Kofi stands in the yard."
    client = _CountingClient()
    refs.generate_still(client, cfg, caller_prompt, 896, 512,
                        str(tmp_path / "still.png"), 1, ref_images=[str(r1)])
    wf, patches, _ = client.calls[0]
    assert wf == "flux2_klein_ref"
    blob = json.dumps(patches)
    assert caller_prompt in blob
