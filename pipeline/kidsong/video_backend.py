"""kidsong.render_backend — selects which client renders shot VIDEO.

The kidsong director loop (``pipeline/kidsong/generate.py``) opens a
``ComfyClient`` for a run and uses it for two very different jobs: (1)
Z-Image stills (the authored-negative baseline, the run fingerprint, cast
reference bootstrap, and keyframe-first renders) and (2) the per-shot LTX
video render. Only job (2) is swappable — job (1) always stays on the local
ComfyUI client, because Z-Image/keyframe machinery has no cloud equivalent
here. This module is the seam for job (2) only.

``make_video_backend(cfg, comfy_client)`` reads ``kidsong.render_backend``
from config and returns the object the shot-render loop should call instead
of ``client`` directly:

- absent key, or ``"comfy"``: returns ``comfy_client`` UNCHANGED (the exact
  same object, not a wrapper). This identity return is what makes the
  default path byte-identical to before this seam existed — there is
  nothing new in the call chain to change behavior.
- ``"seedance"``: lazily imports ``pipeline.kidsong.seedance`` (by name, at
  call time, so this module itself always imports cleanly even in a tree
  where that file is absent or mid-edit) and returns a constructed
  ``SeedanceBackend(cfg)``. A misconfigured/unavailable Seedance backend
  raises immediately, here, before the song is even sung — never partway
  through an episode's render loop.
- anything else: raises ``ValueError`` naming the allowed values.

A video backend is any object providing this duck-typed shape (the shape
``kidsong.comfy.ComfyClient`` already has, and that
``kidsong.seedance.SeedanceBackend`` mirrors on purpose):

- ``render(workflow_name, patches, out_path)`` -> out_path
- ``stage_input_image(src)`` -> handle
- ``free()``
- ``restart_if_hung()``
- a ``url`` attribute

A backend MAY also declare ``accepts_reference_images = True`` to opt into
multi-image reference patches; ``ComfyClient`` deliberately does not (its
patch applier raises ``KeyError: No node titled …`` for a graph title it
does not recognize), so the render loop must check
``getattr(backend, "accepts_reference_images", False)`` rather than assume
every backend supports it.
"""
import logging

log = logging.getLogger("kidsong.render_backend")

# The only values kidsong.render_backend may take.
_ALLOWED_BACKENDS = ("comfy", "seedance")


def make_video_backend(cfg, comfy_client):
    """Return the client the shot-render loop should call for shot VIDEO.

    ``comfy_client`` is the already-constructed ``ComfyClient`` for this run
    (still used directly for stills — see the module docstring). This
    function never constructs a ComfyClient itself; it either returns the
    one it was given, or builds an alternative backend for the video render
    only.
    """
    ks = (cfg or {}).get("kidsong", {}) if isinstance(cfg, dict) else {}
    ks = ks or {}
    name = ks.get("render_backend") or "comfy"

    if name == "comfy":
        return comfy_client

    if name == "seedance":
        try:
            from pipeline.kidsong import seedance
        except ImportError as exc:
            raise RuntimeError(
                "kidsong.render_backend is 'seedance' but "
                f"pipeline.kidsong.seedance could not be imported: {exc}"
            ) from exc
        if not seedance.is_available(cfg):
            raise RuntimeError(
                "kidsong.render_backend is 'seedance' but the backend is not "
                "available -- seedance.enabled must be true and "
                "SEEDANCE_API_KEY must be set. Fix the config/environment "
                "before rendering, not mid-episode."
            )
        return seedance.SeedanceBackend(cfg)

    raise ValueError(
        f"kidsong.render_backend={name!r} is not supported; allowed values "
        f"are {_ALLOWED_BACKENDS}."
    )
