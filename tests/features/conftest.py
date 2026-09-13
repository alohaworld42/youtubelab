"""Fixtures for the feature suite: a REAL studio, in a temp world.

The 90 files in `tests/` are named after modules — `test_alignment`,
`test_gpu_lock`, `test_kidsong_cast`. They test units. Even `test_auth_gate.py`,
the closest thing to an application test, builds a *bare Flask app with a stub
route* rather than the real one. Nothing anywhere drives the app a person
actually uses.

So a defect that lives in the seams between units — a view that never passes
`cfg` to the function under it, a template whose form field name does not match
what the route reads, a status page that disagrees with the queue it links to —
passes every one of those 1971 tests.

`studio` here is the real `create_app()`: real blueprints, real templates, real
`load_config()` merge, real SQLite. Only two things are moved: `output_dir`
points at a tmp directory and the DB at a tmp file, so a test can plant a
world on disk and then use the app exactly as the operator does.

Nothing is mocked. A feature test that needs a review gate writes the same
`*.request.json` the pipeline writes, and asserts on the same
`*.response.json` the pipeline polls for.
"""
import json
import os
import time

import pytest


@pytest.fixture()
def world(tmp_path):
    """The on-disk state the studio reads: an output dir plus builders for it.

    Every builder writes the REAL file shape (see the module docstring of
    `pipeline.kidsong.review_queue`, which documents the two gate layouts and
    notes they were confirmed against live `output/` state). A feature test
    that invents its own shape here would pass while the app breaks.
    """
    out = tmp_path / "output"
    (out / "Final").mkdir(parents=True)

    class World:
        root = tmp_path
        output_dir = out

        def episode(self, base="20260731-120000-kidsong-test-song", shots=2):
            """A planned episode: a sung wav and a shot list, nothing rendered."""
            (out / f"{base}.wav").write_bytes(b"RIFF\0\0\0\0WAVE")
            (out / f"{base}-shots.json").write_text(json.dumps({
                "shots": [
                    {"id": f"s{i:02d}", "shot_type": "wide" if i % 2 == 0 else "closeup",
                     "action": f"the kids do thing {i}", "setting": "a sunny backyard",
                     "characters": ["all"], "camera": "static"}
                    for i in range(shots)
                ]
            }), encoding="utf-8")
            (out / f"{base}-song.json").write_text(json.dumps({
                "title": "Test Song", "description": "d", "tags": ["t"],
                "characters": "Three toddlers: Zuri; Kofi; Nala",
                "verses": [{"scene": "backyard", "lines": ["la la", "la la la"]}],
            }), encoding="utf-8")
            return base

        def rendered_take(self, base, shot_id="s00", attempt=0):
            d = out / f"{base}-shots"
            d.mkdir(exist_ok=True)
            path = d / f"{shot_id}_a{attempt}.mp4"
            path.write_bytes(b"\0" * 64)
            return path

        def shot_gate(self, base, shot_id="s00", attempt=0, age_seconds=0):
            """A per-take review request with no response — what the pipeline
            writes before it blocks polling (review.ExternalReviewer.review)."""
            review = out / f"{base}-shots" / "review"
            review.mkdir(parents=True, exist_ok=True)
            req = review / f"{shot_id}_a{attempt}.request.json"
            req.write_text(json.dumps({
                "shot_id": shot_id, "take": attempt,
                "contact_sheet": str(review / f"{shot_id}.png"),
                "shot": {"action": "the kids clap", "characters": ["all"],
                         "shot_type": "wide", "camera": "static",
                         "setting": "a sunny backyard"},
            }), encoding="utf-8")
            (review / f"{shot_id}.png").write_bytes(b"\x89PNG\r\n\x1a\n")
            if age_seconds:
                old = time.time() - age_seconds
                os.utime(req, (old, old))
            return req

        def cut_gate(self, base, age_seconds=0):
            """The cut gate — the last gate before an episode reaches Final/."""
            review = out / f"{base}-shots" / "review"
            review.mkdir(parents=True, exist_ok=True)
            staging = out / f"{base}-cut.mp4"
            staging.write_bytes(b"\0" * 64)
            req = review / "cut.request.json"
            req.write_text(json.dumps({
                "video_path": str(staging),
                "cut_list": [{"shot_id": "s00", "start": 0.0, "end": 2.0, "src": "s00"},
                             {"shot_id": "s01", "start": 2.0, "end": 4.0, "src": "s01"}],
            }), encoding="utf-8")
            if age_seconds:
                old = time.time() - age_seconds
                os.utime(req, (old, old))
            return req

        def answered(self, request_path):
            """Read the response the operator's click was supposed to write."""
            resp = str(request_path).replace(".request.json", ".response.json")
            if not os.path.exists(resp):
                return None
            with open(resp, encoding="utf-8") as f:
                return json.load(f)

        def finished(self, base):
            (out / "Final" / f"{base}.mp4").write_bytes(b"\0" * 64)

    return World()


@pytest.fixture()
def studio(world, tmp_path, monkeypatch):
    """The real application, pointed at `world`.

    `create_app()` resolves its config exactly once and hands it to both the
    view closures and `app.config["STUDIO_CFG"]`, so patching `load_config` is
    the one seam that redirects the whole app — blueprints included — without
    touching a line of production code.
    """
    # A real installation's root carries the shipped config.example.json —
    # `setup.set_llm_backend` seeds config.json from it on first save. A temp
    # root without it would make setup fail here and only here, which is
    # fixture infidelity rather than a product defect.
    import shutil

    repo_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    shutil.copyfile(os.path.join(repo_root, "config.example.json"),
                    str(tmp_path / "config.example.json"))

    monkeypatch.setenv("STUDIO_DB_PATH", str(tmp_path / "studio.db"))
    monkeypatch.setenv("STUDIO_SECRET_KEY", "feature-suite-fixed-key")
    monkeypatch.delenv("STUDIO_PASSWORD", raising=False)

    import app as app_module
    from pipeline.config import load_config as _real_load_config

    def _tmp_config():
        cfg = _real_load_config()
        cfg["_root"] = str(tmp_path)
        cfg.setdefault("paths", {})["output_dir"] = str(world.output_dir)
        return cfg

    monkeypatch.setattr(app_module, "load_config", _tmp_config)
    monkeypatch.setattr("studio.views.setup.load_config", _tmp_config, raising=False)

    from studio import db

    db.reset_for_tests()
    flask_app = app_module.create_app()
    flask_app.config["TESTING"] = True
    yield flask_app
    db.reset_for_tests()


@pytest.fixture()
def operator(studio):
    """A browser. Feature tests speak through this and nothing else."""
    return studio.test_client()
