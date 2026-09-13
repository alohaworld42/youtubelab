import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


@pytest.fixture(autouse=True)
def _clean_review_sessions():
    """`pipeline.kidsong.review` keeps a module-level registry of per-run
    external-review sessions (the shared "is anyone answering?" state). It is
    rotated per run in production, but tests that build ad-hoc cfg dicts all
    land in the same fallback slot — clear it around every test so no test can
    inherit another's unattended state."""
    from pipeline.kidsong import review

    review.reset_sessions()
    yield
    review.reset_sessions()


@pytest.fixture()
def tmp_db(tmp_path, monkeypatch):
    """Fresh SQLite DB per test."""
    from studio import db

    monkeypatch.setenv("STUDIO_DB_PATH", str(tmp_path / "test.db"))
    db.reset_for_tests()
    yield
    db.reset_for_tests()


@pytest.fixture(autouse=True)
def _no_inherited_studio_password(monkeypatch):
    """The password gate is opt-in via STUDIO_PASSWORD, which
    `pipeline.config.load_dotenv` reads from the developer's `.env`. A test
    that builds the real `create_app()` would then be redirected to /login on
    every request — on that developer's machine only.

    `studio.auth._gate` used to paper over this by opening for any app with
    `TESTING` set, which voided a configured password and made the gate
    impossible to test on the real app. Clearing the variable here is where
    that concern belongs: the suite becomes independent of the environment it
    runs in, and a test that WANTS the gate sets the password itself
    (tests/features/test_locking_the_studio.py does).
    """
    monkeypatch.delenv("STUDIO_PASSWORD", raising=False)
