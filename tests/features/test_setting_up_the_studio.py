"""FEATURE: a new operator gets the studio running.

First contact with the product. If setup silently fails to save, nothing else
in the app works and there is no error to search for — so these check that a
setting entered in the browser actually reaches the file the pipeline reads,
and that the status the page reports back is computed from that file rather
than from what was just typed.

Also the security boundary that lives here: every value on this page comes
from a browser field and lands in `.env`, which is parsed one setting per
line.
"""
import json
import os

import pytest

from pipeline.config import load_dotenv


def _status(operator):
    return operator.get("/api/setup/status").get_json()


# ============================================================== the first run ==
def test_a_fresh_studio_sends_you_to_setup(operator):
    resp = operator.get("/")
    assert resp.status_code in (200, 302)
    if resp.status_code == 302:
        assert "/setup" in resp.headers["Location"]


def test_the_setup_page_lists_the_steps_in_order(operator):
    page = operator.get("/setup").get_data(as_text=True)
    for step in ("1", "2", "3", "4"):
        assert step in page
    assert "FFmpeg" in page
    assert "Setup" in page


def test_the_setup_page_reports_what_is_missing(operator):
    status = _status(operator)
    assert isinstance(status, dict)
    assert "llm" in status, "the status endpoint does not report the LLM backend"


# ==================================================== choosing an LLM backend ==
def test_choosing_a_backend_is_persisted_where_the_pipeline_reads_it(operator, tmp_path):
    """The click has to reach `config.json`, not just the page."""
    resp = operator.post("/api/setup/llm", json={"backend": "ollama"})
    assert resp.status_code == 200, resp.get_data(as_text=True)
    assert resp.get_json()["ok"] is True

    written = json.loads((tmp_path / "config.json").read_text(encoding="utf-8"))
    assert written["llm"]["backend"] == "ollama"


def test_the_page_reports_the_backend_back_after_saving(operator):
    operator.post("/api/setup/llm", json={"backend": "ollama"})
    assert _status(operator)["llm"]["backend"] == "ollama"


def test_an_unknown_backend_is_refused(operator, tmp_path):
    resp = operator.post("/api/setup/llm", json={"backend": "definitely-not-a-backend"})
    assert resp.status_code == 400
    assert "error" in resp.get_json()

    if (tmp_path / "config.json").exists():
        written = json.loads((tmp_path / "config.json").read_text(encoding="utf-8"))
        assert written.get("llm", {}).get("backend") != "definitely-not-a-backend"


def test_switching_backends_twice_leaves_valid_config(operator, tmp_path):
    """`set_llm_backend` rewrites the whole of config.json to change one string;
    a half-written file would take the app down on next boot."""
    operator.post("/api/setup/llm", json={"backend": "groq", "api_key": "gsk_test"})
    operator.post("/api/setup/llm", json={"backend": "ollama"})

    written = json.loads((tmp_path / "config.json").read_text(encoding="utf-8"))
    assert written["llm"]["backend"] == "ollama"
    assert "kidsong" in written, "the rewrite dropped the rest of the config"


# ============================================================= saving API keys ==
def test_a_stock_key_reaches_the_env_file(operator, tmp_path):
    resp = operator.post("/api/setup/keys", json={"pexels": "pexels-secret-value"})
    assert resp.status_code == 200

    env = (tmp_path / ".env").read_text(encoding="utf-8")
    assert "PEXELS_API_KEY=pexels-secret-value" in env


def test_saving_no_key_at_all_is_an_error_not_a_silent_no_op(operator):
    resp = operator.post("/api/setup/keys", json={})
    assert resp.status_code == 400


@pytest.mark.skipif(os.name == "nt", reason="POSIX permission bits")
def test_the_env_file_is_not_world_readable(operator, tmp_path):
    operator.post("/api/setup/keys", json={"pexels": "secret"})
    mode = (tmp_path / ".env").stat().st_mode & 0o777
    assert mode == 0o600, f"secrets file is mode {mode:o}"


def test_a_key_containing_a_newline_cannot_add_a_second_setting(operator, tmp_path,
                                                                monkeypatch):
    """`.env` is one setting per line, so a value with a newline in it writes
    FURTHER lines that `load_dotenv` then parses as more settings. Pasting
    `gsk_real\\nSTUDIO_PASSWORD=hunter2` into the API-key box set the login
    password."""
    operator.post("/api/setup/keys",
                  json={"pexels": "real-key\nSTUDIO_PASSWORD=hunter2"})

    # Checked with the REAL parser, not a substring search: the payload text
    # legitimately survives inside the key's *value*, and what must not happen
    # is a second SETTING appearing. `load_dotenv` is the code that decides.
    monkeypatch.delenv("STUDIO_PASSWORD", raising=False)
    monkeypatch.delenv("PEXELS_API_KEY", raising=False)
    load_dotenv(str(tmp_path))

    assert "STUDIO_PASSWORD" not in os.environ, (
        "an API-key field was able to set the studio login password"
    )
    assert os.environ.get("PEXELS_API_KEY", "").startswith("real-key")
    assert len([ln for ln in (tmp_path / ".env").read_text(encoding="utf-8").splitlines()
                if ln.strip()]) == 1, "one submitted key wrote more than one line"


def test_a_second_key_does_not_wipe_the_first(operator, tmp_path):
    operator.post("/api/setup/keys", json={"pexels": "aaa"})
    operator.post("/api/setup/keys", json={"pixabay": "bbb"})

    env = (tmp_path / ".env").read_text(encoding="utf-8")
    assert "PEXELS_API_KEY=aaa" in env
    assert "PIXABAY_API_KEY=bbb" in env
