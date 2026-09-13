import json
import os

import pytest

from studio import setup


def test_write_env_creates_and_updates(tmp_path, monkeypatch):
    root = str(tmp_path)
    setup.write_env(root, {"GROQ_API_KEY": "abc123"})
    assert setup.read_env(root)["GROQ_API_KEY"] == "abc123"
    assert os.environ["GROQ_API_KEY"] == "abc123"

    setup.write_env(root, {"GROQ_API_KEY": "xyz789"})
    env = setup.read_env(root)
    assert env["GROQ_API_KEY"] == "xyz789"
    content = (tmp_path / ".env").read_text(encoding="utf-8")
    assert content.count("GROQ_API_KEY") == 1


def test_write_env_preserves_foreign_lines(tmp_path):
    (tmp_path / ".env").write_text("# my comment\nOTHER=keepme\n", encoding="utf-8")
    setup.write_env(str(tmp_path), {"PEXELS_API_KEY": "p1"})
    content = (tmp_path / ".env").read_text(encoding="utf-8")
    assert "# my comment" in content
    assert "OTHER=keepme" in content
    assert "PEXELS_API_KEY=p1" in content


def test_save_client_secret_accepts_desktop_json(tmp_path):
    payload = json.dumps({"installed": {"client_id": "x"}}).encode()
    path = setup.save_client_secret(str(tmp_path), payload)
    assert os.path.exists(path)


def test_save_client_secret_rejects_garbage(tmp_path):
    with pytest.raises(ValueError):
        setup.save_client_secret(str(tmp_path), b"not json")
    with pytest.raises(ValueError):
        setup.save_client_secret(str(tmp_path), json.dumps({"foo": 1}).encode())
    assert not os.path.exists(tmp_path / "client_secret.json")


def test_llm_check_groq_key(monkeypatch):
    cfg = {"llm": {"backend": "groq"}}
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    assert setup._check_llm(cfg)["ok"] is False
    monkeypatch.setenv("GROQ_API_KEY", "k")
    assert setup._check_llm(cfg)["ok"] is True


def test_set_llm_backend_creates_config_and_patches(tmp_path):
    example = {"llm": {"backend": "groq", "_backend_options": "groq | ollama"}}
    (tmp_path / "config.example.json").write_text(json.dumps(example), encoding="utf-8")
    cfg = {"_root": str(tmp_path)}

    setup.set_llm_backend(cfg, "ollama")

    written = json.loads((tmp_path / "config.json").read_text(encoding="utf-8"))
    assert written["llm"]["backend"] == "ollama"
    assert written["llm"]["_backend_options"] == "groq | ollama"  # comments kept


def test_set_llm_backend_rejects_unknown(tmp_path):
    with pytest.raises(ValueError):
        setup.set_llm_backend({"_root": str(tmp_path)}, "openai")


def test_setup_status_shape(tmp_db, tmp_path, monkeypatch):
    monkeypatch.delenv("PEXELS_API_KEY", raising=False)
    monkeypatch.delenv("PIXABAY_API_KEY", raising=False)
    monkeypatch.setenv("GROQ_API_KEY", "k")
    cfg = {"_root": str(tmp_path), "llm": {"backend": "groq"}}
    s = setup.setup_status(cfg)
    assert s["llm"]["ok"] is True
    assert s["client_secret"]["ok"] is False
    assert s["channels"]["count"] == 0
    assert s["gameplay"]["ok"] is False
    assert isinstance(s["complete"], bool)


# ------------------------------------------------------ .env value hygiene --
def test_a_newline_in_a_value_cannot_inject_another_setting(tmp_path):
    """`.env` is KEY=VALUE per line, so a value with a newline does not store a
    multi-line value — it writes extra lines that read_env/load_dotenv then
    parse as further settings. Every value here comes from a browser field, so
    pasting `abc\\nSTUDIO_PASSWORD=hunter2` into the API-key box would have set
    the studio's login password. The views only `.strip()`, which catches a
    trailing newline and nothing in the middle."""
    setup.write_env(str(tmp_path), {
        "GROQ_API_KEY": "gsk_real\nSTUDIO_PASSWORD=hunter2",
    })

    env = setup.read_env(str(tmp_path))
    assert "STUDIO_PASSWORD" not in env
    assert "\n" not in env["GROQ_API_KEY"]
    assert env["GROQ_API_KEY"].startswith("gsk_real")


@pytest.mark.parametrize("hostile", [
    "a\rSTUDIO_PASSWORD=x",
    "a\r\nSTUDIO_PASSWORD=x",
    "a STUDIO_PASSWORD=x",
])
def test_every_line_separator_is_neutralised(tmp_path, hostile):
    setup.write_env(str(tmp_path), {"PEXELS_API_KEY": hostile})
    assert "STUDIO_PASSWORD" not in setup.read_env(str(tmp_path))


@pytest.mark.parametrize("bad_key", ["FOO=BAR", "FOO BAR", "1FOO", "FOO\nBAR"])
def test_a_key_that_is_not_a_variable_name_is_refused(tmp_path, bad_key):
    with pytest.raises(ValueError):
        setup.write_env(str(tmp_path), {bad_key: "x"})


def test_unrelated_lines_and_comments_survive_a_write(tmp_path):
    env_path = tmp_path / ".env"
    env_path.write_text("# my keys\nOTHER=keepme\nGROQ_API_KEY=old\n", encoding="utf-8")

    setup.write_env(str(tmp_path), {"GROQ_API_KEY": "new"})

    text = env_path.read_text(encoding="utf-8")
    assert "# my keys" in text
    assert "OTHER=keepme" in text
    assert "GROQ_API_KEY=new" in text
    assert "old" not in text


def test_writing_no_updates_leaves_the_file_alone(tmp_path):
    env_path = tmp_path / ".env"
    env_path.write_text("KEEP=1\n", encoding="utf-8")
    setup.write_env(str(tmp_path), {})
    assert env_path.read_text(encoding="utf-8") == "KEEP=1\n"


# ------------------------------------------------- gameplay starter default --
def test_a_blank_url_falls_back_to_the_starter_pack(tmp_path, monkeypatch):
    """A whitespace-only URL survived `urls or DEFAULT` (a list holding "  " is
    truthy) and then filtered to nothing, so the run finished instantly as
    "error" with no message."""
    started = {}

    class _FakeThread:
        def __init__(self, target=None, args=(), **kw):
            started["urls"] = args[1]

        def start(self):
            pass

    monkeypatch.setattr(setup.threading, "Thread", _FakeThread)
    setup._dl_state.update({"status": "idle", "done": 0, "total": 0, "error": None})

    assert setup.start_gameplay_download({"_root": str(tmp_path)}, urls=["   "]) is True
    assert started["urls"] == setup.DEFAULT_GAMEPLAY_URLS
    assert setup.gameplay_download_status()["total"] == len(setup.DEFAULT_GAMEPLAY_URLS)


# ------------------------------------------------------ credential file mode --
@pytest.mark.skipif(os.name == "nt", reason="POSIX file modes only")
def test_env_is_owner_only(tmp_path):
    """.env holds every API key and the studio password."""
    setup.write_env(str(tmp_path), {"GROQ_API_KEY": "secret"})
    mode = (tmp_path / ".env").stat().st_mode & 0o777
    assert mode == 0o600, oct(mode)


@pytest.mark.skipif(os.name == "nt", reason="POSIX file modes only")
def test_a_saved_oauth_token_is_owner_only(tmp_path, monkeypatch):
    """A refresh token is a live credential — anyone holding the file can
    upload to that YouTube channel."""
    from studio import oauth

    class _Creds:
        def to_json(self):
            return '{"refresh_token": "1//real"}'

    class _Flow:
        @staticmethod
        def from_client_secrets_file(path, scopes):
            return _Flow()

        def run_local_server(self, port=0):
            return _Creds()

    secret = tmp_path / "client_secret.json"
    secret.write_text("{}", encoding="utf-8")

    import sys
    import types

    flow_mod = types.ModuleType("google_auth_oauthlib.flow")
    flow_mod.InstalledAppFlow = _Flow
    disc_mod = types.ModuleType("googleapiclient.discovery")
    disc_mod.build = lambda *a, **k: types.SimpleNamespace(
        channels=lambda: types.SimpleNamespace(
            list=lambda **kw: types.SimpleNamespace(
                execute=lambda: {"items": [{"id": "UC123", "snippet": {"title": "T"}}]}
            )
        )
    )
    monkeypatch.setitem(sys.modules, "google_auth_oauthlib.flow", flow_mod)
    monkeypatch.setitem(sys.modules, "googleapiclient.discovery", disc_mod)
    monkeypatch.setattr(oauth.models, "get_channel_by_yt_id", lambda *a: None)
    monkeypatch.setattr(oauth.models, "create_channel", lambda **k: 1)

    oauth._run_flow({"_root": str(tmp_path)}, str(secret))

    assert oauth.connect_status()["status"] == "done", oauth.connect_status()["error"]
    token = tmp_path / "tokens" / "UC123.json"
    assert token.exists()
    mode = token.stat().st_mode & 0o777
    assert mode == 0o600, oct(mode)
