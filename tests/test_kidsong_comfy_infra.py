"""Unit tests for Contract 2 (pipeline.kidsong.comfy.ComfyClient): detecting a
dead ComfyUI server fast instead of polling out the full render_timeout, and
the self-heal restart path.

CPU-only, no GPU, no network, no real ComfyUI: requests.get/time.time/
time.sleep are all monkeypatched.
"""
import requests
import pytest

from pipeline.kidsong.comfy import ComfyClient, ComfyUnreachableError


def _client(comfy_overrides=None):
    cfg = {"comfy": {"render_timeout": 900, **(comfy_overrides or {})}}
    return ComfyClient(cfg)


# ------------------------------------------------------------- _poll_history --
def test_poll_history_raises_unreachable_at_30s_not_render_timeout(monkeypatch):
    client = _client({"render_timeout": 900})

    fake_now = [1_000_000.0]

    def fake_time():
        return fake_now[0]

    def fake_sleep(seconds):
        fake_now[0] += seconds

    def always_down(url, timeout=None):
        raise requests.ConnectionError("connection refused")

    monkeypatch.setattr("pipeline.kidsong.comfy.time.time", fake_time)
    monkeypatch.setattr("pipeline.kidsong.comfy.time.sleep", fake_sleep)
    monkeypatch.setattr("pipeline.kidsong.comfy.requests.get", always_down)

    start = fake_now[0]
    with pytest.raises(ComfyUnreachableError):
        client._poll_history("prompt-1")

    elapsed = fake_now[0] - start
    # Must give up well before the 900s render_timeout -- specifically once
    # 30s of consecutive unreachable time have elapsed.
    assert 30 <= elapsed < 60


def test_poll_history_reachable_but_unfinished_keeps_polling(monkeypatch):
    """A server that answers (just hasn't finished the render) must NOT trip
    ComfyUnreachableError, even past the 30s threshold -- it should keep
    polling until it either completes or hits the real render_timeout."""
    client = _client({"render_timeout": 40})

    fake_now = [2_000_000.0]

    def fake_time():
        return fake_now[0]

    def fake_sleep(seconds):
        fake_now[0] += seconds

    def always_pending(url, timeout=None):
        class R:
            status_code = 200

            def json(self):
                return {}  # no entry for this prompt_id yet -- still running

        return R()

    monkeypatch.setattr("pipeline.kidsong.comfy.time.time", fake_time)
    monkeypatch.setattr("pipeline.kidsong.comfy.time.sleep", fake_sleep)
    monkeypatch.setattr("pipeline.kidsong.comfy.requests.get", always_pending)

    with pytest.raises(TimeoutError) as excinfo:
        client._poll_history("prompt-2")
    assert not isinstance(excinfo.value, ComfyUnreachableError)


def test_poll_history_recovers_after_a_transient_blip(monkeypatch):
    """A handful of connection failures under the 30s threshold, followed by
    a normal completion, must return normally -- not every failure counts
    toward an unbroken 30s run."""
    client = _client({"render_timeout": 900})

    fake_now = [3_000_000.0]
    calls = {"n": 0}

    def fake_time():
        return fake_now[0]

    def fake_sleep(seconds):
        fake_now[0] += seconds

    def flaky_then_ok(url, timeout=None):
        calls["n"] += 1
        if calls["n"] <= 3:  # 3 failures * 2s sleep each = 6s, well under 30s
            raise requests.ConnectionError("blip")

        class R:
            status_code = 200

            def json(self):
                return {"prompt-3": {"status": {"completed": True}, "outputs": {"x": 1}}}

        return R()

    monkeypatch.setattr("pipeline.kidsong.comfy.time.time", fake_time)
    monkeypatch.setattr("pipeline.kidsong.comfy.time.sleep", fake_sleep)
    monkeypatch.setattr("pipeline.kidsong.comfy.requests.get", flaky_then_ok)

    entry = client._poll_history("prompt-3")
    assert entry["status"]["completed"] is True


# ------------------------------------------------------------- restart_if_hung --
def test_restart_if_hung_returns_false_without_a_managed_path():
    client = _client()
    assert client.path is None
    assert client.restart_if_hung() is False


def test_restart_if_hung_kills_port_owner_then_ensures_up_when_not_owned(monkeypatch):
    client = _client({"path": "/fake/ComfyUI"})
    assert client._proc is None  # we did not start this server ourselves

    calls = []
    monkeypatch.setattr(client, "_kill_port_owner", lambda: calls.append("kill"))
    monkeypatch.setattr(client, "ensure_up", lambda: calls.append("ensure_up"))
    monkeypatch.setattr(client, "stop", lambda: calls.append("stop"))

    assert client.restart_if_hung() is True
    assert calls == ["kill", "ensure_up"]


def test_restart_if_hung_stops_our_own_process_when_owned(monkeypatch):
    client = _client({"path": "/fake/ComfyUI"})
    client._proc = object()  # sentinel: this client started the server

    calls = []
    monkeypatch.setattr(client, "_kill_port_owner", lambda: calls.append("kill"))
    monkeypatch.setattr(client, "ensure_up", lambda: calls.append("ensure_up"))
    monkeypatch.setattr(client, "stop", lambda: calls.append("stop"))

    assert client.restart_if_hung() is True
    assert calls == ["stop", "ensure_up"]


# ----------------------------------------------- ensure_up clears a wedged port --
class _FakeProc:
    """Minimal Popen stand-in: 'running' (poll() is None) so ensure_up's startup
    loop reaches the _alive() check on the next tick."""

    returncode = None

    def poll(self):
        return None


def _stub_autostart(monkeypatch, client, alive_seq, port_in_use):
    """Wire ensure_up for a no-network autostart: _alive() walks alive_seq
    (False first = 'down', then True once the fresh server answers), the port
    occupancy is fixed, and Popen/time are stubbed. Returns the call log."""
    calls = []
    seq = iter(alive_seq)
    monkeypatch.setattr(client, "_alive", lambda: next(seq))
    monkeypatch.setattr(client, "_port_in_use", lambda: port_in_use)
    monkeypatch.setattr(client, "_kill_port_owner", lambda: calls.append("kill"))
    monkeypatch.setattr(client, "stop", lambda: calls.append("stop"))

    def fake_popen(cmd, **kw):
        calls.append("launch")
        return _FakeProc()

    monkeypatch.setattr("pipeline.kidsong.comfy.subprocess.Popen", fake_popen)
    monkeypatch.setattr("pipeline.kidsong.comfy.time.sleep", lambda *_a: None)
    monkeypatch.setattr("pipeline.kidsong.comfy.os.makedirs", lambda *_a, **_k: None)
    # open() is called for the log file; point it at os.devnull.
    import builtins

    real_open = builtins.open
    monkeypatch.setattr(
        "pipeline.kidsong.comfy.open",
        lambda *a, **k: real_open(__import__("os").devnull, "ab"),
        raising=False,
    )
    return calls


def test_ensure_up_clears_a_wedged_port_owner_before_launch(monkeypatch, tmp_path):
    """The incident: a CUDA-wedged ComfyUI holds the port (500, so _alive False).
    ensure_up must kill that owner BEFORE launching, else the new process hits
    'port already in use' and the job fails."""
    client = _client({"path": str(tmp_path), "startup_timeout": 5})
    # down on entry, up after the fresh launch:
    calls = _stub_autostart(monkeypatch, client, [False, True], port_in_use=True)
    client.ensure_up()
    assert calls == ["kill", "launch"], calls  # cleared the wedged owner first


def test_ensure_up_does_not_kill_on_a_cold_port(monkeypatch, tmp_path):
    """Cold start (nothing on the port): no kill attempt -- behaviour unchanged."""
    client = _client({"path": str(tmp_path), "startup_timeout": 5})
    calls = _stub_autostart(monkeypatch, client, [False, True], port_in_use=False)
    client.ensure_up()
    assert calls == ["launch"], calls
    assert "kill" not in calls and "stop" not in calls


def test_ensure_up_stops_our_own_wedged_proc(monkeypatch, tmp_path):
    """If THIS client started the (now wedged) server, stop() it rather than
    hunting the port owner."""
    client = _client({"path": str(tmp_path), "startup_timeout": 5})
    client._proc = _FakeProc()
    calls = _stub_autostart(monkeypatch, client, [False, True], port_in_use=True)
    client.ensure_up()
    assert calls == ["stop", "launch"], calls
