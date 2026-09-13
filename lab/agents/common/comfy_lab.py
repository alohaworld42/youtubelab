"""Gemeinsame Basis aller Lab-Agenten.

- LabClient: duenner Subclass von pipeline.kidsong.comfy.ComfyClient (orca-Worktree),
  akzeptiert fertig gebaute API-Graphen (dict) statt nur Workflow-Namen.
- lab_config(): laedt lab.json, setzt HF_HOME.
- run_paths(): Ordner eines Runs (runs/<id>/...), legt sie an.
- state: append-only JSON (runs/<id>/state.json) via pipeline.atomicio.
- gpu_lock(): pipeline.gpu_lock.GPULock auf output/gpu_render.lock der Pipeline.
"""
import copy
import json
import os
import sys
import time
import logging

LAB_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
LAB_JSON = os.path.join(LAB_DIR, "lab.json")

log = logging.getLogger("lab")


def lab_config():
    with open(LAB_JSON, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    cfg["lab_dir"] = LAB_DIR
    os.environ.setdefault("HF_HOME", cfg["hf_home"])
    os.environ.setdefault("PYTHONUTF8", "1")
    if cfg["orca_root"] not in sys.path:
        sys.path.insert(0, cfg["orca_root"])
    return cfg


def setup_logging(run_dir=None, name="lab"):
    handlers = [logging.StreamHandler(sys.stdout)]
    if run_dir:
        os.makedirs(run_dir, exist_ok=True)
        handlers.append(logging.FileHandler(os.path.join(run_dir, "run.log"), encoding="utf-8"))
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s",
                        handlers=handlers, force=True)
    return logging.getLogger(name)


def run_paths(cfg, run_id, create=True):
    root = os.path.join(cfg["lab_dir"], "runs", run_id)
    p = {
        "root": root,
        "shots": os.path.join(root, "shots.json"),
        "state": os.path.join(root, "state.json"),
        "graphs": os.path.join(root, "graphs"),
        "keyframes": os.path.join(root, "keyframes"),
        "clips": os.path.join(root, "clips_V2"),
        "qc": os.path.join(root, "qc"),
        "final": os.path.join(root, "final"),
        "phone": os.path.join(root, "phone"),
    }
    if create:
        for k, v in p.items():
            if k in ("shots", "state"):
                continue
            os.makedirs(v, exist_ok=True)
    return p


def done(path):
    return os.path.isfile(path) and os.path.getsize(path) > 0


# ------------------------------------------------------------------ state ---
def read_state(paths):
    if not os.path.isfile(paths["state"]):
        return {"entries": []}
    with open(paths["state"], "r", encoding="utf-8") as f:
        return json.load(f)


def mark_state(paths, **entry):
    """Append one entry (never rewrites old ones)."""
    from pipeline.atomicio import atomic_write_json
    st = read_state(paths)
    entry["ts"] = time.strftime("%Y-%m-%d %H:%M:%S")
    st["entries"].append(entry)
    atomic_write_json(paths["state"], st)
    return st


# ------------------------------------------------------------------- lock ---
def gpu_lock(cfg):
    from pipeline.gpu_lock import GPULock
    return GPULock(cfg["lock_file"])


# ----------------------------------------------------------------- client ---
def make_client(cfg, render_timeout):
    lab_config()  # sys.path
    from pipeline.kidsong.comfy import ComfyClient, ComfyUnreachableError  # noqa: F401
    import requests

    class LabClient(ComfyClient):
        def __init__(self, lab_cfg, timeout):
            super().__init__({
                "comfy": {
                    "url": lab_cfg["comfy_url"],
                    "path": lab_cfg["comfy_dir"],
                    "python": lab_cfg["python"],
                    "autostart": False,
                    "render_timeout": int(timeout),
                },
                "_root": lab_cfg["lab_dir"],
            })
            self._graph_dump_dir = None

        # Graph als dict akzeptieren (render() ruft load_workflow(name) auf).
        def load_workflow(self, workflow_name):
            if isinstance(workflow_name, dict):
                return copy.deepcopy(workflow_name)
            return super().load_workflow(workflow_name)

        def queue_idle(self):
            try:
                r = requests.get(self.url + "/queue", timeout=10)
                r.raise_for_status()
                q = r.json()
                return not q.get("queue_running") and not q.get("queue_pending")
            except requests.RequestException:
                return False

        def wait_idle(self, timeout=600):
            t0 = time.time()
            while not self.queue_idle():
                if time.time() - t0 > timeout:
                    raise RuntimeError("ComfyUI queue not idle after %ss (foreign job running?)" % timeout)
                time.sleep(5)

        def render_headless(self, graph, dump_name=None):
            """Submit a graph that has no retrievable output file (OUTPUT_NODE
            like H3SaveConditioning/H3SaveLatentAV write to disk themselves).
            POST /prompt, poll history, raise on node error. Returns elapsed s."""
            import uuid, requests as _rq
            if self._graph_dump_dir and dump_name:
                os.makedirs(self._graph_dump_dir, exist_ok=True)
                with open(os.path.join(self._graph_dump_dir, dump_name + ".json"), "w", encoding="utf-8") as f:
                    json.dump(graph, f, indent=2, ensure_ascii=False)
            self.wait_idle()
            self.validate(graph)
            t0 = time.time()
            r = _rq.post(self.url + "/prompt", json={"prompt": graph, "client_id": str(uuid.uuid4())}, timeout=30)
            if r.status_code != 200:
                raise RuntimeError("ComfyUI rejected the prompt (HTTP %d): %s" % (r.status_code, r.text))
            pid = r.json().get("prompt_id")
            if not pid:
                raise RuntimeError("no prompt_id: %s" % r.text)
            history = self._poll_history(pid)
            self._raise_on_node_error(history, pid)
            return time.time() - t0

        def render_graph(self, graph, out_path, dump_name=None, retries=1):
            """Render a fully built API graph; optional dump for traceability."""
            if self._graph_dump_dir and dump_name:
                os.makedirs(self._graph_dump_dir, exist_ok=True)
                with open(os.path.join(self._graph_dump_dir, dump_name + ".json"), "w", encoding="utf-8") as f:
                    json.dump(graph, f, indent=2, ensure_ascii=False)
            self.wait_idle()
            last = None
            for attempt in range(retries + 1):
                try:
                    t0 = time.time()
                    path = self.render(graph, {}, out_path)
                    return path, time.time() - t0
                except (TimeoutError, ComfyUnreachableError, RuntimeError) as exc:
                    last = exc
                    log.warning("render attempt %d failed: %s", attempt + 1, exc)
                    if attempt < retries:
                        self.free()
                        time.sleep(10)
            raise last

    c = LabClient(cfg, render_timeout)
    return c


def stage_ref(client, path):
    """Copy an image into ComfyUI/input and return the LoadImage name."""
    return client.stage_input_image(path)


def title(node, t):
    node["_meta"] = {"title": t}
    return node


# -------------------------------------------------------------- gpu guard ---
class gpu_guard:
    """Context manager: holds the pipeline's GPU lock (output/gpu_render.lock).
    Refuses to start if a LIVE process holds it; steals stale (dead pid) locks."""

    def __init__(self, cfg, logger=None):
        self.lock = gpu_lock(cfg)
        self.log = logger or log

    def __enter__(self):
        if not self.lock.try_acquire():
            info = self.lock._read() or {}
            raise SystemExit(f"GPU lock held by live pid {info.get('pid')} ({self.lock.path}) - not starting.")
        self.log.info("gpu lock acquired: %s", self.lock.path)
        return self

    def __exit__(self, *exc):
        self.lock.release()
        self.log.info("gpu lock released")
        return False
