"""Expose a localhost-only service on the LAN, without restarting it.

Brainrot Studio binds 127.0.0.1 by default in older/running processes, which
makes the review queue unreachable from a phone on the same WLAN. Restarting
the app to rebind is not always acceptable — the scheduler renders episodes
*inside* that process, so a restart throws away an in-progress render.

This forwards 0.0.0.0:<listen> to 127.0.0.1:<target> so the phone can reach it
now. It is a plain TCP relay: no TLS, no auth, no rewriting. It trusts the
local network exactly as much as binding the app to 0.0.0.0 would.

    python tools/lan_bridge.py                 # 5001 -> 5000
    python tools/lan_bridge.py --listen 8080 --target 5000
"""
import argparse
import socket
import sys
import threading

_BUF = 65536


def _pump(src, dst):
    """Copy one direction until it closes, then half-close the far side."""
    try:
        while True:
            data = src.recv(_BUF)
            if not data:
                break
            dst.sendall(data)
    except OSError:
        pass
    finally:
        # Half-close so the peer sees EOF instead of hanging on a dead socket.
        for sock in (src, dst):
            try:
                sock.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass


def _serve_client(client, target_host, target_port):
    try:
        upstream = socket.create_connection((target_host, target_port), timeout=10)
    except OSError as exc:
        # The app is down or still starting; drop this connection quietly
        # rather than killing the bridge.
        print(f"  upstream unreachable ({exc}) — dropping connection", flush=True)
        try:
            client.close()
        except OSError:
            pass
        return
    threading.Thread(target=_pump, args=(client, upstream), daemon=True).start()
    threading.Thread(target=_pump, args=(upstream, client), daemon=True).start()


def lan_addresses():
    """LAN IPv4 addresses of this machine, best effort (see app.py twin)."""
    found = []
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            s.connect(("8.8.8.8", 80))
            found.append(s.getsockname()[0])
        finally:
            s.close()
    except OSError:
        pass
    return found


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--listen", type=int, default=5001, help="port to expose on the LAN")
    ap.add_argument("--target", type=int, default=5000, help="localhost port to forward to")
    ap.add_argument("--target-host", default="127.0.0.1")
    args = ap.parse_args(argv)

    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    try:
        server.bind(("0.0.0.0", args.listen))
    except OSError as exc:
        print(f"cannot bind 0.0.0.0:{args.listen}: {exc}", file=sys.stderr)
        return 1
    server.listen(64)

    for ip in lan_addresses():
        print(f"  Reachable from your phone at  http://{ip}:{args.listen}", flush=True)
    print(f"  forwarding -> {args.target_host}:{args.target}\n", flush=True)

    try:
        while True:
            client, _addr = server.accept()
            _serve_client(client, args.target_host, args.target)
    except KeyboardInterrupt:
        pass
    finally:
        server.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
