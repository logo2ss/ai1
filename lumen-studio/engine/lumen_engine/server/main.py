"""Entry point: ``python -m lumen_engine.server.main [--port 0] [--static DIR]``.

Prints ``LUMEN_READY {"port": ..., "token": ...}`` on stdout once listening,
which the Electron shell waits for.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import socket
import sys

os.environ.setdefault("GLOG_minloglevel", "2")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")


def main() -> None:
    import faulthandler

    faulthandler.enable()  # if native code ever crashes, print where (for bug reports)
    ap = argparse.ArgumentParser(description="Lumen Studio local engine")
    ap.add_argument("--port", type=int, default=int(os.environ.get("LUMEN_PORT", 0)))
    ap.add_argument("--static", default=os.environ.get("LUMEN_STATIC"))
    ap.add_argument("--token", default=os.environ.get("LUMEN_TOKEN"))
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    import uvicorn

    from .. import store
    from .app import TOKEN, create_app

    store.cleanup_older_than()
    app = create_app(args.static, args.token)
    token = args.token or TOKEN
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.bind(("127.0.0.1", args.port))  # loopback only - never reachable from the network
    port = sock.getsockname()[1]
    print("LUMEN_READY " + json.dumps({"port": port, "token": token}), flush=True)
    config = uvicorn.Config(app, log_level="warning", access_log=False)
    server = uvicorn.Server(config)
    try:
        server.run(sockets=[sock])
    finally:
        from .. import mpworker

        mpworker.shutdown()


if __name__ == "__main__":
    sys.exit(main())
