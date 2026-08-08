#!/usr/bin/env python3
"""Serve the site from localhost and open it in the default browser."""
from __future__ import annotations

from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import threading
import webbrowser

ROOT = Path(__file__).resolve().parent


class Handler(SimpleHTTPRequestHandler):
    # Avoid stale profile.json/profile.js while tuning the model and alignment.
    def end_headers(self):
        self.send_header("Cache-Control", "no-store, max-age=0")
        self.send_header("Pragma", "no-cache")
        super().end_headers()


def main() -> None:
    handler = partial(Handler, directory=str(ROOT))
    # Port 0 asks the OS for a free localhost port, so another dev server does
    # not make this launcher fail.
    server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    port = server.server_address[1]
    url = f"http://127.0.0.1:{port}/"

    print(f"Serving Ray's site at {url}", flush=True)
    print("Keep this window open while testing. Press Ctrl-C to stop.", flush=True)

    def open_browser():
        try:
            webbrowser.open(url)
        except Exception as exc:
            print(f"Could not open the browser automatically: {exc}", flush=True)
            print(f"Open this URL manually: {url}", flush=True)

    threading.Timer(0.5, open_browser).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping server.")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
