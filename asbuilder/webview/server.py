"""
Serve a local copy of VibeMol (https://github.com/evangelistalab/vibemol,
MIT licensed) for QWebEngineView to load.

VibeMol is "deployable from the repository root" per its own README (a
static site). On first launch the app downloads a copy to
`~/.asbuilder/vibemol`; after that the viewer works with no internet
connection.

Developers may also vendor a copy into `asbuilder/webview/vendor/vibemol/`;
the user-configured download takes precedence.
"""

from __future__ import annotations

import functools
import http.server
import threading
from pathlib import Path

from asbuilder.config import VIBEMOL_DIR

VENDORED_VIBEMOL_DIR = Path(__file__).parent / "vendor" / "vibemol"


def _default_vibemol_root() -> Path:
    """Prefer the user-config download (~/.asbuilder/vibemol), fall back to vendored."""
    return VIBEMOL_DIR if VIBEMOL_DIR.exists() else VENDORED_VIBEMOL_DIR


class VibeMolServer:
    """A background HTTP server for a local static VibeMol build.

    Usage:
        server = VibeMolServer()
        server.start()
        view.load(QUrl(server.url()))
        ...
        server.stop()
    """

    def __init__(self, root: str | Path | None = None, port: int = 0) -> None:
        self.root = Path(root) if root is not None else _default_vibemol_root()
        self.available = (self.root / "index.html").exists()
        self._requested_port = port
        self._httpd: http.server.ThreadingHTTPServer | None = None
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        if not self.available:
            return
        handler = functools.partial(http.server.SimpleHTTPRequestHandler, directory=str(self.root))
        self._httpd = http.server.ThreadingHTTPServer(("127.0.0.1", self._requested_port), handler)
        self._thread = threading.Thread(target=self._httpd.serve_forever, daemon=True)
        self._thread.start()

    @property
    def port(self) -> int:
        if self._httpd is None:
            raise RuntimeError("server not started")
        return self._httpd.server_address[1]

    def url(self, path: str = "/") -> str:
        return f"http://127.0.0.1:{self.port}{path}"

    def stop(self) -> None:
        if self._httpd is not None:
            self._httpd.shutdown()
            self._httpd.server_close()
            self._httpd = None
