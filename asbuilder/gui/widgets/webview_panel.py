"""
Embeds VibeMol (https://vibemol.org, MIT) via QWebEngineView for orbital
visualization. VibeMol is a pure static site — we load its index.html
directly via file:// so no HTTP server is needed.

Loading a molden file uses VibeMol's public embed API:
  window.VibeMolEmbed.loadFiles([{name, text}])
which is registered by app.js at startup via installEmbeddedMessageHandler.
We inject this call via runJavaScript after the page finishes loading.

TIER B (not implemented): QWebChannel bridge for click-to-assign on the
rendered isosurface — reserved in the orbital_clicked signal.
"""

from __future__ import annotations

import json
from pathlib import Path

from PyQt6.QtCore import QUrl, pyqtSignal
from PyQt6.QtWebEngineCore import QWebEngineSettings
from PyQt6.QtWebEngineWidgets import QWebEngineView
from PyQt6.QtWidgets import QVBoxLayout, QWidget

from asbuilder.webview.server import _default_vibemol_root

_PLACEHOLDER_HTML = """\
<body style='font-family:sans-serif;padding:2em;background:#1a1a2e;color:#aaa'>
<h3 style='color:#e0e0e0'>VibeMol not installed</h3>
<p>Restart TPSChemVIS to retry the automatic VibeMol download.</p>
<pre style='font-size:0.85em;background:#111;padding:1em;border-radius:4px'>
asbuilder
</pre>
</body>
"""

# Calls VibeMol's public embed API (window.VibeMolEmbed, set by app.js).
# Fire-and-forget: loadFiles() returns a Promise that we ignore here.
_LOAD_MOLDEN_JS = """\
(function(moldenText, filename) {
    var api = window.VibeMolEmbed;
    if (!api || typeof api.loadFiles !== 'function') {
        console.error('[asbuilder] window.VibeMolEmbed not ready');
        return false;
    }
    api.loadFiles([{name: filename, text: moldenText}]);
    return true;
})(%s, %s);
"""


class WebViewPanel(QWidget):
    orbital_clicked = pyqtSignal(int)  # reserved for Tier B bridge
    molden_loaded = pyqtSignal(bool)   # emitted after JS injection

    def __init__(self, vibemol_root: str | Path | None = None, parent=None) -> None:
        super().__init__(parent)
        root = Path(vibemol_root) if vibemol_root else _default_vibemol_root()
        self._index_html = root / "index.html"
        self._available = self._index_html.exists()
        self._page_ready = False
        self._pending_molden: Path | None = None

        self._view = QWebEngineView(self)
        # Don't let QWebEngineView enforce a large minimum size — it would
        # inflate the QStackedWidget's minimum and prevent the main window
        # from being resized smaller than the screen.
        self._view.setMinimumSize(0, 0)
        self.setMinimumSize(0, 0)

        # Allow local file access to sibling files (assets/, etc.)
        settings = self._view.settings()
        settings.setAttribute(QWebEngineSettings.WebAttribute.LocalContentCanAccessFileUrls, True)
        settings.setAttribute(QWebEngineSettings.WebAttribute.LocalContentCanAccessRemoteUrls, True)
        settings.setAttribute(QWebEngineSettings.WebAttribute.JavascriptEnabled, True)

        self._view.loadFinished.connect(self._on_load_finished)

        if self._available:
            self._view.load(QUrl.fromLocalFile(str(self._index_html)))
        else:
            self._view.setHtml(_PLACEHOLDER_HTML)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self._view)
        self.setLayout(layout)

    @property
    def available(self) -> bool:
        return self._available

    def _on_load_finished(self, ok: bool) -> None:
        self._page_ready = ok
        if ok and self._pending_molden is not None:
            self._inject_molden(self._pending_molden)
            self._pending_molden = None

    def load_molden(self, molden_path: str | Path) -> None:
        """Load a .molden file into VibeMol. Queued if the page isn't ready yet."""
        if not self._available:
            return
        path = Path(molden_path)
        if self._page_ready:
            self._inject_molden(path)
        else:
            self._pending_molden = path

    def _inject_molden(self, path: Path) -> None:
        text = path.read_text(encoding="utf-8", errors="replace")
        js = _LOAD_MOLDEN_JS % (json.dumps(text), json.dumps(path.name))
        self._view.page().runJavaScript(js, self._on_inject_result)

    def _on_inject_result(self, result) -> None:
        self.molden_loaded.emit(bool(result))

    def reload(self) -> None:
        if self._available:
            self._page_ready = False
            self._view.load(QUrl.fromLocalFile(str(self._index_html)))

    def closeEvent(self, event) -> None:  # noqa: N802
        super().closeEvent(event)
