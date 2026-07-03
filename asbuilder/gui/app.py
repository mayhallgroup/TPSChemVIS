"""
Entry point: `python -m asbuilder.gui.app [project_dir]`
              or `asbuilder` (after `pip install -e .`)

On first launch TPSChemVIS downloads VibeMol and bootstraps TPSChem.jl.
Config is persisted at ~/.asbuilder/config.json so subsequent launches need
no flags.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path


def _ensure_vibemol() -> None:
    """Download VibeMol to ~/.asbuilder/vibemol/ on first launch if missing."""
    from asbuilder.webview.server import _default_vibemol_root
    if (_default_vibemol_root() / "index.html").exists():
        return

    import subprocess
    from asbuilder.config import VIBEMOL_REPO, VIBEMOL_DIR
    from PyQt6.QtCore import QThread, QTimer, pyqtSignal
    from PyQt6.QtWidgets import QDialog, QLabel, QPlainTextEdit, QPushButton, QVBoxLayout

    class _Downloader(QThread):
        line = pyqtSignal(str)
        done = pyqtSignal(bool, str)

        def run(self):
            try:
                VIBEMOL_DIR.parent.mkdir(parents=True, exist_ok=True)
                if (VIBEMOL_DIR / ".git").exists():
                    cmd = ["git", "-C", str(VIBEMOL_DIR), "pull", "--ff-only"]
                else:
                    if VIBEMOL_DIR.exists():
                        self.done.emit(False, f"{VIBEMOL_DIR} exists but is not a VibeMol checkout")
                        return
                    cmd = ["git", "clone", "--depth", "1", VIBEMOL_REPO, str(VIBEMOL_DIR)]

                r = subprocess.run(cmd, capture_output=True, text=True)
                for ln in (r.stdout + r.stderr).splitlines():
                    self.line.emit(ln)
                ok = r.returncode == 0 and (VIBEMOL_DIR / "index.html").exists()
                msg = "" if ok else (r.stderr or "VibeMol checkout did not contain index.html")
                self.done.emit(ok, msg)
            except Exception as exc:
                self.done.emit(False, str(exc))

    from asbuilder.gui._screen_util import fit_to_screen
    dlg = QDialog()
    dlg.setWindowTitle("One-time setup: VibeMol orbital viewer")
    fit_to_screen(dlg, 520, 320)
    label = QLabel(
        "<b>VibeMol</b> (orbital viewer) is not installed yet.<br>"
        "Downloading a local copy now — this is a one-time step (~5 MB)."
    )
    label.setWordWrap(True)
    log = QPlainTextEdit()
    log.setReadOnly(True)
    font = log.font(); font.setFamily("monospace"); log.setFont(font)
    skip_btn = QPushButton("Skip (orbital viewer won't work this session)")
    skip_btn.clicked.connect(dlg.reject)
    layout = QVBoxLayout(dlg)
    layout.addWidget(label)
    layout.addWidget(log)
    layout.addWidget(skip_btn)

    worker = _Downloader(dlg)
    worker.line.connect(log.appendPlainText)

    def _on_done(ok: bool, err: str) -> None:
        if ok:
            label.setText("<b>VibeMol downloaded successfully.</b> Continuing…")
            skip_btn.setText("Continue →")
            skip_btn.clicked.disconnect()
            skip_btn.clicked.connect(dlg.accept)
            QTimer.singleShot(800, dlg.accept)
        else:
            label.setText(f"<b>Download failed</b> — orbital viewer won't work.<br>{err}")
            skip_btn.setText("Continue without VibeMol")

    worker.done.connect(_on_done)
    worker.start()
    dlg.exec()


def main() -> int:
    parser = argparse.ArgumentParser(description="TPS-chemistry")
    parser.add_argument(
        "project_dir",
        nargs="?",
        default=None,
        help="Project folder to open or create (default: ~/asbuilder_projects/untitled.qcproj)",
    )
    parser.add_argument("--julia-bin", default=None, help="Julia executable (default: julia)")
    parser.add_argument(
        "--julia-project",
        default=None,
        help="Override path to TPSChem.jl directory (saved to config on first use)",
    )
    parser.add_argument(
        "--vibemol-root",
        default=None,
        help="Path to vendored VibeMol static build",
    )
    parser.add_argument(
        "--setup", action="store_true",
        help="Force the TPSChem.jl setup dialog even if already configured",
    )
    args = parser.parse_args()

    from PyQt6.QtWidgets import QApplication
    import asbuilder.config as cfg
    from asbuilder.gui.main_window import MainWindow

    app = QApplication(sys.argv)
    app.setApplicationName("TPS-chemistry")
    app.setOrganizationName("asbuilder")

    # Show a dialog for unhandled exceptions instead of silently exiting
    _orig_excepthook = sys.excepthook

    def _excepthook(exc_type, exc_value, exc_tb):
        import traceback
        from PyQt6.QtWidgets import QMessageBox
        msg = "".join(traceback.format_exception(exc_type, exc_value, exc_tb))
        dlg = QMessageBox()
        dlg.setWindowTitle("Unexpected error")
        dlg.setText(f"<b>{exc_type.__name__}</b>: {exc_value}")
        dlg.setDetailedText(msg)
        dlg.setIcon(QMessageBox.Icon.Critical)
        dlg.exec()
        _orig_excepthook(exc_type, exc_value, exc_tb)

    sys.excepthook = _excepthook

    # --- one-time VibeMol download (fast, skippable) ---
    if not args.vibemol_root:
        _ensure_vibemol()

    # --- resolve julia_bin ---
    julia_bin = args.julia_bin or cfg.julia_bin()

    # --- resolve julia_project ---
    julia_project: Path | None = None
    if args.julia_project:
        julia_project = Path(args.julia_project)
        cfg.set_value("julia_project", str(julia_project))
    else:
        julia_project = cfg.julia_project()

    # Auto-detect workspace sibling as a convenience (dev layout)
    if julia_project is None:
        sibling = Path(__file__).parent.parent.parent.parent / "TPSChem.jl"
        if (sibling / "Project.toml").exists():
            julia_project = sibling
            cfg.set_value("julia_project", str(julia_project))

    # --- bootstrap TPSChem.jl if not found ---
    if julia_project is None or args.setup:
        from asbuilder.gui.screens.setup_screen import SetupDialog
        dlg = SetupDialog(julia_bin=julia_bin, auto_start=(julia_project is None and not args.setup))
        if dlg.exec() and dlg.chosen_path:
            julia_project = dlg.chosen_path
        elif julia_project is None:
            # User closed without configuring — still open the app, CMF will warn
            pass

    # --- resolve project dir ---
    if args.project_dir:
        project_dir = Path(args.project_dir)
    else:
        from asbuilder.gui.screens.project_picker import ProjectPickerDialog
        picker = ProjectPickerDialog()
        if not picker.exec() or picker.chosen_path is None:
            return 0   # user cancelled the picker → quit cleanly
        project_dir = picker.chosen_path

    cfg.add_recent_project(project_dir)

    window = MainWindow(
        project_root=project_dir,
        julia_bin=julia_bin,
        julia_project=julia_project,
        vibemol_root=args.vibemol_root,
    )
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
