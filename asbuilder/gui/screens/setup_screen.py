"""
First-launch setup dialog.

Shown once when no valid TPSChem.jl project is configured.  The user can
either point to an existing clone or let the app clone and build it.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from PyQt6.QtCore import QThread, QTimer, pyqtSignal
from PyQt6.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPlainTextEdit,
    QPushButton,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)

import asbuilder.config as cfg
from asbuilder.gui._screen_util import fit_to_screen


class _CloneWorker(QThread):
    line = pyqtSignal(str)
    done = pyqtSignal(bool, str)   # success, message

    def __init__(self, dest: Path, julia_bin: str, python_exe: str | None = None, parent=None):
        super().__init__(parent)
        self._dest = dest
        self._julia_bin = julia_bin
        self._python_exe = python_exe or sys.executable

    @staticmethod
    def _julia_string(value: str | Path) -> str:
        import json
        return json.dumps(str(value))

    def _run_logged(self, cmd: list[str], timeout: int | None = None) -> subprocess.CompletedProcess[str]:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        for l in (result.stdout + result.stderr).splitlines():
            self.line.emit(l)
        return result

    def run(self):
        try:
            self._dest.parent.mkdir(parents=True, exist_ok=True)

            # --- 1. checkout / update ---
            if (self._dest / "Project.toml").exists():
                self.line.emit(f"Using existing TPSChem.jl checkout at {self._dest}")
                if (self._dest / ".git").exists():
                    self.line.emit("Updating TPSChem.jl checkout with git pull --ff-only …")
                    result = self._run_logged(
                        ["git", "-C", str(self._dest), "pull", "--ff-only", "origin", "main"],
                        timeout=300,
                    )
                    if result.returncode != 0:
                        self.line.emit("[setup] git pull failed; continuing with existing checkout.")
            elif self._dest.exists() and any(self._dest.iterdir()):
                self.done.emit(False, f"{self._dest} exists but is not a TPSChem.jl checkout")
                return
            else:
                self.line.emit(f"Cloning {cfg.TPSCHEM_REPO} → {self._dest} …")
                result = self._run_logged(
                    ["git", "clone", cfg.TPSCHEM_REPO, str(self._dest)],
                    timeout=600,
                )
                if result.returncode != 0:
                    self.done.emit(False, "git clone failed")
                    return

            # --- 2. Pkg.instantiate ---
            self.line.emit("Running Pkg.instantiate …")
            script = (
                f'import Pkg; Pkg.activate({self._julia_string(self._dest)}); '
                'Pkg.instantiate(); println("[setup] instantiate done")'
            )
            result = self._run_logged(
                [self._julia_bin, "-e", script],
                timeout=600,
            )
            if result.returncode != 0:
                self.done.emit(False, "Pkg.instantiate failed")
                return

            # --- 3. Pkg.build("PyCall") ---
            self.line.emit(f"Building PyCall against Python: {self._python_exe}")
            script = (
                f'import Pkg; ENV["PYTHON"]={self._julia_string(self._python_exe)}; '
                f'Pkg.activate({self._julia_string(self._dest)}); '
                'Pkg.build("PyCall"); println("[setup] PyCall build done")'
            )
            result = self._run_logged(
                [self._julia_bin, "-e", script],
                timeout=300,
            )
            if result.returncode != 0:
                self.done.emit(False, "Pkg.build(PyCall) failed")
                return

            # --- 4. Pkg.precompile ---
            self.line.emit("Precompiling TPSChem.jl environment …")
            script = (
                f'import Pkg; Pkg.activate({self._julia_string(self._dest)}); '
                'Pkg.precompile(); println("[setup] precompile done")'
            )
            result = self._run_logged(
                [self._julia_bin, "-e", script],
                timeout=600,
            )
            if result.returncode != 0:
                self.done.emit(False, "Pkg.precompile failed")
                return

            self.done.emit(True, str(self._dest))
        except Exception as exc:
            self.done.emit(False, str(exc))


class SetupDialog(QDialog):
    """Modal dialog shown on first launch (or when TPSChem.jl is not found)."""

    def __init__(self, julia_bin: str = "julia", auto_start: bool = False, parent=None):
        super().__init__(parent)
        self.setWindowTitle("TPS-chemistry — First-launch setup")
        self.setMinimumWidth(560)
        fit_to_screen(self, 680, 560)
        self._julia_bin = julia_bin
        self._auto_start = auto_start
        self._worker: _CloneWorker | None = None
        self.chosen_path: Path | None = None   # set on success

        info = QLabel(
            "<b>TPSChem.jl is required to run CMF and TPSCI calculations.</b><br>"
            "Either point to an existing clone, or let the app download and build it now."
        )
        info.setWordWrap(True)

        # --- Option A: existing clone ---
        self._existing_edit = QLineEdit()
        self._existing_edit.setPlaceholderText("e.g. /Users/you/workspace/TPSChem.jl")
        browse_btn = QPushButton("Browse…")
        browse_btn.setFixedWidth(80)
        browse_btn.clicked.connect(self._browse)
        use_existing_btn = QPushButton("Use this folder")
        use_existing_btn.clicked.connect(self._use_existing)
        row_a = QHBoxLayout()
        row_a.addWidget(self._existing_edit)
        row_a.addWidget(browse_btn)
        existing_group = QGroupBox("Use existing TPSChem.jl clone")
        existing_layout = QVBoxLayout(existing_group)
        existing_layout.addLayout(row_a)
        existing_layout.addWidget(use_existing_btn)

        # --- Option B: auto-install ---
        self._install_dir_edit = QLineEdit(str(cfg.DEFAULT_INSTALL_DIR))
        browse_install_btn = QPushButton("Browse…")
        browse_install_btn.setFixedWidth(80)
        browse_install_btn.clicked.connect(self._browse_install_dir)
        dir_row = QHBoxLayout()
        dir_row.addWidget(self._install_dir_edit)
        dir_row.addWidget(browse_install_btn)

        self._install_btn = QPushButton(f"Clone from GitHub and build  ({cfg.TPSCHEM_REPO})")
        self._install_btn.clicked.connect(self._start_install)

        install_group = QGroupBox("Download and build automatically")
        install_layout = QVBoxLayout(install_group)
        install_layout.addWidget(QLabel("Install to:"))
        install_layout.addLayout(dir_row)
        install_layout.addWidget(self._install_btn)

        # --- log ---
        self._log = QPlainTextEdit()
        self._log.setReadOnly(True)
        f = self._log.font(); f.setFamily("monospace"); self._log.setFont(f)
        self._log.setMaximumHeight(180)

        # --- status + buttons ---
        self._status = QLabel("")
        self._btns = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
        self._btns.rejected.connect(self.reject)
        self._ok_btn = self._btns.addButton("Continue →", QDialogButtonBox.ButtonRole.AcceptRole)
        self._ok_btn.setEnabled(False)
        self._ok_btn.clicked.connect(self.accept)

        body = QWidget()
        body_layout = QVBoxLayout(body)
        body_layout.addWidget(info)
        body_layout.addWidget(existing_group)
        body_layout.addWidget(install_group)
        body_layout.addWidget(QLabel("Output:"))
        body_layout.addWidget(self._log)
        body_layout.addWidget(self._status)
        body_layout.addStretch(1)

        scroll = QScrollArea()
        scroll.setWidget(body)
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QScrollArea.Shape.NoFrame)

        layout = QVBoxLayout(self)
        layout.addWidget(scroll, stretch=1)
        layout.addWidget(self._btns)

        if self._auto_start:
            self._status.setText("First launch: downloading and building TPSChem.jl automatically…")
            QTimer.singleShot(0, self._start_install)

    def _browse(self):
        d = QFileDialog.getExistingDirectory(self, "Select TPSChem.jl directory")
        if d:
            self._existing_edit.setText(d)

    def _browse_install_dir(self):
        d = QFileDialog.getExistingDirectory(self, "Install TPSChem.jl into…")
        if d:
            self._install_dir_edit.setText(str(Path(d) / "TPSChem.jl"))

    def _use_existing(self):
        p = Path(self._existing_edit.text().strip())
        if not (p / "Project.toml").exists():
            self._status.setText(f"No Project.toml found in {p}")
            return
        cfg.set_value("julia_project", str(p))
        self.chosen_path = p
        self._status.setText(f"Saved: {p}")
        self._ok_btn.setEnabled(True)

    def _start_install(self):
        dest = Path(self._install_dir_edit.text().strip())
        self._install_btn.setEnabled(False)
        self._ok_btn.setEnabled(False)
        self._worker = _CloneWorker(dest, self._julia_bin, python_exe=sys.executable, parent=self)
        self._worker.line.connect(self._log.appendPlainText)
        self._worker.done.connect(self._on_done)
        self._worker.start()

    def _on_done(self, success: bool, message: str):
        self._install_btn.setEnabled(True)
        if success:
            p = Path(message)
            cfg.set_value("julia_project", str(p))
            self.chosen_path = p
            self._status.setText(f"Installed and configured: {p}")
            self._ok_btn.setEnabled(True)
        else:
            self._status.setText(f"Failed: {message}")
