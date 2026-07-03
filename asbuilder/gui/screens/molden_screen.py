"""Screen 2: convert the loaded .chk to .molden in a background thread,
then hand off (chk_contents, molden_path) to the viewer/clustering screen."""

from __future__ import annotations

from pathlib import Path

from PyQt6.QtCore import pyqtSignal
from PyQt6.QtWidgets import QLabel, QProgressBar, QPushButton, QVBoxLayout, QWidget

from asbuilder.gui.workers import MoldenWorker


class MoldenScreen(QWidget):
    molden_ready = pyqtSignal(object, str)  # (ChkContents, molden_path)
    failed = pyqtSignal(str)

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._worker: MoldenWorker | None = None

        self._label = QLabel("Ready to generate .molden from the loaded checkpoint.")
        self._progress = QProgressBar()
        self._progress.setRange(0, 0)  # indeterminate -- chk_to_molden has no fine-grained progress
        self._progress.setVisible(False)

        self._start_btn = QPushButton("Generate Molden")
        self._start_btn.clicked.connect(self._on_start)

        layout = QVBoxLayout(self)
        layout.addWidget(self._label)
        layout.addWidget(self._progress)
        layout.addWidget(self._start_btn)
        layout.addStretch(1)
        self.setLayout(layout)
        self.setMinimumSize(0, 0)

        self._chk_path: Path | None = None
        self._molden_out: Path | None = None

    def set_paths(self, chk_path: str | Path, molden_out: str | Path) -> None:
        self._chk_path = Path(chk_path)
        self._molden_out = Path(molden_out)

    def _on_start(self) -> None:
        if self._chk_path is None or self._molden_out is None:
            self._label.setText("No checkpoint path set -- go back to Load.")
            return
        self._start_btn.setEnabled(False)
        self._progress.setVisible(True)
        self._label.setText(f"Converting {self._chk_path.name}...")

        self._worker = MoldenWorker(self._chk_path, self._molden_out, parent=self)
        self._worker.finished_ok.connect(self._on_finished)
        self._worker.failed.connect(self._on_failed)
        self._worker.start()

    def _on_finished(self, result) -> None:
        chk, molden_path = result
        self._progress.setVisible(False)
        self._label.setText(f"Wrote {molden_path} ({chk.n_orb} orbitals).")
        self.molden_ready.emit(chk, str(molden_path))

    def _on_failed(self, message: str) -> None:
        self._progress.setVisible(False)
        self._start_btn.setEnabled(True)
        self._label.setText(f"Failed: {message}")
        self.failed.emit(message)
