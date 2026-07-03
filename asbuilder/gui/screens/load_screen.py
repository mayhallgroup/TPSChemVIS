"""
Screen 1: Load .chk or resume from any saved intermediate.

Saved checkpoints detected automatically from the project folder:
  - active_space/h0.npy + clusters.json  → "Jump to CMF"
  - cmf/cmf_result.jld2                  → "Jump to TPSCI / Export"

The user can also browse for files from a *different* project folder using
the manual jump buttons.
"""

from __future__ import annotations

from pathlib import Path

from PyQt6.QtCore import pyqtSignal
from PyQt6.QtWidgets import (
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPlainTextEdit,
    QPushButton,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)

from asbuilder.gui.screens.new_calc_dialog import NewCalculationDialog


class LoadScreen(QWidget):
    chk_selected      = pyqtSignal(str)   # .chk path → full pipeline from start
    continue_requested = pyqtSignal()

    # Jump signals carry the paths needed by the target screen.
    jump_to_cmf    = pyqtSignal(str, str)  # (active_space_dir, clusters_json)
    jump_to_export = pyqtSignal(str)        # cmf_result.jld2 path

    def __init__(self, project_root: Path, parent=None) -> None:
        super().__init__(parent)
        self._project_root = project_root
        self._chk_path: Path | None = None

        title = QLabel("<h2>Load Calculation</h2>")

        load_btn = QPushButton("Load existing .chk / .fchk...")
        load_btn.clicked.connect(self._on_load_existing)
        new_btn = QPushButton("New Calculation (run PySCF)...")
        new_btn.clicked.connect(self._on_new_calculation)
        start_row = QHBoxLayout()
        start_row.addWidget(load_btn)
        start_row.addWidget(new_btn)

        self._path_label = QLabel("No checkpoint loaded.")

        self._prov_box = QGroupBox("Provenance (read-only)")
        prov_content = QWidget()
        self._prov_form = QFormLayout(prov_content)
        self._prov_labels: dict[str, QLabel | QPlainTextEdit] = {}
        for key in ("formula", "basis", "charge", "spin", "n_atoms",
                    "n_orb", "n_electrons", "e_tot", "method"):
            if key == "formula":
                widget: QLabel | QPlainTextEdit = QPlainTextEdit("-")
                widget.setReadOnly(True)
                widget.setFixedHeight(110)
                widget.setLineWrapMode(QPlainTextEdit.LineWrapMode.NoWrap)
                font = widget.font()
                font.setFamily("monospace")
                widget.setFont(font)
            else:
                widget = QLabel("-")
            self._prov_labels[key] = widget
            self._prov_form.addRow(key, widget)

        prov_scroll = QScrollArea()
        prov_scroll.setWidget(prov_content)
        prov_scroll.setWidgetResizable(True)
        prov_scroll.setMaximumHeight(300)
        prov_box_layout = QVBoxLayout(self._prov_box)
        prov_box_layout.addWidget(prov_scroll)

        self._continue_btn = QPushButton("Continue → Generate Molden")
        self._continue_btn.setEnabled(False)
        self._continue_btn.clicked.connect(self.continue_requested.emit)

        # --- resume / jump section ---
        resume_group = QGroupBox("Resume from saved intermediate")
        resume_layout = QVBoxLayout(resume_group)

        self._stage_label = QLabel("Scanning project folder...")
        resume_layout.addWidget(self._stage_label)

        self._jump_cmf_auto_btn = QPushButton(
            "▶  Jump to CMF  (active_space/ + clusters.json)"
        )
        self._jump_cmf_auto_btn.setEnabled(False)
        self._jump_cmf_auto_btn.clicked.connect(self._on_jump_cmf_auto)

        self._jump_export_auto_btn = QPushButton(
            "▶  Jump to TPSCI / Export  (cmf/cmf_result.jld2)"
        )
        self._jump_export_auto_btn.setEnabled(False)
        self._jump_export_auto_btn.clicked.connect(self._on_jump_export_auto)

        resume_layout.addWidget(self._jump_cmf_auto_btn)
        resume_layout.addWidget(self._jump_export_auto_btn)

        resume_layout.addWidget(QLabel("Or browse manually (different project folder):"))
        browse_cmf_btn = QPushButton("Browse active_space/ directory → Jump to CMF...")
        browse_cmf_btn.clicked.connect(self._on_browse_cmf)
        browse_export_btn = QPushButton("Browse cmf_result.jld2 → Jump to TPSCI...")
        browse_export_btn.clicked.connect(self._on_browse_export)
        resume_layout.addWidget(browse_cmf_btn)
        resume_layout.addWidget(browse_export_btn)

        # Scrollable body — keeps the Continue button always visible even on
        # small screens or when the formula/geometry block is very long.
        body = QWidget()
        body_layout = QVBoxLayout(body)
        body_layout.addWidget(title)
        body_layout.addLayout(start_row)
        body_layout.addWidget(self._path_label)
        body_layout.addWidget(self._prov_box)
        body_layout.addWidget(resume_group)
        body_layout.addStretch(1)

        scroll = QScrollArea()
        scroll.setWidget(body)
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QScrollArea.Shape.NoFrame)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 4)
        layout.addWidget(scroll, stretch=1)
        layout.addWidget(self._continue_btn)
        self.setLayout(layout)
        self.setMinimumSize(0, 0)

        self._refresh_resume_buttons()

    # ------------------------------------------------------------------
    # Public

    def show_provenance(self, provenance: dict) -> None:
        for key, widget in self._prov_labels.items():
            value = str(provenance.get(key, "-"))
            if isinstance(widget, QPlainTextEdit):
                widget.setPlainText(value)
            else:
                widget.setText(value)
        self._continue_btn.setEnabled(True)

    def refresh(self) -> None:
        """Re-scan project folder (call after any step completes)."""
        self._refresh_resume_buttons()

    # ------------------------------------------------------------------
    # Internal

    def _refresh_resume_buttons(self) -> None:
        root = self._project_root
        has_integrals = (
            (root / "active_space" / "h0.npy").exists() and
            (root / "active_space" / "h1.npy").exists() and
            (root / "clusters.json").exists()
        )
        has_cmf = (root / "cmf" / "cmf_result.jld2").exists()

        self._jump_cmf_auto_btn.setEnabled(has_integrals)
        self._jump_export_auto_btn.setEnabled(has_cmf)

        if has_cmf:
            self._stage_label.setText(
                "Found: active_space/ + cmf/cmf_result.jld2 — can jump to CMF or TPSCI/Export"
            )
        elif has_integrals:
            self._stage_label.setText(
                "Found: active_space/ + clusters.json — can jump to CMF"
            )
        else:
            self._stage_label.setText("No intermediate files found in project folder.")

    # ------------------------------------------------------------------
    # Slots

    def _set_chk(self, path: str | Path) -> None:
        self._chk_path = Path(path)
        self._path_label.setText(f"Checkpoint: {self._chk_path}")
        self.chk_selected.emit(str(self._chk_path))

    def _on_load_existing(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Load PySCF checkpoint", "",
            "PySCF checkpoint (*.chk *.fchk);;All files (*)"
        )
        if path:
            self._set_chk(path)

    def _on_new_calculation(self) -> None:
        default_path = self._project_root / "input.chk"
        dialog = NewCalculationDialog(default_path, parent=self)
        if dialog.exec() and dialog.result_chk_path is not None:
            self._set_chk(dialog.result_chk_path)

    def _on_jump_cmf_auto(self) -> None:
        root = self._project_root
        self.jump_to_cmf.emit(
            str(root / "active_space"),
            str(root / "clusters.json"),
        )

    def _on_jump_export_auto(self) -> None:
        self.jump_to_export.emit(
            str(self._project_root / "cmf" / "cmf_result.jld2")
        )

    def _on_browse_cmf(self) -> None:
        d = QFileDialog.getExistingDirectory(
            self, "Select active_space/ directory (containing h0/h1/h2.npy)", ""
        )
        if not d:
            return
        clusters_json = Path(d).parent / "clusters.json"
        if not clusters_json.exists():
            path, _ = QFileDialog.getOpenFileName(
                self, "Select clusters.json", str(Path(d).parent),
                "JSON (*.json);;All files (*)"
            )
            if not path:
                return
            clusters_json = Path(path)
        self.jump_to_cmf.emit(d, str(clusters_json))

    def _on_browse_export(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Select cmf_result.jld2", "",
            "JLD2 files (*.jld2);;All files (*)"
        )
        if path:
            self.jump_to_export.emit(path)
