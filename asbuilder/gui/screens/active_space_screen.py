"""
Screen 4: run localization + integrals in the background, show a log, and
(once available) re-render localized orbitals per cluster for a sanity
check before committing to h0/h1/h2.npy.

NOTE: asbuilder.active_space.localize_integrals.build_active_space is
still a stub (NotImplementedError) pending the notebook's SPADE driver
cells -- ActiveSpaceWorker's generic exception handling means that error
message surfaces directly in this screen's log pane rather than crashing,
which is deliberate: it's a clear, actionable "not wired up yet" message
instead of a silent failure.
"""

from __future__ import annotations

from pathlib import Path

from PyQt6.QtCore import pyqtSignal
from PyQt6.QtWidgets import (
    QButtonGroup,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QRadioButton,
    QVBoxLayout,
    QWidget,
)

from asbuilder.cluster.state import ClusterSet
from asbuilder.gui.widgets.log_pane import LogPane
from asbuilder.gui.workers import ActiveSpaceWorker
from asbuilder.io.chk_to_molden import ChkContents


class ActiveSpaceScreen(QWidget):
    active_space_built = pyqtSignal(str)  # emits the active_space/ output dir on success
    redo_clustering_requested = pyqtSignal()

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._worker: ActiveSpaceWorker | None = None
        self._output_dir: Path | None = None

        self._label = QLabel("Ready to build the active space from the current clusters.")

        # --- partitioning mode ---
        self._radio_manual = QRadioButton(
            "Manual  — use orbital assignments from viewer (fspace auto-derived from mo_occ)"
        )
        self._radio_spade = QRadioButton(
            "SPADE   — automatic partitioning by atom indices (for bimetallic / fragment systems)"
        )
        self._radio_manual.setChecked(True)
        self._mode_group = QButtonGroup(self)
        self._mode_group.addButton(self._radio_manual)
        self._mode_group.addButton(self._radio_spade)

        self._log = LogPane()

        self._build_btn = QPushButton("Build active space + integrals")
        self._build_btn.clicked.connect(self._on_build)
        self._back_btn = QPushButton("← Redo clustering")
        self._back_btn.clicked.connect(self.redo_clustering_requested.emit)

        btn_row = QHBoxLayout()
        btn_row.addWidget(self._back_btn)
        btn_row.addStretch(1)
        btn_row.addWidget(self._build_btn)

        layout = QVBoxLayout(self)
        layout.addWidget(self._label)
        layout.addWidget(self._radio_manual)
        layout.addWidget(self._radio_spade)
        layout.addWidget(self._log)
        layout.addLayout(btn_row)
        self.setLayout(layout)
        self.setMinimumSize(0, 0)
        self._log.setMinimumSize(0, 0)

        self._chk: ChkContents | None = None
        self._clusters: ClusterSet | None = None

    def set_inputs(self, chk: ChkContents, clusters: ClusterSet, output_dir: str | Path) -> None:
        self._chk = chk
        self._clusters = clusters
        self._output_dir = Path(output_dir)
        # Auto-select mode based on whether any cluster has atom_indices set (SPADE)
        has_atom_indices = any(c.atom_indices is not None for c in clusters.clusters)
        if has_atom_indices:
            self._radio_spade.setChecked(True)
        else:
            self._radio_manual.setChecked(True)
        self._label.setText(
            f"{'SPADE' if has_atom_indices else 'Manual'} mode detected from cluster definitions. "
            "Change below if needed."
        )

    def _on_build(self) -> None:
        if self._chk is None or self._clusters is None or self._output_dir is None:
            self._log.append_line("[active_space] no inputs set -- go back and build clusters first.")
            return

        nao = self._chk.mol.nao
        n_active = sum(len(c.orbitals) for c in self._clusters.clusters)
        if nao > 300:
            self._log.append_line(
                f"[active_space] warning: large molecule ({nao} AOs). "
                "Integral computation may take several minutes."
            )
        if n_active > 30:
            import math
            mem_mb = (n_active ** 4) * 8 / 1e6
            self._log.append_line(
                f"[active_space] warning: {n_active} active orbitals → "
                f"h2.npy ~{mem_mb:.0f} MB. Consider a smaller active space."
            )

        self._build_btn.setEnabled(False)
        self._log.append_line("[active_space] starting build_active_space...")

        chk, clusters, output_dir = self._chk, self._clusters, self._output_dir
        mode = "spade" if self._radio_spade.isChecked() else "manual"
        self._log.append_line(f"[active_space] mode: {mode}  |  {nao} AOs  |  {n_active} active MOs")

        def _run():
            from asbuilder.active_space.localize_integrals import build_active_space

            # Compute overlap inside the worker thread — blocks on the main
            # thread for large molecules and freezes the GUI.
            overlap_ao = chk.mol.intor("int1e_ovlp")
            return build_active_space(
                mol=chk.mol,
                mo_coeff=chk.mo_coeff,
                mo_occ=chk.mo_occ,
                fock_ao=None,
                overlap_ao=overlap_ao,
                clusters=clusters.clusters,
                output_dir=output_dir,
                mode=mode,
            )

        log_path = output_dir.parent / "asbuilder.log"
        self._worker = ActiveSpaceWorker(_run, parent=self, log_path=log_path)
        self._worker.line_received.connect(self._log.append_line)
        self._worker.finished_ok.connect(self._on_finished)
        self._worker.failed.connect(self._on_failed)
        self._worker.start()

    def _on_finished(self, _result) -> None:
        # SPADE may have updated cluster.orbitals/fspace in-place; persist.
        if self._clusters is not None and self._output_dir is not None:
            try:
                self._clusters.save(self._output_dir.parent)
            except Exception as exc:
                self._log.append_line(f"[active_space] warning: could not save clusters.json: {exc}")
        self._log.append_line("[active_space] done.")
        self._build_btn.setEnabled(True)
        self.active_space_built.emit(str(self._output_dir))

    def _on_failed(self, message: str) -> None:
        self._log.append_line(f"[active_space] FAILED: {message}")
        self._build_btn.setEnabled(True)
