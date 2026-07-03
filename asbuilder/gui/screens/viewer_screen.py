"""
Screen 3: the main orbital viewer / clustering screen.

Two modes toggled by a tab widget at the top:

  Manual mode  -- user clicks MOs in the orbital table and assigns them to
                  clusters (Tier A click-to-assign). fspace is derived from
                  mo_occ automatically when the active space is built.

  SPADE mode   -- user assigns *atoms* to clusters from the atom table, and
                  optionally filters which AO types (s/p/d/f) contribute to
                  the SPADE projector per cluster. Useful for bimetallic /
                  fragment systems (Fe₂S₂, dimer, etc.).
"""

from __future__ import annotations

import re
import sys
import tempfile
from pathlib import Path

from PyQt6.QtCore import QProcess, pyqtSignal
from PyQt6.QtWidgets import (
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPlainTextEdit,
    QPushButton,
    QSplitter,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from asbuilder.cluster.state import ClusterSet
from asbuilder.gui.widgets.atom_table import AtomTable
from asbuilder.gui.widgets.cluster_manager import ClusterManager
from asbuilder.gui.widgets.orbital_table import OrbitalTable
from asbuilder.gui.widgets.webview_panel import WebViewPanel
from asbuilder.io.chk_to_molden import ChkContents

_HA2EV = 27.211386245


def _shells_from_mol(mol) -> list[str]:
    """Return sorted unique shell labels (e.g. '1s','2p','3d') from the mol basis."""
    try:
        labels = mol.ao_labels(fmt=False)
        shells = set(ao[2] for ao in labels)
        return sorted(shells, key=lambda s: (int(re.match(r"(\d+)", s).group(1)), s[-1]))
    except Exception:
        return ["s", "p", "d", "f"]


def _mf_summary(chk: ChkContents) -> str:
    """Instant MF summary from numpy data only — no C extensions, safe on any thread."""
    import numpy as np
    mol = chk.mol
    mo_e = np.asarray(chk.mo_energy).ravel()
    mo_o = np.asarray(chk.mo_occ).ravel()
    occ  = mo_o > 1e-6
    virt = ~occ

    lines = [
        f"Total energy : {chk.e_tot:.10f} Ha",
        f"Basis        : {mol.basis if isinstance(mol.basis, str) else 'custom/mixed'}",
        f"Charge / Spin: {mol.charge} / {mol.spin}  (2S = n_α - n_β)",
        f"n_orb        : {chk.n_orb}   n_elec: {mol.nelectron}",
    ]

    homo_i = lumo_i = None
    if occ.any():
        homo_i = int(np.where(occ)[0].max())
        lines.append(
            f"\nHOMO  MO #{homo_i+1:<4d}:  {mo_e[homo_i]:+.6f} Ha"
            f"  ({mo_e[homo_i]*_HA2EV:+.4f} eV)  occ={mo_o[homo_i]:.1f}"
        )
    if virt.any():
        lumo_i = int(np.where(virt)[0].min())
        lines.append(
            f"LUMO  MO #{lumo_i+1:<4d}:  {mo_e[lumo_i]:+.6f} Ha"
            f"  ({mo_e[lumo_i]*_HA2EV:+.4f} eV)  occ={mo_o[lumo_i]:.1f}"
        )
    if homo_i is not None and lumo_i is not None:
        gap = mo_e[lumo_i] - mo_e[homo_i]
        lines.append(f"Gap          :  {gap:.6f} Ha  ({gap*_HA2EV:.4f} eV)")

    lines += [
        "",
        f"{'MO':>5}  {'Energy (Ha)':>14}  {'Energy (eV)':>12}  {'Occ':>5}  Note",
        "-" * 54,
    ]
    for i, (e, o) in enumerate(zip(mo_e, mo_o)):
        note = "<-- HOMO" if i == homo_i else ("<-- LUMO" if i == lumo_i else "")
        lines.append(f"{i+1:>5}  {e:>14.8f}  {e*_HA2EV:>12.6f}  {o:>5.1f}  {note}")

    return "\n".join(lines)


# Subprocess script: receives chk_path on argv[1], prints mf.analyze() to stdout.
# Runs in a *separate* Python process so libcint C extensions are safe.
_ANALYZE_SCRIPT = """\
import sys
from pyscf import lib, scf
chk_path = sys.argv[1]
mol = lib.chkfile.load_mol(chk_path)
data = lib.chkfile.load(chk_path, "scf")
import numpy as np
mo_coeff = np.asarray(data["mo_coeff"])
mf = scf.UHF(mol) if mo_coeff.ndim == 3 else scf.RHF(mol)
mf.mo_coeff = mo_coeff
mf.mo_energy = np.asarray(data["mo_energy"])
mf.mo_occ   = np.asarray(data["mo_occ"])
mf.e_tot    = float(data.get("e_tot", float("nan")))
mol.verbose = 4
mol.stdout  = sys.stdout
mf.analyze()
"""


class ViewerScreen(QWidget):
    build_requested = pyqtSignal(object, object)  # (ChkContents, ClusterSet)

    def __init__(self, vibemol_root: str | Path | None = None, parent=None) -> None:
        super().__init__(parent)
        self._chk: ChkContents | None = None
        self.cluster_set = ClusterSet()
        self._analyze_proc: QProcess | None = None
        self._analyze_script_path: Path | None = None

        # --- MF analysis pane (collapsible, top) ---
        self._analysis_group = QGroupBox("MF Analysis  (mf.analyze)")
        self._analysis_group.setCheckable(True)
        self._analysis_group.setChecked(False)
        analysis_inner = QVBoxLayout(self._analysis_group)
        self._analysis_pane = QPlainTextEdit()
        self._analysis_pane.setReadOnly(True)
        self._analysis_pane.setMaximumBlockCount(5000)
        self._analysis_pane.setMaximumHeight(200)
        self._analysis_pane.setMinimumHeight(0)
        font = self._analysis_pane.font()
        font.setFamily("monospace")
        self._analysis_pane.setFont(font)
        analysis_inner.addWidget(self._analysis_pane)
        # hide content when unchecked (the QGroupBox checkable toggle)
        self._analysis_group.toggled.connect(self._analysis_pane.setVisible)
        self._analysis_pane.setVisible(False)

        # --- orbital table (manual mode) ---
        self._orbital_table = OrbitalTable()
        self._orbital_table.orbital_clicked.connect(self._on_orbital_clicked)

        # --- atom table (SPADE mode) ---
        self._atom_table = AtomTable()
        self._atom_table.atom_clicked.connect(self._on_atom_clicked)

        # --- webview (center) ---
        self._webview = WebViewPanel(vibemol_root=vibemol_root)

        # --- cluster manager (right) ---
        self._cluster_manager = ClusterManager(self.cluster_set)
        self._cluster_manager.clusters_changed.connect(self._on_clusters_changed)

        # --- mode tabs ---
        self._tabs = QTabWidget()
        self._tabs.addTab(self._orbital_table, "Manual — assign MOs")
        self._tabs.addTab(self._atom_table,    "SPADE  — assign atoms")
        self._tabs.currentChanged.connect(self._on_mode_changed)

        self._mode_label = QLabel(
            "<b>Manual mode</b>: click an MO row to assign it to the active cluster."
        )
        self._mode_label.setWordWrap(True)

        self._status = QLabel("")
        self._build_btn = QPushButton("Build Active Space")
        self._build_btn.setEnabled(False)
        self._build_btn.clicked.connect(self._on_build_clicked)

        splitter = QSplitter()
        left = QWidget()
        left_layout = QVBoxLayout(left)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.addWidget(self._mode_label)
        left_layout.addWidget(self._tabs)
        splitter.addWidget(left)
        splitter.addWidget(self._webview)
        splitter.addWidget(self._cluster_manager)
        splitter.setStretchFactor(0, 2)
        splitter.setStretchFactor(1, 1)
        splitter.setStretchFactor(2, 1)

        bottom = QHBoxLayout()
        bottom.addWidget(self._status)
        bottom.addStretch(1)
        bottom.addWidget(self._build_btn)

        layout = QVBoxLayout(self)
        layout.addWidget(self._analysis_group)
        layout.addWidget(splitter)
        layout.addLayout(bottom)
        self.setLayout(layout)
        self.setMinimumSize(0, 0)

    # ------------------------------------------------------------------

    def load(self, chk: ChkContents, molden_path: str, chk_path: str | None = None) -> None:
        self._chk = chk
        self.cluster_set = ClusterSet()
        self._cluster_manager.reset(self.cluster_set)
        self._cluster_manager.set_available_shells(_shells_from_mol(chk.mol))
        self._orbital_table.set_orbitals(chk.orbital_table())
        self._orbital_table.refresh_cluster_assignments(self.cluster_set)
        self._atom_table.set_molecule(chk.mol, self.cluster_set)
        self._webview.load_molden(molden_path)
        self._update_status()
        # Show instant numpy summary, then kick off Mulliken in a subprocess
        self._analysis_pane.setPlainText(_mf_summary(chk))
        if chk_path:
            self._launch_mulliken(chk_path)

    def _launch_mulliken(self, chk_path: str) -> None:
        """Run mf.analyze() (including Mulliken population) in a background
        subprocess. QProcess streams stdout line-by-line so the GUI stays
        responsive. A fresh Python interpreter means libcint is single-threaded
        and won't segfault."""
        # Kill any previous run
        if self._analyze_proc is not None:
            self._analyze_proc.kill()
            self._analyze_proc = None
        if self._analyze_script_path is not None:
            try:
                self._analyze_script_path.unlink()
            except Exception:
                pass

        # Write the runner script to a temp file
        tmp = tempfile.NamedTemporaryFile(
            mode="w", suffix=".py", delete=False, prefix="asbuilder_analyze_"
        )
        tmp.write(_ANALYZE_SCRIPT)
        tmp.close()
        self._analyze_script_path = Path(tmp.name)

        self._analysis_pane.appendPlainText(
            "\n" + "─" * 54 + "\nMulliken population analysis (running in background)…\n"
        )

        proc = QProcess(self)
        proc.setProcessChannelMode(QProcess.ProcessChannelMode.MergedChannels)
        proc.readyReadStandardOutput.connect(self._on_analyze_output)
        proc.finished.connect(self._on_analyze_finished)
        proc.start(sys.executable, [str(self._analyze_script_path), chk_path])
        self._analyze_proc = proc

    def _on_analyze_output(self) -> None:
        if self._analyze_proc is None:
            return
        data = bytes(self._analyze_proc.readAllStandardOutput()).decode("utf-8", errors="replace")
        for line in data.splitlines():
            self._analysis_pane.appendPlainText(line)

    def _on_analyze_finished(self, exit_code: int, _status) -> None:
        if exit_code != 0:
            self._analysis_pane.appendPlainText(f"\n[analyze exited with code {exit_code}]")
        if self._analyze_script_path is not None:
            try:
                self._analyze_script_path.unlink()
            except Exception:
                pass
            self._analyze_script_path = None
        self._analyze_proc = None

    # ------------------------------------------------------------------

    def _is_spade_mode(self) -> bool:
        return self._tabs.currentIndex() == 1

    def _on_mode_changed(self, index: int) -> None:
        spade = (index == 1)
        self._cluster_manager.set_spade_mode(spade)
        if spade:
            self._mode_label.setText(
                "<b>SPADE mode</b>: click an atom row to assign it to the active cluster. "
                "Select AO types (s/p/d/f) per cluster in the panel on the right — "
                "leave all unchecked to include all AO types."
            )
        else:
            self._mode_label.setText(
                "<b>Manual mode</b>: click an MO row to assign it to the active cluster."
            )
        self._update_status()

    def _on_orbital_clicked(self, orbital_index: int) -> None:
        active_id = self._cluster_manager.active_cluster_id()
        if active_id is None:
            self._status.setText("Add/select a cluster first.")
            return
        self.cluster_set.assign_orbital(active_id, orbital_index)
        self._orbital_table.refresh_cluster_assignments(self.cluster_set)
        self._cluster_manager.refresh()
        self._update_status()

    def _on_atom_clicked(self, atom_index: int) -> None:
        active_id = self._cluster_manager.active_cluster_id()
        if active_id is None:
            self._status.setText("Add/select a cluster first.")
            return
        # Toggle: if already assigned to this cluster, unassign
        c = self.cluster_set.get(active_id)
        if c.atom_indices and atom_index in c.atom_indices:
            self.cluster_set.unassign_atom(atom_index)
        else:
            self.cluster_set.assign_atom(active_id, atom_index)
        self._atom_table.refresh_cluster_assignments(self.cluster_set)
        self._cluster_manager.refresh()
        self._update_status()

    def _on_clusters_changed(self) -> None:
        if self._is_spade_mode():
            self._atom_table.refresh_cluster_assignments(self.cluster_set)
        else:
            self._orbital_table.refresh_cluster_assignments(self.cluster_set)
        self._update_status()

    def _update_status(self) -> None:
        problems = self.cluster_set.validate()
        if problems:
            self._status.setText("; ".join(problems))
            self._build_btn.setEnabled(False)
        else:
            self._status.setText("Clusters look valid — ready to build.")
            self._build_btn.setEnabled(self._chk is not None)

    def _on_build_clicked(self) -> None:
        if self._chk is not None:
            self.build_requested.emit(self._chk, self.cluster_set)
