"""
Screen 7: export / run calculation.

Methods available:
  TPSCI-GS          closed-shell ground state
  TPSCI-OpenShell   bimetallics / low-spin, spin_adapt + add_spin_focksectors
  TPSCI-CT          excited states with charge-transfer Fock sectors
  SPT               SPTstate seed → subspace_product_tucker; optional post-SPT PT2/variance
  SPT+PT2           SPTstate seed → subspace_product_tucker + PT2 correction
  SPT+Variance      SPTstate seed → subspace_product_tucker + variance estimate
  CEPA              FOIS-CEPA convergence sweep (bimetallics)
  PT2/CMF           tps_ci_direct + compute_pt2_energy
  QDPT2/CMF         tps_ci_direct + compute_qdpt2_energy
  PT2/TPSCI         tpsci_ci + compute_pt2_energy
  FOIS-CI           do_fois_ci first-order interacting-space CI
  Single diagonalization
                    tps_ci_direct once in the current TPSCI space

All results save cluster_bases + energy + wavefunction for post-processing.
Each method can be run locally or packaged for HPC (SLURM).
"""

from __future__ import annotations

import shutil
import zipfile
from pathlib import Path

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPlainTextEdit,
    QPushButton,
    QSpinBox,
    QSplitter,
    QStackedWidget,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from asbuilder.gui.widgets.collapsible_section import CollapsibleSection
from asbuilder.gui.widgets.log_pane import LogPane
from asbuilder.julia_bridge.runner import PINNED_JULIA_VERSION, write_driver

# ------------------------------------------------------------------
# Method registry:  key  →  (display label,  template filename)
# ------------------------------------------------------------------
_METHODS = [
    ("TPSCI-GS",        "TPSCI — ground state (closed-shell)",         "driver_tpsci.jl.j2"),
    ("TPSCI-OpenShell", "TPSCI — open-shell / low-spin bimetallics",   "driver_tpsci.jl.j2"),
    ("TPSCI-CT",        "TPSCI — excited states / CT sectors",          "driver_tpsci_ct.jl.j2"),
    ("SPT",             "SPT — Subspace Product Tucker",                "driver_spt.jl.j2"),
    ("SPT-PT2",         "SPT+PT2 — SPT with PT2 correction",            "driver_spt.jl.j2"),
    ("SPT-Variance",    "SPT+Variance — SPT sigma norm",                "driver_spt.jl.j2"),
    ("CEPA",            "CEPA — convergence sweep (bimetallics)",       "driver_cepa.jl.j2"),
    ("PT2-CMF",         "PT2 / CMF  (tps_ci_direct + PT2)",            "driver_pt2.jl.j2"),
    ("QDPT2-CMF",       "QDPT2 / CMF  (tps_ci_direct + QDPT2)",       "driver_pt2.jl.j2"),
    ("PT2-TPSCI",       "PT2 / TPSCI  (tpsci_ci + PT2)",              "driver_pt2.jl.j2"),
    ("FCI-solve",       "FOIS-CI — first-order interacting-space CI",   "driver_fci_solve.jl.j2"),
    ("Single-Diagonalization", "Single diagonalization — tps_ci_direct", "driver_single_diagonalization.jl.j2"),
]
_METHOD_KEYS   = [m[0] for m in _METHODS]
_METHOD_LABELS = [m[1] for m in _METHODS]
_METHOD_TPL    = {m[0]: m[2] for m in _METHODS}
_METHOD_PANEL_INDEX = {
    "TPSCI-GS": 0,
    "TPSCI-OpenShell": 1,
    "TPSCI-CT": 2,
    "SPT": 3,
    "SPT-PT2": 3,
    "SPT-Variance": 3,
    "CEPA": 4,
    "PT2-CMF": 5,
    "QDPT2-CMF": 6,
    "PT2-TPSCI": 7,
    "FCI-solve": 8,
    "Single-Diagonalization": 9,
}


def _parse_cluster_list(text: str) -> list[int]:
    """Parse '3' or '1, 3' into [3] / [1, 3]. Silently drops non-integers."""
    out: list[int] = []
    for tok in text.replace(";", ",").split(","):
        tok = tok.strip()
        if not tok:
            continue
        try:
            out.append(int(tok))
        except ValueError:
            continue
    return out


class ExportScreen(QWidget):
    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._cmf_result_path: Path | None = None
        self._export_dir: Path | None = None
        self._julia_project: Path | None = None
        self._worker = None
        self._cancelled = False          # user pressed Cancel
        self._restart_pending = False    # user pressed Restart mid-run

        # ── method selector ──────────────────────────────────────────
        self._method = QComboBox()
        for label in _METHOD_LABELS:
            self._method.addItem(label)
        self._method.currentIndexChanged.connect(self._on_method_changed)

        # ── shared options ───────────────────────────────────────────
        self._max_roots  = QSpinBox(); self._max_roots.setRange(1, 500); self._max_roots.setValue(100)
        self._nroots     = QSpinBox(); self._nroots.setRange(1, 50);     self._nroots.setValue(4)
        self._thresh_foi = QLineEdit("1e-4")
        self._delta_elec = QSpinBox(); self._delta_elec.setRange(1, 8); self._delta_elec.setValue(3)
        self._delta_elec.setToolTip(
            "delta_elec passed to compute_cluster_eigenbasis_spin: number of electrons\n"
            "added/removed from each cluster's reference Fock sector when building its\n"
            "spin-adapted eigenbasis. 2–3 per cluster is typical for bimetallics.\n"
            "Only used by the spin-adapted (open-shell / bimetallic) methods.")
        self._flip_clusters = QLineEdit("")
        self._flip_clusters.setPlaceholderText("blank = auto broken-symmetry; or e.g. 3")
        self._flip_clusters.setToolTip(
            "Open-shell reference: clusters whose (nα,nβ) is spin-flipped to (nβ,nα) when\n"
            "building the reference Fock config, relative to the CMF init_fspace.\n"
            "add_spin_focksectors conserves the GLOBAL (nα,nβ), so reaching the low-spin\n"
            "(antiferromagnetic) manifold requires flipping metals — e.g. flip cluster 3 to\n"
            "turn [(3,0),(3,3),(3,0)] (M_S=3) into [(3,0),(3,3),(0,3)] (M_S=0).\n\n"
            "DEFAULT (blank): auto broken-symmetry — alternate the spin of the open-shell\n"
            "clusters to minimize global M_S. Enter cluster #s here only to override.\n"
            "Applies to every spin-adapted method (ignored by closed-shell TPSCI-GS).")
        self._verbose    = QSpinBox(); self._verbose.setRange(0, 5); self._verbose.setValue(1)
        self._compute_s2 = QCheckBox("Compute ⟨S²⟩ after solve")

        shared_form = QFormLayout()
        shared_form.addRow("Method", self._method)
        shared_form.addRow("max_roots (eigenbasis)", self._max_roots)
        shared_form.addRow("nroots", self._nroots)
        shared_form.addRow("thresh_foi", self._thresh_foi)
        shared_form.addRow("Δn_elec per cluster (spin basis)", self._delta_elec)
        shared_form.addRow("Spin-flip clusters (open-shell)", self._flip_clusters)
        shared_form.addRow("Verbose", self._verbose)
        shared_form.addRow("", self._compute_s2)

        # ── method-specific panels (stacked) ─────────────────────────
        self._method_stack = QStackedWidget()

        # Panel 0: TPSCI-GS
        p0 = QWidget(); f0 = QFormLayout(p0)
        self._thresh_cipsi_gs = QLineEdit("1e-3")
        f0.addRow("thresh_cipsi", self._thresh_cipsi_gs)
        f0.addRow(self._note_panel(
            "Closed-shell TPSCI using compute_cluster_eigenbasis.\n"
            "Use 'TPSCI — open-shell' for bimetallics."))
        self._method_stack.addWidget(p0)

        # Panel 1: TPSCI-OpenShell
        p1 = QWidget(); f1 = QFormLayout(p1)
        self._spin_adapt_os    = QCheckBox("compute_cluster_eigenbasis_spin  (bimetallics / open-shell)")
        self._spin_adapt_os.setChecked(True)
        self._add_spin_fock_os = QCheckBox("add_spin_focksectors  (M_S ± 1 sectors)")
        self._add_spin_fock_os.setChecked(True)
        self._thresh_spin      = QLineEdit("1e-5")
        self._thresh_cipsi_os  = QLineEdit("1e-3")
        f1.addRow("", self._spin_adapt_os)
        f1.addRow("", self._add_spin_fock_os)
        f1.addRow("thresh_spin", self._thresh_spin)
        f1.addRow("thresh_cipsi", self._thresh_cipsi_os)
        self._method_stack.addWidget(p1)

        # Panel 2: TPSCI-CT
        p2 = QWidget(); f2 = QFormLayout(p2)
        self._max_local_roots = QSpinBox(); self._max_local_roots.setRange(1, 10); self._max_local_roots.setValue(3)
        self._add_double_exc  = QCheckBox("Double-cluster excitations in seed")
        self._add_double_exc.setChecked(True)
        self._thresh_cipsi_ct = QLineEdit("5e-3")
        f2.addRow("max local roots / seed", self._max_local_roots)
        f2.addRow("", self._add_double_exc)
        f2.addRow("thresh_cipsi", self._thresh_cipsi_ct)
        note = QLabel("  ⓘ Edit ct_fspaces in the rendered script to\n"
                      "  add molecule-specific CT Fock sectors.")
        note.setStyleSheet("color: #888; font-size: 10px;")
        f2.addRow(note)
        self._method_stack.addWidget(p2)

        # Panel 3: SPT
        p3 = QWidget(); f3 = QFormLayout(p3)
        self._spt_space_mode    = QComboBox()
        self._spt_space_mode.addItems(["Low-spin open-shell P-space", "CT / excitonic seed"])
        self._thresh_var        = QLineEdit("1e-2")
        self._thresh_spin_spt   = QLineEdit("1e-2")
        self._thresh_spin_spt.setPlaceholderText("blank = thresh_var")
        self._thresh_pt_spt     = QLineEdit("5e-3")
        self._p_space_roots     = QSpinBox(); self._p_space_roots.setRange(1, 10); self._p_space_roots.setValue(1)
        self._L_single          = QSpinBox(); self._L_single.setRange(1, 50); self._L_single.setValue(4)
        self._L_double          = QSpinBox(); self._L_double.setRange(1, 50); self._L_double.setValue(4)
        self._max_iter_spt      = QSpinBox(); self._max_iter_spt.setRange(1, 500); self._max_iter_spt.setValue(50)
        self._post_analysis     = QComboBox()
        self._post_analysis.addItems(["None", "PT2 correction", "Variance"])
        self._post_thresh_foi   = QLineEdit("1e-4")
        f3.addRow("SPT seed space", self._spt_space_mode)
        f3.addRow("thresh_var", self._thresh_var)
        f3.addRow("thresh_spin", self._thresh_spin_spt)
        f3.addRow("thresh_pt", self._thresh_pt_spt)
        f3.addRow("P roots / sector", self._p_space_roots)
        f3.addRow("single exciton roots (CT)", self._L_single)
        f3.addRow("double exciton roots (CT)", self._L_double)
        f3.addRow("max_iter SPT", self._max_iter_spt)
        sep = QLabel("Post-SPT analysis"); sep.setStyleSheet("font-weight: bold; margin-top: 6px;")
        f3.addRow(sep)
        f3.addRow("Post-SPT step", self._post_analysis)
        f3.addRow("thresh_foi (post)", self._post_thresh_foi)
        self._method_stack.addWidget(p3)

        # Panel 4: CEPA
        p4 = QWidget(); f4 = QFormLayout(p4)
        self._spin_adapt_cepa    = QCheckBox("compute_cluster_eigenbasis_spin")
        self._spin_adapt_cepa.setChecked(True)
        self._add_spin_fock_cepa = QCheckBox("add_spin_focksectors")
        self._add_spin_fock_cepa.setChecked(True)
        self._cepa_shift      = QComboBox()
        self._cepa_shift.addItems(["cepa", "cisd", "acpf", "aqcc"])
        self._cepa_thresholds = QLineEdit("[1e-4]")
        self._cepa_tol        = QLineEdit("1e-8")
        f4.addRow("", self._spin_adapt_cepa)
        f4.addRow("", self._add_spin_fock_cepa)
        f4.addRow("cepa_shift", self._cepa_shift)
        f4.addRow("thresh_foi sweep", self._cepa_thresholds)
        f4.addRow("solver tol", self._cepa_tol)
        self._method_stack.addWidget(p4)

        # Panel 5: PT2/CMF
        p5 = QWidget(); f5 = QFormLayout(p5)
        self._spin_adapt_pt2_cmf = QCheckBox("compute_cluster_eigenbasis_spin")
        self._pt2_cmf_thresh     = QLineEdit("1e-4")
        f5.addRow("", self._spin_adapt_pt2_cmf)
        f5.addRow("thresh_foi (PT2)", self._pt2_cmf_thresh)
        f5.addRow(self._note_panel("Reference: tps_ci_direct (CMF, no CIPSI expansion).\n"
                                   "PT2 correction via compute_pt2_energy."))
        self._method_stack.addWidget(p5)

        # Panel 6: QDPT2/CMF
        p6 = QWidget(); f6 = QFormLayout(p6)
        self._spin_adapt_qdpt2 = QCheckBox("compute_cluster_eigenbasis_spin")
        self._qdpt2_thresh     = QLineEdit("1e-4")
        f6.addRow("", self._spin_adapt_qdpt2)
        f6.addRow("thresh_foi (QDPT2)", self._qdpt2_thresh)
        f6.addRow(self._note_panel("Reference: tps_ci_direct (CMF, no CIPSI expansion).\n"
                                   "QDPT2 correction via compute_qdpt2_energy."))
        self._method_stack.addWidget(p6)

        # Panel 7: PT2/TPSCI
        p7 = QWidget(); f7 = QFormLayout(p7)
        self._spin_adapt_pt2_tpsci = QCheckBox("compute_cluster_eigenbasis_spin")
        self._thresh_cipsi_pt2     = QLineEdit("1e-3")
        self._pt2_tpsci_thresh     = QLineEdit("1e-4")
        f7.addRow("", self._spin_adapt_pt2_tpsci)
        f7.addRow("thresh_cipsi (TPSCI)", self._thresh_cipsi_pt2)
        f7.addRow("thresh_foi (PT2)", self._pt2_tpsci_thresh)
        f7.addRow(self._note_panel("Reference: full tpsci_ci.\n"
                                   "PT2 correction via compute_pt2_energy."))
        self._method_stack.addWidget(p7)

        # Panel 8: FOIS-CI
        p8 = QWidget(); f8 = QFormLayout(p8)
        self._spin_adapt_fois = QCheckBox("compute_cluster_eigenbasis_spin")
        self._spin_adapt_fois.setChecked(True)
        self._add_spin_fock_fois = QCheckBox("add_spin_focksectors")
        self._add_spin_fock_fois.setChecked(True)
        self._fois_max_iter = QSpinBox(); self._fois_max_iter.setRange(1, 500); self._fois_max_iter.setValue(50)
        self._fois_nbody = QSpinBox(); self._fois_nbody.setRange(1, 6); self._fois_nbody.setValue(4)
        self._fois_tol = QLineEdit("1e-5")
        self._fois_thresh_clip = QLineEdit("1e-6")
        self._fois_threaded = QCheckBox("threaded")
        self._fois_prescreen = QCheckBox("prescreen")
        self._fois_compress = QCheckBox("compress FOIS")
        self._fois_pt = QCheckBox("compute PT after FOIS-CI")
        f8.addRow("", self._spin_adapt_fois)
        f8.addRow("", self._add_spin_fock_fois)
        f8.addRow("max_iter", self._fois_max_iter)
        f8.addRow("nbody", self._fois_nbody)
        f8.addRow("tol", self._fois_tol)
        f8.addRow("thresh_clip", self._fois_thresh_clip)
        f8.addRow("", self._fois_threaded)
        f8.addRow("", self._fois_prescreen)
        f8.addRow("", self._fois_compress)
        f8.addRow("", self._fois_pt)
        self._method_stack.addWidget(p8)

        # Panel 9: Single diagonalization
        p9 = QWidget(); f9 = QFormLayout(p9)
        self._spin_adapt_single = QCheckBox("compute_cluster_eigenbasis_spin")
        self._spin_adapt_single.setChecked(True)
        self._add_spin_fock_single = QCheckBox("add_spin_focksectors")
        self._add_spin_fock_single.setChecked(True)
        self._single_conv_thresh = QLineEdit("1e-5")
        self._single_lindep_thresh = QLineEdit("1e-12")
        self._single_max_ss_vecs = QSpinBox(); self._single_max_ss_vecs.setRange(1, 200); self._single_max_ss_vecs.setValue(12)
        self._single_max_iter = QSpinBox(); self._single_max_iter.setRange(1, 500); self._single_max_iter.setValue(40)
        self._single_precond = QCheckBox("preconditioner")
        self._single_precond.setChecked(True)
        f9.addRow("", self._spin_adapt_single)
        f9.addRow("", self._add_spin_fock_single)
        f9.addRow("conv_thresh", self._single_conv_thresh)
        f9.addRow("lindep_thresh", self._single_lindep_thresh)
        f9.addRow("max_ss_vecs", self._single_max_ss_vecs)
        f9.addRow("max_iter", self._single_max_iter)
        f9.addRow("", self._single_precond)
        self._method_stack.addWidget(p9)

        calc_group = CollapsibleSection("Calculation settings")
        calc_body  = QVBoxLayout()
        calc_body.addLayout(shared_form)
        calc_body.addWidget(self._method_stack)
        calc_group.set_body_layout(calc_body)

        # ── HPC / SLURM ──────────────────────────────────────────────
        self._job_name  = QLineEdit("tpsci_job")
        self._account   = QLineEdit()
        self._partition = QLineEdit("normal")
        self._nodes     = QSpinBox(); self._nodes.setRange(1, 1000); self._nodes.setValue(1)
        self._walltime  = QLineEdit("24:00:00")
        self._threads   = QLineEdit("auto")
        self._mem       = QLineEdit("64G")

        hpc_form = QFormLayout()
        hpc_form.addRow("Job name",     self._job_name)
        hpc_form.addRow("Account",      self._account)
        hpc_form.addRow("Partition",    self._partition)
        hpc_form.addRow("Nodes",        self._nodes)
        hpc_form.addRow("Walltime",     self._walltime)
        hpc_form.addRow("Threads (-t)", self._threads)
        hpc_form.addRow("Memory",       self._mem)

        hpc_group = CollapsibleSection("HPC submission (SLURM)")
        hpc_group.set_body_layout(hpc_form)

        # ── preview tabs ─────────────────────────────────────────────
        self._driver_preview = QPlainTextEdit(); self._driver_preview.setReadOnly(True)
        self._slurm_preview  = QPlainTextEdit(); self._slurm_preview.setReadOnly(True)
        for w in (self._driver_preview, self._slurm_preview):
            f = w.font(); f.setFamily("monospace"); w.setFont(f)
        preview_tabs = QTabWidget()
        preview_tabs.addTab(self._driver_preview, "driver_*.jl")
        preview_tabs.addTab(self._slurm_preview,  "submit.slurm")

        self._log = LogPane()

        render_btn          = QPushButton("Render script")
        render_btn.clicked.connect(self._on_render)
        self._run_local_btn = QPushButton("▶  Run locally")
        self._run_local_btn.clicked.connect(self._on_run_local)
        self._restart_btn   = QPushButton("Restart")
        self._restart_btn.clicked.connect(self._on_restart)
        self._restart_btn.setEnabled(False)
        self._cancel_btn    = QPushButton("Cancel")
        self._cancel_btn.clicked.connect(self._on_cancel)
        self._cancel_btn.setEnabled(False)
        package_btn         = QPushButton("Package for HPC…")
        package_btn.clicked.connect(self._on_package)

        btn_row = QHBoxLayout()
        btn_row.addWidget(render_btn)
        btn_row.addWidget(self._run_local_btn)
        btn_row.addWidget(self._restart_btn)
        btn_row.addWidget(self._cancel_btn)
        btn_row.addStretch(1)
        btn_row.addWidget(package_btn)

        self._status = QLabel(f"Target: Julia {PINNED_JULIA_VERSION}")

        splitter = QSplitter(Qt.Orientation.Vertical)
        splitter.setHandleWidth(8)
        splitter.setChildrenCollapsible(False)
        splitter.addWidget(calc_group)
        splitter.addWidget(hpc_group)
        splitter.addWidget(self._log)
        splitter.addWidget(preview_tabs)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 0)
        splitter.setStretchFactor(2, 4)
        splitter.setStretchFactor(3, 1)

        layout = QVBoxLayout(self)
        layout.addWidget(splitter, stretch=1)
        layout.addLayout(btn_row)
        layout.addWidget(self._status)
        self.setLayout(layout)
        self.setMinimumSize(0, 0)
        self._method_stack.setMinimumSize(0, 0)
        self._log.setMinimumSize(0, 0)
        self._driver_preview.setMinimumSize(0, 0)
        self._slurm_preview.setMinimumSize(0, 0)

        self._on_method_changed(0)

    # ------------------------------------------------------------------

    @staticmethod
    def _note_panel(text: str) -> QWidget:
        w = QWidget()
        lbl = QLabel(text)
        lbl.setStyleSheet("color: #888; font-style: italic; padding: 4px;")
        lbl.setWordWrap(True)
        QVBoxLayout(w).addWidget(lbl)
        return w

    def set_inputs(self, cmf_result_path: str | Path, export_dir: str | Path,
                   julia_project: str | Path | None = None) -> None:
        self._cmf_result_path = Path(cmf_result_path)
        self._export_dir = Path(export_dir)
        self._julia_project = Path(julia_project) if julia_project else None
        self._export_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------

    def _method_key(self) -> str:
        return _METHOD_KEYS[self._method.currentIndex()]

    def _on_method_changed(self, idx: int) -> None:
        key = _METHOD_KEYS[idx]
        self._method_stack.setCurrentIndex(_METHOD_PANEL_INDEX[key])
        if key == "SPT":
            self._post_analysis.setCurrentText("None")
        elif key == "SPT-PT2":
            self._post_analysis.setCurrentText("PT2 correction")
        elif key == "SPT-Variance":
            self._post_analysis.setCurrentText("Variance")

    def _build_context(self) -> tuple[str, dict]:
        key    = self._method_key()
        tpl    = _METHOD_TPL[key]
        cmf    = str(self._cmf_result_path)
        outdir = str(self._export_dir)

        ctx = dict(
            cmf_result_path = cmf,
            output_dir      = outdir,
            max_roots       = self._max_roots.value(),
            nroots          = self._nroots.value(),
            thresh_foi      = self._thresh_foi.text(),
            delta_elec      = self._delta_elec.value(),
            flip_clusters   = _parse_cluster_list(self._flip_clusters.text()),
            verbose         = self._verbose.value(),
            compute_s2      = self._compute_s2.isChecked(),
        )

        if key == "TPSCI-GS":
            ctx.update(
                spin_adapt=False,
                add_spin_fock=False,
                auto_flip=False,
                thresh_cipsi=self._thresh_cipsi_gs.text(),
            )

        elif key == "TPSCI-OpenShell":
            spin_adapt = self._spin_adapt_os.isChecked()
            ctx.update(
                spin_adapt    = spin_adapt,
                add_spin_fock = self._add_spin_fock_os.isChecked(),
                thresh_cipsi  = self._thresh_cipsi_os.text(),
                thresh_spin   = self._thresh_spin.text() if self._add_spin_fock_os.isChecked() else None,
                auto_flip     = spin_adapt,   # open-shell default: alternate open-shell clusters
            )

        elif key == "TPSCI-CT":
            ctx.update(
                thresh_cipsi           = self._thresh_cipsi_ct.text(),
                max_local_roots        = self._max_local_roots.value(),
                add_double_excitations = self._add_double_exc.isChecked(),
                auto_flip              = True,   # CT is always spin-adapted
            )

        elif key in {"SPT", "SPT-PT2", "SPT-Variance"}:
            post_map = {"None": "none", "PT2 correction": "pt2", "Variance": "variance"}
            spt_space_mode = "ct" if self._spt_space_mode.currentText().startswith("CT") else "open_shell"
            spt_thresh_var = self._thresh_var.text().strip() or "1e-2"
            spt_thresh_spin = self._thresh_spin_spt.text().strip() or spt_thresh_var
            post_analysis = post_map[self._post_analysis.currentText()]
            if key == "SPT-PT2":
                post_analysis = "pt2"
            elif key == "SPT-Variance":
                post_analysis = "variance"
            ctx.update(
                spt_space_mode = spt_space_mode,
                thresh_var      = spt_thresh_var,
                thresh_spin     = spt_thresh_spin,
                thresh_pt       = self._thresh_pt_spt.text(),
                p_space_roots   = self._p_space_roots.value(),
                single_excitons_roots = self._L_single.value(),
                double_excitons_roots = self._L_double.value(),
                max_iter        = self._max_iter_spt.value(),
                post_analysis   = post_analysis,
                post_thresh_foi = self._post_thresh_foi.text(),
                auto_flip       = True,   # SPT is always spin-adapted
            )

        elif key == "CEPA":
            spin_adapt = self._spin_adapt_cepa.isChecked()
            ctx.update(
                spin_adapt      = spin_adapt,
                add_spin_fock   = self._add_spin_fock_cepa.isChecked(),
                cepa_shift      = self._cepa_shift.currentText(),
                thresh_foi_list = self._cepa_thresholds.text(),
                tol             = self._cepa_tol.text(),
                auto_flip       = spin_adapt,
            )

        elif key == "PT2-CMF":
            spin_adapt = self._spin_adapt_pt2_cmf.isChecked()
            ctx.update(
                ref_type       = "cmf_pt2",
                spin_adapt     = spin_adapt,
                pt2_thresh_foi = self._pt2_cmf_thresh.text(),
                auto_flip      = spin_adapt,
            )

        elif key == "QDPT2-CMF":
            spin_adapt = self._spin_adapt_qdpt2.isChecked()
            ctx.update(
                ref_type       = "cmf_qdpt2",
                spin_adapt     = spin_adapt,
                pt2_thresh_foi = self._qdpt2_thresh.text(),
                auto_flip      = spin_adapt,
            )

        elif key == "PT2-TPSCI":
            spin_adapt = self._spin_adapt_pt2_tpsci.isChecked()
            ctx.update(
                ref_type       = "tpsci_pt2",
                spin_adapt     = spin_adapt,
                thresh_cipsi   = self._thresh_cipsi_pt2.text(),
                pt2_thresh_foi = self._pt2_tpsci_thresh.text(),
                auto_flip      = spin_adapt,
            )

        elif key == "FCI-solve":
            spin_adapt = self._spin_adapt_fois.isChecked()
            ctx.update(
                spin_adapt    = spin_adapt,
                add_spin_fock = self._add_spin_fock_fois.isChecked(),
                auto_flip     = spin_adapt,
                fois_max_iter = self._fois_max_iter.value(),
                nbody         = self._fois_nbody.value(),
                tol           = self._fois_tol.text(),
                thresh_clip   = self._fois_thresh_clip.text(),
                threaded      = self._fois_threaded.isChecked(),
                prescreen     = self._fois_prescreen.isChecked(),
                compress      = self._fois_compress.isChecked(),
                pt            = self._fois_pt.isChecked(),
            )

        elif key == "Single-Diagonalization":
            spin_adapt = self._spin_adapt_single.isChecked()
            ctx.update(
                spin_adapt     = spin_adapt,
                add_spin_fock  = self._add_spin_fock_single.isChecked(),
                auto_flip      = spin_adapt,
                conv_thresh    = self._single_conv_thresh.text(),
                lindep_thresh  = self._single_lindep_thresh.text(),
                max_ss_vecs    = self._single_max_ss_vecs.value(),
                diag_max_iter  = self._single_max_iter.value(),
                precond        = self._single_precond.isChecked(),
            )

        return tpl, ctx

    def _driver_name(self) -> str:
        key = self._method_key().lower().replace("-", "_")
        return f"driver_{key}.jl"

    def _on_render(self) -> None:
        if self._cmf_result_path is None or self._export_dir is None:
            self._status.setText("No CMF result set — finish the CMF step first.")
            return
        try:
            from asbuilder.julia_bridge.runner import render_driver
            template, ctx = self._build_context()
            self._driver_preview.setPlainText(render_driver(template, ctx))

            slurm_ctx = dict(
                job_name      = self._job_name.text() or "tpsci_job",
                account       = self._account.text(),
                partition     = self._partition.text(),
                nodes         = self._nodes.value(),
                walltime      = self._walltime.text(),
                mem           = self._mem.text(),
                threads       = self._threads.text(),
                driver_script = self._driver_name(),
                julia_project = str(self._julia_project or "."),
                use_juliaup   = False,
                julia_version = PINNED_JULIA_VERSION,
            )
            self._slurm_preview.setPlainText(render_driver("submit.slurm.j2", slurm_ctx))
            self._status.setText(f"Rendered {template} for Julia {PINNED_JULIA_VERSION}.")
        except Exception as exc:
            self._status.setText(f"Render failed: {exc}")

    def _set_running(self, running: bool) -> None:
        """Toggle the button row between idle and running states."""
        self._run_local_btn.setEnabled(not running)
        self._cancel_btn.setEnabled(running)
        self._restart_btn.setEnabled(running)

    def _on_cancel(self) -> None:
        """Stop the running local job."""
        self._cancelled = True
        self._restart_pending = False
        if self._worker is not None and self._worker.is_running():
            self._worker.kill()
        self._log.append_line("[export] cancelled by user.")
        self._set_running(False)

    def _on_restart(self) -> None:
        """Kill the current job and immediately start a fresh run."""
        self._log.append_line("[export] restarting…")
        if self._worker is not None and self._worker.is_running():
            self._restart_pending = True
            self._worker.kill()   # _on_run_finished re-launches once it stops
        else:
            self._on_run_local()

    def _on_run_local(self) -> None:
        if self._cmf_result_path is None or self._export_dir is None:
            self._log.append_line("[export] No CMF result — finish CMF step first.")
            return
        if self._worker is not None and self._worker.is_running():
            self._log.append_line("[export] A job is already running.")
            return

        template, ctx = self._build_context()
        driver_path = write_driver(template, ctx, self._export_dir / self._driver_name())
        self._log.append_line(f"[export] Wrote {driver_path.name}, launching Julia…")
        self._cancelled = False
        self._set_running(True)

        from asbuilder.gui.workers import JuliaProcessWorker
        julia_project = self._julia_project or self._cmf_result_path.parent.parent
        self._worker = JuliaProcessWorker(
            driver_path, julia_project,
            parent=self,
            log_path=self._export_dir / "export.log",
        )
        self._worker.line_received.connect(self._log.append_line)
        self._worker.finished_ok.connect(self._on_run_finished)
        self._worker.failed.connect(self._on_run_failed)
        self._worker.start()

    def _on_run_finished(self, exit_code: int) -> None:
        # Restart requested mid-run: the kill just landed, start a fresh run.
        if self._restart_pending:
            self._restart_pending = False
            self._on_run_local()
            return
        self._set_running(False)
        if self._cancelled:
            self._cancelled = False
            self._log.append_line("[export] run stopped.")
            return
        msg = "Julia finished OK." if exit_code == 0 else f"Julia exited with code {exit_code}."
        self._log.append_line(f"[export] {msg}")

    def _on_run_failed(self, message: str) -> None:
        if self._restart_pending:
            self._restart_pending = False
            self._on_run_local()
            return
        self._set_running(False)
        if self._cancelled:
            self._cancelled = False
            return
        self._log.append_line(f"[export] FAILED: {message}")

    def _on_package(self) -> None:
        if self._export_dir is None or self._cmf_result_path is None:
            self._status.setText("Nothing to package — render first.")
            return

        from asbuilder.julia_bridge.runner import render_driver
        template, ctx = self._build_context()
        write_driver(template, ctx, self._export_dir / self._driver_name())

        slurm_ctx = dict(
            job_name      = self._job_name.text() or "tpsci_job",
            account       = self._account.text(),
            partition     = self._partition.text(),
            nodes         = self._nodes.value(),
            walltime      = self._walltime.text(),
            mem           = self._mem.text(),
            threads       = self._threads.text(),
            driver_script = self._driver_name(),
            julia_project = ".",
            use_juliaup   = False,
            julia_version = PINNED_JULIA_VERSION,
        )
        (self._export_dir / "submit.slurm").write_text(render_driver("submit.slurm.j2", slurm_ctx))

        if self._cmf_result_path.exists():
            shutil.copy2(self._cmf_result_path, self._export_dir / self._cmf_result_path.name)

        zip_path, _ = QFileDialog.getSaveFileName(
            self, "Save export package",
            f"{self._method_key().lower()}_export.zip",
            "Zip files (*.zip)",
        )
        if not zip_path:
            return
        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
            for path in self._export_dir.rglob("*"):
                if path.is_file():
                    zf.write(path, arcname=path.relative_to(self._export_dir.parent))
        self._status.setText(f"Packaged → {zip_path}")
        self._log.append_line(f"[export] HPC package saved to {zip_path}")
        self._log.append_line("[export] Transfer and submit with: sbatch submit.slurm")
