"""
Screen 6: "Run CMF before continuing?" -- launches driver_cmf.jl as a
Julia subprocess with live-streamed log output.

Supports three CMF methods exposed in the GUI:
  - BFGS   : cmf_oo(ints, clusters, init_fspace, d1; method="bfgs", ...)
  - Newton  : cmf_oo_newton(ints, clusters, init_fspace, ansatze, d1; ...)
  - DIIS    : cmf_oo(ints, clusters, init_fspace, d1; method="diis", ...)
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPlainTextEdit,
    QPushButton,
    QSpinBox,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

import asbuilder.config as cfg
from asbuilder.cluster.state import ClusterSet
from asbuilder.gui.widgets.collapsible_section import CollapsibleSection
from asbuilder.gui.widgets.log_pane import LogPane
from asbuilder.gui.workers import JuliaProcessWorker, _CallableWorker
from asbuilder.julia_bridge.runner import write_driver


class CMFScreen(QWidget):
    cmf_done = pyqtSignal(str)   # emits cmf_result.jld2 path
    cmf_skipped = pyqtSignal()

    def __init__(self, julia_project_dir: str | Path, julia_bin: str = "julia", parent=None) -> None:
        super().__init__(parent)
        self._julia_project_dir = Path(julia_project_dir)
        self._julia_bin = julia_bin
        self._worker: JuliaProcessWorker | None = None
        self._output_dir: Path | None = None
        self._active_space_dir: Path | None = None
        self._clusters: ClusterSet | None = None
        self._cancelled = False          # user pressed Cancel
        self._restart_pending = False    # user pressed Restart mid-run

        # --- Julia project path ---
        self._julia_project_edit = QLineEdit(str(julia_project_dir))
        browse_btn = QPushButton("Browse...")
        browse_btn.setFixedWidth(80)
        browse_btn.clicked.connect(self._browse_julia_project)
        julia_row = QHBoxLayout()
        julia_row.addWidget(self._julia_project_edit)
        julia_row.addWidget(browse_btn)

        pycall_btn = QPushButton("Build PyCall (run once per environment)")
        pycall_btn.clicked.connect(self._on_build_pycall)

        env_group = CollapsibleSection("Julia environment")
        env_inner = QVBoxLayout()
        env_form = QFormLayout()
        env_form.addRow("TPSChem.jl project", julia_row)
        env_inner.addLayout(env_form)
        env_inner.addWidget(pycall_btn)
        env_group.set_body_layout(env_inner)

        # --- method selector ---
        self._method = QComboBox()
        self._method.addItems(["Newton (cmf_oo_newton)", "BFGS (cmf_oo)", "DIIS (cmf_oo)"])
        self._method.currentTextChanged.connect(self._on_method_changed)

        # --- shared options ---
        self._max_iter = QSpinBox()
        self._max_iter.setRange(1, 2000)
        self._max_iter.setValue(400)

        self._tol_oo = QLineEdit("1e-8")
        self._verbose = QSpinBox()
        self._verbose.setRange(0, 10)
        self._verbose.setValue(4)

        # --- Newton-specific options ---
        self._newton_box = QGroupBox("Newton options")
        self._tol_d1 = QLineEdit("1e-9")
        self._tol_ci = QLineEdit("1e-11")
        self._zero_intra = QCheckBox("zero_intra_rots")
        self._zero_intra.setChecked(False)
        self._sequential = QCheckBox("sequential")
        self._sequential.setChecked(True)
        newton_form = QFormLayout(self._newton_box)
        newton_form.addRow("tol_d1", self._tol_d1)
        newton_form.addRow("tol_ci", self._tol_ci)
        newton_form.addRow("", self._zero_intra)
        newton_form.addRow("", self._sequential)

        opt_form = QFormLayout()
        opt_form.addRow("Method", self._method)
        opt_form.addRow("Max iterations", self._max_iter)
        opt_form.addRow("tol_oo / gconv", self._tol_oo)
        opt_form.addRow("Verbose", self._verbose)

        options_group = CollapsibleSection("CMF Options")
        opt_inner = QVBoxLayout()
        opt_inner.addLayout(opt_form)
        opt_inner.addWidget(self._newton_box)
        options_group.set_body_layout(opt_inner)

        # --- cluster / init_fspace summary (filled by set_inputs) ---
        self._summary_pane = QPlainTextEdit()
        self._summary_pane.setReadOnly(True)
        self._summary_pane.setMaximumHeight(160)
        self._summary_pane.setPlaceholderText("(cluster summary will appear here after the active space is built)")
        font = self._summary_pane.font()
        font.setFamily("monospace")
        self._summary_pane.setFont(font)
        summary_group = QGroupBox("Active space — clusters & init_fspace")
        sg_layout = QVBoxLayout(summary_group)
        sg_layout.addWidget(self._summary_pane)

        self._log = LogPane()

        self._setup_worker: _CallableWorker | None = None
        self._run_btn = QPushButton("Run CMF")
        self._run_btn.clicked.connect(self._on_run)
        self._restart_btn = QPushButton("Restart")
        self._restart_btn.clicked.connect(self._on_restart)
        self._restart_btn.setEnabled(False)
        self._cancel_btn = QPushButton("Cancel")
        self._cancel_btn.clicked.connect(self._on_cancel)
        self._cancel_btn.setEnabled(False)
        self._skip_btn = QPushButton("Skip -- use bare integrals")
        self._skip_btn.clicked.connect(self.cmf_skipped.emit)

        btn_row = QHBoxLayout()
        btn_row.addWidget(self._skip_btn)
        btn_row.addStretch(1)
        btn_row.addWidget(self._restart_btn)
        btn_row.addWidget(self._cancel_btn)
        btn_row.addWidget(self._run_btn)

        splitter = QSplitter(Qt.Orientation.Vertical)
        splitter.setHandleWidth(8)
        splitter.setChildrenCollapsible(False)
        splitter.addWidget(env_group)
        splitter.addWidget(options_group)
        splitter.addWidget(summary_group)
        splitter.addWidget(self._log)
        splitter.setStretchFactor(0, 0)   # env — collapses to header
        splitter.setStretchFactor(1, 0)   # options — collapses to header
        splitter.setStretchFactor(2, 0)   # summary — fixed height
        splitter.setStretchFactor(3, 1)   # log expands to fill freed space

        layout = QVBoxLayout(self)
        layout.addWidget(QLabel("Run Cluster Mean-Field (CMF) orbital optimization before continuing?"))
        layout.addWidget(splitter, stretch=1)
        layout.addLayout(btn_row)
        self.setLayout(layout)
        self.setMinimumSize(0, 0)
        self._log.setMinimumSize(0, 0)

        self._on_method_changed(self._method.currentText())

    def _browse_julia_project(self) -> None:
        d = QFileDialog.getExistingDirectory(self, "Select TPSChem.jl project directory",
                                              self._julia_project_edit.text())
        if d:
            self._julia_project_edit.setText(d)
            cfg.set_value("julia_project", d)   # persist for next launch

    def _on_build_pycall(self) -> None:
        import subprocess
        julia_project = self._julia_project_edit.text()
        self._log.append_line(f"[setup] Instantiating & building PyCall in {julia_project} ...")
        # Step 1: Pkg.instantiate() to resolve/download all dependencies
        # Step 2: Pkg.build("PyCall") to link against the current Python
        script = (
            'import Pkg; '
            f'ENV["PYTHON"] = {json.dumps(sys.executable)}; '
            f'Pkg.activate({json.dumps(julia_project)}); '
            'Pkg.instantiate(); '
            'Pkg.build("PyCall"); '
            'Pkg.precompile(); '
            'println("[setup] done")'
        )
        cmd = [self._julia_bin, "-e", script]
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
            for line in (result.stdout + result.stderr).splitlines():
                self._log.append_line(line)
            if result.returncode == 0:
                self._log.append_line("[setup] Pkg.instantiate + PyCall build OK.")
            else:
                self._log.append_line(f"[setup] failed (code {result.returncode}).")
        except Exception as exc:
            self._log.append_line(f"[setup] error: {exc}")

    def set_inputs(self, active_space_dir: str | Path, clusters: ClusterSet, output_dir: str | Path) -> None:
        self._active_space_dir = Path(active_space_dir)
        self._clusters = clusters
        self._output_dir = Path(output_dir)
        self._summary_pane.setPlainText(self._cluster_summary(clusters))

    @staticmethod
    def _cluster_summary(clusters: ClusterSet) -> str:
        lines = [
            f"{'ID':<4} {'Name':<14} {'n_orb':<7} {'init_fspace (α,β)':<20} {'orbitals'}",
            "-" * 72,
        ]
        for c in clusters.clusters:
            orb_str = str(c.orbitals) if len(c.orbitals) <= 8 else (
                "[" + ", ".join(str(o) for o in c.orbitals[:6]) + f", … ({len(c.orbitals)} total)]"
            )
            lines.append(
                f"{c.id:<4} {c.name:<14} {c.n_orb:<7} {str(c.fspace):<20} {orb_str}"
            )
        lines += [
            "",
            f"clusters    = {clusters.as_mocluster_literal()}",
            f"init_fspace = {clusters.as_init_fspace_literal()}",
            f"ansatze     = {clusters.as_ansatze_literal()}",
        ]
        return "\n".join(lines)

    def _on_method_changed(self, text: str) -> None:
        self._newton_box.setVisible("Newton" in text)

    def _set_running(self, running: bool) -> None:
        """Toggle the button row between idle and running states."""
        self._run_btn.setEnabled(not running)
        self._skip_btn.setEnabled(not running)
        self._cancel_btn.setEnabled(running)
        self._restart_btn.setEnabled(running)

    def _on_cancel(self) -> None:
        """Stop the running CMF job (or abort a pending setup phase)."""
        self._cancelled = True
        self._restart_pending = False
        if self._worker is not None and self._worker.is_running():
            self._worker.kill()
        self._log.append_line("[cmf] cancelled by user.")
        self._set_running(False)

    def _on_restart(self) -> None:
        """Kill the current job and immediately start a fresh CMF run."""
        self._log.append_line("[cmf] restarting…")
        if self._worker is not None and self._worker.is_running():
            self._restart_pending = True
            self._worker.kill()   # _on_finished re-launches once it stops
        else:
            self._on_run()

    def _on_run(self) -> None:
        if self._active_space_dir is None or self._clusters is None or self._output_dir is None:
            self._log.append_line("[cmf] inputs not set.")
            return

        self._cancelled = False
        self._set_running(True)
        julia_project = self._julia_project_edit.text()

        # --- Phase 1: Pkg.instantiate + Pkg.build("PyCall") in a background
        # thread so the GUI stays responsive. Phase 2 (CMF) starts on success.
        self._log.append_line(f"[setup] Pkg.instantiate + Pkg.build(PyCall) in {julia_project} ...")

        julia_bin = self._julia_bin

        def _do_setup():
            import subprocess
            script = (
                'import Pkg; '
                f'ENV["PYTHON"] = {json.dumps(sys.executable)}; '
                f'Pkg.activate({json.dumps(julia_project)}); '
                'Pkg.instantiate(); '
                'Pkg.build("PyCall"); '
                'Pkg.precompile(); '
                'println("[setup] environment ready")'
            )
            result = subprocess.run(
                [julia_bin, "-e", script],
                capture_output=True, text=True, timeout=600,
            )
            # Print all output so it reaches line_received
            for line in (result.stdout + result.stderr).splitlines():
                print(line)
            if result.returncode != 0:
                raise RuntimeError(f"Pkg setup failed (code {result.returncode})")

        self._setup_worker = _CallableWorker(_do_setup, parent=self,
                                              log_path=self._output_dir.parent / "asbuilder.log")
        self._setup_worker.line_received.connect(self._log.append_line)
        self._setup_worker.finished_ok.connect(lambda _: self._launch_cmf())
        self._setup_worker.failed.connect(self._on_setup_failed)
        self._setup_worker.start()

    def _launch_cmf(self) -> None:
        """Phase 2: render and run the CMF driver (called after setup succeeds)."""
        if self._cancelled:
            # User cancelled while Pkg setup was still running.
            self._cancelled = False
            self._set_running(False)
            return
        method_text = self._method.currentText()
        use_newton = "Newton" in method_text
        julia_method = "diis" if "DIIS" in method_text else "bfgs"

        ctx = dict(
            h0_path=str(self._active_space_dir / "h0.npy"),
            h1_path=str(self._active_space_dir / "h1.npy"),
            h2_path=str(self._active_space_dir / "h2.npy"),
            clusters_literal=self._clusters.as_mocluster_literal(),
            init_fspace_literal=self._clusters.as_init_fspace_literal(),
            ansatze_literal=self._clusters.as_ansatze_literal(),
            output_dir=str(self._output_dir),
            use_newton=use_newton,
            method=julia_method,
            max_iter_oo=self._max_iter.value(),
            gconv=self._tol_oo.text(),
            tol_oo=self._tol_oo.text(),
            tol_d1=self._tol_d1.text(),
            tol_ci=self._tol_ci.text(),
            zero_intra_rots=str(self._zero_intra.isChecked()).lower(),
            sequential=str(self._sequential.isChecked()).lower(),
            verbose=self._verbose.value(),
        )
        script_path = write_driver("driver_cmf.jl.j2", ctx, self._output_dir / "driver_cmf.jl")
        self._log.append_line(f"[cmf] wrote {script_path}, launching Julia...")

        log_path = self._output_dir.parent / "asbuilder.log"
        julia_project = self._julia_project_edit.text()
        self._worker = JuliaProcessWorker(
            script_path, julia_project,
            julia_bin=self._julia_bin, parent=self, log_path=log_path,
            out_path=self._output_dir / "cmf.out",
            threads=1,
        )
        self._worker.line_received.connect(self._log.append_line)
        self._worker.finished_ok.connect(self._on_finished)
        self._worker.failed.connect(self._on_failed)
        self._worker.start()

    def _on_setup_failed(self, message: str) -> None:
        self._set_running(False)
        if self._cancelled:
            self._cancelled = False
            return
        self._log.append_line(f"[setup] FAILED: {message}")

    def _on_finished(self, exit_code: int) -> None:
        # Restart requested mid-run: the kill just landed, start a fresh run.
        if self._restart_pending:
            self._restart_pending = False
            self._on_run()
            return
        self._set_running(False)
        # Cancelled run: the non-zero exit code is expected, stay quiet.
        if self._cancelled:
            self._cancelled = False
            self._log.append_line("[cmf] run stopped.")
            return
        if exit_code == 0:
            result_path = self._output_dir / "cmf_result.jld2"
            self._log.append_line(f"[cmf] finished OK -> {result_path}")
            self.cmf_done.emit(str(result_path))
        else:
            self._log.append_line(f"[cmf] julia exited with code {exit_code}")

    def _on_failed(self, message: str) -> None:
        if self._restart_pending:
            self._restart_pending = False
            self._on_run()
            return
        self._set_running(False)
        if self._cancelled:
            self._cancelled = False
            return
        self._log.append_line(f"[cmf] FAILED: {message}")
