"""
"New Calculation" dialog: collects xyz / basis / method / charge / spin and
runs asbuilder.io.run_scf.run_scf in a background thread (ScfWorker), so
the GUI doesn't need its own copy of the SCF-setup logic -- it calls the
exact same function the notebook does (once run_scf.py is filled in with
the real notebook logic; currently a generic RHF/UHF/ROHF placeholder,
see io/run_scf.py's module docstring).
"""

from __future__ import annotations

from pathlib import Path

from PyQt6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFormLayout,
    QLabel,
    QLineEdit,
    QPlainTextEdit,
    QScrollArea,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from asbuilder.gui._screen_util import fit_to_screen
from asbuilder.gui.widgets.log_pane import LogPane
from asbuilder.gui.workers import ScfWorker


class NewCalculationDialog(QDialog):
    def __init__(self, default_chk_path: str | Path, parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("New Calculation")
        fit_to_screen(self, 540, 560)

        self.result_chk_path: Path | None = None
        self._worker: ScfWorker | None = None

        self._xyz = QPlainTextEdit()
        self._xyz.setPlaceholderText("Cr 0.0 0.0 0.0\nCr 0.0 0.0 2.5\n...")
        self._xyz.setFixedHeight(120)

        self._basis = QLineEdit("sto-3g")
        self._method = QComboBox()
        self._method.addItems(["RHF", "UHF", "ROHF"])
        self._charge = QSpinBox()
        self._charge.setRange(-10, 10)
        self._spin = QSpinBox()
        self._spin.setRange(0, 20)
        self._spin.setToolTip("PySCF convention: 2S = n_alpha - n_beta")

        self._density_fit = QCheckBox("Density fitting (DF/RI)")
        self._density_fit.setToolTip("mf.density_fit() — use RI approximation for 2e integrals")
        self._auxbasis = QLineEdit()
        self._auxbasis.setPlaceholderText("auto (PySCF default)")
        self._auxbasis.setToolTip("Auxiliary basis for DF; leave blank for PySCF auto-select")
        self._auxbasis.setEnabled(False)
        self._density_fit.toggled.connect(self._auxbasis.setEnabled)

        self._newton = QCheckBox("Newton solver (2nd order)")
        self._newton.setToolTip("mf.newton() — second-order NR, more robust for difficult convergence")

        form = QFormLayout()
        form.addRow("XYZ (atom lines)", self._xyz)
        form.addRow("Basis set", self._basis)
        form.addRow("Method", self._method)
        form.addRow("Charge", self._charge)
        form.addRow("Spin (2S)", self._spin)
        form.addRow("", self._density_fit)
        form.addRow("Aux basis", self._auxbasis)
        form.addRow("", self._newton)

        self._status = QLabel("")
        self._log = LogPane()
        self._log.setFixedHeight(140)

        self._buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Cancel)
        self._run_btn = self._buttons.addButton("Run SCF", QDialogButtonBox.ButtonRole.ActionRole)
        self._run_btn.clicked.connect(self._on_run)
        self._buttons.rejected.connect(self.reject)

        self._default_chk_path = Path(default_chk_path)

        body = QWidget()
        body_layout = QVBoxLayout(body)
        body_layout.addLayout(form)
        body_layout.addWidget(self._status)
        body_layout.addWidget(self._log)
        body_layout.addStretch(1)

        scroll = QScrollArea()
        scroll.setWidget(body)
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QScrollArea.Shape.NoFrame)

        layout = QVBoxLayout(self)
        layout.addWidget(scroll, stretch=1)
        layout.addWidget(self._buttons)
        self.setLayout(layout)

    def _on_run(self) -> None:
        xyz = self._xyz.toPlainText().strip()
        if not xyz:
            self._status.setText("Enter atom coordinates first.")
            return

        use_df = self._density_fit.isChecked()
        auxbasis_text = self._auxbasis.text().strip()
        auxbasis = auxbasis_text if auxbasis_text else None
        use_newton = self._newton.isChecked()

        self._run_btn.setEnabled(False)
        self._status.setText("Running SCF...")
        self._log.append_line(
            f"[new_calc] basis={self._basis.text()} method={self._method.currentText()}"
            f" df={use_df} newton={use_newton}"
            + (f" auxbasis={auxbasis}" if auxbasis else "")
        )

        self._worker = ScfWorker(
            xyz=xyz,
            basis=self._basis.text(),
            method=self._method.currentText(),
            charge=self._charge.value(),
            spin=self._spin.value(),
            chk_path=self._default_chk_path,
            log_path=self._default_chk_path.with_suffix(".log"),
            density_fit=use_df,
            auxbasis=auxbasis,
            newton=use_newton,
            parent=self,
        )
        self._worker.line_received.connect(self._log.append_line)
        self._worker.finished_ok.connect(self._on_finished)
        self._worker.failed.connect(self._on_failed)
        self._worker.start()

    def _on_finished(self, result) -> None:
        self._log.append_line(f"[new_calc] converged={result.converged} E_tot={result.e_tot:.10f}")
        self._status.setText("Done.")
        self.result_chk_path = result.chk_path
        self.accept()

    def _on_failed(self, message: str) -> None:
        self._log.append_line(f"[new_calc] FAILED: {message}")
        self._status.setText("SCF failed -- see log.")
        self._run_btn.setEnabled(True)
