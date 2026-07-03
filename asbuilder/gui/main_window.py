"""
Main window: a QStackedWidget routing between the 7 pipeline screens, with
a single ProjectState + ClusterSet as the shared source of truth that gets
passed between them (see project.STAGES for the canonical stage list this
routing mirrors).

This file owns the wiring; each screen only knows about its own inputs/
outputs (via constructor args + Qt signals), not about its neighbors, so
screens stay testable/reusable in isolation -- see the design doc's
"backend engine" section for why the same principle applies one layer
down (asbuilder.* has no Qt dependency at all).
"""

from __future__ import annotations

from pathlib import Path

from PyQt6.QtCore import QEvent, QSize, QTimer
from PyQt6.QtGui import QAction
from PyQt6.QtWidgets import QMainWindow, QMenuBar, QMessageBox, QStackedWidget, QStatusBar, QToolBar


class _FlexStack(QStackedWidget):
    """QStackedWidget that never forces the window taller than the screen.

    A plain QStackedWidget reports minimumSizeHint() = max over all pages of
    each page's own minimumSizeHint(). Our tallest pages (ExportScreen ~1059px,
    CMFScreen ~730px) would therefore force QMainWindow to grow the whole
    window past the screen height, no matter how many resize() calls we make.
    Overriding the hint to (0, 0) lets the window open at whatever size we
    resize() it to; pages themselves scroll / use splitters for their content.
    """

    def minimumSizeHint(self) -> QSize:  # noqa: D102
        return QSize(0, 0)

    def sizeHint(self) -> QSize:  # noqa: D102
        return QSize(0, 0)

from asbuilder.gui.screens.active_space_screen import ActiveSpaceScreen
from asbuilder.gui.screens.cmf_screen import CMFScreen
from asbuilder.gui.screens.export_screen import ExportScreen
from asbuilder.gui.screens.integrals_summary_screen import IntegralsSummaryScreen
from asbuilder.gui.screens.load_screen import LoadScreen
from asbuilder.gui.screens.molden_screen import MoldenScreen
from asbuilder.gui.screens.viewer_screen import ViewerScreen
from asbuilder.io.chk_to_molden import load_chk, provenance
from asbuilder.project import ProjectState


class MainWindow(QMainWindow):
    def __init__(
        self,
        project_root: str | Path,
        julia_bin: str = "julia",
        julia_project: Path | None = None,
        vibemol_root: str | Path | None = None,
    ) -> None:
        super().__init__()
        self.setWindowTitle(f"TPS-chemistry — {Path(project_root).name}")

        project_root = Path(project_root)
        self.project = (
            ProjectState.load(project_root)
            if (project_root / "project.json").exists()
            else ProjectState.create(project_root)
        )

        # Julia project: prefer explicit arg, else look for TPSChem.jl sibling,
        # else fall back to project root (will fail gracefully with a clear error).
        if julia_project is None:
            candidate = Path(__file__).parent.parent.parent.parent / "TPSChem.jl"
            julia_project = candidate if (candidate / "Project.toml").exists() else project_root

        # -- build screens ---------------------------------------------
        self.load_screen = LoadScreen(self.project.root)
        self.molden_screen = MoldenScreen()
        self.viewer_screen = ViewerScreen(vibemol_root=vibemol_root)
        self.active_space_screen = ActiveSpaceScreen()
        self.integrals_screen = IntegralsSummaryScreen(vibemol_root=vibemol_root)
        self.cmf_screen = CMFScreen(julia_project_dir=julia_project, julia_bin=julia_bin)
        self.export_screen = ExportScreen()

        self.stack = _FlexStack()
        # Allow the stack to be any height so QTimer-deferred resize works.
        # Without this, QStackedWidget enforces the maximum minimumSizeHint()
        # of all pages (dominated by QWebEngineView / OrbitalTable) and Qt
        # silently ignores resize() calls that are smaller than that minimum.
        self.stack.setMinimumSize(0, 0)
        self._screen_meta = [
            (self.load_screen,          "1. Load"),
            (self.molden_screen,        "2. Molden"),
            (self.viewer_screen,        "3. Clusters"),
            (self.active_space_screen,  "4. Active Space"),
            (self.integrals_screen,     "5. Integrals"),
            (self.cmf_screen,           "6. CMF"),
            (self.export_screen,        "7. TPSCI/Export"),
        ]
        for screen, _ in self._screen_meta:
            self.stack.addWidget(screen)
        self.setCentralWidget(self.stack)

        # -- Navigation toolbar (always visible, lets user jump back) --------
        self._nav_toolbar = QToolBar("Pipeline steps")
        self._nav_toolbar.setMovable(False)
        self._nav_actions: list[QAction] = []
        for screen, label in self._screen_meta:
            act = QAction(label, self)
            act.setCheckable(True)
            act.triggered.connect(lambda checked, s=screen: self._goto(s))
            self._nav_toolbar.addAction(act)
            self._nav_actions.append(act)
        self.addToolBar(self._nav_toolbar)

        self.setStatusBar(QStatusBar())
        self._update_status()

        # -- menu bar ---------------------------------------------------
        menubar = self.menuBar()
        tools_menu = menubar.addMenu("Tools")
        setup_act = QAction("Julia / TPSChem.jl setup…", self)
        setup_act.triggered.connect(self._open_setup)
        tools_menu.addAction(setup_act)

        self._wire_signals()

        # Set size BEFORE show() so Qt never displays the window taller than
        # the screen.  setMaximumHeight cap is lifted in showEvent after Qt
        # has finished positioning, leaving the window freely resizable.
        from asbuilder.gui._screen_util import screen_capped_size
        w, h = screen_capped_size(1400, 700, pct=0.65)
        self.setMaximumHeight(h)
        self.resize(w, h)
        self._initial_h = h

    # -- initial sizing -------------------------------------------------

    def showEvent(self, event: QEvent) -> None:
        super().showEvent(event)
        if hasattr(self, '_initial_h'):
            from asbuilder.gui._screen_util import center_on_screen
            center_on_screen(self)
            # Release the height cap so user can resize freely
            QTimer.singleShot(300, lambda: self.setMaximumHeight(16777215))
            del self._initial_h

    # -- routing --------------------------------------------------------

    def _goto(self, screen) -> None:
        self.stack.setCurrentWidget(screen)
        # Highlight the active step in the toolbar
        for i, (s, _) in enumerate(self._screen_meta):
            self._nav_actions[i].setChecked(s is screen)
        self._update_status()

    def _update_status(self) -> None:
        self.statusBar().showMessage(f"project: {self.project.root}  |  stage: {self.project.stage}")

    def _wire_signals(self) -> None:
        # Screen 1 -> 2
        self.load_screen.chk_selected.connect(self._on_chk_selected)
        self.load_screen.continue_requested.connect(self._on_load_continue)
        # Resume / jump shortcuts
        self.load_screen.jump_to_cmf.connect(self._on_jump_to_cmf)
        self.load_screen.jump_to_export.connect(self._on_jump_to_export)

        # Screen 2 -> 3
        self.molden_screen.molden_ready.connect(self._on_molden_ready)
        self.molden_screen.failed.connect(self._on_error)

        # Screen 3 -> 4
        self.viewer_screen.build_requested.connect(self._on_build_requested)

        # Screen 4 -> 5 (or back to 3)
        self.active_space_screen.active_space_built.connect(self._on_active_space_built)
        self.active_space_screen.redo_clustering_requested.connect(lambda: self._goto(self.viewer_screen))

        # Screen 5 -> 6
        self.integrals_screen.continue_requested.connect(lambda: self._goto(self.cmf_screen))

        # Screen 6 -> 7
        self.cmf_screen.cmf_done.connect(self._on_cmf_done)
        self.cmf_screen.cmf_skipped.connect(self._on_cmf_skipped)

    # -- slots ------------------------------------------------------------

    def _on_chk_selected(self, chk_path: str) -> None:
        self.project.import_chk(Path(chk_path))
        try:
            chk = load_chk(self.project.root / "input.chk")
        except Exception as exc:  # noqa: BLE001
            self._on_error(f"could not load checkpoint: {exc}")
            return
        prov = provenance(chk)
        self.project.set_provenance(prov)
        self.project.advance("chk_loaded")
        self.load_screen.show_provenance(prov)
        self._update_status()

    def _on_load_continue(self) -> None:
        self.molden_screen.set_paths(
            self.project.root / "input.chk",
            self.project.root / "orbitals.molden",
        )
        self._goto(self.molden_screen)

    def _on_molden_ready(self, chk, molden_path: str) -> None:
        self.project.advance("molden_generated")
        self.viewer_screen.load(chk, molden_path, chk_path=str(self.project.root / "input.chk"))
        self._goto(self.viewer_screen)

    def _on_build_requested(self, chk, clusters) -> None:
        clusters.save(self.project.root)
        self.project.advance("clusters_defined")
        self.active_space_screen.set_inputs(chk, clusters, self.project.root / "active_space")
        self._goto(self.active_space_screen)

    def _on_active_space_built(self, output_dir: str) -> None:
        self.project.advance("active_space_built")
        self.integrals_screen.set_summary(self.viewer_screen.cluster_set, output_dir)
        self._goto(self.integrals_screen)
        self.cmf_screen.set_inputs(output_dir, self.viewer_screen.cluster_set, self.project.root / "cmf")
        self.load_screen.refresh()   # enable "Jump to CMF" button

    def _on_cmf_done(self, cmf_result_path: str) -> None:
        self.project.advance("cmf_run", {"skipped": False})
        julia_project = self.cmf_screen._julia_project_edit.text()
        self.export_screen.set_inputs(
            cmf_result_path, self.project.root / "export",
            julia_project=julia_project,
        )
        self._goto(self.export_screen)
        self.load_screen.refresh()   # enable "Jump to TPSCI/Export" button

    def _on_cmf_skipped(self) -> None:
        self.project.advance("cmf_run", {"skipped": True})
        # TODO: driver_tpsci.jl.j2/driver_spt.jl.j2 currently assume a
        # cmf_result.jld2 bundle (ints already CMF-rotated, per
        # ClusterMeanField's own README pattern). Skipping CMF means there
        # is no such bundle -- exporting straight from the bare
        # h0/h1/h2.npy needs either a third template that builds
        # clustered_ham directly from InCoreInts without a CMF reference,
        # or a tiny Julia shim that wraps the bare ints into the same
        # jld2 shape the export templates expect. Not built yet.
        QMessageBox.information(
            self,
            "Not yet implemented",
            "Skipping CMF isn't wired up to the export step yet -- the TPSCI/SPT "
            "driver templates currently assume a CMF-produced cmf_result.jld2. "
            "Run CMF for now, or extend driver_tpsci.jl.j2/driver_spt.jl.j2 to "
            "accept bare integrals directly.",
        )

    def _on_jump_to_cmf(self, active_space_dir: str, clusters_json: str) -> None:
        """Resume from saved h0/h1/h2.npy + clusters.json → jump straight to CMF."""
        from asbuilder.cluster.state import ClusterSet
        try:
            clusters = ClusterSet.load(Path(clusters_json).parent)
        except Exception as exc:
            QMessageBox.warning(self, "Resume error", f"Could not load clusters.json: {exc}")
            return
        cmf_output = self.project.root / "cmf"
        self.cmf_screen.set_inputs(active_space_dir, clusters, cmf_output)
        self.project.advance("active_space_built")
        self._goto(self.cmf_screen)
        self._update_status()

    def _on_jump_to_export(self, cmf_result_path: str) -> None:
        """Resume from saved cmf_result.jld2 → jump straight to TPSCI/Export."""
        julia_project = self.cmf_screen._julia_project_edit.text()
        self.export_screen.set_inputs(
            cmf_result_path,
            self.project.root / "export",
            julia_project=julia_project,
        )
        self.project.advance("cmf_run")
        self._goto(self.export_screen)
        self._update_status()

    def _open_setup(self) -> None:
        from asbuilder.gui.screens.setup_screen import SetupDialog
        julia_bin = self.cmf_screen._julia_bin
        dlg = SetupDialog(julia_bin=julia_bin, parent=self)
        if dlg.exec() and dlg.chosen_path:
            self.cmf_screen._julia_project_edit.setText(str(dlg.chosen_path))

    def _on_error(self, message: str) -> None:
        QMessageBox.warning(self, "Error", message)
