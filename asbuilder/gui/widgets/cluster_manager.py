"""
Cluster manager dock: add/remove/rename/color-code clusters, set
(n_alpha, n_beta) per cluster, and pick which cluster is "active" (i.e.
which one the orbital table / VibeMol click bridge assigns clicked
orbitals to).

Owns a `ClusterSet` (asbuilder.cluster.state) and is the single place that
mutates it from the GUI side -- other widgets (OrbitalTable, the future
webview click bridge) read from it via `clusters_changed` but don't mutate
it directly, so there's one code path for "an orbital moved cluster."
"""

from __future__ import annotations

from PyQt6.QtCore import pyqtSignal
from PyQt6.QtGui import QColor, QIcon, QPixmap
from PyQt6.QtWidgets import (
    QAbstractItemView,
    QCheckBox,
    QColorDialog,
    QFormLayout,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from asbuilder.cluster.state import Cluster, ClusterSet


def _swatch(color: str) -> QIcon:
    pix = QPixmap(16, 16)
    pix.fill(QColor(color))
    return QIcon(pix)


class ClusterManager(QWidget):
    clusters_changed = pyqtSignal()          # any add/remove/rename/color/fspace change
    active_cluster_changed = pyqtSignal(int)  # cluster id now active, for click-to-assign

    def __init__(self, cluster_set: ClusterSet | None = None, parent=None) -> None:
        super().__init__(parent)
        self.clusters = cluster_set if cluster_set is not None else ClusterSet()

        self._list = QListWidget()
        self._list.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self._list.currentRowChanged.connect(self._on_selection_changed)

        add_btn = QPushButton("Add cluster")
        add_btn.clicked.connect(self._on_add)
        remove_btn = QPushButton("Remove cluster")
        remove_btn.clicked.connect(self._on_remove)
        btn_row = QHBoxLayout()
        btn_row.addWidget(add_btn)
        btn_row.addWidget(remove_btn)

        # -- selected-cluster editor -----------------------------------
        self._name_edit = QLineEdit()
        self._name_edit.editingFinished.connect(self._on_name_edited)

        self._color_btn = QPushButton("Choose color")
        self._color_btn.clicked.connect(self._on_pick_color)

        self._n_alpha = QSpinBox()
        self._n_alpha.setRange(0, 999)
        self._n_alpha.valueChanged.connect(self._on_fspace_edited)
        self._n_beta = QSpinBox()
        self._n_beta.setRange(0, 999)
        self._n_beta.valueChanged.connect(self._on_fspace_edited)

        # -- SPADE orbital type filter (shown only in SPADE mode) ----------
        self._ao_group = QGroupBox("AO shells for SPADE projector")
        self._ao_group.setVisible(False)
        ao_inner = QVBoxLayout(self._ao_group)
        ao_inner.addWidget(QLabel("Which shells from this cluster's atoms to include:"))
        self._ao_shell_widget = QWidget()
        self._ao_shell_grid = QGridLayout(self._ao_shell_widget)
        self._ao_shell_grid.setContentsMargins(0, 0, 0, 0)
        ao_inner.addWidget(self._ao_shell_widget)
        ao_inner.addWidget(QLabel("(leave all unchecked = include all shells)"))
        self._ao_checks: dict[str, QCheckBox] = {}
        self._build_ao_checks(["s", "p", "d", "f"])  # sensible default before mol loads

        form = QFormLayout()
        form.addRow("Name", self._name_edit)
        form.addRow("Color", self._color_btn)
        form.addRow("n_alpha", self._n_alpha)
        form.addRow("n_beta", self._n_beta)

        layout = QVBoxLayout(self)
        layout.addWidget(self._list)
        layout.addLayout(btn_row)
        layout.addLayout(form)
        layout.addWidget(self._ao_group)
        self.setLayout(layout)

        self._refresh_list()
        self._set_editor_enabled(False)

    # -- public API used by screens ---------------------------------------

    def reset(self, cluster_set: ClusterSet | None = None) -> None:
        """Replace the internal ClusterSet with a fresh one (or the given one)
        and clear the list. Call this when a new molecule is loaded."""
        self.clusters = cluster_set if cluster_set is not None else ClusterSet()
        self._refresh_list()
        self._set_editor_enabled(False)

    def set_available_shells(self, shells: list[str]) -> None:
        """Rebuild the AO shell checkboxes from the molecule's actual shells.
        Called from ViewerScreen.load() after mol.ao_labels() is parsed."""
        self._build_ao_checks(shells if shells else ["s", "p", "d", "f"])

    def _build_ao_checks(self, shells: list[str]) -> None:
        for cb in self._ao_checks.values():
            self._ao_shell_grid.removeWidget(cb)
            cb.deleteLater()
        self._ao_checks.clear()
        for i, shell in enumerate(shells):
            cb = QCheckBox(shell)
            cb.stateChanged.connect(self._on_ao_types_changed)
            self._ao_checks[shell] = cb
            self._ao_shell_grid.addWidget(cb, i // 5, i % 5)

    def set_spade_mode(self, spade: bool) -> None:
        """Show/hide the SPADE-specific AO type filter."""
        self._ao_group.setVisible(spade)

    def active_cluster_id(self) -> int | None:
        c = self._current_cluster()
        return c.id if c is not None else None

    def validate(self) -> list[str]:
        return self.clusters.validate()

    def refresh(self) -> None:
        """Re-render the cluster list (e.g. orbital/electron counts) after
        something outside this widget mutated `self.clusters` -- e.g. an
        orbital assignment made via the orbital table or a future Tier B
        click-on-isosurface bridge."""
        active = self._current_cluster()
        self._refresh_list(select_id=active.id if active is not None else None)

    # -- internal --------------------------------------------------------

    def _current_cluster(self) -> Cluster | None:
        row = self._list.currentRow()
        if 0 <= row < len(self.clusters.clusters):
            return self.clusters.clusters[row]
        return None

    def _refresh_list(self, select_id: int | None = None) -> None:
        self._list.blockSignals(True)
        self._list.clear()
        select_row = -1
        for i, c in enumerate(self.clusters.clusters):
            item = QListWidgetItem(_swatch(c.color), f"{c.name}  ({c.n_orb} orb, {c.n_elec} e-)")
            self._list.addItem(item)
            if select_id is not None and c.id == select_id:
                select_row = i
        self._list.blockSignals(False)
        if select_row >= 0:
            self._list.setCurrentRow(select_row)
        elif self._list.count() > 0 and self._list.currentRow() < 0:
            self._list.setCurrentRow(0)

    def _set_editor_enabled(self, enabled: bool) -> None:
        for w in (self._name_edit, self._color_btn, self._n_alpha, self._n_beta):
            w.setEnabled(enabled)

    def _load_editor_from(self, c: Cluster) -> None:
        self._name_edit.blockSignals(True)
        self._n_alpha.blockSignals(True)
        self._n_beta.blockSignals(True)
        self._name_edit.setText(c.name)
        self._n_alpha.setValue(c.fspace[0])
        self._n_beta.setValue(c.fspace[1])
        self._name_edit.blockSignals(False)
        self._n_alpha.blockSignals(False)
        self._n_beta.blockSignals(False)
        for t, cb in self._ao_checks.items():
            cb.blockSignals(True)
            cb.setChecked(t in c.ao_types)
            cb.blockSignals(False)

    # -- slots -------------------------------------------------------------

    def _on_add(self) -> None:
        c = self.clusters.add_cluster()
        self._refresh_list(select_id=c.id)
        self.clusters_changed.emit()

    def _on_remove(self) -> None:
        c = self._current_cluster()
        if c is None:
            return
        self.clusters.remove_cluster(c.id)
        self._refresh_list()
        self.clusters_changed.emit()

    def _on_selection_changed(self, _row: int) -> None:
        c = self._current_cluster()
        if c is None:
            self._set_editor_enabled(False)
            return
        self._set_editor_enabled(True)
        self._load_editor_from(c)
        self.active_cluster_changed.emit(c.id)

    def _on_name_edited(self) -> None:
        c = self._current_cluster()
        if c is None:
            return
        c.name = self._name_edit.text() or c.name
        self._refresh_list(select_id=c.id)
        self.clusters_changed.emit()

    def _on_pick_color(self) -> None:
        c = self._current_cluster()
        if c is None:
            return
        color = QColorDialog.getColor(QColor(c.color), self, "Cluster color")
        if color.isValid():
            c.color = color.name()
            self._refresh_list(select_id=c.id)
            self.clusters_changed.emit()

    def _on_fspace_edited(self, _value: int) -> None:
        c = self._current_cluster()
        if c is None:
            return
        c.fspace = (self._n_alpha.value(), self._n_beta.value())
        self._refresh_list(select_id=c.id)
        self.clusters_changed.emit()

    def _on_ao_types_changed(self, _state: int) -> None:
        c = self._current_cluster()
        if c is None:
            return
        c.ao_types = [t for t, cb in self._ao_checks.items() if cb.isChecked()]
        self.clusters_changed.emit()
