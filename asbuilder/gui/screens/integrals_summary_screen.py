"""Screen 5: summary table + orbital viewer after the active space build.

Shows:
  - A dropdown to switch between "All active (Cact.molden)" and per-cluster
    moldens.  Both are written by localize_integrals._save_moldens before
    h0/h1/h2.npy so the user can inspect orbitals first.
  - A VibeMol pane (same widget as the viewer screen) showing the selected
    molden.
  - An editable fspace table: n_alpha / n_beta cells can be corrected before
    clicking "Save & Continue", which writes the values back to the ClusterSet.
"""

from __future__ import annotations

import json
from pathlib import Path

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtWidgets import (
    QComboBox,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from asbuilder.cluster.state import ClusterSet
from asbuilder.gui.widgets.webview_panel import WebViewPanel

_COLUMNS = ["Cluster", "n_orb", "n_alpha (edit)", "n_beta (edit)", "Orbitals"]
_COL_NA = 2
_COL_NB = 3


class IntegralsSummaryScreen(QWidget):
    continue_requested = pyqtSignal()

    def __init__(self, vibemol_root=None, parent=None) -> None:
        super().__init__(parent)
        self._clusters: ClusterSet | None = None
        self._output_dir: Path | None = None

        # --- left panel: table + controls ---
        self._path_label = QLabel("")
        self._info_label = QLabel(
            "Inspect the orbitals in the viewer, then verify n_alpha / n_beta "
            "in the table. Edit if needed, then click Save & Continue."
        )
        self._info_label.setWordWrap(True)

        self._molden_selector = QComboBox()
        self._molden_selector.currentIndexChanged.connect(self._on_molden_changed)

        self._table = QTableWidget(0, len(_COLUMNS))
        self._table.setHorizontalHeaderLabels(_COLUMNS)
        self._table.verticalHeader().setVisible(False)

        self._continue_btn = QPushButton("Save & Continue →")
        self._continue_btn.clicked.connect(self._on_continue)

        left = QWidget()
        left_layout = QVBoxLayout(left)
        left_layout.addWidget(QLabel("<h3>Active-space orbitals — verify before CMF</h3>"))
        left_layout.addWidget(self._info_label)
        left_layout.addWidget(self._path_label)
        selector_row = QHBoxLayout()
        selector_row.addWidget(QLabel("Show orbitals:"))
        selector_row.addWidget(self._molden_selector)
        selector_row.addStretch(1)
        left_layout.addLayout(selector_row)
        left_layout.addWidget(self._table)
        left_layout.addWidget(self._continue_btn)

        # --- right panel: VibeMol viewer ---
        self._webview = WebViewPanel(vibemol_root=vibemol_root)

        splitter = QSplitter()
        splitter.setHandleWidth(8)
        splitter.setChildrenCollapsible(False)
        splitter.addWidget(left)
        splitter.addWidget(self._webview)
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([5000, 5000])   # force 50/50 initial split

        layout = QVBoxLayout(self)
        layout.addWidget(splitter)
        self.setLayout(layout)
        self.setMinimumSize(0, 0)

        self._molden_entries: list[tuple[str, Path]] = []   # (label, path)

    # ------------------------------------------------------------------

    def set_summary(self, clusters: ClusterSet, output_dir: str | Path) -> None:
        self._clusters = clusters
        self._output_dir = Path(output_dir)
        self._path_label.setText(f"h0/h1/h2.npy → {self._output_dir}")
        self._populate_table(clusters)
        self._populate_molden_selector(self._output_dir, clusters)

    def _populate_table(self, clusters: ClusterSet) -> None:
        self._table.setRowCount(len(clusters.clusters))
        for row, c in enumerate(clusters.clusters):
            name_item = QTableWidgetItem(c.name)
            name_item.setFlags(name_item.flags() & ~Qt.ItemFlag.ItemIsEditable)
            norb_item = QTableWidgetItem(str(c.n_orb))
            norb_item.setFlags(norb_item.flags() & ~Qt.ItemFlag.ItemIsEditable)
            orb_item  = QTableWidgetItem(", ".join(str(o) for o in c.orbitals))
            orb_item.setFlags(orb_item.flags() & ~Qt.ItemFlag.ItemIsEditable)

            self._table.setItem(row, 0, name_item)
            self._table.setItem(row, 1, norb_item)
            self._table.setItem(row, _COL_NA, QTableWidgetItem(str(c.fspace[0])))
            self._table.setItem(row, _COL_NB, QTableWidgetItem(str(c.fspace[1])))
            self._table.setItem(row, 4, orb_item)
        self._table.resizeColumnsToContents()

    def _populate_molden_selector(self, out: Path, clusters: ClusterSet) -> None:
        self._molden_selector.blockSignals(True)
        self._molden_selector.clear()
        self._molden_entries.clear()

        # All active combined
        cact = out / "Cact.molden"
        if cact.exists():
            self._molden_entries.append(("All active orbitals (Cact.molden)", cact))
            self._molden_selector.addItem("All active orbitals (Cact.molden)")

        # Per-cluster — read cluster_map.json if present, else guess filenames
        cluster_map_path = out / "cluster_map.json"
        if cluster_map_path.exists():
            try:
                cmap = json.loads(cluster_map_path.read_text())
                for cid, info in cmap.items():
                    molden_path = out / info["molden"]
                    if molden_path.exists():
                        label = f"Cluster {cid}: {info['name']}  ({info['n_orb']} orb, fspace={info['fspace']})"
                        self._molden_entries.append((label, molden_path))
                        self._molden_selector.addItem(label)
            except Exception:
                pass
        else:
            # Fallback: look for cluster_N_name.molden files
            for c in clusters.clusters:
                p = out / f"cluster_{c.id}_{c.name}.molden"
                if p.exists():
                    label = f"Cluster {c.id}: {c.name}"
                    self._molden_entries.append((label, p))
                    self._molden_selector.addItem(label)

        self._molden_selector.blockSignals(False)

        # Load the first available molden into the viewer
        if self._molden_entries:
            self._load_molden(self._molden_entries[0][1])

    def _load_molden(self, path: Path) -> None:
        try:
            self._webview.load_molden(str(path))
        except Exception as e:
            self._info_label.setText(f"Viewer error: {e}")

    def _on_molden_changed(self, index: int) -> None:
        if 0 <= index < len(self._molden_entries):
            self._load_molden(self._molden_entries[index][1])

    def _on_continue(self) -> None:
        if self._clusters is not None:
            for row, c in enumerate(self._clusters.clusters):
                try:
                    na = int(self._table.item(row, _COL_NA).text())
                    nb = int(self._table.item(row, _COL_NB).text())
                    c.fspace = (na, nb)
                except (ValueError, AttributeError):
                    pass
        self.continue_requested.emit()
