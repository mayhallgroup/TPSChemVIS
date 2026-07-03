"""
Project picker dialog — shown on launch when no project_dir is given on the CLI.

The user can:
  - Double-click a recent project to open it immediately.
  - Click "Open existing…" to browse to any .qcproj folder.
  - Click "New project…" to choose a parent folder and give the project a name.
"""
from __future__ import annotations

from pathlib import Path

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QGroupBox,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QVBoxLayout,
)

import asbuilder.config as cfg
from asbuilder.gui._screen_util import fit_to_screen


class ProjectPickerDialog(QDialog):
    """Modal dialog that lets the user pick or create a project folder."""

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("TPS-chemistry — Open or create project")
        self.setMinimumWidth(480)
        fit_to_screen(self, 620, 460)
        self.chosen_path: Path | None = None

        # --- recent list ---
        recent_group = QGroupBox("Recent projects")
        recent_layout = QVBoxLayout(recent_group)
        self._recent_list = QListWidget()
        self._recent_list.setAlternatingRowColors(True)
        self._populate_recent()
        self._recent_list.itemDoubleClicked.connect(self._on_recent_double_click)
        self._recent_list.currentItemChanged.connect(self._on_recent_selection)
        recent_layout.addWidget(self._recent_list)

        # --- action buttons ---
        new_btn = QPushButton("New project…")
        new_btn.clicked.connect(self._on_new)
        open_btn = QPushButton("Open existing project…")
        open_btn.clicked.connect(self._on_open)
        btn_row = QHBoxLayout()
        btn_row.addWidget(new_btn)
        btn_row.addWidget(open_btn)
        btn_row.addStretch(1)

        # --- selection label ---
        self._sel_label = QLabel("No project selected.")
        self._sel_label.setWordWrap(True)

        # --- OK / Cancel ---
        self._box = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        self._box.button(QDialogButtonBox.StandardButton.Ok).setEnabled(False)
        self._box.accepted.connect(self.accept)
        self._box.rejected.connect(self.reject)

        layout = QVBoxLayout(self)
        layout.addWidget(recent_group)
        layout.addLayout(btn_row)
        layout.addWidget(self._sel_label)
        layout.addWidget(self._box)

    # ------------------------------------------------------------------

    def _populate_recent(self) -> None:
        self._recent_list.clear()
        for p in cfg.recent_projects():
            item = QListWidgetItem(str(p))
            item.setData(Qt.ItemDataRole.UserRole, p)
            item.setToolTip(str(p))
            self._recent_list.addItem(item)
        if self._recent_list.count() == 0:
            placeholder = QListWidgetItem("(no recent projects)")
            placeholder.setFlags(Qt.ItemFlag.NoItemFlags)
            self._recent_list.addItem(placeholder)

    def _set_path(self, path: Path) -> None:
        self.chosen_path = path
        self._sel_label.setText(f"<b>Selected:</b> {path}")
        self._box.button(QDialogButtonBox.StandardButton.Ok).setEnabled(True)

    # ------------------------------------------------------------------

    def _on_recent_selection(self, current: QListWidgetItem | None, _prev) -> None:
        if current is None:
            return
        path = current.data(Qt.ItemDataRole.UserRole)
        if path is not None:
            self._set_path(path)

    def _on_recent_double_click(self, item: QListWidgetItem) -> None:
        path = item.data(Qt.ItemDataRole.UserRole)
        if path is not None:
            self._set_path(path)
            self.accept()

    def _on_open(self) -> None:
        d = QFileDialog.getExistingDirectory(
            self, "Open existing project folder", str(Path.home())
        )
        if d:
            self._set_path(Path(d))

    def _on_new(self) -> None:
        parent_dir = QFileDialog.getExistingDirectory(
            self, "Choose where to create the project", str(Path.home())
        )
        if not parent_dir:
            return
        name, ok = QInputDialog.getText(
            self, "Project name", "Project folder name:",
            text="untitled.qcproj",
        )
        if ok and name.strip():
            path = Path(parent_dir) / name.strip()
            self._set_path(path)
