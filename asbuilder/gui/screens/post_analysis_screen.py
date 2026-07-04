"""Screen 8: post-analysis of a TPSChem wavefunction."""

from __future__ import annotations

import csv
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path

from PyQt6.QtCore import QPointF, QRectF, Qt, pyqtSignal
from PyQt6.QtGui import QColor, QPainter, QPainterPath, QPen, QPolygonF, QRadialGradient
from PyQt6.QtWidgets import (
    QAbstractItemView,
    QComboBox,
    QFileDialog,
    QFormLayout,
    QHeaderView,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPlainTextEdit,
    QPushButton,
    QSpinBox,
    QSplitter,
    QTabWidget,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from asbuilder.gui.widgets.log_pane import LogPane
from asbuilder.julia_bridge.runner import PINNED_JULIA_VERSION, render_driver, write_driver

_RESULT_CANDIDATES = [
    "tpsci_ct_result.jld2",
    "tpsci_result.jld2",
    "pt2_result.jld2",
    "fois_ci_result.jld2",
    "single_diagonalization_result.jld2",
    "spt_result.jld2",
    "cepa_result.jld2",
]

_FSPACE_PAIR_RE = re.compile(r"\(\s*(-?\d+)\s*,\s*(-?\d+)\s*\)")

_CT_TYPE_COLORS = {
    "MLCT": "#dc143c",
    "LMCT": "#1e90ff",
    "LLCT": "#2e8b57",
    "MMCT": "#9932cc",
    "Unknown": "#6b7280",
}

_TRANSITION_METALS = {
    "Sc",
    "Ti",
    "V",
    "Cr",
    "Mn",
    "Fe",
    "Co",
    "Ni",
    "Cu",
    "Zn",
    "Y",
    "Zr",
    "Nb",
    "Mo",
    "Tc",
    "Ru",
    "Rh",
    "Pd",
    "Ag",
    "Cd",
    "Hf",
    "Ta",
    "W",
    "Re",
    "Os",
    "Ir",
    "Pt",
    "Au",
    "Hg",
}


@dataclass
class _Atom:
    symbol: str
    x: float
    y: float
    z: float


@dataclass
class _ClusterNode:
    id: int
    label: str
    role: str
    color: str
    ref_electrons: int
    x: float | None = None
    y: float | None = None


@dataclass
class _TransferArrow:
    donor: int
    acceptor: int
    weight: float
    ct_type: str
    color: str


def _parse_fock_space(text: str) -> list[tuple[int, int]]:
    return [(int(a), int(b)) for a, b in _FSPACE_PAIR_RE.findall(text)]


def _classify_transfer(donor: int, acceptor: int, role_by_id: dict[int, str]) -> tuple[str, str]:
    donor_role = role_by_id.get(donor, "Ligand")
    acceptor_role = role_by_id.get(acceptor, "Ligand")
    if donor_role == "Metal" and acceptor_role == "Metal":
        return "MMCT", _CT_TYPE_COLORS["MMCT"]
    if donor_role == "Ligand" and acceptor_role == "Metal":
        return "LMCT", _CT_TYPE_COLORS["LMCT"]
    if donor_role == "Metal" and acceptor_role == "Ligand":
        return "MLCT", _CT_TYPE_COLORS["MLCT"]
    if donor_role == "Ligand" and acceptor_role == "Ligand":
        return "LLCT", _CT_TYPE_COLORS["LLCT"]
    return "Unknown", _CT_TYPE_COLORS["Unknown"]


def _parse_atoms(formula: str) -> list[_Atom]:
    atoms: list[_Atom] = []
    for raw_line in formula.replace(";", "\n").splitlines():
        parts = raw_line.replace(",", " ").split()
        if len(parts) < 4:
            continue
        symbol = parts[0]
        try:
            x, y, z = (float(parts[1]), float(parts[2]), float(parts[3]))
        except ValueError:
            continue
        atoms.append(_Atom(symbol=symbol, x=x, y=y, z=z))
    return atoms


def _fragment_label(indices: list[int], atoms: list[_Atom]) -> str:
    counts: dict[str, int] = {}
    order: list[str] = []
    for index in indices:
        if 0 <= index < len(atoms):
            symbol = atoms[index].symbol
            if symbol not in counts:
                order.append(symbol)
                counts[symbol] = 0
            counts[symbol] += 1
    if not order:
        return ""
    return "".join(symbol if counts[symbol] == 1 else f"{symbol}{counts[symbol]}" for symbol in order)


def _looks_like_default_cluster_name(name: str, cluster_id: int) -> bool:
    lowered = name.strip().lower()
    return lowered in {"", f"cluster{cluster_id}", f"cluster {cluster_id}", f"c{cluster_id}"}


def _infer_role(cluster: dict, label: str, atoms: list[_Atom]) -> str:
    name = str(cluster.get("name", ""))
    atom_indices = cluster.get("atom_indices") or []
    text = f"{name} {label}".lower()
    if "metal" in text:
        return "Metal"
    if "ligand" in text:
        return "Ligand"
    for index in atom_indices:
        if 0 <= index < len(atoms) and atoms[index].symbol in _TRANSITION_METALS:
            return "Metal"
    for symbol in _TRANSITION_METALS:
        if re.search(rf"(^|[^A-Za-z]){re.escape(symbol.lower())}([^A-Za-z]|$)", text):
            return "Metal"
    return "Ligand"


def _project_centroids(nodes: list[_ClusterNode], centroids: dict[int, tuple[float, float, float]]) -> None:
    if len(centroids) != len(nodes):
        return
    ranges = []
    for axis in range(3):
        values = [coords[axis] for coords in centroids.values()]
        ranges.append((max(values) - min(values), axis))
    axes = [axis for _, axis in sorted(ranges, reverse=True)[:2]]
    if len(axes) < 2 or ranges[0][0] == 0:
        return
    first, second = axes
    for node in nodes:
        coords = centroids.get(node.id)
        if coords is None:
            continue
        node.x = coords[first]
        node.y = -coords[second]


def _assign_fallback_layout(nodes: list[_ClusterNode]) -> None:
    metal_nodes = [node for node in nodes if node.role == "Metal"]
    ligand_nodes = [node for node in nodes if node.role != "Metal"]
    if not nodes:
        return
    if metal_nodes:
        if len(metal_nodes) == 1:
            metal_nodes[0].x = 0.0
            metal_nodes[0].y = 0.0
        else:
            for index, node in enumerate(metal_nodes):
                angle = 2.0 * math.pi * index / len(metal_nodes)
                node.x = 0.55 * math.cos(angle)
                node.y = 0.55 * math.sin(angle)
        ring = ligand_nodes
        radius = 1.7 if len(nodes) <= 6 else 2.1
    else:
        ring = nodes
        radius = 1.5 if len(nodes) <= 6 else 2.0
    if not ring:
        return
    start_angle = math.pi / 2 if metal_nodes else 0.0
    for index, node in enumerate(ring):
        angle = start_angle + 2.0 * math.pi * index / len(ring)
        node.x = radius * math.cos(angle)
        node.y = radius * math.sin(angle)


class _HorizontalBarChart(QWidget):
    def __init__(self, title: str, empty_text: str, parent=None) -> None:
        super().__init__(parent)
        self._title = title
        self._empty_text = empty_text
        self._rows: list[tuple[str, float, str]] = []
        self._bar_color = QColor("#0f766e")
        self.setMinimumHeight(180)

    def set_rows(self, rows: list[tuple[str, float, str]]) -> None:
        self._rows = rows
        self.update()

    def set_bar_color(self, color: str) -> None:
        self._bar_color = QColor(color)
        self.update()

    def paintEvent(self, event) -> None:  # noqa: N802, ANN001
        super().paintEvent(event)
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.fillRect(self.rect(), self.palette().base())

        margin = 16
        title_rect = QRectF(margin, 8, self.width() - margin * 2, 24)
        painter.setPen(QColor("#111827"))
        title_font = painter.font()
        title_font.setBold(True)
        painter.setFont(title_font)
        painter.drawText(title_rect, Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter, self._title)

        plot = QRectF(margin, 40, self.width() - margin * 2, self.height() - 52)
        body_font = painter.font()
        body_font.setBold(False)
        painter.setFont(body_font)

        if not self._rows:
            painter.setPen(QColor("#6b7280"))
            painter.drawText(plot, Qt.AlignmentFlag.AlignCenter, self._empty_text)
            return

        max_value = max(abs(value) for _, value, _ in self._rows) or 1.0
        row_count = len(self._rows)
        row_h = max(22.0, min(36.0, plot.height() / max(row_count, 1)))
        label_w = min(260.0, max(90.0, plot.width() * 0.36))
        value_w = 96.0
        bar_x = plot.left() + label_w + 10
        bar_w = max(20.0, plot.width() - label_w - value_w - 18)
        fm = painter.fontMetrics()

        for index, (label, value, detail) in enumerate(self._rows):
            y = plot.top() + index * row_h
            if y + row_h > plot.bottom() + 1:
                break

            label_rect = QRectF(plot.left(), y, label_w, row_h)
            value_rect = QRectF(bar_x + bar_w + 8, y, value_w, row_h)
            bar_rect = QRectF(bar_x, y + 5, bar_w, max(4.0, row_h - 10))
            filled = QRectF(bar_rect.left(), bar_rect.top(), bar_rect.width() * abs(value) / max_value, bar_rect.height())

            painter.setPen(QColor("#374151"))
            label_text = label if not detail else f"{label}  {detail}"
            painter.drawText(
                label_rect,
                Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter,
                fm.elidedText(label_text, Qt.TextElideMode.ElideRight, int(label_rect.width())),
            )

            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(QColor("#e5e7eb"))
            painter.drawRoundedRect(bar_rect, 3, 3)
            painter.setBrush(self._bar_color)
            painter.drawRoundedRect(filled, 3, 3)

            painter.setPen(QColor("#111827"))
            painter.drawText(
                value_rect,
                Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter,
                f"{value:.4g}",
            )


class _CTNetworkWidget(QWidget):
    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._nodes: list[_ClusterNode] = []
        self._arrows: list[_TransferArrow] = []
        self._root: int | None = None
        self._message = "No charge-transfer arrows loaded yet."
        self.setMinimumHeight(330)

    def set_data(self, nodes: list[_ClusterNode], arrows: list[_TransferArrow], root: int | None) -> None:
        self._nodes = nodes
        self._arrows = sorted(arrows, key=lambda arrow: abs(arrow.weight), reverse=True)[:40]
        self._root = root
        self.update()

    def set_message(self, message: str) -> None:
        self._message = message
        self._nodes = []
        self._arrows = []
        self._root = None
        self.update()

    def paintEvent(self, event) -> None:  # noqa: N802, ANN001
        super().paintEvent(event)
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.fillRect(self.rect(), self.palette().base())

        margin = 18
        title = "Charge-Transfer Map" if self._root is None else f"Charge-Transfer Map - Root {self._root}"
        title_rect = QRectF(margin, 8, self.width() - margin * 2, 28)
        title_font = painter.font()
        title_font.setBold(True)
        title_font.setPointSize(max(11, title_font.pointSize() + 1))
        painter.setFont(title_font)
        painter.setPen(QColor("#111827"))
        painter.drawText(title_rect, Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter, title)

        plot = QRectF(margin, 48, self.width() - margin * 2, self.height() - 88)
        body_font = painter.font()
        body_font.setBold(False)
        body_font.setPointSize(max(9, body_font.pointSize() - 1))
        painter.setFont(body_font)

        if not self._nodes:
            painter.setPen(QColor("#6b7280"))
            painter.drawText(plot, Qt.AlignmentFlag.AlignCenter, self._message)
            return

        positions = self._screen_positions(plot)
        radius = max(22.0, min(42.0, min(plot.width(), plot.height()) / (4.8 + math.sqrt(len(self._nodes)))))

        for arrow in self._arrows:
            if arrow.donor in positions and arrow.acceptor in positions:
                self._draw_arrow(painter, arrow, positions[arrow.donor], positions[arrow.acceptor], radius)

        for node in self._nodes:
            point = positions.get(node.id)
            if point is not None:
                self._draw_node(painter, node, point, radius)

        self._draw_legend(painter, QRectF(margin, self.height() - 34, self.width() - margin * 2, 24))

    def _screen_positions(self, plot: QRectF) -> dict[int, QPointF]:
        xs = [node.x if node.x is not None else 0.0 for node in self._nodes]
        ys = [node.y if node.y is not None else 0.0 for node in self._nodes]
        min_x, max_x = min(xs), max(xs)
        min_y, max_y = min(ys), max(ys)
        if max_x - min_x < 1e-9:
            min_x -= 1.0
            max_x += 1.0
        if max_y - min_y < 1e-9:
            min_y -= 1.0
            max_y += 1.0
        pad_x = (max_x - min_x) * 0.16
        pad_y = (max_y - min_y) * 0.18
        min_x -= pad_x
        max_x += pad_x
        min_y -= pad_y
        max_y += pad_y
        scale = min(plot.width() / (max_x - min_x), plot.height() / (max_y - min_y))
        used_w = (max_x - min_x) * scale
        used_h = (max_y - min_y) * scale
        left = plot.left() + (plot.width() - used_w) / 2
        top = plot.top() + (plot.height() - used_h) / 2
        positions: dict[int, QPointF] = {}
        for node in self._nodes:
            x = node.x if node.x is not None else 0.0
            y = node.y if node.y is not None else 0.0
            positions[node.id] = QPointF(left + (x - min_x) * scale, top + (y - min_y) * scale)
        return positions

    def _draw_arrow(
        self,
        painter: QPainter,
        arrow: _TransferArrow,
        start: QPointF,
        end: QPointF,
        node_radius: float,
    ) -> None:
        dx = end.x() - start.x()
        dy = end.y() - start.y()
        dist = math.hypot(dx, dy)
        if dist < 1e-6:
            return
        ux, uy = dx / dist, dy / dist
        px, py = -uy, ux
        start = QPointF(start.x() + ux * node_radius * 0.86, start.y() + uy * node_radius * 0.86)
        end = QPointF(end.x() - ux * node_radius * 1.08, end.y() - uy * node_radius * 1.08)
        curve = 0.18 * dist
        control1 = QPointF(start.x() + dx * 0.33 + px * curve, start.y() + dy * 0.33 + py * curve)
        control2 = QPointF(start.x() + dx * 0.67 + px * curve, start.y() + dy * 0.67 + py * curve)

        max_weight = max((abs(item.weight) for item in self._arrows), default=1.0)
        width = min(10.0, max(1.8, 1.5 + 7.0 * math.sqrt(abs(arrow.weight) / max_weight)))
        color = QColor(arrow.color)
        color.setAlpha(220)
        pen = QPen(color, width, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap, Qt.PenJoinStyle.RoundJoin)
        painter.setPen(pen)
        painter.setBrush(Qt.BrushStyle.NoBrush)
        path = QPainterPath(start)
        path.cubicTo(control1, control2, end)
        painter.drawPath(path)

        tx = end.x() - control2.x()
        ty = end.y() - control2.y()
        tdist = math.hypot(tx, ty) or 1.0
        tux, tuy = tx / tdist, ty / tdist
        tpx, tpy = -tuy, tux
        head_len = 13.0 + width
        head_w = 7.0 + width * 0.55
        head = QPolygonF(
            [
                end,
                QPointF(end.x() - tux * head_len + tpx * head_w, end.y() - tuy * head_len + tpy * head_w),
                QPointF(end.x() - tux * head_len - tpx * head_w, end.y() - tuy * head_len - tpy * head_w),
            ]
        )
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(color)
        painter.drawPolygon(head)

        label = f"{arrow.weight:.4g}"
        mid = QPointF(
            (start.x() + end.x()) / 2 + px * curve * 0.82,
            (start.y() + end.y()) / 2 + py * curve * 0.82,
        )
        fm = painter.fontMetrics()
        label_rect = QRectF(
            mid.x() - fm.horizontalAdvance(label) / 2 - 5,
            mid.y() - fm.height() / 2 - 3,
            fm.horizontalAdvance(label) + 10,
            fm.height() + 6,
        )
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QColor(255, 255, 255, 222))
        painter.drawRoundedRect(label_rect, 4, 4)
        painter.setPen(color)
        painter.drawText(label_rect, Qt.AlignmentFlag.AlignCenter, label)

    def _draw_node(self, painter: QPainter, node: _ClusterNode, point: QPointF, radius: float) -> None:
        base = QColor(node.color)
        if not base.isValid():
            base = QColor("#d1d5db")
        gradient = QRadialGradient(QPointF(point.x() - radius * 0.32, point.y() - radius * 0.36), radius * 1.35)
        gradient.setColorAt(0.0, QColor("#ffffff"))
        gradient.setColorAt(0.36, base.lighter(135))
        gradient.setColorAt(1.0, base.darker(120))

        painter.setPen(QPen(QColor("#111827"), 1.4))
        painter.setBrush(gradient)
        painter.drawEllipse(point, radius, radius)

        text_rect = QRectF(point.x() - radius * 0.86, point.y() - radius * 0.54, radius * 1.72, radius * 1.08)
        label_font = painter.font()
        label_font.setBold(True)
        label_font.setPointSize(max(8, int(radius / 3.7)))
        painter.setFont(label_font)
        painter.setPen(QColor("#111827"))
        painter.drawText(text_rect, Qt.AlignmentFlag.AlignCenter, node.label)

        role_font = painter.font()
        role_font.setBold(False)
        role_font.setPointSize(max(7, int(radius / 4.8)))
        painter.setFont(role_font)
        role_rect = QRectF(point.x() - radius, point.y() + radius * 0.48, radius * 2.0, radius * 0.55)
        painter.setPen(QColor("#374151"))
        painter.drawText(role_rect, Qt.AlignmentFlag.AlignCenter, node.role)

    def _draw_legend(self, painter: QPainter, rect: QRectF) -> None:
        x = rect.left()
        painter.setPen(QColor("#374151"))
        for ct_type in ("MLCT", "LMCT", "LLCT", "MMCT"):
            color = QColor(_CT_TYPE_COLORS[ct_type])
            painter.setBrush(color)
            painter.setPen(Qt.PenStyle.NoPen)
            painter.drawRoundedRect(QRectF(x, rect.top() + 7, 16, 6), 3, 3)
            painter.setPen(QColor("#374151"))
            painter.drawText(QRectF(x + 22, rect.top(), 86, rect.height()), Qt.AlignmentFlag.AlignVCenter, ct_type)
            x += 92


class PostAnalysisScreen(QWidget):
    analysis_done = pyqtSignal(str)

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._output_dir: Path | None = None
        self._julia_project: Path | None = None
        self._worker = None
        self._sector_rows: list[dict] = []
        self._cluster_nodes: list[_ClusterNode] = []

        self._cmf_result = QLineEdit()
        self._wavefunction_result = QLineEdit()
        self._output_path = QLineEdit()

        browse_cmf = QPushButton("Browse...")
        browse_cmf.clicked.connect(self._browse_cmf)
        browse_wfn = QPushButton("Browse...")
        browse_wfn.clicked.connect(self._browse_wavefunction)

        cmf_row = QHBoxLayout()
        cmf_row.addWidget(self._cmf_result)
        cmf_row.addWidget(browse_cmf)
        wfn_row = QHBoxLayout()
        wfn_row.addWidget(self._wavefunction_result)
        wfn_row.addWidget(browse_wfn)

        self._wavefunction_key = QComboBox()
        self._wavefunction_key.setEditable(True)
        self._wavefunction_key.addItems(["auto", "v0", "v_var", "v0a", "v_guess", "v_updated_pt1", "v_ci"])

        self._nroots = QSpinBox()
        self._nroots.setRange(1, 100)
        self._nroots.setValue(4)

        self._ct_thresh = QLineEdit("1e-5")

        form = QFormLayout()
        form.addRow("CMF result", cmf_row)
        form.addRow("Wavefunction result", wfn_row)
        form.addRow("Wavefunction key", self._wavefunction_key)
        form.addRow("nroots", self._nroots)
        form.addRow("CT print threshold", self._ct_thresh)
        form.addRow("Output directory", self._output_path)

        self._driver_preview = QPlainTextEdit()
        self._driver_preview.setReadOnly(True)
        font = self._driver_preview.font()
        font.setFamily("monospace")
        self._driver_preview.setFont(font)

        self._results = QPlainTextEdit()
        self._results.setReadOnly(True)
        font = self._results.font()
        font.setFamily("monospace")
        self._results.setFont(font)

        self._visual_status = QLabel("Run analysis or load an existing post-analysis folder.")
        self._visual_status.setWordWrap(True)
        self._root_selector = QComboBox()
        self._root_selector.currentIndexChanged.connect(self._refresh_transfer_map)
        self._ct_summary = QLabel("")
        self._ct_summary.setWordWrap(True)
        self._network = _CTNetworkWidget()
        self._transfer_table = QTableWidget(0, 5)
        self._transfer_table.setHorizontalHeaderLabels(["Root", "Donor", "Acceptor", "Type", "Weight"])
        self._transfer_table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self._transfer_table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self._transfer_table.setAlternatingRowColors(True)
        self._transfer_table.verticalHeader().setVisible(False)
        self._transfer_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
        self._transfer_table.horizontalHeader().setStretchLastSection(True)

        self._ct_chart = _HorizontalBarChart("Root CT Totals", "No ct_table.csv loaded yet.")
        self._sector_chart = _HorizontalBarChart(
            "Largest Fock-Space Sector Weights",
            "No ct_sectors.csv loaded yet.",
        )
        self._sector_chart.set_bar_color("#b45309")
        self._sector_table = QTableWidget(0, 4)
        self._sector_table.setHorizontalHeaderLabels(["Root", "Weight", "Configs", "Fock space"])
        self._sector_table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self._sector_table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self._sector_table.setAlternatingRowColors(True)
        self._sector_table.verticalHeader().setVisible(False)
        self._sector_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
        self._sector_table.horizontalHeader().setStretchLastSection(True)

        root_row = QHBoxLayout()
        root_row.addWidget(QLabel("Root"))
        root_row.addWidget(self._root_selector)
        root_row.addWidget(self._ct_summary, stretch=1)

        ct_map = QWidget()
        ct_map_layout = QVBoxLayout(ct_map)
        ct_map_layout.addLayout(root_row)
        ct_map_layout.addWidget(self._network, stretch=2)
        ct_map_layout.addWidget(self._transfer_table, stretch=1)
        ct_map.setLayout(ct_map_layout)

        weights = QWidget()
        weights_layout = QVBoxLayout(weights)
        weights_layout.addWidget(self._ct_chart)
        weights_layout.addWidget(self._sector_chart)
        weights_layout.addWidget(self._sector_table, stretch=1)
        weights.setLayout(weights_layout)

        visual_tabs = QTabWidget()
        visual_tabs.addTab(ct_map, "CT Map")
        visual_tabs.addTab(weights, "Weights")

        visualization = QWidget()
        visualization_layout = QVBoxLayout(visualization)
        visualization_layout.addWidget(self._visual_status)
        visualization_layout.addWidget(visual_tabs, stretch=1)
        visualization.setLayout(visualization_layout)

        self._log = LogPane()

        render_btn = QPushButton("Render analysis script")
        render_btn.clicked.connect(self._on_render)
        self._run_btn = QPushButton("Run analysis")
        self._run_btn.clicked.connect(self._on_run)
        self._cancel_btn = QPushButton("Cancel")
        self._cancel_btn.clicked.connect(self._on_cancel)
        self._cancel_btn.setEnabled(False)
        load_visual_btn = QPushButton("Load visualization")
        load_visual_btn.clicked.connect(self._on_load_visualization)

        btn_row = QHBoxLayout()
        btn_row.addWidget(render_btn)
        btn_row.addWidget(self._run_btn)
        btn_row.addWidget(self._cancel_btn)
        btn_row.addWidget(load_visual_btn)
        btn_row.addStretch(1)

        left = QWidget()
        left_layout = QVBoxLayout(left)
        left_layout.addWidget(QLabel("<h3>Post-Analysis of Wavefunction</h3>"))
        left_layout.addLayout(form)
        left_layout.addLayout(btn_row)
        left_layout.addWidget(QLabel(f"Target: Julia {PINNED_JULIA_VERSION}"))
        left_layout.addWidget(self._log, stretch=1)

        right = QTabWidget()
        right.addTab(visualization, "Visualization")
        right.addTab(self._results, "Text")
        right.addTab(self._driver_preview, "Driver")

        splitter = QSplitter()
        splitter.setHandleWidth(8)
        splitter.addWidget(left)
        splitter.addWidget(right)
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 1)

        layout = QVBoxLayout(self)
        layout.addWidget(splitter)
        self.setLayout(layout)
        self.setMinimumSize(0, 0)

    def set_inputs(
        self,
        cmf_result_path: str | Path,
        export_dir: str | Path,
        julia_project: str | Path | None = None,
        wavefunction_result_path: str | Path | None = None,
    ) -> None:
        export_dir = Path(export_dir)
        self._output_dir = export_dir.parent / "post_analysis"
        self._output_dir.mkdir(parents=True, exist_ok=True)
        self._julia_project = Path(julia_project) if julia_project else None

        self._cmf_result.setText(str(cmf_result_path))
        self._output_path.setText(str(self._output_dir))

        result_path = Path(wavefunction_result_path) if wavefunction_result_path else self._find_result(export_dir)
        if result_path is not None:
            self._wavefunction_result.setText(str(result_path))
        self._load_existing_outputs()

    def _find_result(self, export_dir: Path) -> Path | None:
        for name in _RESULT_CANDIDATES:
            path = export_dir / name
            if path.exists():
                return path
        found = sorted(export_dir.glob("*result.jld2"))
        return found[0] if found else None

    def _build_context(self) -> dict:
        return dict(
            cmf_result_path=self._cmf_result.text().strip(),
            wavefunction_result_path=self._wavefunction_result.text().strip(),
            output_dir=self._output_path.text().strip(),
            wavefunction_key=self._wavefunction_key.currentText().strip() or "auto",
            nroots=self._nroots.value(),
            ct_thresh=self._ct_thresh.text().strip() or "1e-5",
        )

    def _driver_path(self) -> Path:
        out = Path(self._output_path.text().strip())
        return out / "driver_wavefunction_post_analysis.jl"

    def _on_render(self) -> None:
        try:
            self._driver_preview.setPlainText(
                render_driver("driver_wavefunction_post_analysis.jl.j2", self._build_context())
            )
        except Exception as exc:  # noqa: BLE001
            self._log.append_line(f"[post] render failed: {exc}")

    def _on_run(self) -> None:
        if self._worker is not None and self._worker.is_running():
            self._log.append_line("[post] analysis is already running.")
            return
        cmf = Path(self._cmf_result.text().strip())
        wfn = Path(self._wavefunction_result.text().strip())
        if not cmf.exists():
            self._log.append_line(f"[post] CMF result not found: {cmf}")
            return
        if not wfn.exists():
            self._log.append_line(f"[post] wavefunction result not found: {wfn}")
            return

        ctx = self._build_context()
        driver_path = write_driver(
            "driver_wavefunction_post_analysis.jl.j2",
            ctx,
            self._driver_path(),
        )
        self._driver_preview.setPlainText(driver_path.read_text())
        self._log.append_line(f"[post] wrote {driver_path.name}, launching Julia...")
        self._set_running(True)

        from asbuilder.gui.workers import JuliaProcessWorker

        julia_project = self._julia_project or cmf.parent.parent
        self._worker = JuliaProcessWorker(
            driver_path,
            julia_project,
            parent=self,
            log_path=Path(ctx["output_dir"]) / "post_analysis.log",
            out_path=Path(ctx["output_dir"]) / "post_analysis.out",
        )
        self._worker.line_received.connect(self._log.append_line)
        self._worker.finished_ok.connect(self._on_finished)
        self._worker.failed.connect(self._on_failed)
        self._worker.start()

    def _on_cancel(self) -> None:
        if self._worker is not None and self._worker.is_running():
            self._worker.kill()
        self._set_running(False)
        self._log.append_line("[post] cancelled by user.")

    def _on_finished(self, exit_code: int) -> None:
        self._set_running(False)
        if exit_code == 0:
            out = Path(self._output_path.text().strip())
            self._load_outputs(out)
            self._log.append_line("[post] analysis finished OK.")
            self.analysis_done.emit(str(out))
        else:
            self._log.append_line(f"[post] Julia exited with code {exit_code}.")

    def _on_failed(self, message: str) -> None:
        self._set_running(False)
        self._log.append_line(f"[post] FAILED: {message}")

    def _set_running(self, running: bool) -> None:
        self._run_btn.setEnabled(not running)
        self._cancel_btn.setEnabled(running)

    def _browse_cmf(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Select cmf_result.jld2", "", "JLD2 files (*.jld2);;All files (*)"
        )
        if path:
            self._cmf_result.setText(path)

    def _browse_wavefunction(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Select wavefunction result", "", "JLD2 files (*.jld2);;All files (*)"
        )
        if path:
            self._wavefunction_result.setText(path)

    def _on_load_visualization(self) -> None:
        out_text = self._output_path.text().strip()
        if not out_text:
            self._log.append_line("[post] no output directory selected.")
            return
        out = Path(out_text)
        self._load_outputs(out)

    def _load_existing_outputs(self) -> None:
        out_text = self._output_path.text().strip()
        if out_text:
            self._load_outputs(Path(out_text), quiet=True)

    def _load_outputs(self, out: Path, quiet: bool = False) -> None:
        txt = out / "ct_analysis.txt"
        if txt.exists():
            self._results.setPlainText(txt.read_text())
        self._load_visualization(out, quiet=quiet)

    def _load_visualization(self, out: Path, quiet: bool = False) -> None:
        totals = self._read_ct_totals(out / "ct_table.csv")
        sectors = self._read_ct_sectors(out / "ct_sectors.csv")
        previous_root = self._root_selector.currentData()
        self._sector_rows = sectors
        self._cluster_nodes = self._load_cluster_nodes(out, sectors)

        self._ct_chart.set_rows([(f"Root {root}", value, "") for root, value in totals])
        top_sector_rows = [
            (f"Root {row['root']}", row["weight"], row["fock_space"])
            for row in sorted(sectors, key=lambda item: abs(item["weight"]), reverse=True)[:24]
        ]
        self._sector_chart.set_rows(top_sector_rows)
        self._populate_sector_table(sectors)
        roots = sorted({root for root, _ in totals} | {row["root"] for row in sectors})
        self._root_selector.blockSignals(True)
        self._root_selector.clear()
        for root in roots:
            self._root_selector.addItem(str(root), root)
        if previous_root in roots:
            self._root_selector.setCurrentIndex(roots.index(previous_root))
        self._root_selector.blockSignals(False)
        self._refresh_transfer_map()

        if totals or sectors:
            self._visual_status.setText(
                f"Loaded {len(totals)} root totals and {len(sectors)} fock-space sectors from {out}."
            )
            if not quiet:
                self._log.append_line("[post] visualization loaded.")
        else:
            self._visual_status.setText(f"No CT CSV outputs found in {out}.")
            if not quiet:
                self._log.append_line(f"[post] no visualization CSVs found in {out}.")

    def _load_cluster_nodes(self, out: Path, sectors: list[dict]) -> list[_ClusterNode]:
        project_root = out.parent
        clusters_path = project_root / "clusters.json"
        if not clusters_path.exists():
            return self._fallback_nodes_from_sectors(sectors)

        atoms = self._read_project_atoms(project_root)
        try:
            cluster_data = json.loads(clusters_path.read_text())
        except Exception as exc:  # noqa: BLE001
            self._log.append_line(f"[post] could not read clusters.json: {exc}")
            return self._fallback_nodes_from_sectors(sectors)

        nodes: list[_ClusterNode] = []
        centroids: dict[int, tuple[float, float, float]] = {}
        for raw_cluster in cluster_data:
            cluster_id = int(raw_cluster.get("id", len(nodes) + 1))
            raw_name = str(raw_cluster.get("name") or f"cluster{cluster_id}")
            atom_indices = [int(index) for index in (raw_cluster.get("atom_indices") or [])]
            fragment = _fragment_label(atom_indices, atoms)
            if fragment and _looks_like_default_cluster_name(raw_name, cluster_id):
                label = fragment
            elif _looks_like_default_cluster_name(raw_name, cluster_id):
                label = f"C{cluster_id}"
            else:
                label = raw_name
            if len(label) > 12:
                label = label[:9] + "..."

            fspace = raw_cluster.get("fspace") or [0, 0]
            role = _infer_role(raw_cluster, label, atoms)
            nodes.append(
                _ClusterNode(
                    id=cluster_id,
                    label=label,
                    role=role,
                    color=str(raw_cluster.get("color") or "#d1d5db"),
                    ref_electrons=int(fspace[0]) + int(fspace[1]),
                )
            )
            coords = [
                (atoms[index].x, atoms[index].y, atoms[index].z)
                for index in atom_indices
                if 0 <= index < len(atoms)
            ]
            if coords:
                ncoords = len(coords)
                centroids[cluster_id] = (
                    sum(coord[0] for coord in coords) / ncoords,
                    sum(coord[1] for coord in coords) / ncoords,
                    sum(coord[2] for coord in coords) / ncoords,
                )

        _project_centroids(nodes, centroids)
        if any(node.x is None or node.y is None for node in nodes):
            _assign_fallback_layout(nodes)
        return nodes

    def _fallback_nodes_from_sectors(self, sectors: list[dict]) -> list[_ClusterNode]:
        n_clusters = 0
        for row in sectors:
            n_clusters = max(n_clusters, len(_parse_fock_space(row.get("fock_space", ""))))
        nodes = [
            _ClusterNode(
                id=index,
                label=f"C{index}",
                role="Ligand",
                color="#d1d5db",
                ref_electrons=-1,
            )
            for index in range(1, n_clusters + 1)
        ]
        _assign_fallback_layout(nodes)
        return nodes

    def _read_project_atoms(self, project_root: Path) -> list[_Atom]:
        project_json = project_root / "project.json"
        if not project_json.exists():
            return []
        try:
            data = json.loads(project_json.read_text())
        except Exception:
            return []
        formula = str(data.get("provenance", {}).get("formula", ""))
        return _parse_atoms(formula)

    def _refresh_transfer_map(self, *_args) -> None:
        root = self._root_selector.currentData()
        if root is None:
            self._network.set_message("No roots are available for charge-transfer mapping.")
            self._transfer_table.setRowCount(0)
            self._ct_summary.setText("")
            return
        if not self._cluster_nodes:
            self._network.set_message("No clusters are available for charge-transfer mapping.")
            self._transfer_table.setRowCount(0)
            self._ct_summary.setText("")
            return

        arrows = self._compute_transfers_for_root(int(root))
        if any(node.ref_electrons < 0 for node in self._cluster_nodes):
            self._network.set_message("clusters.json is needed to compute reference electron counts.")
            self._populate_transfer_table([])
            self._ct_summary.setText("")
            return
        self._network.set_data(self._cluster_nodes, arrows, int(root))
        self._populate_transfer_table(arrows)
        self._ct_summary.setText(self._transfer_type_summary(arrows))

    def _compute_transfers_for_root(self, root: int) -> list[_TransferArrow]:
        pair_weights: dict[tuple[int, int], float] = {}
        node_order = self._cluster_nodes
        if not node_order or any(node.ref_electrons < 0 for node in node_order):
            return []
        for row in self._sector_rows:
            if int(row["root"]) != root:
                continue
            fspace = _parse_fock_space(row["fock_space"])
            if len(fspace) != len(node_order):
                continue
            diffs = [
                (alpha + beta) - node.ref_electrons
                for (alpha, beta), node in zip(fspace, node_order)
            ]
            donors = [node_order[index].id for index, diff in enumerate(diffs) if diff < 0]
            acceptors = [node_order[index].id for index, diff in enumerate(diffs) if diff > 0]
            if not donors or not acceptors:
                continue
            share = float(row["weight"]) / (len(donors) * len(acceptors))
            for donor in donors:
                for acceptor in acceptors:
                    pair_weights[(donor, acceptor)] = pair_weights.get((donor, acceptor), 0.0) + share

        role_by_id = {node.id: node.role for node in node_order}
        arrows: list[_TransferArrow] = []
        for (donor, acceptor), weight in pair_weights.items():
            ct_type, color = _classify_transfer(donor, acceptor, role_by_id)
            arrows.append(_TransferArrow(donor, acceptor, weight, ct_type, color))
        return sorted(arrows, key=lambda arrow: abs(arrow.weight), reverse=True)

    def _transfer_type_summary(self, arrows: list[_TransferArrow]) -> str:
        if not arrows:
            return "No donor/acceptor CT pathways for this root."
        totals = {ct_type: 0.0 for ct_type in ("MLCT", "LMCT", "LLCT", "MMCT")}
        for arrow in arrows:
            if arrow.ct_type in totals:
                totals[arrow.ct_type] += arrow.weight
        return "   ".join(f"{ct_type}: {value:.4g}" for ct_type, value in totals.items() if value > 0)

    def _populate_transfer_table(self, arrows: list[_TransferArrow]) -> None:
        root = self._root_selector.currentData()
        label_by_id = {node.id: node.label for node in self._cluster_nodes}
        ordered = sorted(arrows, key=lambda arrow: -abs(arrow.weight))
        self._transfer_table.setSortingEnabled(False)
        self._transfer_table.setRowCount(len(ordered))
        for row_index, arrow in enumerate(ordered):
            values = [
                str(root if root is not None else ""),
                f"{arrow.donor}: {label_by_id.get(arrow.donor, f'C{arrow.donor}')}",
                f"{arrow.acceptor}: {label_by_id.get(arrow.acceptor, f'C{arrow.acceptor}')}",
                arrow.ct_type,
                f"{arrow.weight:.8g}",
            ]
            for col, value in enumerate(values):
                item = QTableWidgetItem(value)
                if col in (0, 4):
                    item.setTextAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
                if col == 3:
                    item.setBackground(QColor(arrow.color).lighter(180))
                self._transfer_table.setItem(row_index, col, item)
        self._transfer_table.setSortingEnabled(True)

    def _read_ct_totals(self, path: Path) -> list[tuple[int, float]]:
        if not path.exists():
            return []
        totals: list[tuple[int, float]] = []
        try:
            with path.open(newline="") as handle:
                for row in csv.DictReader(handle):
                    totals.append((int(row.get("root", "0")), float(row.get("total_ct", "0"))))
        except Exception as exc:  # noqa: BLE001
            self._log.append_line(f"[post] could not read {path.name}: {exc}")
            return []
        return totals

    def _read_ct_sectors(self, path: Path) -> list[dict]:
        if not path.exists():
            return []
        sectors: list[dict] = []
        try:
            with path.open(newline="") as handle:
                for row in csv.DictReader(handle):
                    sectors.append(
                        {
                            "root": int(row.get("root", "0")),
                            "weight": float(row.get("weight", "0")),
                            "n_configs": int(row.get("n_configs", "0")),
                            "fock_space": row.get("fock_space", ""),
                        }
                    )
        except Exception as exc:  # noqa: BLE001
            self._log.append_line(f"[post] could not read {path.name}: {exc}")
            return []
        return sectors

    def _populate_sector_table(self, sectors: list[dict]) -> None:
        ordered = sorted(sectors, key=lambda row: (row["root"], -abs(row["weight"]), row["fock_space"]))
        self._sector_table.setSortingEnabled(False)
        self._sector_table.setRowCount(len(ordered))
        max_weight = max((abs(row["weight"]) for row in ordered), default=0.0)
        for row_index, row in enumerate(ordered):
            values = [
                str(row["root"]),
                f"{row['weight']:.8g}",
                str(row["n_configs"]),
                row["fock_space"],
            ]
            for col, value in enumerate(values):
                item = QTableWidgetItem(value)
                if col in (0, 1, 2):
                    item.setTextAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
                if col == 1 and max_weight > 0:
                    strength = min(1.0, abs(row["weight"]) / max_weight)
                    item.setBackground(QColor(255, int(247 - strength * 42), int(237 - strength * 95)))
                self._sector_table.setItem(row_index, col, item)
        self._sector_table.setSortingEnabled(True)
