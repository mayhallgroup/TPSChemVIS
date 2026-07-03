"""Helpers for screen-aware window sizing."""
from __future__ import annotations
from PyQt6.QtWidgets import QApplication


def screen_capped_size(preferred_w: int, preferred_h: int, pct: float = 0.75) -> tuple[int, int]:
    """Return (w, h) = min(preferred, pct × available screen)."""
    screen = QApplication.primaryScreen()
    if screen is None:
        return preferred_w, preferred_h
    avail = screen.availableGeometry()
    w = min(preferred_w, int(avail.width()  * pct))
    h = min(preferred_h, int(avail.height() * pct))
    return w, h


def center_on_screen(win) -> None:
    """Move *win* to the center of the primary screen's available area."""
    screen = QApplication.primaryScreen()
    if screen is None:
        return
    avail = screen.availableGeometry()
    win.move(
        avail.x() + (avail.width()  - win.width())  // 2,
        avail.y() + (avail.height() - win.height()) // 2,
    )


def fit_to_screen(win, preferred_w: int, preferred_h: int, pct: float = 0.75) -> None:
    """Resize *win* to min(preferred, pct × screen) and center it."""
    w, h = screen_capped_size(preferred_w, preferred_h, pct)
    win.resize(w, h)
    center_on_screen(win)
