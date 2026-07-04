"""
Background workers so the UI thread never blocks.

Two kinds, matching the design doc's threading model:

- `*Worker(QThread)` for in-process Python steps (SCF, molden generation,
  active-space build) -- these call straight into `asbuilder.*` functions
  and emit Qt signals for progress/log/result/error.
- `JuliaProcessWorker(QObject)` for external Julia steps (CMF, TPSCI, SPT)
  -- wraps `QProcess` so stdout streams into the log pane live, rather than
  blocking until the whole job finishes.

None of these classes import anything from `asbuilder.gui.screens` -- they
only depend on the plain `asbuilder` backend package, so they're usable
from a script/test harness too, not just the screens that own them.
"""

from __future__ import annotations

import io
import sys
from pathlib import Path
from typing import Any, Callable

from PyQt6.QtCore import QObject, QProcess, QThread, pyqtSignal


class _TeeStream(io.TextIOBase):
    """Mirrors writes to the original stdout, an optional log file, and a
    per-line signal callback -- so Python worker output appears in the
    GUI log pane and on disk at the same time."""

    def __init__(self, original, file_obj, on_line: Callable[[str], None]) -> None:
        super().__init__()
        self._original = original
        self._file = file_obj
        self._on_line = on_line
        self._buf = ""

    def write(self, s: str) -> int:
        if self._original:
            try:
                self._original.write(s)
                self._original.flush()
            except Exception:
                pass
        if self._file:
            try:
                self._file.write(s)
                self._file.flush()
            except Exception:
                pass
        self._buf += s
        while "\n" in self._buf:
            line, self._buf = self._buf.split("\n", 1)
            try:
                self._on_line(line)
            except Exception:
                pass
        return len(s)

    def flush(self) -> None:
        if self._original:
            try:
                self._original.flush()
            except Exception:
                pass
        if self._file:
            try:
                self._file.flush()
            except Exception:
                pass


class _CallableWorker(QThread):
    """Generic QThread that runs a single callable and reports back via
    signals. Screens subclass this only when they need step-specific
    signals (see ScfWorker, MoldenWorker below); otherwise this is used
    directly for anything that doesn't need intermediate progress updates.

    stdout is tee'd to `log_path` (if given) and line_received is emitted
    per output line so the GUI log pane can show Python-side output live.
    """

    finished_ok = pyqtSignal(object)   # emits the callable's return value
    failed = pyqtSignal(str)            # emits str(exception)
    line_received = pyqtSignal(str)     # one stdout line at a time

    def __init__(
        self,
        fn: Callable[[], Any],
        parent: QObject | None = None,
        log_path: str | Path | None = None,
    ) -> None:
        super().__init__(parent)
        self._fn = fn
        self._log_path = Path(log_path) if log_path else None

    def run(self) -> None:
        orig_stdout = sys.stdout
        file_obj = None
        if self._log_path:
            try:
                self._log_path.parent.mkdir(parents=True, exist_ok=True)
                file_obj = open(self._log_path, "a", encoding="utf-8")
            except Exception:
                file_obj = None

        tee = _TeeStream(orig_stdout, file_obj, self.line_received.emit)
        sys.stdout = tee
        try:
            result = self._fn()
        except BaseException as exc:
            sys.stdout = orig_stdout
            if file_obj:
                try:
                    file_obj.close()
                except Exception:
                    pass
            self.failed.emit(f"{type(exc).__name__}: {exc}")
            return
        finally:
            sys.stdout = orig_stdout
            if file_obj:
                try:
                    file_obj.close()
                except Exception:
                    pass

        self.finished_ok.emit(result)


class ScfWorker(_CallableWorker):
    """Runs asbuilder.io.run_scf.run_scf in a background thread for the
    "New Calculation" dialog. Emits finished_ok(SCFResult) or failed(str).
    """

    def __init__(
        self,
        xyz: str,
        basis: str,
        method: str,
        charge: int,
        spin: int,
        chk_path: str | Path,
        parent: QObject | None = None,
        log_path: str | Path | None = None,
        density_fit: bool = False,
        auxbasis: str | None = None,
        newton: bool = False,
    ) -> None:
        from asbuilder.io.run_scf import run_scf

        def _run():
            return run_scf(
                xyz=xyz, basis=basis, method=method, charge=charge, spin=spin,
                chk_path=chk_path, density_fit=density_fit, auxbasis=auxbasis, newton=newton,
            )

        super().__init__(_run, parent, log_path=log_path)


class MoldenWorker(_CallableWorker):
    """Runs asbuilder.io.chk_to_molden.chk_to_molden in a background thread.
    Emits finished_ok((ChkContents, Path)) or failed(str)."""

    def __init__(self, chk_path: str | Path, out_path: str | Path, parent: QObject | None = None) -> None:
        from asbuilder.io.chk_to_molden import chk_to_molden

        def _run():
            return chk_to_molden(chk_path, out_path)

        super().__init__(_run, parent)


class ActiveSpaceWorker(_CallableWorker):
    """Runs the localization + integrals step in a background thread."""

    def __init__(
        self,
        fn: Callable[[], Any],
        parent: QObject | None = None,
        log_path: str | Path | None = None,
    ) -> None:
        super().__init__(fn, parent, log_path=log_path)


class JuliaProcessWorker(QObject):
    """QProcess-based wrapper for CMF/TPSCI/SPT driver scripts.

    Unlike the QThread workers above, this streams output line-by-line as
    it arrives (QProcess's readyReadStandardOutput), which is what lets the
    CMF/export screens show a live log instead of a blocked spinner. Version
    checking happens synchronously before the process is started, via
    asbuilder.julia_bridge.runner.check_julia_version, so a wrong Julia
    version fails fast with a clear message instead of a confusing crash
    partway through a long CMF run.

    If `log_path` is given, cleaned line output is also written there in real
    time. If `out_path` is given, raw Julia stdout/stderr is written there,
    preserving progress-bar output for post-run inspection.
    """

    line_received = pyqtSignal(str)
    finished_ok = pyqtSignal(int)     # exit code
    failed = pyqtSignal(str)          # version check or launch failure

    def __init__(
        self,
        script_path: str | Path,
        project_dir: str | Path,
        julia_bin: str = "julia",
        parent: QObject | None = None,
        log_path: str | Path | None = None,
        out_path: str | Path | None = None,
        threads: str | int | None = None,
    ) -> None:
        super().__init__(parent)
        self._script_path = str(script_path)
        self._project_dir = str(project_dir)
        self._julia_bin = julia_bin
        self._log_path = Path(log_path) if log_path else None
        self._out_path = Path(out_path) if out_path else None
        self._threads = threads
        self._log_file = None
        self._out_file = None
        self._line_buffer = ""
        self._process = QProcess(self)
        self._process.setProcessChannelMode(QProcess.ProcessChannelMode.MergedChannels)
        self._process.readyReadStandardOutput.connect(self._on_ready_read)
        self._process.finished.connect(self._on_finished)
        self._process.errorOccurred.connect(lambda err: self.failed.emit(f"QProcess error: {err}"))

    def start(self) -> None:
        from asbuilder.julia_bridge.runner import JuliaVersionError, check_julia_version, julia_thread_args

        try:
            version = check_julia_version(self._julia_bin)
            thread_args = julia_thread_args(self._threads)
        except JuliaVersionError as exc:
            self.failed.emit(str(exc))
            return
        except ValueError as exc:
            self.failed.emit(str(exc))
            return

        if self._log_path:
            try:
                self._log_path.parent.mkdir(parents=True, exist_ok=True)
                self._log_file = open(self._log_path, "a", encoding="utf-8")
            except Exception:
                self._log_file = None

        if self._out_path:
            try:
                self._out_path.parent.mkdir(parents=True, exist_ok=True)
                self._out_file = open(self._out_path, "ab")
            except Exception:
                self._out_file = None

        self._emit_clean_line(f"[julia] using Julia {version}")
        if thread_args:
            self._emit_clean_line(f"[julia] launch threads: {thread_args[0].split('=', 1)[1]}")

        args = [f"--project={self._project_dir}", *thread_args, self._script_path]
        self._process.start(self._julia_bin, args)

    def _emit_clean_line(self, line: str) -> None:
        self.line_received.emit(line)
        if self._log_file:
            try:
                self._log_file.write(line + "\n")
                self._log_file.flush()
            except Exception:
                pass

    def _on_ready_read(self) -> None:
        raw = bytes(self._process.readAllStandardOutput())
        if self._out_file:
            try:
                self._out_file.write(raw)
                self._out_file.flush()
            except Exception:
                pass

        from asbuilder.julia_bridge.runner import consume_output_lines

        data = raw.decode("utf-8", errors="replace")
        lines, self._line_buffer = consume_output_lines(self._line_buffer, data)
        for line in lines:
            self._emit_clean_line(line)

    def _on_finished(self, exit_code: int, _exit_status: Any) -> None:
        from asbuilder.julia_bridge.runner import flush_output_buffer

        for line in flush_output_buffer(self._line_buffer):
            self._emit_clean_line(line)
        self._line_buffer = ""

        if self._log_file:
            try:
                self._log_file.close()
            except Exception:
                pass
            self._log_file = None
        if self._out_file:
            try:
                self._out_file.close()
            except Exception:
                pass
            self._out_file = None
        self.finished_ok.emit(exit_code)

    def is_running(self) -> bool:
        return self._process.state() != QProcess.ProcessState.NotRunning

    def kill(self) -> None:
        if self._process.state() != QProcess.ProcessState.NotRunning:
            self._process.kill()
