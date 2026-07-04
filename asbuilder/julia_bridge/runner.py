"""
Render Jinja2 Julia driver templates and launch them.

Deliberately plain `subprocess`, not `QProcess` -- this module has no Qt
dependency so it's usable from a headless script/test, and the future GUI
code wraps `run_julia_streaming` (or shells out the same command) inside a
`QProcess` for live log streaming into the app.

Version pinning: every call checks `julia --version` reports the pinned
version (1.11.x by default) before running anything, since a silent
version mismatch against TPSChem.jl's compat bounds is a bad failure mode
to discover on a supercomputer allocation.
"""

from __future__ import annotations

import re
import subprocess
from pathlib import Path
from typing import Any, Iterator

from jinja2 import Environment, FileSystemLoader, StrictUndefined

TEMPLATES_DIR = Path(__file__).parent / "templates"

PINNED_JULIA_VERSION = "1.11"


class JuliaVersionError(RuntimeError):
    pass


def julia_thread_args(threads: str | int | None) -> list[str]:
    """Return Julia CLI args for the requested thread count.

    Accepts ``auto`` or a positive integer. Blank/``None`` leaves Julia's
    default unchanged, which is useful for non-calculation helper scripts.
    """
    if threads is None:
        return []

    value = str(threads).strip().lower()
    if not value:
        return []
    if value == "auto":
        return ["--threads=auto"]

    try:
        nthreads = int(value)
    except ValueError as exc:
        raise ValueError("Julia threads must be 'auto' or a positive integer.") from exc

    if nthreads < 1:
        raise ValueError("Julia threads must be 'auto' or a positive integer.")
    return [f"--threads={nthreads}"]


def consume_output_lines(buffer: str, text: str) -> tuple[list[str], str]:
    """Split streamed process output into display lines without chunk noise.

    QProcess may deliver a single Julia progress-bar line as many small chunks.
    This helper emits only newline-terminated lines and keeps the incomplete
    tail buffered for the next chunk. Carriage-return redraws are collapsed to
    the latest visible line.
    """
    text = text.replace("\r\n", "\n")
    parts = text.split("\n")
    lines: list[str] = []

    for part in parts[:-1]:
        current = buffer + part
        if "\r" in current:
            current = current.rsplit("\r", 1)[-1]
        lines.append(current)
        buffer = ""

    tail = buffer + parts[-1]
    if "\r" in tail:
        tail = tail.rsplit("\r", 1)[-1]
    return lines, tail


def flush_output_buffer(buffer: str) -> list[str]:
    """Return the final unterminated display line, if any."""
    if not buffer:
        return []
    if "\r" in buffer:
        buffer = buffer.rsplit("\r", 1)[-1]
    return [buffer] if buffer else []


def _env() -> Environment:
    # StrictUndefined: fail loudly on a missing template variable instead of
    # silently rendering "" into a driver script -- wrong integrals paths
    # failing silently would be a nasty bug to chase.
    return Environment(loader=FileSystemLoader(str(TEMPLATES_DIR)), undefined=StrictUndefined)


def render_driver(template_name: str, context: dict[str, Any]) -> str:
    """Render one of driver_cmf.jl.j2 / driver_tpsci.jl.j2 / driver_spt.jl.j2
    / submit.slurm.j2 with the given context dict."""
    return _env().get_template(template_name).render(**context)


def write_driver(template_name: str, context: dict[str, Any], out_path: str | Path) -> Path:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(render_driver(template_name, context))
    return out_path


def check_julia_version(julia_bin: str = "julia", pinned: str = PINNED_JULIA_VERSION) -> str:
    """Run `<julia_bin> --version` and verify it starts with `pinned`
    (e.g. "1.11"). Raises JuliaVersionError if julia isn't found or the
    version doesn't match. Returns the full version string on success."""
    try:
        result = subprocess.run([julia_bin, "--version"], capture_output=True, text=True, timeout=15)
    except FileNotFoundError as exc:
        raise JuliaVersionError(
            f"could not find a `{julia_bin}` executable on PATH. "
            f"If you're managing versions with juliaup, try julia_bin='julia +{pinned}' "
            f"or resolve the path with `juliaup which {pinned}` first."
        ) from exc

    m = re.search(r"julia version (\S+)", result.stdout)
    if not m:
        raise JuliaVersionError(f"unexpected `{julia_bin} --version` output: {result.stdout!r}")
    version = m.group(1)
    if not version.startswith(pinned):
        raise JuliaVersionError(
            f"found Julia {version}, but this project is pinned to {pinned}.x. "
            f"Use juliaup to install/select {pinned}, or pass julia_bin= pointing at the right binary."
        )
    return version


def ensure_pycall_built(
    project_dir: str | Path,
    python_path: str | None = None,
    julia_bin: str = "julia",
    check_version: bool = True,
) -> None:
    """One-time environment setup: `Pkg.build("PyCall")` against the given
    Python interpreter (should be the same one with pyscf installed).

    Only needed if a driver script actually does `using PyCall` -- our
    default driver_cmf.jl.j2 doesn't (it reads h0/h1/h2.npy directly and
    never touches PySCF from Julia), but TPSChem.jl's "direct" CMF mode
    (pyscf_do_scf/pyscf_build_ints, see test_direct_cmf.jl) does, and if
    that's the route actually used, PyCall must be built before the first
    run -- otherwise `using PyCall` fails or silently points at the wrong
    Python. Not something to run on every job submission, just once per
    environment.
    """
    if check_version:
        check_julia_version(julia_bin)

    script = write_driver(
        "setup_pycall.jl.j2",
        {"python_path": python_path},
        Path(project_dir) / "setup_pycall.jl",
    )
    cmd = [julia_bin, f"--project={project_dir}", str(script)]
    result = subprocess.run(cmd, capture_output=True, text=True)
    print(result.stdout)
    if result.returncode != 0:
        print(result.stderr)
        raise subprocess.CalledProcessError(result.returncode, cmd)


def run_julia_streaming(
    script_path: str | Path,
    project_dir: str | Path,
    julia_bin: str = "julia",
    extra_args: list[str] | None = None,
    threads: str | int | None = None,
    check_version: bool = True,
) -> Iterator[str]:
    """Launch `julia --project=<project_dir> <script_path>` and yield stdout
    lines as they arrive. This is what a QThread/QProcess wrapper in the GUI
    iterates over to stream into the log pane; here it's a plain generator
    so it can be driven from a script or a unit test without Qt.

    Raises JuliaVersionError up front (before launching anything) if the
    installed julia isn't the pinned version.
    """
    if check_version:
        check_julia_version(julia_bin)

    cmd = [
        julia_bin,
        f"--project={project_dir}",
        *julia_thread_args(threads),
        *(extra_args or []),
        str(script_path),
    ]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
    assert proc.stdout is not None
    try:
        for line in proc.stdout:
            yield line.rstrip("\n")
    finally:
        proc.stdout.close()
        returncode = proc.wait()
        if returncode != 0:
            raise subprocess.CalledProcessError(returncode, cmd)
