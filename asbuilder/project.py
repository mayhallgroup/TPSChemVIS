"""
Project state / history for an Active Space Builder run.

A "project" is a folder (conventionally named ``*.qcproj``) that holds every
artifact produced by the pipeline, plus a JSON manifest (``project.json``)
that records what stage the project is at and an append-only log of events
so nothing gets silently overwritten when the user redoes a step (e.g.
re-clustering after looking at localized orbitals).

This module has no Qt dependency and no dependency on PySCF/Julia — it is
pure file/JSON bookkeeping so it can be unit tested and used from a plain
script or notebook, not just the GUI.

Layout on disk::

    project.qcproj/
      project.json
      input.chk
      orbitals.molden
      clusters.json
      active_space/
        h0.npy, h1.npy, h2.npy
        localized_orbitals.molden
      cmf/
        driver_cmf.jl
        cmf.log, cmf_result.npz
      export/
        driver_tpsci.jl | driver_spt.jl
        submit.slurm
"""

from __future__ import annotations

import json
import shutil
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

PROJECT_MANIFEST = "project.json"

# The pipeline stages, in order. Stored on the project so the GUI can figure
# out which screen to resume on, and so "redo this stage" can be validated
# (you can't jump back to a stage that was never reached).
STAGES = [
    "created",          # project folder exists, nothing computed yet
    "chk_loaded",       # input.chk present, provenance parsed
    "molden_generated",  # orbitals.molden written
    "clusters_defined",  # clusters.json written (orbitals + fspace per cluster)
    "active_space_built",  # h0/h1/h2.npy + localized_orbitals.molden written
    "cmf_run",          # cmf/ populated (optional stage -- may be skipped)
    "exported",         # export/ populated with driver + submission script
]


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class ProjectState:
    """In-memory view of a project.json manifest."""

    root: Path
    provenance: dict[str, Any] = field(default_factory=dict)
    stage: str = "created"
    stage_log: list[dict[str, Any]] = field(default_factory=list)
    created_at: str = field(default_factory=_now)
    updated_at: str = field(default_factory=_now)

    # -- construction -----------------------------------------------------

    @classmethod
    def create(cls, root: Path) -> "ProjectState":
        """Initialize a project folder at `root`, creating it if needed.
        Safe to call on a folder that already has files (e.g. after a crash
        that left input.chk but no project.json): existing artifacts are kept."""
        root = Path(root)
        root.mkdir(parents=True, exist_ok=True)
        (root / "active_space").mkdir(exist_ok=True)
        (root / "cmf").mkdir(exist_ok=True)
        (root / "export").mkdir(exist_ok=True)

        state = cls(root=root)
        state._log_event("project_created", {})
        state.save()
        return state

    @classmethod
    def load(cls, root: Path) -> "ProjectState":
        """Reopen an existing project folder, resuming from its manifest."""
        root = Path(root)
        manifest_path = root / PROJECT_MANIFEST
        if not manifest_path.exists():
            raise FileNotFoundError(f"no {PROJECT_MANIFEST} found under {root}")
        data = json.loads(manifest_path.read_text())
        return cls(
            root=root,
            provenance=data.get("provenance", {}),
            stage=data.get("stage", "created"),
            stage_log=data.get("stage_log", []),
            created_at=data.get("created_at", _now()),
            updated_at=data.get("updated_at", _now()),
        )

    # -- persistence --------------------------------------------------------

    def save(self) -> None:
        self.updated_at = _now()
        manifest = {
            "provenance": self.provenance,
            "stage": self.stage,
            "stage_log": self.stage_log,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }
        (self.root / PROJECT_MANIFEST).write_text(json.dumps(manifest, indent=2))

    # -- stage/event tracking ------------------------------------------------

    def _log_event(self, event: str, details: dict[str, Any]) -> None:
        self.stage_log.append({"time": _now(), "event": event, "details": details})

    def advance(self, stage: str, details: dict[str, Any] | None = None) -> None:
        """Record that the project has reached `stage`. Does not enforce
        strict ordering (redoing an earlier stage, e.g. re-clustering, is
        allowed and just gets its own log entry) but validates the stage
        name is one we recognize."""
        if stage not in STAGES:
            raise ValueError(f"unknown stage {stage!r}, expected one of {STAGES}")
        self._log_event("stage_advanced", {"stage": stage, **(details or {})})
        self.stage = stage
        self.save()

    def redo_from(self, stage: str, reason: str = "") -> None:
        """Record that the user is redoing the pipeline starting at `stage`
        (e.g. going back from active-space review to re-clustering). Old
        artifacts for later stages are NOT deleted automatically -- see
        `clear_downstream_artifacts` -- this just logs the intent."""
        if stage not in STAGES:
            raise ValueError(f"unknown stage {stage!r}, expected one of {STAGES}")
        self._log_event("redo_from", {"stage": stage, "reason": reason})
        self.stage = stage
        self.save()

    def set_provenance(self, provenance: dict[str, Any]) -> None:
        """Record calculation provenance parsed from the .chk (or produced
        by the New Calculation dialog): method, basis, charge, multiplicity,
        atoms, source (uploaded vs generated), etc."""
        self.provenance = provenance
        self._log_event("provenance_set", provenance)
        self.save()

    # -- artifact helpers -----------------------------------------------------

    def path_for(self, *parts: str) -> Path:
        """Resolve a path inside the project folder, creating parent dirs."""
        p = self.root.joinpath(*parts)
        p.parent.mkdir(parents=True, exist_ok=True)
        return p

    def import_chk(self, source_chk: Path) -> Path:
        """Copy an externally-produced .chk into the project as input.chk."""
        dest = self.root / "input.chk"
        if source_chk.resolve() != dest.resolve():
            shutil.copy2(source_chk, dest)
        self._log_event("chk_imported", {"source": str(source_chk)})
        self.save()
        return dest

    def summary(self) -> dict[str, Any]:
        """A compact dict for display in the GUI's project overview panel."""
        return {
            "root": str(self.root),
            "stage": self.stage,
            "provenance": self.provenance,
            "n_events": len(self.stage_log),
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }
