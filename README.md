# TPSChemVIS

**TPSChemVIS** is a desktop GUI for building active spaces, running Cluster Mean-Field (CMF) orbital optimization, and launching TPSCI/SPT calculations through [TPSChem.jl](https://github.com/arnab82/TPSChem.jl).

It is designed as a point-and-click frontend to the quantum chemistry pipeline described in:

> *Tensor Product State Configuration Interaction (TPSCI)* — see TPSChem.jl for references.

---

## Features

| Step | What you do | What the app does |
|------|-------------|-------------------|
| **Load** | Pick a PySCF `.chk` checkpoint | Reads geometry, basis, MO coefficients |
| **Visualize** | Inspect canonical MOs in 3D | Renders isosurfaces via VibeMol (in-browser, offline) |
| **Cluster** | Assign MOs or atoms to clusters | Two modes: Manual MO assignment or SPADE atom-based partitioning |
| **Active Space** | Click "Build" | Runs SPADE SVD, writes `h0/h1/h2.npy` integrals, saves per-cluster `.molden` files |
| **Inspect** | Review cluster orbitals in 3D | Dropdown switches between all-active and per-cluster moldens; edit `n_α / n_β` if needed |
| **CMF** | Choose Newton / BFGS / DIIS | Streams Julia output live; saves `cmf_result.jld2` |
| **TPSCI / Export** | Set thresholds, click Run | Renders Julia driver from Jinja2 templates, runs locally or packages for HPC (SLURM) |

**Additional capabilities:**

- **Resume anywhere** — Jump straight to CMF from saved integrals, or to TPSCI/Export from a saved CMF result, without restarting from scratch.
- **Persistent config** — TPSChem.jl path is saved to `~/.asbuilder/config.json` after first setup.
- **First-launch wizard** — Clones and builds TPSChem.jl automatically (or point to an existing clone).
- **Navigation toolbar** — Click any pipeline step in the toolbar to jump back to it.
- **Collapsible panels** — Settings sections collapse to a header so the log output gets more screen space.

---

## Requirements

| Dependency | Version | Notes |
|---|---|---|
| Python | ≥ 3.11 | |
| PyQt6 | ≥ 6.6 | GUI framework |
| PyQt6-WebEngine | ≥ 6.6 | Embedded VibeMol orbital viewer |
| PySCF | ≥ 2.4 | SCF + integral generation |
| NumPy | ≥ 1.24 | |
| SciPy | ≥ 1.11 | SPADE SVD |
| h5py | ≥ 3.9 | Checkpoint reading |
| Jinja2 | ≥ 3.1 | Julia driver templating |
| Julia | ≥ 1.11 | Required for CMF and TPSCI |
| TPSChem.jl | latest | Installed by the setup wizard |

---

## Installation

### 1. Clone this repository

```bash
git clone --recurse-submodules https://github.com/arnab82/TPSChemVIS.git
cd TPSChemVIS
```

> `--recurse-submodules` fetches VibeMol (the orbital viewer) automatically.

### 2. Install Python dependencies

```bash
pip install -e .
```

This installs all Python dependencies and registers the `asbuilder` command.

### 3. Install Julia

Download Julia 1.11+ from [julialang.org](https://julialang.org/downloads/) or via `juliaup`:

```bash
curl -fsSL https://install.julialang.org | sh
```

### 4. Launch — the setup wizard handles the rest

```bash
asbuilder
```

On first launch a setup wizard appears:

- **Use existing clone** — Browse to a TPSChem.jl directory you already have.
- **Clone from GitHub** — The app clones `https://github.com/arnab82/TPSChem.jl.git`, runs `Pkg.instantiate`, and builds PyCall automatically. Progress is shown live in the wizard.

The configured path is saved to `~/.asbuilder/config.json` — subsequent launches skip the wizard entirely.

To re-run the wizard at any time: **Tools → Julia / TPSChem.jl setup…**

---

## Usage

### Starting a new project

```bash
asbuilder ~/my_calculations/h6_project.qcproj
```

If the directory doesn't exist it is created automatically.

### Typical workflow

```
Load checkpoint (.chk)
  ↓
Visualize MOs (VibeMol 3D viewer)
  ↓
Define clusters
  ├── Manual mode  — click MO rows to assign to clusters
  └── SPADE mode   — assign atoms per cluster; select AO types (s/p/d/f)
  ↓
Build Active Space  →  inspects cluster moldens, verify n_α / n_β
  ↓
Run CMF (Newton / BFGS / DIIS)
  ↓
Run TPSCI / SPT / PT2 / CEPA  →  local or package for HPC
```

### Resuming from a saved calculation

From the **Load** screen:

- **Jump to CMF** — enabled automatically when `active_space/h0.npy` and `clusters.json` are found.
- **Jump to TPSCI/Export** — enabled automatically when `cmf/cmf_result.jld2` is found.

Use the **Browse** buttons to load intermediate results from a different project directory.

### SPADE mode (bimetallic / fragment systems)

Switch to the **SPADE — assign atoms** tab in the Clusters screen.  
Click atom rows to assign atoms to clusters. For each cluster, check which AO types (s / p / d / f) should contribute to the SPADE projector — useful for, e.g., assigning only Fe 3d orbitals to a metal cluster.

---

## Command-line options

```
asbuilder [project_dir] [options]

Arguments:
  project_dir          Project folder (default: ~/asbuilder_projects/untitled.qcproj)

Options:
  --julia-bin PATH     Julia executable (default: julia from PATH)
  --julia-project PATH Override TPSChem.jl directory (saved to config)
  --vibemol-root PATH  Path to a custom VibeMol build
  --setup              Force the setup wizard even if already configured
```

---

## Project directory layout

After a full run, a project folder looks like:

```
my_project.qcproj/
├── project.json           # stage tracker
├── input.chk              # PySCF checkpoint (copied on load)
├── orbitals.molden        # canonical MO molden
├── clusters.json          # cluster definitions
├── active_space/
│   ├── h0.npy             # core energy
│   ├── h1.npy             # one-electron integrals
│   ├── h2.npy             # two-electron integrals
│   ├── Cact.molden        # all active MOs
│   ├── cluster_1_name.molden
│   ├── cluster_2_name.molden
│   └── cluster_map.json
├── cmf/
│   ├── driver_cmf.jl      # rendered CMF driver
│   └── cmf_result.jld2    # CMF output bundle
└── export/
    ├── driver_tpsci.jl    # rendered TPSCI driver
    └── export.log
```

---

## HPC / SLURM submission

The **TPSCI/Export** screen generates a `submit.slurm` script alongside the Julia driver. Fill in the job name, account, partition, nodes, and walltime, click **Package for HPC…** to download a `.zip` containing everything needed to run on a cluster.

---

## Dependencies and licenses

| Package | License | Notes |
|---|---|---|
| [PySCF](https://github.com/pyscf/pyscf) | Apache 2.0 | SCF and integral back-end |
| [VibeMol](https://github.com/evangelistalab/vibemol) | MIT | 3D orbital viewer (git submodule) |
| [TPSChem.jl](https://github.com/arnab82/TPSChem.jl) | See repo | CMF / TPSCI / SPT engine |
| [PyQt6](https://www.riverbankcomputing.com/software/pyqt/) | GPL v3 | GUI framework |

---

## Contributing

Pull requests are welcome. The Python back-end (`asbuilder/active_space/`, `asbuilder/cluster/`, `asbuilder/julia_bridge/`) has no Qt dependency and can be tested in isolation. The GUI (`asbuilder/gui/`) requires a display.

```bash
# Run the app from source
pip install -e .
asbuilder
```

---

## Citation

If you use TPSChemVIS in your research, please cite the underlying TPSChem.jl methodology (see the TPSChem.jl repository for the appropriate references).
