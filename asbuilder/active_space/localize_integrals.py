"""
Orbital partitioning (SPADE) + active-space integral generation.

The partitioning functions (`svd_subspace_partitioning`, `sym_ortho`, etc.)
follow the notebook TPSChem.jl/examples/notes/scf_spade.ipynb exactly.
`build_active_space` implements the driver cells of that notebook:

  1. Extract Cdocc/Csing/Cvirt from the MF solution.
  2. Build per-cluster AO projectors via sqrtm(S)[:, ao_indices].
  3. One global SVD pass to get the full (Oact/Sact/Vact, Cenv) split.
  4. Per-cluster SVD pass to assign active orbitals to clusters.
  5. sym_ortho across clusters, then build embedded h0/h1/h2.

Two modes are supported:
  - SPADE mode  : cluster.atom_indices is set; SPADE runs and fills in
                  cluster.orbitals / cluster.fspace automatically.
  - Manual mode : cluster.orbitals already set by the user in the viewer;
                  SPADE is skipped and integrals are built directly.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import scipy.linalg


# --------------------------------------------------------------------------
# SPADE partitioning -- verbatim from scf_spade.ipynb
# --------------------------------------------------------------------------


def canonicalize(orbital_blocks, F):
    """Rotate each orbital block to diagonalize the AO Fock matrix F."""
    out = []
    for ob in orbital_blocks:
        fi = ob.T @ F @ ob
        fi = 0.5 * (fi + fi.T)
        _, U = np.linalg.eig(fi)
        out.append(ob @ U)
    return out


def extract_frontier_orbitals(orbital_blocks, F, dims):
    """Split each orbital block into (env, act, virt) sub-blocks.

    dims = [(NDocc, NAct, Nvirt), ...] -- one tuple per block.
    """
    NAOs = F.shape[0]
    tmp = canonicalize(orbital_blocks, F)
    env_blocks, act_blocks, vir_blocks = [], [], []
    for ob in tmp:
        assert ob.shape[0] == NAOs
        env_blocks.append(np.zeros((NAOs, 0)))
        act_blocks.append(np.zeros((NAOs, 0)))
        vir_blocks.append(np.zeros((NAOs, 0)))
    for obi, ob in enumerate(tmp):
        assert np.sum(dims[obi]) == ob.shape[1]
        d = dims[obi]
        env_blocks[obi] = ob[:, :d[0]]
        act_blocks[obi] = ob[:, d[0]:d[0] + d[1]]
        vir_blocks[obi] = ob[:, d[0] + d[1]:]
    return env_blocks, act_blocks, vir_blocks


def svd_subspace_partitioning(orbitals_blocks, Pv, S):
    """Partition each orbital block via SVD overlap onto projector Pv.

    Returns (Cf, Ce) where Cf[i] are the fragment orbitals in block i
    and Ce[i] are the remainder (environment) orbitals in block i.
    """
    nfrag = Pv.shape[1]
    nbas = S.shape[0]
    assert Pv.shape[0] == nbas

    nmo = sum(ob.shape[1] for ob in orbitals_blocks)
    print(" Partition %4i orbitals into a total of %4i orbitals" % (nmo, nfrag))

    PS = Pv.T @ S @ Pv
    P = Pv @ np.linalg.inv(PS) @ Pv.T

    s, Clist, spaces = [], [], []
    Cf = [np.zeros((nbas, 0)) for _ in orbitals_blocks]
    Ce = [np.zeros((nbas, 0)) for _ in orbitals_blocks]

    for obi, ob in enumerate(orbitals_blocks):
        _, sob, Vob = np.linalg.svd(P @ S @ ob, full_matrices=True)
        s.extend(sob)
        Clist.append(ob @ Vob.T)
        spaces.extend([obi] * ob.shape[1])

    spaces = np.array(spaces)
    s = np.array(s)
    perm = np.argsort(s)[::-1]
    s, spaces = s[perm], spaces[perm]
    Ctot = np.hstack(Clist)[:, perm]

    print(" %16s %12s %-12s" % ("Index", "Sing. Val.", "Space"))
    for i in range(nfrag):
        print(" %16i %12.8f %12s*" % (i, s[i], spaces[i]))
        Cf[spaces[i]] = np.hstack((Cf[spaces[i]], Ctot[:, i:i + 1]))
    for i in range(nfrag, nmo):
        if s[i] > 1e-6:
            print(" %16i %12.8f %12s" % (i, s[i], spaces[i]))
        Ce[spaces[i]] = np.hstack((Ce[spaces[i]], Ctot[:, i:i + 1]))

    return Cf, Ce


def sym_ortho(frags, S, thresh=1e-8):
    """Symmetrically orthogonalize a list of MO coefficient matrices."""
    inds, shift = [], 0
    for f in frags:
        inds.append(list(range(shift, shift + f.shape[1])))
        shift += f.shape[1]

    Cnonorth = np.hstack(frags)
    Smo = Cnonorth.T @ S @ Cnonorth
    X = np.linalg.inv(scipy.linalg.sqrtm(Smo))
    Corth = Cnonorth @ X
    return [Corth[:, idx] for idx in inds]


# --------------------------------------------------------------------------
# Active-space integrals
# --------------------------------------------------------------------------


@dataclass
class ActiveSpaceIntegrals:
    """h0/h1/h2 matching TPSChem.jl's InCoreInts contract."""

    h0: float
    h1: np.ndarray   # (n_act, n_act)
    h2: np.ndarray   # (n_act, n_act, n_act, n_act), chemist notation (pq|rs)

    def save(self, out_dir: str | Path) -> None:
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        np.save(out / "h0.npy", np.asarray(self.h0))
        np.save(out / "h1.npy", self.h1)
        np.save(out / "h2.npy", self.h2)


# --------------------------------------------------------------------------
# Orchestration
# --------------------------------------------------------------------------


def build_active_space(
    mol,
    mo_coeff: np.ndarray,
    mo_occ: np.ndarray,
    fock_ao,                  # unused for SPADE; kept for API stability
    overlap_ao: np.ndarray,
    clusters: list[Any],      # list[Cluster] from asbuilder.cluster.state
    output_dir: str | Path | None = None,
    mode: str = "auto",       # "auto" | "spade" | "manual"
) -> ActiveSpaceIntegrals:
    """Run SPADE partitioning and build embedded active-space integrals.

    SPADE mode  (cluster.atom_indices is set):
        Builds per-cluster AO projectors from atom membership, runs two
        rounds of svd_subspace_partitioning (global then per-cluster),
        orthogonalises with sym_ortho, updates cluster.orbitals and
        cluster.fspace in-place, then builds h0/h1/h2.

    Manual mode (cluster.orbitals already set, atom_indices is None):
        Skips SPADE; assembles Cact from the user's orbital assignments
        and builds integrals directly.

    Saves h0/h1/h2.npy (+ mo_coeffs.npy, Cact.molden) to output_dir if
    provided. Returns ActiveSpaceIntegrals regardless.
    """
    from pyscf import ao2mo
    from pyscf.scf import hf as scf_hf

    S = overlap_ao

    # --- Determine MO blocks (RHF: nsing=0, ROHF: nsing>0) ----------------
    n_alpha, n_beta = mol.nelec
    ndocc = n_beta          # doubly occupied = n_beta
    nsing = n_alpha - n_beta
    Cdocc = mo_coeff[:, :ndocc]
    Csing = mo_coeff[:, ndocc:ndocc + nsing]
    Cvirt = mo_coeff[:, ndocc + nsing:]

    # --- Branch on SPADE vs manual ----------------------------------------
    if mode == "auto":
        spade_mode = any(c.atom_indices is not None for c in clusters)
    else:
        spade_mode = (mode == "spade")

    print(f" ---- Partitioning mode: {'SPADE' if spade_mode else 'Manual'} ----")

    if spade_mode:
        Cact, Cenv = _run_spade(mol, clusters, Cdocc, Csing, Cvirt, S, nsing)
        # Cact = np.hstack(sym_ortho'd Cfrags) from SPADE — NOT the original SCF MOs.
        # cluster.orbitals and cluster.fspace are updated in-place by _run_spade.
    else:
        Cact, Cenv = _cact_manual(mol, mo_coeff, clusters, ndocc, mo_occ)

    # --- Embedded integrals (notebook recipe) -----------------------------
    hcore = mol.intor("int1e_kin") + mol.intor("int1e_nuc")
    h0 = mol.energy_nuc()

    if Cenv.shape[1] > 0:
        d1_embed = 2.0 * Cenv @ Cenv.T
        j, k = scf_hf.get_jk(mol, d1_embed, hermi=1)
        h0 += float(np.trace(d1_embed @ (hcore + 0.5 * j - 0.25 * k)))
        h_eff = hcore + j - 0.5 * k
    else:
        h_eff = hcore

    nact = Cact.shape[1]
    h1 = Cact.T @ h_eff @ Cact
    h2 = ao2mo.kernel(mol, Cact, aosym="s4", compact=False)
    h2 = h2.reshape(nact, nact, nact, nact)

    ints = ActiveSpaceIntegrals(h0=float(h0), h1=h1, h2=h2)

    if output_dir is not None:
        out = Path(output_dir)
        # Save moldens first so the user can inspect orbitals before integrals
        _save_moldens(mol, Cact, Cenv, clusters, out)
        # Then save integrals and auxiliary arrays
        ints.save(out)
        np.save(out / "mo_coeffs.npy", Cact)
        np.save(out / "overlap_mat.npy", S)

    return ints


def _save_moldens(mol, Cact, Cenv, clusters, out: Path) -> None:
    """Save Cact.molden (all active combined), one molden per cluster, and cluster_map.json."""
    import json
    try:
        from pyscf.tools import molden as pyscf_molden
    except ImportError:
        print(" [warn] pyscf.tools.molden not available — skipping molden output")
        return

    # --- All active orbitals combined ---
    try:
        pyscf_molden.from_mo(mol, str(out / "Cact.molden"), Cact)
        print(f" wrote Cact.molden ({Cact.shape[1]} active orbitals)")
    except Exception as e:
        print(f" [warn] Cact.molden: {e}")

    # --- Environment ---
    if Cenv.shape[1] > 0:
        try:
            pyscf_molden.from_mo(mol, str(out / "Cenv.molden"), Cenv)
        except Exception as e:
            print(f" [warn] Cenv.molden: {e}")

    # --- Per-cluster moldens ---
    # Cact columns are ordered by orbital index (from _cact_from_clusters).
    # Rebuild the column → cluster mapping.
    sorted_by_orb = sorted(
        ((o, c) for c in clusters for o in c.orbitals),
        key=lambda x: x[0],
    )
    cluster_cols: dict[int, list[int]] = {c.id: [] for c in clusters}
    for col, (_, c) in enumerate(sorted_by_orb):
        cluster_cols[c.id].append(col)

    cluster_map: dict[str, dict] = {}
    for c in clusters:
        cols = cluster_cols.get(c.id, [])
        if not cols:
            continue
        Cc = Cact[:, cols]
        fname = out / f"cluster_{c.id}_{c.name}.molden"
        try:
            pyscf_molden.from_mo(mol, str(fname), Cc)
            print(f" wrote {fname.name} ({len(cols)} orbitals)")
        except Exception as e:
            print(f" [warn] {fname.name}: {e}")
        cluster_map[str(c.id)] = {
            "name": c.name,
            "color": c.color,
            "n_orb": len(cols),
            "molden": fname.name,
            "fspace": list(c.fspace),
        }

    (out / "cluster_map.json").write_text(json.dumps(cluster_map, indent=2))
    print(f" wrote cluster_map.json ({len(cluster_map)} clusters)")


# --------------------------------------------------------------------------
# Internal helpers
# --------------------------------------------------------------------------


def _run_spade(mol, clusters, Cdocc, Csing, Cvirt, S, nsing):
    """Run two-level SPADE; update cluster.orbitals/fspace in-place.

    Returns (Cact, Cenv):
      Cact — np.hstack of sym_ortho'd per-cluster orbital blocks (NOT original SCF MOs).
      Cenv — environment doubly-occupied MOs (from global SVD, same subspace as
             mo_coeff columns not selected into the active space).
    """
    X = scipy.linalg.sqrtm(S).real
    ao_labels = mol.ao_labels(fmt=False)  # (atom_idx, symbol, ao_type, component)

    # Build per-cluster AO index lists
    frag_aos = []
    full_ao_set = set()
    for cluster in clusters:
        if cluster.atom_indices is None:
            raise ValueError(
                f"cluster {cluster.id} has no atom_indices set; "
                "all clusters must use atom-based (SPADE) mode together"
            )
        atom_set = set(cluster.atom_indices)
        ao_type_filter = set(cluster.ao_types) if cluster.ao_types else None
        frag_ao = [
            i for i, ao in enumerate(ao_labels)
            if ao[0] in atom_set and (
                ao_type_filter is None or
                any(
                    ao[2] == t or ao[2].lstrip("0123456789") == t
                    for t in ao_type_filter
                )
            )
        ]
        if not frag_ao:
            raise ValueError(
                f"cluster {cluster.id}: no AOs matched atom_indices={cluster.atom_indices} "
                f"ao_types={cluster.ao_types or 'all'}. Check your selection."
            )
        frag_aos.append(frag_ao)
        full_ao_set.update(frag_ao)

    Pfull = X[:, sorted(full_ao_set)]
    Pf = [X[:, fao] for fao in frag_aos]

    # Global partition: all orbital blocks vs full AO projector
    all_blocks = [Cdocc, Csing, Cvirt]  # some may be (nao, 0) for RHF
    Cf, Ce = svd_subspace_partitioning(all_blocks, Pfull, S)
    Oact, Sact, Vact = Cf[0], Cf[1], Cf[2]
    # Ce[0] = environment doubly-occupied MOs (low overlap with Pfull)
    Cenv = Ce[0]

    # Per-cluster partition of the active space
    act_blocks = [Oact, Sact, Vact]
    Cfrags, init_fspace, cluster_orb_lists = [], [], []
    orb_index = 1

    for cluster, pf in zip(clusters, Pf):
        Cf_i, _ = svd_subspace_partitioning(act_blocks, pf, S)
        Of, Sf, Vf = Cf_i[0], Cf_i[1], Cf_i[2]

        parts = [b for b in [Of, Sf, Vf] if b.shape[1] > 0]
        Cfrag = np.hstack(parts) if parts else np.zeros((Oact.shape[0], 0))
        Cfrags.append(Cfrag)

        ndocc_f = Of.shape[1]
        nsing_f = Sf.shape[1]
        init_fspace.append((ndocc_f + nsing_f, ndocc_f))

        nmof = Of.shape[1] + Sf.shape[1] + Vf.shape[1]
        cluster_orb_lists.append(list(range(orb_index, orb_index + nmof)))
        orb_index += nmof

    # Orthogonalize fragment orbital blocks across clusters
    Cfrags = sym_ortho(Cfrags, S)
    # SPADE active orbital coefficients: columns are the new SPADE-rotated MOs,
    # NOT columns of the original SCF mo_coeff matrix.
    Cact = np.hstack(Cfrags)

    # Write results back into Cluster objects
    for cluster, orb_list, fsp in zip(clusters, cluster_orb_lists, init_fspace):
        cluster.orbitals = orb_list
        cluster.fspace = fsp

    print()
    print(" ---- SPADE result ----")
    print(f" init_fspace = {init_fspace}")
    print(f" clusters    = {cluster_orb_lists}")
    print(" ----------------------")

    return Cact, Cenv


def _cact_from_clusters(mol, mo_coeff, clusters, ndocc):
    """Build Cact grouped by cluster, then renumber cluster.orbitals sequentially.

    Before: cluster 1 → [1,4], cluster 2 → [2,5], cluster 3 → [3,6]
            Cact columns: [0,1,2,3,4,5] (sorted globally)
    After:  Cact columns grouped: cluster1(cols 0,1) | cluster2(cols 2,3) | cluster3(cols 4,5)
            cluster.orbitals updated to [1,2], [3,4], [5,6] (1-based sequential)
    """
    # Track original assignment before we mutate cluster.orbitals
    original_assigned_0based = {o - 1 for c in clusters for o in c.orbitals}

    # Build Cact column list: cluster order preserved, sorted within each cluster
    col_indices = []
    for c in clusters:
        for o in sorted(c.orbitals):
            col_indices.append(o - 1)   # 0-based into mo_coeff

    Cact = mo_coeff[:, col_indices]

    # Renumber cluster.orbitals to consecutive 1-based indices
    new_idx = 1
    for c in clusters:
        n = len(c.orbitals)
        c.orbitals = list(range(new_idx, new_idx + n))
        new_idx += n

    print()
    print(" ---- Orbital reordering ----")
    for c in clusters:
        print(f"  cluster {c.id} ({c.name}): orbitals {c.orbitals}")
    print(" ----------------------------")

    env_cols = [i for i in range(ndocc) if i not in original_assigned_0based]
    Cenv = mo_coeff[:, env_cols] if env_cols else np.zeros((mol.nao, 0))
    return Cact, Cenv


def _cact_manual(mol, mo_coeff, clusters, ndocc, mo_occ):
    """Build Cact/Cenv from manually assigned cluster.orbitals; derive fspace from mo_occ."""
    for cluster in clusters:
        n_alpha = sum(1 for o in cluster.orbitals if mo_occ[o - 1] >= 1.0)
        n_beta  = sum(1 for o in cluster.orbitals if mo_occ[o - 1] >= 2.0)
        if cluster.fspace == (0, 0):
            cluster.fspace = (n_alpha, n_beta)
            print(f" cluster {cluster.id} fspace auto-derived from mo_occ: {cluster.fspace}")
        else:
            print(f" cluster {cluster.id} fspace from user: {cluster.fspace}")

    print()
    print(" ---- Manual clustering result ----")
    print(f" init_fspace = {[c.fspace for c in clusters]}")
    print(f" clusters    = {[c.orbitals for c in clusters]}")
    print(" ----------------------------------")

    return _cact_from_clusters(mol, mo_coeff, clusters, ndocc)
