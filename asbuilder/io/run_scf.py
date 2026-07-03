"""
Run a PySCF SCF calculation from (xyz, basis, level of theory) and write a
checkpoint -- this is what the GUI's "New Calculation" dialog calls in a
QThread, and it should be the *same* function your notebook calls, not a
GUI-only reimplementation.

STATUS: stub. The signature/behavior below is what the design doc's
contract expects (xyz + basis + level of theory in, a .chk with mo_coeff/
mo_energy/mo_occ/e_tot out, loadable by asbuilder.io.chk_to_molden.load_chk).
The actual notebook cells that build the Mole/set up the calculation
haven't been shared yet -- only the SPADE partitioning functions
(now in asbuilder/active_space/localize_integrals.py) were. Once shared,
replace `run_scf` below with (or have it call directly into) the real
notebook logic instead of this generic RHF/UHF implementation.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal


@dataclass
class SCFResult:
    chk_path: Path
    e_tot: float
    converged: bool
    method: str


def run_scf(
    xyz: str,
    basis: str,
    method: Literal["RHF", "UHF", "ROHF"] = "RHF",
    charge: int = 0,
    spin: int = 0,
    chk_path: str | Path = "calc.chk",
    verbose: int = 4,
    density_fit: bool = False,
    auxbasis: str | None = None,
    newton: bool = False,
) -> SCFResult:
    """Build a PySCF Mole from `xyz`/`basis`/`charge`/`spin`, run the given
    SCF `method`, and dump a checkpoint to `chk_path`.

    `xyz` is plain xyz-format atom lines (PySCF's `mol.atom` string format),
    e.g. "Cr 0 0 0\\nCr 0 0 2.5". `spin` is PySCF's convention: 2S = n_alpha
    - n_beta.

    `density_fit`: wrap the SCF object with DF/RI approximation. `auxbasis`
    sets the auxiliary basis (None → PySCF auto-selects, typically weigend or
    the cc-pVDZ-jkfit family matching the AO basis).

    `newton`: wrap with the second-order Newton-Raphson solver after any DF
    wrapping. Useful for difficult convergence cases (open-shell metals, etc.).
    """
    from pyscf import gto, scf

    chk_path = Path(chk_path)
    chk_path.parent.mkdir(parents=True, exist_ok=True)
    mol = gto.M(atom=xyz, basis=basis, charge=charge, spin=spin, verbose=verbose)
    mol.output = str(chk_path.with_suffix(".pyscf.log"))

    method = method.upper()
    if method == "RHF":
        mf = scf.RHF(mol)
    elif method == "UHF":
        mf = scf.UHF(mol)
    elif method == "ROHF":
        mf = scf.ROHF(mol)
    else:
        raise ValueError(f"unsupported method {method!r}; extend run_scf() for DFT/other methods")

    if density_fit:
        mf = mf.density_fit(auxbasis=auxbasis)
    if newton:
        mf = mf.newton()

    mf.chkfile = str(chk_path)
    mf.kernel()

    return SCFResult(chk_path=chk_path, e_tot=float(mf.e_tot), converged=bool(mf.converged), method=method)
