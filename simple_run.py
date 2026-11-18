import gc

import numpy as np
from ase.io import read

import symmetrix
from symmetrix import Symmetrix
from mace.calculators import MACECalculator

atoms = read("test.xyz@503",format="extxyz")

# --- Symmetrix ---
atoms.calc = Symmetrix("MACE_RPBED3zero_com1.json")
e_sym = atoms.get_potential_energy()
f_sym = atoms.get_forces().copy()
print("Symmetrix energy:", e_sym)
print(f_sym)

# Release device data, then finalize Kokkos explicitly to avoid the
# "Failed to call Kokkos::Cuda::finalize()" error at interpreter exit.
atoms.calc = None
gc.collect()
symmetrix._finalize_kokkos()

# --- Reference: original MACE model ---
atoms.calc = MACECalculator(
    model_paths="MACE_RPBED3zero_com1.model", device="cuda", default_dtype="float64"
)
e_mace = atoms.get_potential_energy()
f_mace = atoms.get_forces()
print("MACE energy:     ", e_mace)

# --- Comparison ---
df = f_sym - f_mace
print(f"Energy diff:            {e_sym - e_mace:.3e} eV "
      f"({(e_sym - e_mace) / len(atoms) * 1000:.3e} meV/atom)")
print(f"Force diff max abs:     {np.abs(df).max():.3e} eV/A")
print(f"Force diff RMS:         {np.sqrt(np.mean(df**2)):.3e} eV/A")
# Symmetrix uses spline-tabulated radial functions (and float32 by default),
# so agreement is ~1e-5 eV in energy and ~1e-4 eV/A in forces, not bit-exact.
E_TOL = 1e-5  # eV/atom
F_TOL = 1e-3  # eV/A
ok = abs(e_sym - e_mace) / len(atoms) < E_TOL and np.abs(df).max() < F_TOL
print(f"Match (|dE| < {E_TOL} eV/atom, max|dF| < {F_TOL} eV/A): {ok}")
