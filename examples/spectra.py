"""Comparison of spectra from different solvers."""

import matplotlib.pyplot as plt
import numpy
from pyscf import gto, scf

from dyson.expressions import ADC2
from dyson.solvers import Exact, Downfolded, MBLSE, MBLGF, CorrectionVector, CPGF
from dyson.grids import GridRF

# Get a molecule and mean-field from PySCF
mol = gto.M(atom="Li 0 0 0; Li 0 0 1.64", basis="sto3g", verbose=0)
mf = scf.RHF(mol)
mf.kernel()

# Define a grid for the spectra
grid = GridRF.from_uniform(-3.0, 3.0, 256, eta=1e-1)

# Get a complete self-energy (identity overlap) to solve for demonstration purposes
exp_h = ADC2.h.from_mf(mf)
exp_p = ADC2.p.from_mf(mf)
exact_h = Exact.from_expression(exp_h)
exact_h.kernel()
exact_p = Exact.from_expression(exp_p)
exact_p.kernel()
result = exact_h.result.combine(exact_p.result)
static = result.get_static_self_energy()
self_energy = result.get_self_energy()

# Solve the self-energy using each static solver -- since ADC(2) is non-Dyson, we can just add
# the Green's function rather than using the spectral combination utility
spectra = {}
for key, solver_cls, kwargs in [
    ("Exact", Exact, dict()),
    ("Downfolded", Downfolded, dict()),
    ("MBLSE(1)", MBLSE, dict(max_cycle=1)),
    ("MBLGF(1)", MBLGF, dict(max_cycle=1)),
]:
    solver = solver_cls.from_self_energy(static, self_energy, **kwargs)
    solver.kernel()
    gf = solver.result.get_greens_function()
    spectra[key] = -grid.evaluate_lehmann(gf, ordering="retarded", trace=True).imag / numpy.pi

# Solve the self-energy using each dynamic solver
for key, solver_cls, kwargs in [
    ("CorrectionVector", CorrectionVector, dict()),
    ("CPGF(256)", CPGF, dict(max_cycle=256)),
]:
    solver = solver_cls.from_self_energy(
        static,
        self_energy,
        grid=grid,
        ordering="retarded",
        trace=True,
        **kwargs,
    )
    gf = solver.kernel()
    spectra[key] = -gf.imag / numpy.pi

# Plot the spectra
plt.figure()
for i, (key, spectrum) in enumerate(spectra.items()):
    plt.plot(grid, spectrum, f"C{i}", label=key)
plt.xlabel("Frequency")
plt.ylabel("Spectral function")
plt.grid()
plt.legend()
plt.tight_layout()
plt.show()
