"""
Utility functions.
"""

import numpy as np


def format_value(val, prec=8):
    """
    Format a float or complex value.
    """

    if not np.iscomplexobj(val):
        return "%.*f" % (prec, val)
    else:
        op = "+" if val.imag >= 0 else "-"
        return "%.*f %s %.*f" % (prec, val.real, op, prec, np.abs(val.imag))


def print_eigenvalues(eigvals, nroots=5, abs_sort=True, header=True):
    """
    Return a string summarising some eigenvalues.
    """

    lines = ["-" * 25]
    if header:
        lines += ["{:^25s}".format("Eigenvalue summary")]
        lines += ["-" * 25]
    lines += [
            "%4s %20s" % ("Root", "Value"),
            "%4s %20s" % ("-" * 4, "-" * 20),
    ]

    inds = np.argsort(np.abs(eigvals.real)) if abs_sort else np.argsort(eigvals.real)
    for i in inds[:min(nroots, len(eigvals))]:
        lines.append("%4d %20s" % (i, format_value(eigvals[i])))

    if nroots < len(eigvals):
        lines.append(" ...")

    lines.append("-" * 25)

    return "\n".join(lines)


def print_dyson_orbitals(eigvals, eigvecs, nphys, nroots=5, abs_sort=True, phys_threshold=1e-8, header=True):
    """
    Returns a string summarising the projection of some eigenfunctions
    into the physical space, resulting in Dyson orbitals.
    """

    lines = ["-" * 93]
    if header:
        lines += ["{:^93s}".format("Dyson orbital summary")]
        lines += ["-" * 93]
    lines += [
            "{:>4s} {:^20s} {:^33s} {:^33s}".format("", "", "Weight", ""),
            "{:>4s} {:^20s} {:^33s} {:^33s}".format(
                "Orb", "Energy", "-" * 33, "Dominant physical states",
            ),
            "{:>4s} {:^20s} {:>16s} {:>16s} {:>16s}".format("", "", "Physical", "Auxiliary", ""),
            "{:>4s} {:^20s} {:>16s} {:>16s} {:>16s}".format(
                "-" * 4, "-" * 20, "-" * 16, "-" * 16, "-" * 33,
            ),
    ]

    mask = np.linalg.norm(eigvecs[:nphys], axis=0)**2 > phys_threshold
    eigvals, eigvecs = eigvals[mask], eigvecs[:, mask]

    inds = np.argsort(np.abs(eigvals.real)) if abs_sort else np.argsort(eigvals.real)
    for i in inds[:min(nroots, len(eigvals))]:
        v = eigvecs[:, i]
        phys = np.linalg.norm(np.abs(v[:nphys]))**2
        aux = np.linalg.norm(np.abs(v[nphys:]))**2
        chars = []
        for j in np.argsort(np.abs(v[:nphys])**2):
            if eigvecs[j, i]**2 > 0.2:
                chars.append("%d (%.2f)" % (j, np.abs(eigvecs[j, i])**2))
        chars = ", ".join(chars)
        lines.append("%4d %20s %16.3g %16.3g %33s" % (
            i, format_value(eigvals[i]), phys, aux, chars,
        ))

    lines += ["-" * 93]

    return "\n".join(lines)


def cache(function):
    """
    Caches return values according to positional and keyword arguments
    in the `_cache` property of an object.
    """

    def wrapper(obj, *args, **kwargs):
        if (function.__name__, args, tuple(kwargs.items())) in obj._cache:
            return obj._cache[function.__name__, args, tuple(kwargs.items())]
        else:
            out = function(obj, *args, **kwargs)
            obj._cache[function.__name__, args, tuple(kwargs.items())] = out
            return out

    return wrapper


def matrix_power(m, power, hermitian=True, threshold=1e-10, return_error=False):
    """
    Compute the power of the matrix `m` via the eigenvalue
    decomposition.
    """

    if hermitian:
        eigvals, eigvecs = np.linalg.eigh(m)
    else:
        eigvals, eigvecs = np.linalg.eig(m)

    if np.abs(power) < 1:
        mask = np.abs(eigvals) > threshold
    else:
        mask = np.ones_like(eigvals, dtype=bool)

    if hermitian and not np.iscomplexobj(m):
        if np.abs(power) < 0:
            mask = np.logical_and(mask, eigvals > 0)
        eigvecs_right = (eigvecs.T)
    elif hermitian and np.iscomplexobj(m):
        power = power + 0.0j
        eigvecs_right = (eigvecs.T.conj())
    else:
        power = power + 0.0j
        eigvecs_right = np.linalg.inv(eigvecs)

    left = eigvecs[:, mask] * eigvals[mask][None]**power
    right = eigvecs_right[mask]
    m_pow = np.dot(left, right)

    if return_error:
        left = eigvecs[:, ~mask] * eigvals[~mask][None]
        right = eigvecs_right[~mask]
        m_res = np.dot(left, right)
        error = np.linalg.norm(np.linalg.norm(m_res))
        return m_pow, error
    else:
        return m_pow


def hermi_sum(m):
    """
    Return m + m^†
    """

    return m + m.T.conj()


def build_block_tridiagonal(on_diagonal, off_diagonal_upper, off_diagonal_lower=None):
    """
    Build a block tridiagonal matrix.
    """

    zero = np.zeros_like(on_diagonal[0])

    if off_diagonal_lower is None:
        off_diagonal_lower = [m.T.conj() for m in off_diagonal_upper]

    m = np.block([[
        on_diagonal[i] if i == j else
        off_diagonal_upper[j] if j == i-1 else
        off_diagonal_lower[i] if i == j-1 else zero
        for j in range(len(on_diagonal))]
        for i in range(len(on_diagonal))]
    )

    return m


def scaled_error(a, b):
    """
    Return the scaled error between two matrices.
    """

    a = a / max(np.max(np.abs(a)), 1)
    b = b / max(np.max(np.abs(b)), 1)

    return np.linalg.norm(a - b)
