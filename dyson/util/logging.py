"""
Logging utilities.
"""

import warnings

import numpy as np


def format_value(val, prec=8):
    """
    Format a float or complex value.
    """

    if not np.iscomplexobj(val):
        return "%.*f" % (prec, val)
    else:
        op = "+" if val.imag >= 0 else "-"
        return "%.*f%s%.*fj" % (prec, val.real, op, prec, np.abs(val.imag))


def print_eigenvalues(eigvals, nroots=5, abs_sort=True, header=True):
    """
    Return a string summarising some eigenvalues.
    """

    lines = ["-" * 30]
    if header:
        lines += ["{:^30s}".format("Eigenvalue summary")]
        lines += ["-" * 30]
    lines += [
        "%4s %25s" % ("Root", "Value"),
        "%4s %25s" % ("-" * 4, "-" * 25),
    ]

    inds = np.argsort(np.abs(eigvals.real)) if abs_sort else np.argsort(eigvals.real)
    for i in inds[: min(nroots, len(eigvals))]:
        lines.append("%4d %25s" % (i, format_value(eigvals[i])))

    if nroots < len(eigvals):
        lines.append(" ...")

    lines.append("-" * 30)

    return "\n".join(lines)


def print_dyson_orbitals(
    eigvals, eigvecs, nphys, nroots=5, abs_sort=True, phys_threshold=1e-8, header=True
):
    """
    Returns a string summarising the projection of some eigenfunctions
    into the physical space, resulting in Dyson orbitals.
    """

    lines = ["-" * 98]
    if header:
        lines += ["{:^98s}".format("Dyson orbital summary")]
        lines += ["-" * 98]
    lines += [
        "{:>4s} {:^25s} {:^33s} {:^33s}".format("", "", "Weight", ""),
        "{:>4s} {:^25s} {:^33s} {:^33s}".format(
            "Orb",
            "Energy",
            "-" * 33,
            "Dominant physical contributions",
        ),
        "{:>4s} {:^25s} {:>16s} {:>16s} {:>16s}".format("", "", "Physical", "Auxiliary", ""),
        "{:>4s} {:^25s} {:>16s} {:>16s} {:>16s}".format(
            "-" * 4,
            "-" * 25,
            "-" * 16,
            "-" * 16,
            "-" * 33,
        ),
    ]

    if isinstance(eigvecs, tuple):
        eigvecs_l, eigvecs_r = eigvecs
    else:
        eigvecs_l = eigvecs_r = eigvecs

    mask = np.sum(np.abs(eigvecs_l * eigvecs_r.conj()), axis=0) > phys_threshold
    inds = np.arange(eigvals.size)[mask]
    inds = inds[
        np.argsort(np.abs(eigvals[inds].real)) if abs_sort else np.argsort(eigvals[inds].real)
    ]
    for i in inds[: min(nroots, len(eigvals))]:
        v = np.abs(eigvecs_l[:, i] * eigvecs_r[:, i].conj())
        phys = np.sum(v[:nphys])
        aux = np.sum(v[nphys:])
        chars = []
        for j in np.argsort(v[:nphys]):
            if v[j] > 0.2:
                chars.append("%d (%.2f)" % (j, v[j]))
        chars = ", ".join(chars)
        lines.append(
            "%4d %25s %16.3g %16.3g %33s"
            % (
                i,
                format_value(eigvals[i]),
                phys,
                aux,
                chars,
            )
        )

    if nroots < len(inds):
        lines.append(" ...")

    lines += ["-" * 98]

    return "\n".join(lines)
