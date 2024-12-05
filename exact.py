import math
from collections import namedtuple

import numpy as np
from scipy.linalg import expm, kron
from scipy.sparse import csr_matrix, diags, eye
from scipy.sparse import kron as krons
from scipy.sparse.linalg import expm as expms


def spin_operators(S, *, to_dense_array=False, format=None, dtype=np.float_):
    Sz = diags([m for m in np.arange(-S, S + 1)], format=format, dtype=dtype)
    Sp = diags(
        [math.sqrt(S * (S + 1) - m * (m + 1)) for m in np.arange(-S, S)],
        offsets=-1,
        format=format,
        dtype=dtype,
    )
    Sm = Sp.T
    Seye = eye(2 * S + 1, format=format, dtype=dtype)

    Spin_operators = namedtuple("Spin_operators", "Sz Sp Sm Seye")
    ops = Spin_operators(Sz, Sp, Sm, Seye)
    if to_dense_array:
        ops = Spin_operators(*[o.toarray() for o in ops])

    return ops


def boson_operators(
    N_photons_cutoff, *, to_dense_array=False, format=None, dtype=np.float_
):
    a = diags(
        [math.sqrt(n) for n in range(1, N_photons_cutoff + 1)],
        offsets=1,
        format=format,
        dtype=dtype,
    )
    ad = a.T
    beye = eye(N_photons_cutoff + 1, format=format, dtype=dtype)

    Boson_operators = namedtuple("Boson_operators", "a ad beye")
    ops = Boson_operators(a, ad, beye)
    if to_dense_array:
        ops = Boson_operators(*[o.toarray() for o in ops])

    return ops


def sparse_kron_factory(n):
    """
    function factory: returns a function that computes the kronecker product of n sparse matrices
    """

    import scipy.sparse as sp

    if n == 2:
        return sp.kron
    else:

        def inner(*ops, format=None):
            if len(ops) != n:
                raise TypeError(
                    f"sparse_kron_factory({n})() takes exactly {n} arguments"
                )

            return sp.kron(ops[0], sparse_kron_factory(n - 1)(*ops[1:]), format=format)

        inner.__name__ = f"sparse_kron_factory({n})"
        inner.__module__ = __name__
        inner.__doc__ = (
            f"kronecker product of sparse matrices A1, ..., An with n={n}\n\n"
        )

        return inner


def sparse_kron(*ops, format=None):
    """
    kronecker product of an arbitrary number of sparse matrices
    """
    if len(ops) < 2:
        raise TypeError("sparse_kron takes at least two arguments")

    return sparse_kron_factory(len(ops))(*ops, format=format)


def sort_eigensystem(vals, vects=None):
    if vects is None:
        return np.sort(vals)
    else:
        idx = np.argsort(vals)
        return vals[idx], vects[:, idx]


from scipy.sparse import spmatrix


def lanczos_ed(
    operator: spmatrix,
    *,
    k: int = 1,
    compute_eigenvectors: bool = False,
    scipy_args: dict = None,
):
    r"""
    *** Adapted from Netket ***

    Computes `first_n` smallest eigenvalues and, optionally, eigenvectors
    of a Hermitian operator using :meth:`scipy.sparse.linalg.eigsh`.

    Args:
        operator: Scipy sparse matrix to diagonalize.
        k: The number of eigenvalues to compute.
        compute_eigenvectors: Whether or not to return the
            eigenvectors of the operator. With ARPACK, not requiring the
            eigenvectors has almost no performance benefits.
        scipy_args: Additional keyword arguments passed to
            :meth:`scipy.sparse.linalg.eigvalsh`. See the Scipy documentation for further
            information.

    Returns:
        Either `w` or the tuple `(w, v)` depending on whether `compute_eigenvectors`
        is True.

        - w: Array containing the lowest `first_n` eigenvalues.
        - v: Array containing the eigenvectors as columns, such that`v[:, i]`
          corresponds to `w[i]`.
    """
    from scipy.sparse.linalg import eigsh

    actual_scipy_args = {}
    if scipy_args:
        actual_scipy_args.update(scipy_args)
    actual_scipy_args["which"] = "SA"
    actual_scipy_args["k"] = k
    actual_scipy_args["return_eigenvectors"] = compute_eigenvectors

    result = eigsh(operator, **actual_scipy_args)
    if not compute_eigenvectors:
        # for some reason scipy does a terrible job
        # ordering the eigenvalues and eigenvectors
        # therefore we do it ourselves
        return sort_eigensystem(result)
    else:
        return sort_eigensystem(*result)


def dicke_ising(J, wc, wx, lam, N, n_bosons):
    Sz, Sp, Sm, Seye = spin_operators(1 / 2)
    Sx = 0.5 * (Sp + Sm)
    a, ad, beye = boson_operators(n_bosons)

    sz = 2 * Sz
    sx = 2 * Sx
    # ising interaction
    Hi = csr_matrix((2**N * (n_bosons + 1), 2**N * (n_bosons + 1)))
    for i in range(N - 1):
        op_chain = [Seye] * i + [sz, sz] + [Seye] * (N - i - 2) + [beye]
        Hi += -J * sparse_kron(*op_chain)
    op_chain = [sz] + [Seye] * (N - 2) + [sz] + [beye]
    Hi += -J * sparse_kron(*op_chain)  # PBC

    # classical field
    Hb = csr_matrix((2**N * (n_bosons + 1), 2**N * (n_bosons + 1)))
    for i in range(N):
        op_chain = [Seye] * i + [sx] + [Seye] * (N - i - 1) + [beye]
        Hb += 0.5 * wx * sparse_kron(*op_chain)

    # cavity energy
    op_chain = [Seye] * N + [ad @ a]
    Hcav = wc * sparse_kron(*op_chain)

    # dicke interaction
    Hc = csr_matrix((2**N * (n_bosons + 1), 2**N * (n_bosons + 1)))
    for i in range(N):
        op_chain = [Seye] * i + [sx] + [Seye] * (N - i - 1) + [a + ad]
        Hb += lam / np.sqrt(N) * sparse_kron(*op_chain)

    return Hi + Hb + Hc + Hcav


def open_dicke_ising(J, wc, wx, lam, N, n_bosons):
    Sz, Sp, Sm, Seye = spin_operators(1 / 2)
    Sx = 0.5 * (Sp + Sm)
    a, ad, beye = boson_operators(n_bosons)

    sz = 2 * Sz
    sx = 2 * Sx
    # ising interaction
    Hi = csr_matrix((2**N * (n_bosons + 1), 2**N * (n_bosons + 1)))
    for i in range(N - 1):
        op_chain = [Seye] * i + [sz, sz] + [Seye] * (N - i - 2) + [beye]
        Hi += -J * sparse_kron(*op_chain)

    # classical field
    Hb = csr_matrix((2**N * (n_bosons + 1), 2**N * (n_bosons + 1)))
    for i in range(N):
        op_chain = [Seye] * i + [sx] + [Seye] * (N - i - 1) + [beye]
        Hb += 0.5 * wx * sparse_kron(*op_chain)

    # cavity energy
    op_chain = [Seye] * N + [ad @ a]
    Hcav = wc * sparse_kron(*op_chain)

    # dicke interaction
    Hc = csr_matrix((2**N * (n_bosons + 1), 2**N * (n_bosons + 1)))
    for i in range(N):
        op_chain = [Seye] * i + [sx] + [Seye] * (N - i - 1) + [a + ad]
        Hb += lam / np.sqrt(N) * sparse_kron(*op_chain)

    return Hi + Hb + Hc + Hcav


def open_dicke_ising_ext(J, wc, wx, wz, lam, N, n_bosons):
    Sz, Sp, Sm, Seye = spin_operators(1 / 2)
    Sx = 0.5 * (Sp + Sm)
    a, ad, beye = boson_operators(n_bosons)

    sz = 2 * Sz
    sx = 2 * Sx
    # ising interaction
    Hi = csr_matrix((2**N * (n_bosons + 1), 2**N * (n_bosons + 1)))
    for i in range(N - 1):
        op_chain = [Seye] * i + [sz, sz] + [Seye] * (N - i - 2) + [beye]
        Hi += -J * sparse_kron(*op_chain)

    # classical field
    Hb = csr_matrix((2**N * (n_bosons + 1), 2**N * (n_bosons + 1)))
    for i in range(N):
        op_chain = [Seye] * i + [sx] + [Seye] * (N - i - 1) + [beye]
        Hb += 0.5 * wx * sparse_kron(*op_chain)
        op_chain = [Seye] * i + [sz] + [Seye] * (N - i - 1) + [beye]
        Hb += 0.5 * wz * sparse_kron(*op_chain)

    # cavity energy
    op_chain = [Seye] * N + [ad @ a]
    Hcav = wc * sparse_kron(*op_chain)

    # dicke interaction
    Hc = csr_matrix((2**N * (n_bosons + 1), 2**N * (n_bosons + 1)))
    for i in range(N):
        op_chain = [Seye] * i + [sx] + [Seye] * (N - i - 1) + [a + ad]
        Hb += lam / np.sqrt(N) * sparse_kron(*op_chain)

    return Hi + Hb + Hc + Hcav


def spectral_dec(ws, A, B, vals, vects):
    e0 = vals[0]
    v0 = vects[:, 0]
    chi = np.zeros(len(ws), dtype=complex)

    for n, val in enumerate(vals):
        v = vects[:, n]
        delta = val - e0

        t1 = np.vdot(v0, np.dot(A.toarray(), v))
        t2 = np.vdot(v, np.dot(B.toarray(), v0))
        t3 = np.vdot(v, np.dot(A.toarray(), v0))
        t4 = np.vdot(v0, np.dot(B.toarray(), v))

        chi += t1 * t2 / (ws - delta) - t3 * t4 / (ws + delta)

    return chi


def sparse_spectral_dec(ws, A, B, vals, vects):
    e0 = vals[0]
    v0 = vects[:, 0]
    chi = np.zeros(len(ws), dtype=complex)

    for n, val in enumerate(vals):
        v = vects[:, n]
        delta = val - e0

        t1 = np.vdot(v0, A.dot(v))
        t2 = np.vdot(v, B.dot(v0))
        t3 = np.vdot(v, A.dot(v0))
        t4 = np.vdot(v0, B.dot(v))

        chi += t1 * t2 / (ws - delta) - t3 * t4 / (ws + delta)

    return chi
