import numpy as np
import exact as ed

J = 0.25
W = 1
wx = 0.001
# lam0s = np.linspace(1.1, 1.6, 100)
lam0s = np.linspace(0, 1, 100)
n_bosons = 40
Ns = [8, 10, 12]


for N in Ns:
    print(f"Computing N={N}...")
    mxs = []
    mzs = []
    nphots = []
    energies = []
    e1 = []
    e2 = []
    for lam in lam0s:
        H = ed.dicke_ising(J, W, wx, lam, N, n_bosons)
        vals, vects = ed.lanczos_ed(H, k=8, compute_eigenvectors=True)
        energies.append(vals[0] / N)
        e1.append(vals[1] / N)
        e2.append(vals[2] / N)
        v0 = vects[:, 0]

        Sz, Sp, Sm, Seye = ed.spin_operators(1 / 2)
        Sx = 0.5 * (Sp + Sm)
        a, ad, beye = ed.boson_operators(n_bosons)

        Sz_full = ed.csr_matrix((2**N * (n_bosons + 1), 2**N * (n_bosons + 1)))
        for i in range(N):
            op_chain = [Seye] * i + [Sz] + [Seye] * (N - i - 1) + [beye]
            Sz_full += ed.sparse_kron(*op_chain)

        Sx_full = ed.csr_matrix((2**N * (n_bosons + 1), 2**N * (n_bosons + 1)))
        for i in range(N):
            op_chain = [Seye] * i + [Sx] + [Seye] * (N - i - 1) + [beye]
            Sx_full += ed.sparse_kron(*op_chain)

        op_chain = [Seye] * N + [ad @ a]
        nphot_full = ed.sparse_kron(*op_chain)

        # mzs.append(np.dot(v0, Sz_full.dot(v0)))
        # mxs.append(np.dot(v0, Sx_full.dot(v0)))
        # nphots.append(np.vdot(v0, nphot_full.dot(v0)) / N)
        mzs.append(np.vdot(v0, np.dot(Sz_full.toarray(), v0)) / N)
        mxs.append(np.vdot(v0, np.dot(Sx_full.toarray(), v0)) / N)
        nphots.append(np.vdot(v0, np.dot(nphot_full.toarray(), v0)) / N)

    energies = np.array(energies)
    mzs = np.array(mzs)
    mxs = np.array(mxs)
    nphots = np.array(nphots)
    e1 = np.array(e1)
    e2 = np.array(e2)

    np.savez(
        f"data/exact_{J}_{W}_{wx}_{N}_{n_bosons}",
        lam0s=lam0s,
        e0s=energies,
        mzs=mzs,
        mxs=mxs,
        nphots=nphots,
        e1=e1,
        e2=e2,
    )
