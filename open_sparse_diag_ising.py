import numpy as np
import exact as ed
from scipy.linalg import eigh
from tqdm import tqdm

J = 0.25
W = 1
wx = 0.2
#lam0s = np.linspace(1.1, 1.6, 100)
lam0s = np.linspace(0, 1, 100)
n_bosons = 40
Ns = [14]

eta = 0.01

ws = np.linspace(0, 2, 100)

for N in Ns:
    k = 20*N
    print(f'Computing N={N}...')
    mxs = []
    mzs = []
    nphots = []
    energies = []
    e1 = []
    e2 = []
    Ds = np.empty((len(lam0s), len(ws)), dtype=complex)
    
    for i in tqdm(range(len(lam0s))):
        lam = lam0s[i]
        H = ed.open_dicke_ising(J, W, wx, lam, N, n_bosons)
        vals, vects = ed.lanczos_ed(H, k=k, compute_eigenvectors=True)
        energies.append(vals[0] / N)
        e1.append(vals[1]/N)
        e2.append(vals[2]/N)
        v0 = vects[:, 0]
        
        Sz, Sp, Sm, Seye = ed.spin_operators(1/2)
        Sx = 0.5 * (Sp + Sm)
        a, ad, beye = ed.boson_operators(n_bosons)
        
        Sz_full = ed.csr_matrix((2**N*(n_bosons + 1), 2**N*(n_bosons + 1)))
        for n in range(N):
            op_chain = [Seye]*n + [Sz] + [Seye]*(N - n - 1) + [beye]
            Sz_full += ed.sparse_kron(*op_chain)
        
        Sx_full = ed.csr_matrix((2**N*(n_bosons + 1), 2**N*(n_bosons + 1)))
        for n in range(N):
            op_chain = [Seye]*n + [Sx] + [Seye]*(N - n - 1) + [beye]
            Sx_full += ed.sparse_kron(*op_chain)
        
        op_chain = [Seye]*N + [ad @ a]
        nphot_full = ed.sparse_kron(*op_chain)
        
        op_chain = [Seye]*N + [a]
        a_full = ed.sparse_kron(*op_chain)
        
        op_chain = [Seye]*N + [ad]
        ad_full = ed.sparse_kron(*op_chain)
        
        Ds[i, :] = ed.sparse_spectral_dec(ws+1j*eta, a_full, ad_full, vals, vects)
        
        # mzs.append(np.dot(v0, Sz_full.dot(v0)))
        # mxs.append(np.dot(v0, Sx_full.dot(v0)))
        #nphots.append(np.vdot(v0, nphot_full.dot(v0)) / N)
        mzs.append(np.vdot(v0, Sz_full.dot(v0)) / N)
        mxs.append(np.vdot(v0, Sx_full.dot(v0)) / N)
        nphots.append(np.vdot(v0, nphot_full.dot(v0)) / N)
        
    energies = np.array(energies)
    mzs = np.array(mzs)
    mxs = np.array(mxs)
    nphots = np.array(nphots)
    e1 = np.array(e1)
    e2 = np.array(e2)
    
    np.savez(f'data/open_sparse_exact_{J}_{W}_{wx}_{N}_{n_bosons}',
             lam0s=lam0s,
             e0s=energies,
             mzs=mzs,
             mxs=mxs,
             nphots=nphots,
             e1=e1,
             e2=e2,
             Ds=Ds,
             )