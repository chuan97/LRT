# %%

import matplotlib.pyplot as plt
import numpy as np
from scipy.fft import fft, fftshift
from tqdm import tqdm

import exact as ed
import ising
import polaritons

# %%

J = 0.25
W = 1
wx = 0.0
# lam0s = np.linspace(1.1, 1.6, 100)
lam0s = np.linspace(0.0, 0.5, 20)
n_bosons = 40
N = 8
k = 4 * N
j = 2 * N

energies = []
wupperbound = []
wlowerbound = []
for i in tqdm(range(len(lam0s))):
    lam = lam0s[i]

    if lam == 0:
        mx = ising.mx_free(J, wx)
    else:
        mx = ising.variational_mx(J, wx, W, lam)

    H = ed.open_dicke_ising(J, W, wx, lam, N, n_bosons)
    vals, vects = ed.lanczos_ed(H, k=k, compute_eigenvectors=True)
    energies.append(vals - vals[0])
    wupperbound.append(2 * ((2 * J) + wx - 4 * lam**2 * mx / W))
    wlowerbound.append(2 * np.abs((2 * J) - (wx - 4 * lam**2 * mx / W)))

plt.plot(lam0s, energies, c="k")
plt.plot(lam0s, [levels[j] for levels in energies], c="r")
plt.plot(lam0s, wupperbound, c="b", ls=(0, (1, 10)), lw="1")
plt.plot(lam0s, wlowerbound, c="b", ls=(0, (1, 10)), lw="1")
plt.ylim(0, 1.2)
plt.xlim(0, 0.5)
plt.xlabel(r"$\lambda/\Omega$")
plt.ylabel(r"$E - E_0$")
plt.title(
    rf"$N={N}, \omega_x/\Omega={wx}, J/\Omega={J}, \,$" + rf"eigvect index$\,={j}$"
)

# ---------- two oscillator polaritons -------------
up_twoosc = []
lp_twoosc = []
for i, lam in enumerate(lam0s):
    if lam < 0.46:
        pm, pp = polaritons.normal(W, 4 * J, lam)
    else:
        pm, pp = None, None
    up_twoosc.append(pp)
    lp_twoosc.append(pm)

plt.plot(lam0s, up_twoosc, c="gold", label=r"polariton fit", ls="--")
plt.plot(lam0s, lp_twoosc, c="gold", ls="--")
# ---------- two oscillator polaritons -------------

plt.show()

# %%

lam = 0.01
wx = 0.01
wz = 0.01
N = 8
k = 4 * N

i = 2 * N + 14
# i = 7
# i = 0

H = ed.open_dicke_ising_ext(J, W, wx, wz, lam, N, n_bosons)
vals, vects = ed.lanczos_ed(H, k=k, compute_eigenvectors=True)

v = vects[:, i]
Sz, Sp, Sm, Seye = ed.spin_operators(1 / 2)
Sx = 0.5 * (Sp + Sm)
a, ad, beye = ed.boson_operators(n_bosons)

mzs = np.empty(N)
for n in range(N):
    op_chain = [Seye] * n + [Sz] + [Seye] * (N - n - 1) + [beye]
    Sz_n = ed.sparse_kron(*op_chain)
    mzs[n] = np.vdot(v, Sz_n.dot(v))

mxs = np.empty(N)
for n in range(N):
    op_chain = [Seye] * n + [Sx] + [Seye] * (N - n - 1) + [beye]
    Sx_n = ed.sparse_kron(*op_chain)
    mxs[n] = np.vdot(v, Sx_n.dot(v))

op_chain = [Seye] * N + [ad @ a]
nphot_full = ed.sparse_kron(*op_chain)
nphot = np.vdot(v, nphot_full.dot(v)) / N
print(nphot * N)

plt.plot(2 * mxs, label=r"$m_x(n)$")
plt.plot(2 * mzs, label=r"$m_z(n)$")
plt.xlabel(r"$n$")
plt.title(
    rf"$N={N}, \lambda/\Omega={lam}, \omega_x/\Omega={wx}, J/\Omega={J}, \,$"
    + rf"eigvect index$\,={i}$"
)
plt.legend()

plt.show()

mzks = fft(2 * mzs) / N
mxks = fft(2 * mxs) / N

plt.plot(np.arange(N) - N // 2, fftshift(mxks), label=r"$m_x(k)$")
plt.plot(np.arange(N) - N // 2, fftshift(mzks), label=r"$m_z(k)$")
plt.xlabel(r"$k/\pi$")
plt.title(
    rf"$N={N}, \lambda/\Omega={lam}, \omega_x/\Omega={wx}, J/\Omega={J}, \,$"
    + rf"eigvect index$\,={i}$"
)
plt.legend()
plt.show()
# %%
