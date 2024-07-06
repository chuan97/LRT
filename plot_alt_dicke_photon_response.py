import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

import dicke
import plot
import green
import polaritons

plot.set_rcParams(size=(6, 5), lw=2, fs=14)

fig, axes = plt.subplots(1, 1, constrained_layout=True)
ax = axes

W = 1
wzs = np.linspace(-2, 2, 200)
z = 0

lam = 0.1
ws = np.linspace(0, 2, 200)
eta = 0.001

Dm = np.empty((len(wzs), len(ws)), dtype=complex)
for i, wz in enumerate(wzs):
    for j, w in enumerate(ws):
        mx = dicke.f_mx(np.abs(wz), z, W, lam)

        chixx0 = dicke.f_chixx0(w + 1j * eta, np.abs(wz), 2 * lam**2 * mx / W)
        Vind = dicke.f_Vind(w + 1j * eta, W, lam, z)
        chixx = 2 * green.f_chixx(Vind, chixx0)

        Dm[i, j] = green.f_Dm(w + 1j * eta, W, lam, chixx)

cm = ax.pcolormesh(wzs, ws, np.abs(Dm.T.imag), cmap="BuPu", norm=mpl.colors.LogNorm())
cbar = fig.colorbar(cm, pad=0.0, aspect=40)
# ax.axhline(W, c='k', ls=(0, (1, 5)), lw=1)
# ax.axvline(4*lam**2/W,  c='k', ls=(0, (1, 5)), lw=1)
# plt.plot(wzs, wzs, c='k', ls=(0, (1, 5)), lw=1)

# # ---------- two oscillator polaritons -------------
# up_twoosc = []
# lp_twoosc = []
# for i, wz in enumerate(wzs):
#     if z == 0:
#         pm, pp = polaritons.dicke(wz, W, lam)
#     elif z == 1:
#         pm, pp = polaritons.dicke(np.sqrt(wz*(wz + 4*lam**2/W)),
#                                 W,
#                                 lam*(1 + 4*lam**2/(W*wz))**(-1/4)
#                                 )
#     up_twoosc.append(pp)
#     lp_twoosc.append(pm)

# ax.plot(wzs, up_twoosc, c='gold', label="Poles Two Osc. model", ls='--')
# ax.plot(wzs, lp_twoosc, c='gold', ls='--')

# # ax.legend()
# # ---------- two oscillator polaritons -------------

# ax.set_ylim(0, 2)
ax.set_xlabel(r"$\omega_z / \Omega$")
ax.set_ylabel(r"$\omega / \Omega$")
ax.set_title(r"$-{\rm Im}D(\omega) \Omega$")

fig.savefig("plots/alt_dicke_photon_response.jpeg", bbox_inches="tight", dpi=300)
