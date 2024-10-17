import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

import dicke
import green
import LMG
import plot
import polaritons

plot.set_rcParams(size=(5.75, 4.5), lw=2, fs=16)

fig, axes = plt.subplots(1, 1, constrained_layout=True)
ax = axes

beta = 5
W = 1.0
J = 0.1
lam = 0.05

wzs = np.linspace(0.0001, 2, 200)
ws = np.linspace(0, 2, 200)
eta = 0.01

Dm = np.empty((len(wzs), len(ws)), dtype=complex)
mxs = []
for i, wz in enumerate(wzs):
    mx = LMG.f_mx_temp(wz, J, W, lam, beta)
    mxs.append(mx)
    for j, w in enumerate(ws):
        chixx0 = dicke.f_chixx0_temp(w + 1j * eta, wz, 2 * (lam**2 / W + J) * mx, beta)
        Vind = LMG.f_Vind(w + 1j * eta, W, lam, J)
        chixx = green.f_chixx(Vind, chixx0)

        Dm[i, j] = green.f_Dm(w + 1j * eta, W, lam, chixx)

cm = ax.pcolormesh(wzs, ws, -Dm.T.imag, cmap="BuPu", norm=mpl.colors.LogNorm())

cbar = fig.colorbar(cm, pad=-0.0, aspect=40, label=r"$-{\rm Im}D(\omega) \Omega$")

axin = inset_axes(ax, width="30%", height="20%", loc=1)
axin.plot(wzs, np.abs(mxs), c="b", label=r"$|m_x|$")
axin.set_xticklabels([])
axin.tick_params(axis="y", which="major", labelsize=12)
# axin.set_xlabel(r'$\lambda / \Omega$', labelpad=-10)
# axin.set_ylabel(r'$|m_x|$')
axin.text(
    0.95,
    0.75,
    r"$|m_x|$",
    fontsize=12,
    horizontalalignment="right",
    verticalalignment="center",
    transform=axin.transAxes,
)
axin.set_ylim(-0.1, 1.1)

# # ---------- two oscillator polaritons -------------
# up_twoosc = []
# lp_twoosc = []
# for i, wz in enumerate(wzs):
#     pm, pp = polaritons.LMG(wz, W, lam, J)

#     up_twoosc.append(pp)
#     lp_twoosc.append(pm)

# ax.plot(wzs, up_twoosc, c="gold", label=r"$\Omega_\pm$ (exact polaritons)", ls="--")
# ax.plot(wzs, lp_twoosc, c="gold", ls="--")

# ax.legend(loc="lower right", fontsize=10)
# # ---------- two oscillator polaritons -------------

ax.set_ylim(0, 2)
ax.set_xlabel(r"$\omega_z / \Omega$")
ax.set_ylabel(r"$\omega / \Omega$")
ax.set_title(
    rf"$\lambda/\Omega = {lam / W} \,,\; J / \Omega = {J / W} \,,\; \beta \Omega = {beta*W}$",
    fontsize=14,
)


fig.savefig("plots/temp_transmission_LMG.jpeg", bbox_inches="tight", dpi=300)
