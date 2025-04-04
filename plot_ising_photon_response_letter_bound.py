import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

import ising
import dicke
import plot
import green
import polaritons

plot.set_rcParams(size=(9.5, 4.5), lw=2, fs=16)
Ns = []
fig, axes = plt.subplots(1, 2, constrained_layout=True)

ax = axes[0]

J = 0.25
wx = 0.0
W = 1
eta = 0.001

ws_upperlimit = 2
lam_rightlimit = 1
lam0s = np.linspace(0, lam_rightlimit, 100)
ws = np.linspace(0, ws_upperlimit, 100)
idx = 10

Dm = np.empty((len(lam0s), len(ws)), dtype=complex)
mxs = []
mzs = []
wupperbound = []
wlowerbound = []
wmidband = []
for i, lam in enumerate(lam0s):
    if lam == 0:
        mx = ising.mx_free(J, wx)
    else:
        mx = ising.variational_mx(J, wx, W, lam)
    mz = ising.mz_exact(J, wx - 4 * lam**2 * mx / W)
    mxs.append(mx)
    mzs.append(mz)

    wupperbound.append(2 * ((2 * J) + wx - 4 * lam**2 * mx / W))
    wlowerbound.append(2 * np.abs((2 * J) - (wx - 4 * lam**2 * mx / W)))
    wmidband.append(2 * ising.f_ek(np.pi / 2, J, wx - 4 * lam**2 * mx / W))

    for j, w in enumerate(ws):
        chixx0 = ising.f_chixx0(w + 1j * eta, J, wx - 4 * lam**2 * mx / W)
        Vind = dicke.f_Vind(w + 1j * eta, W, lam, 0)
        chixx = green.f_chixx(Vind, chixx0)

        # Dm[i, j] = green.f_Dm(w + 1j*eta, W, lam, chixx)
        Dm[i, j] = -chixx

vmin = np.amin(-Dm.imag)
vmax = np.amax(-Dm.imag)
print(vmin, vmax)
cm = ax.pcolormesh(lam0s, ws, -Dm.T.imag, cmap="BuPu", norm=mpl.colors.LogNorm())
cax = ax.inset_axes([0.65, 0.1, 0.3, 0.03], transform=ax.transAxes)
cbar = fig.colorbar(cm, cax=cax, ticks=[1e-2, 1e0, 1e2], orientation="horizontal")
cbar.set_label(label=r"$-{\rm Im}D(\omega) \Omega$", fontsize=12)
cbar.ax.xaxis.set_label_position("top")
cax.tick_params(labelsize=12)

ax.set_ylim(0, ws_upperlimit)
ax.set_xlim(0, lam_rightlimit)
ax.set_xlabel(r"$\lambda / \Omega$")
# ax.set_xticklabels([])
ax.set_ylabel(r"$\omega / \Omega$")
ax.set_xticks(np.arange(0, lam_rightlimit + 0.1, 0.2))
ax.set_yticks(np.arange(0, ws_upperlimit + 0.1, 1.0))
# ax.set_title(rf'$\omega_x/\Omega = {wx} \,,\; \omega_z/\Omega = {wz} \,,\; J / \Omega = {J / W}$',
#              fontsize=14)

ax.text(
    0.05,
    0.05,
    rf"$\omega_x/\Omega = {wx}$",
    fontsize=12,
    horizontalalignment="left",
    verticalalignment="center",
    transform=ax.transAxes,
)
# ax.text(0.05,
#         0.05,
#         rf'$J/\Omega = {J}$',
#         fontsize=12,
#         horizontalalignment='left',
#         verticalalignment='center',
#         transform=ax.transAxes
#         )
ax.plot(lam0s, wupperbound, c="k", ls=(0, (1, 10)), lw="1")
ax.plot(lam0s, wlowerbound, c="k", ls=(0, (1, 10)), lw="1")
# ax.axvline(0.461, c='k', ls=(0, (1, 10)), lw='1')

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

ax.plot(lam0s, up_twoosc, c="gold", label=r"polariton fit", ls="--")
ax.plot(lam0s, lp_twoosc, c="gold", ls="--")

ax.legend(loc="lower right", fontsize=10, frameon=False)
# ---------- two oscillator polaritons -------------

axin = inset_axes(ax, width="30%", height="20%", loc=1)
axin.plot(lam0s, np.abs(mzs), c="r", label=r"$|m_z|$")
axin.plot(lam0s, np.abs(mxs), c="b", label=r"$|m_x|$")
axin.set_xticklabels([])
axin.set_xticks(np.arange(0, lam_rightlimit + 0.1, 0.2))
axin.tick_params(axis="y", which="major", labelsize=12)
axin.text(
    0.05,
    0.7,
    r"$|m_z|$",
    c="r",
    fontsize=12,
    horizontalalignment="left",
    verticalalignment="center",
    transform=axin.transAxes,
)
axin.text(
    0.95,
    0.7,
    r"$|m_x|$",
    c="b",
    fontsize=12,
    horizontalalignment="right",
    verticalalignment="center",
    transform=axin.transAxes,
)
axin.set_ylim(-0.1, 1.1)
axin.set_xlim(0, lam_rightlimit)


ax = axes[1]

wx = 0.09

Dm = np.empty((len(lam0s), len(ws)), dtype=complex)
mxs = []
mzs = []
wupperbound = []
wlowerbound = []
wmidband = []
for i, lam in enumerate(lam0s):
    if lam == 0:
        mx = ising.mx_free(J, wx)
    else:
        mx = ising.variational_mx(J, wx, W, lam)
    mz = ising.mz_exact(J, wx - 4 * lam**2 * mx / W)
    mxs.append(mx)
    mzs.append(mz)

    wupperbound.append(2 * ((2 * J) + wx - 4 * lam**2 * mx / W))
    wlowerbound.append(2 * np.abs((2 * J) - (wx - 4 * lam**2 * mx / W)))
    wmidband.append(2 * ising.f_ek(np.pi / 2, J, wx - 4 * lam**2 * mx / W))

    for j, w in enumerate(ws):
        chixx0 = ising.f_chixx0(w + 1j * eta, J, wx - 4 * lam**2 * mx / W)
        Vind = dicke.f_Vind(w + 1j * eta, W, lam, 0)
        chixx = green.f_chixx(Vind, chixx0)

        # Dm[i, j] = green.f_Dm(w + 1j*eta, W, lam, chixx)
        Dm[i, j] = -chixx

cm = ax.pcolormesh(lam0s, ws, -Dm.T.imag, cmap="BuPu", norm=mpl.colors.LogNorm())
cax = ax.inset_axes([0.65, 0.1, 0.3, 0.03], transform=ax.transAxes)
cbar = fig.colorbar(cm, cax=cax, ticks=[1e-2, 1e0, 1e2], orientation="horizontal")
cbar.set_label(label=r"$-{\rm Im}D(\omega) \Omega$", fontsize=12)
cbar.ax.xaxis.set_label_position("top")
cax.tick_params(labelsize=12)
# ax.plot(lam0s, N*(data['e2'] - data['e1']), c='r', ls='--')
ax.set_ylim(0, ws_upperlimit)
ax.set_xlim(0, lam_rightlimit)
ax.set_xlabel(r"$\lambda / \Omega$")
# ax.set_xticklabels([])
# ax.set_ylabel(r'$\omega / \Omega$')
ax.set_yticklabels([])
ax.set_xticks(np.arange(0, lam_rightlimit + 0.1, 0.2))
ax.set_yticks(np.arange(0, ws_upperlimit + 0.1, 1.0))
# ax.set_title(rf'$\omega_x/\Omega = {wx} \,,\; \omega_z/\Omega = {wz} \,,\; J / \Omega = {J / W}$',
#              fontsize=14)

ax.text(
    0.05,
    0.05,
    rf"$\omega_x/\Omega = {wx}$",
    fontsize=12,
    horizontalalignment="left",
    verticalalignment="center",
    transform=ax.transAxes,
)
# ax.text(0.05,
#         0.05,
#         rf'$J/\Omega = {J}$',
#         fontsize=12,
#         horizontalalignment='left',
#         verticalalignment='center',
#         transform=ax.transAxes
#         )
ax.plot(lam0s, wupperbound, c="k", ls=(0, (1, 10)), lw="1")
ax.plot(lam0s, wlowerbound, c="k", ls=(0, (1, 10)), lw="1")
ax.plot(lam0s, wupperbound, c="k", ls=(0, (1, 10)), lw="1")
ax.plot(lam0s, wlowerbound, c="k", ls=(0, (1, 10)), lw="1")
# ax.axvline(0.461, c='k', ls=(0, (1, 10)), lw='1')

# ---------- two oscillator polaritons -------------
up_twoosc = []
lp_twoosc = []
for i, lam in enumerate(lam0s):
    if lam < 0.41:
        pm, pp = polaritons.normal(W, 4 * J, lam)
    else:
        pm, pp = None, None
    up_twoosc.append(pp)
    lp_twoosc.append(pm)

ax.plot(lam0s, up_twoosc, c="gold", label=r"polariton fit", ls="--")
ax.plot(lam0s, lp_twoosc, c="gold", ls="--")

# ax.legend(loc='upper left', fontsize=10)
# ---------- two oscillator polaritons -------------

axin = inset_axes(ax, width="30%", height="20%", loc=1)
axin.plot(lam0s, np.abs(mzs), c="r", label=r"$|m_z|$")
axin.plot(lam0s, np.abs(mxs), c="b", label=r"$|m_x|$")
axin.set_xticklabels([])
axin.set_xticks(np.arange(0, lam_rightlimit + 0.1, 0.2))
axin.tick_params(axis="y", which="major", labelsize=12)
axin.text(
    0.05,
    0.7,
    r"$|m_z|$",
    c="r",
    fontsize=12,
    horizontalalignment="left",
    verticalalignment="center",
    transform=axin.transAxes,
)
axin.text(
    0.95,
    0.7,
    r"$|m_x|$",
    c="b",
    fontsize=12,
    horizontalalignment="right",
    verticalalignment="center",
    transform=axin.transAxes,
)
axin.set_ylim(-0.1, 1.1)
axin.set_xlim(0, lam_rightlimit)

fig.savefig(
    "plots/ising_photon_response_letter_bound.jpeg", bbox_inches="tight", dpi=300
)
