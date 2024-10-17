import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

import dicke
import green
import LMG
import plot

plot.set_rcParams(size=(9.5, 9), lw=2, fs=16)

fig, axes = plt.subplots(2, 2, constrained_layout=True)
ax = axes[0, 0]


W = 1.0
J = 0.1
beta = 1
lam = 0.02

wzs_upperlimit = 2
ws_upperlimit = 2

wzs = np.linspace(0.0, wzs_upperlimit, 200)
ws = np.linspace(0, ws_upperlimit, 200)
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

vmin = np.amin(-Dm.imag)
vmax = np.amax(-Dm.imag)

cm = ax.pcolormesh(wzs, ws, -Dm.T.imag, cmap="BuPu", norm=mpl.colors.LogNorm())
# cax = ax.inset_axes([0.65, 0.1, 0.3, 0.03], transform=ax.transAxes)
# cbar = fig.colorbar(cm, cax=cax, ticks=[1e-2, 1e0, 1e2], orientation="horizontal")
# cbar.set_label(label=r"$-{\rm Im}D(\omega) \Omega$", fontsize=12)
# cbar.ax.xaxis.set_label_position("top")
# cax.tick_params(labelsize=12)

ax.set_ylim(0, ws_upperlimit)
ax.set_xlim(0, wzs_upperlimit)
# ax.set_xlabel(r'$\lambda / \Omega$')
ax.set_xticklabels([])
ax.set_ylabel(r"$\omega / \Omega$")
ax.set_xticks(np.arange(0, wzs_upperlimit + 0.1, 0.5))
ax.set_yticks(np.arange(0, ws_upperlimit + 0.1, 1.0))

ax.text(
    0.05,
    0.95,
    "(a)",
    fontsize=16,
    horizontalalignment="left",
    verticalalignment="top",
    transform=ax.transAxes,
)
ax.text(
    0.05,
    0.8,
    rf"$\lambda/\Omega = {lam}$",
    fontsize=12,
    horizontalalignment="left",
    verticalalignment="center",
    transform=ax.transAxes,
)
ax.text(
    0.05,
    0.725,
    rf"$\beta \Omega = {beta}$",
    fontsize=12,
    horizontalalignment="left",
    verticalalignment="center",
    transform=ax.transAxes,
)

axin = inset_axes(ax, width="30%", height="20%", loc=1)
axin.plot(wzs, np.abs(mxs), c="b", label=r"$|m_x|$")
axin.set_xticklabels([])
axin.tick_params(axis="y", which="major", labelsize=12)
# axin.set_xlabel(r'$\lambda / \Omega$', labelpad=-10)
# axin.set_ylabel(r'$|m_x|$')
axin.text(
    0.95,
    0.25,
    r"$|m_x|$",
    c="b",
    fontsize=12,
    horizontalalignment="right",
    verticalalignment="center",
    transform=axin.transAxes,
)
axin.set_ylim(-0.1, 1.1)
axin.set_xticks(np.arange(0, wzs_upperlimit + 0.1, 0.5))
axin.set_xlim(0, wzs_upperlimit)


ax = axes[0, 1]

beta = 1
lam = 0.2

wzs_upperlimit = 2
ws_upperlimit = 2

wzs = np.linspace(0.0, wzs_upperlimit, 200)
ws = np.linspace(0, ws_upperlimit, 200)
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

vmin = np.amin(-Dm.imag)
vmax = np.amax(-Dm.imag)

cm = ax.pcolormesh(wzs, ws, -Dm.T.imag, cmap="BuPu", norm=mpl.colors.LogNorm())
# cax = ax.inset_axes([0.65, 0.1, 0.3, 0.03], transform=ax.transAxes)
# cbar = fig.colorbar(cm, cax=cax, ticks=[1e-2, 1e0, 1e2], orientation="horizontal")
# cbar.set_label(label=r"$-{\rm Im}D(\omega) \Omega$", fontsize=12)
# cbar.ax.xaxis.set_label_position("top")
# cax.tick_params(labelsize=12)

ax.set_ylim(0, ws_upperlimit)
ax.set_xlim(0, wzs_upperlimit)
# ax.set_xlabel(r'$\lambda / \Omega$')
ax.set_xticklabels([])
ax.set_yticklabels([])
# ax.set_ylabel(r"$\omega / \Omega$")
ax.set_xticks(np.arange(0, wzs_upperlimit + 0.1, 0.5))
ax.set_yticks(np.arange(0, ws_upperlimit + 0.1, 1.0))

ax.text(
    0.05,
    0.95,
    "(b)",
    fontsize=16,
    horizontalalignment="left",
    verticalalignment="top",
    transform=ax.transAxes,
)
ax.text(
    0.05,
    0.8,
    rf"$\lambda/\Omega = {lam}$",
    fontsize=12,
    horizontalalignment="left",
    verticalalignment="center",
    transform=ax.transAxes,
)
ax.text(
    0.05,
    0.725,
    rf"$\beta \Omega = {beta}$",
    fontsize=12,
    horizontalalignment="left",
    verticalalignment="center",
    transform=ax.transAxes,
)

axin = inset_axes(ax, width="30%", height="20%", loc=1)
axin.plot(wzs, np.abs(mxs), c="b", label=r"$|m_x|$")
axin.set_xticklabels([])
axin.tick_params(axis="y", which="major", labelsize=12)
# axin.set_xlabel(r'$\lambda / \Omega$', labelpad=-10)
# axin.set_ylabel(r'$|m_x|$')
axin.text(
    0.95,
    0.25,
    r"$|m_x|$",
    c="b",
    fontsize=12,
    horizontalalignment="right",
    verticalalignment="center",
    transform=axin.transAxes,
)
axin.set_ylim(-0.1, 1.1)
axin.set_xticks(np.arange(0, wzs_upperlimit + 0.1, 0.5))
axin.set_xlim(0, wzs_upperlimit)


ax = axes[1, 0]

beta = 10
lam = 0.02

wzs_upperlimit = 2
ws_upperlimit = 2

wzs = np.linspace(0.0, wzs_upperlimit, 200)
ws = np.linspace(0, ws_upperlimit, 200)
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

vmin = np.amin(-Dm.imag)
vmax = np.amax(-Dm.imag)

cm = ax.pcolormesh(wzs, ws, -Dm.T.imag, cmap="BuPu", norm=mpl.colors.LogNorm())
# cax = ax.inset_axes([0.65, 0.1, 0.3, 0.03], transform=ax.transAxes)
# cbar = fig.colorbar(cm, cax=cax, ticks=[1e-2, 1e0, 1e2], orientation="horizontal")
# cbar.set_label(label=r"$-{\rm Im}D(\omega) \Omega$", fontsize=12)
# cbar.ax.xaxis.set_label_position("top")
# cax.tick_params(labelsize=12)

ax.set_ylim(0, ws_upperlimit)
ax.set_xlim(0, wzs_upperlimit)
ax.set_xlabel(r"$\omega_z / \Omega$")
# ax.set_xticklabels([])
# ax.set_yticklabels([])
ax.set_ylabel(r"$\omega / \Omega$")
ax.set_xticks(np.arange(0, wzs_upperlimit + 0.1, 0.5))
ax.set_yticks(np.arange(0, ws_upperlimit + 0.1, 1.0))

ax.text(
    0.05,
    0.95,
    "(c)",
    fontsize=16,
    horizontalalignment="left",
    verticalalignment="top",
    transform=ax.transAxes,
)
ax.text(
    0.05,
    0.8,
    rf"$\lambda/\Omega = {lam}$",
    fontsize=12,
    horizontalalignment="left",
    verticalalignment="center",
    transform=ax.transAxes,
)
ax.text(
    0.05,
    0.725,
    rf"$\beta \Omega = {beta}$",
    fontsize=12,
    horizontalalignment="left",
    verticalalignment="center",
    transform=ax.transAxes,
)

axin = inset_axes(ax, width="30%", height="20%", loc=1)
axin.plot(wzs, np.abs(mxs), c="b", label=r"$|m_x|$")
axin.set_xticklabels([])
axin.tick_params(axis="y", which="major", labelsize=12)
# axin.set_xlabel(r'$\lambda / \Omega$', labelpad=-10)
# axin.set_ylabel(r'$|m_x|$')
axin.text(
    0.95,
    0.25,
    r"$|m_x|$",
    c="b",
    fontsize=12,
    horizontalalignment="right",
    verticalalignment="center",
    transform=axin.transAxes,
)
axin.set_ylim(-0.1, 1.1)
axin.set_xticks(np.arange(0, wzs_upperlimit + 0.1, 0.5))
axin.set_xlim(0, wzs_upperlimit)

ax = axes[1, 1]

beta = 10
lam = 0.2

wzs_upperlimit = 2
ws_upperlimit = 2

wzs = np.linspace(0.0, wzs_upperlimit, 200)
ws = np.linspace(0, ws_upperlimit, 200)
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

vmin = np.amin(-Dm.imag)
vmax = np.amax(-Dm.imag)

cm = ax.pcolormesh(wzs, ws, -Dm.T.imag, cmap="BuPu", norm=mpl.colors.LogNorm())
cax = ax.inset_axes([0.65, 0.1, 0.3, 0.03], transform=ax.transAxes)
cbar = fig.colorbar(cm, cax=cax, ticks=[1e-2, 1e0, 1e2], orientation="horizontal")
cbar.set_label(label=r"$-{\rm Im}D(\omega) \Omega$", fontsize=12)
cbar.ax.xaxis.set_label_position("top")
cax.tick_params(labelsize=12)

ax.set_ylim(0, ws_upperlimit)
ax.set_xlim(0, wzs_upperlimit)
ax.set_xlabel(r"$\omega_z / \Omega$")
# ax.set_xticklabels([])
ax.set_yticklabels([])
# ax.set_ylabel(r"$\omega / \Omega$")
ax.set_xticks(np.arange(0, wzs_upperlimit + 0.1, 0.5))
ax.set_yticks(np.arange(0, ws_upperlimit + 0.1, 1.0))

ax.text(
    0.05,
    0.95,
    "(d)",
    fontsize=16,
    horizontalalignment="left",
    verticalalignment="top",
    transform=ax.transAxes,
)
ax.text(
    0.05,
    0.8,
    rf"$\lambda/\Omega = {lam}$",
    fontsize=12,
    horizontalalignment="left",
    verticalalignment="center",
    transform=ax.transAxes,
)
ax.text(
    0.05,
    0.725,
    rf"$\beta \Omega = {beta}$",
    fontsize=12,
    horizontalalignment="left",
    verticalalignment="center",
    transform=ax.transAxes,
)

axin = inset_axes(ax, width="30%", height="20%", loc=1)
axin.plot(wzs, np.abs(mxs), c="b", label=r"$|m_x|$")
axin.set_xticklabels([])
axin.tick_params(axis="y", which="major", labelsize=12)
# axin.set_xlabel(r'$\lambda / \Omega$', labelpad=-10)
# axin.set_ylabel(r'$|m_x|$')
axin.text(
    0.95,
    0.25,
    r"$|m_x|$",
    c="b",
    fontsize=12,
    horizontalalignment="right",
    verticalalignment="center",
    transform=axin.transAxes,
)
axin.set_ylim(-0.1, 1.1)
axin.set_xticks(np.arange(0, wzs_upperlimit + 0.1, 0.5))
axin.set_xlim(0, wzs_upperlimit)


fig.savefig("plots/temp_transmission_LMG_thesis.jpeg", bbox_inches="tight", dpi=300)
