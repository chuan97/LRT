import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

import transverse_LMG as tLMG
import plot
import green

plot.set_rcParams(size=(10, 8.5), lw=2, fs=16)

fig, axes = plt.subplots(2, 2, constrained_layout=True)
ax = axes[0, 0]

W = 1.0
wz = 0.0
wx = 0.0
J = 0.25

lam0s = np.linspace(0.0001, 1, 200)
ws = np.linspace(0, 3, 200)
eta = 0.01

Dm = np.empty((len(lam0s), len(ws)), dtype=complex)
chizzs = np.empty((len(lam0s), len(ws)), dtype=complex)
mxs = []
mzs = []
for i, lam in enumerate(lam0s):
    mz = tLMG.variational_mz(wx, wz, J, W, lam)
    mx = -np.sqrt(1 - mz**2)
    mxs.append(mx)
    mzs.append(mz)

    wztilde = wz - 4 * J * mz
    h = wx / 2 - 2 * lam**2 * mx / W

    for j, w in enumerate(ws):
        chixx0 = tLMG.f_chixx0(w + 1j * eta, wztilde, h)
        chixz0 = chizx0 = tLMG.f_chixz0(w + 1j * eta, wztilde, h)
        chizz0 = tLMG.f_chizz0(w + 1j * eta, wztilde, h)
        Vindx = tLMG.f_Vindx(w + 1j * eta, W, lam)
        Vindz = tLMG.f_Vindz(w + 1j * eta, J)

        chixx = green.f_chixx_twomode(Vindx, Vindz, chixx0, chixz0, chizx0, chizz0)

        chizz = green.f_chizz_twomode(Vindx, Vindz, chixx0, chixz0, chizx0, chizz0)

        Dm[i, j] = green.f_Dm(w + 1j * eta, W, lam, chixx)
        chizzs[i, j] = chizz

vmin = np.amin(-Dm.imag)
vmax = np.amax(-Dm.imag)

cm = ax.pcolormesh(
    lam0s, ws, -Dm.T.imag, cmap="BuPu", norm=mpl.colors.LogNorm(vmin=vmin, vmax=vmax)
)

ax.set_ylim(0, 3)
# ax.set_xlabel(r'$\lambda / \Omega$')
ax.set_xticklabels([])
ax.set_ylabel(r"$\omega / \Omega$")
ax.set_title(
    rf"$\omega_x/\Omega = {wx} \,,\; \omega_z/\Omega = {wz} \,,\; J / \Omega = {J / W}$",
    fontsize=14,
)

axin = inset_axes(ax, width="30%", height="20%", loc=1)
axin.plot(lam0s, np.abs(mxs), c="b", label=r"$|m_x|$")
axin.plot(lam0s, np.abs(mzs), c="r", label=r"$|m_z|$")
axin.plot(lam0s, np.array(mxs) ** 2 + np.array(mzs) ** 2, c="g")
axin.set_xticklabels([])
axin.tick_params(axis="y", which="major", labelsize=12)
# axin.set_xlabel(r'$\lambda / \Omega$', labelpad=-10)
# axin.set_ylabel(r'$|m_x|$')
axin.text(
    0.05,
    0.75,
    r"$|m_x|$",
    fontsize=12,
    horizontalalignment="left",
    verticalalignment="center",
    transform=axin.transAxes,
)
axin.set_ylim(-0.1, 1.1)

vminz = np.amin(chizzs.imag[chizzs.imag > 1e-7])
vmaxz = np.amax(chizzs.imag)

axin2 = inset_axes(ax, width="30%", height="30%", loc=2)
cm2 = axin2.pcolormesh(
    lam0s,
    ws,
    chizzs.T.imag,
    cmap="OrRd",
    norm=mpl.colors.LogNorm(vmin=vminz, vmax=vmaxz),
)
axin2.text(
    0.5,
    0.8,
    r"${\rm Im}\chi_{zz}(\omega) \Omega$",
    fontsize=12,
    horizontalalignment="center",
    verticalalignment="center",
    transform=axin2.transAxes,
)
axin2.set_xticklabels([])
axin2.set_yticklabels([])


ax = axes[0, 1]

W = 1.0
wz = 0.0
wx = 0.2
J = 0.25

lam0s = np.linspace(0.0001, 1, 200)
ws = np.linspace(0, 3, 200)
eta = 0.01

Dm = np.empty((len(lam0s), len(ws)), dtype=complex)
chizzs = np.empty((len(lam0s), len(ws)), dtype=complex)
mxs = []
mzs = []
for i, lam in enumerate(lam0s):
    mz = tLMG.variational_mz(wx, wz, J, W, lam)
    mx = -np.sqrt(1 - mz**2)
    mxs.append(mx)
    mzs.append(mz)

    wztilde = wz - 4 * J * mz
    h = wx / 2 - 2 * lam**2 * mx / W

    for j, w in enumerate(ws):
        chixx0 = tLMG.f_chixx0(w + 1j * eta, wztilde, h)
        chixz0 = chizx0 = tLMG.f_chixz0(w + 1j * eta, wztilde, h)
        chizz0 = tLMG.f_chizz0(w + 1j * eta, wztilde, h)
        Vindx = tLMG.f_Vindx(w + 1j * eta, W, lam)
        Vindz = tLMG.f_Vindz(w + 1j * eta, J)

        chixx = green.f_chixx_twomode(Vindx, Vindz, chixx0, chixz0, chizx0, chizz0)

        chizz = green.f_chizz_twomode(Vindx, Vindz, chixx0, chixz0, chizx0, chizz0)

        Dm[i, j] = green.f_Dm(w + 1j * eta, W, lam, chixx)
        chizzs[i, j] = chizz

cm = ax.pcolormesh(
    lam0s, ws, -Dm.T.imag, cmap="BuPu", norm=mpl.colors.LogNorm(vmin=vmin, vmax=vmax)
)
# cbar = fig.colorbar(cm,
#                     pad = -0.0,
#                     aspect = 60,
#                     label=r'$-{\rm Im}D(\omega) \Omega$')

ax.set_ylim(0, 3)
# ax.set_xlabel(r'$\lambda / \Omega$')
ax.set_xticklabels([])
# ax.set_ylabel(r'$\omega / \Omega$')
ax.set_yticklabels([])
ax.set_title(
    rf"$\omega_x/\Omega = {wx} \,,\; \omega_z/\Omega = {wz} \,,\; J / \Omega = {J / W}$",
    fontsize=14,
)

axin = inset_axes(ax, width="30%", height="20%", loc=1)
axin.plot(lam0s, np.abs(mxs), c="b", label=r"$|m_x|$")
axin.plot(lam0s, np.abs(mzs), c="r", label=r"$|m_z|$")
axin.plot(lam0s, np.array(mxs) ** 2 + np.array(mzs) ** 2, c="g")
axin.set_xticklabels([])
axin.tick_params(axis="y", which="major", labelsize=12)
# axin.set_xlabel(r'$\lambda / \Omega$', labelpad=-10)
# axin.set_ylabel(r'$|m_x|$')
axin.text(
    0.05,
    0.75,
    r"$|m_x|$",
    fontsize=12,
    horizontalalignment="left",
    verticalalignment="center",
    transform=axin.transAxes,
)
axin.set_ylim(-0.1, 1.1)

axin2 = inset_axes(ax, width="30%", height="30%", loc=2)
cm2 = axin2.pcolormesh(
    lam0s,
    ws,
    chizzs.T.imag,
    cmap="OrRd",
    norm=mpl.colors.LogNorm(vmin=vminz, vmax=vmaxz),
)
axin2.text(
    0.5,
    0.8,
    r"${\rm Im}\chi_{zz}(\omega) \Omega$",
    fontsize=12,
    horizontalalignment="center",
    verticalalignment="center",
    transform=axin2.transAxes,
)
axin2.set_xticklabels([])
axin2.set_yticklabels([])

ax = axes[1, 0]

W = 1.0
wz = 0.2
wx = 0.0
J = 0.25

lam0s = np.linspace(0.0001, 1, 200)
ws = np.linspace(0, 3, 200)
eta = 0.01

Dm = np.empty((len(lam0s), len(ws)), dtype=complex)
chizzs = np.empty((len(lam0s), len(ws)), dtype=complex)
mxs = []
mzs = []
for i, lam in enumerate(lam0s):
    mz = tLMG.variational_mz(wx, wz, J, W, lam)
    mx = -np.sqrt(1 - mz**2)
    mxs.append(mx)
    mzs.append(mz)

    wztilde = wz - 4 * J * mz
    h = wx / 2 - 2 * lam**2 * mx / W

    for j, w in enumerate(ws):
        chixx0 = tLMG.f_chixx0(w + 1j * eta, wztilde, h)
        chixz0 = chizx0 = tLMG.f_chixz0(w + 1j * eta, wztilde, h)
        chizz0 = tLMG.f_chizz0(w + 1j * eta, wztilde, h)
        Vindx = tLMG.f_Vindx(w + 1j * eta, W, lam)
        Vindz = tLMG.f_Vindz(w + 1j * eta, J)

        chixx = green.f_chixx_twomode(Vindx, Vindz, chixx0, chixz0, chizx0, chizz0)

        chizz = green.f_chizz_twomode(Vindx, Vindz, chixx0, chixz0, chizx0, chizz0)

        Dm[i, j] = green.f_Dm(w + 1j * eta, W, lam, chixx)
        chizzs[i, j] = chizz

cm = ax.pcolormesh(
    lam0s, ws, -Dm.T.imag, cmap="BuPu", norm=mpl.colors.LogNorm(vmin=vmin, vmax=vmax)
)

ax.set_ylim(0, 3)
ax.set_xlabel(r"$\lambda / \Omega$")
ax.set_ylabel(r"$\omega / \Omega$")
ax.set_title(
    rf"$\omega_x/\Omega = {wx} \,,\; \omega_z/\Omega = {wz} \,,\; J / \Omega = {J / W}$",
    fontsize=14,
)

axin = inset_axes(ax, width="30%", height="20%", loc=1)
axin.plot(lam0s, np.abs(mxs), c="b", label=r"$|m_x|$")
axin.plot(lam0s, np.abs(mzs), c="r", label=r"$|m_z|$")
axin.plot(lam0s, np.array(mxs) ** 2 + np.array(mzs) ** 2, c="g")
axin.set_xticklabels([])
axin.tick_params(axis="y", which="major", labelsize=12)
# axin.set_xlabel(r'$\lambda / \Omega$', labelpad=-10)
# axin.set_ylabel(r'$|m_x|$')
axin.text(
    0.05,
    0.75,
    r"$|m_x|$",
    fontsize=12,
    horizontalalignment="left",
    verticalalignment="center",
    transform=axin.transAxes,
)
axin.set_ylim(-0.1, 1.1)

axin2 = inset_axes(ax, width="30%", height="30%", loc=2)
cm2 = axin2.pcolormesh(
    lam0s,
    ws,
    chizzs.T.imag,
    cmap="OrRd",
    norm=mpl.colors.LogNorm(vmin=vminz, vmax=vmaxz),
)
axin2.text(
    0.5,
    0.8,
    r"${\rm Im}\chi_{zz}(\omega) \Omega$",
    fontsize=12,
    horizontalalignment="center",
    verticalalignment="center",
    transform=axin2.transAxes,
)
axin2.set_xticklabels([])
axin2.set_yticklabels([])

ax = axes[1, 1]

W = 1.0
wz = 0.2
wx = 0.2
J = 0.25

lam0s = np.linspace(0.0001, 1, 200)
ws = np.linspace(0, 3, 200)
eta = 0.01

Dm = np.empty((len(lam0s), len(ws)), dtype=complex)
chizzs = np.empty((len(lam0s), len(ws)), dtype=complex)
mxs = []
mzs = []
for i, lam in enumerate(lam0s):
    mz = tLMG.variational_mz(wx, wz, J, W, lam)
    mx = -np.sqrt(1 - mz**2)
    mxs.append(mx)
    mzs.append(mz)

    wztilde = wz - 4 * J * mz
    h = wx / 2 - 2 * lam**2 * mx / W

    for j, w in enumerate(ws):
        chixx0 = tLMG.f_chixx0(w + 1j * eta, wztilde, h)
        chixz0 = chizx0 = tLMG.f_chixz0(w + 1j * eta, wztilde, h)
        chizz0 = tLMG.f_chizz0(w + 1j * eta, wztilde, h)
        Vindx = tLMG.f_Vindx(w + 1j * eta, W, lam)
        Vindz = tLMG.f_Vindz(w + 1j * eta, J)

        chixx = green.f_chixx_twomode(Vindx, Vindz, chixx0, chixz0, chizx0, chizz0)

        chizz = green.f_chizz_twomode(Vindx, Vindz, chixx0, chixz0, chizx0, chizz0)

        Dm[i, j] = green.f_Dm(w + 1j * eta, W, lam, chixx)
        chizzs[i, j] = chizz

cm = ax.pcolormesh(
    lam0s, ws, -Dm.T.imag, cmap="BuPu", norm=mpl.colors.LogNorm(vmin=vmin, vmax=vmax)
)
cbar = fig.colorbar(
    cm,
    ax=axes.ravel().tolist(),
    pad=0.01,
    aspect=60,
    label=r"$-{\rm Im}D(\omega) \Omega$",
)

ax.set_ylim(0, 3)
ax.set_xlabel(r"$\lambda / \Omega$")
# ax.set_ylabel(r'$\omega / \Omega$')
ax.set_yticklabels([])
ax.set_title(
    rf"$\omega_x/\Omega = {wx} \,,\; \omega_z/\Omega = {wz} \,,\; J / \Omega = {J / W}$",
    fontsize=14,
)

axin = inset_axes(ax, width="30%", height="20%", loc=1)
axin.plot(lam0s, np.abs(mxs), c="b", label=r"$|m_x|$")
axin.plot(lam0s, np.abs(mzs), c="r", label=r"$|m_z|$")
axin.plot(lam0s, np.array(mxs) ** 2 + np.array(mzs) ** 2, c="g")
axin.set_xticklabels([])
axin.tick_params(axis="y", which="major", labelsize=12)
# axin.set_xlabel(r'$\lambda / \Omega$', labelpad=-10)
# axin.set_ylabel(r'$|m_x|$')
axin.text(
    0.05,
    0.75,
    r"$|m_x|$",
    fontsize=12,
    horizontalalignment="left",
    verticalalignment="center",
    transform=axin.transAxes,
)
axin.set_ylim(-0.1, 1.1)

axin2 = inset_axes(ax, width="30%", height="30%", loc=2)
cm2 = axin2.pcolormesh(
    lam0s,
    ws,
    chizzs.T.imag,
    cmap="OrRd",
    norm=mpl.colors.LogNorm(vmin=vminz, vmax=vmaxz),
)
axin2.text(
    0.5,
    0.8,
    r"${\rm Im}\chi_{zz}(\omega) \Omega$",
    fontsize=12,
    horizontalalignment="center",
    verticalalignment="center",
    transform=axin2.transAxes,
)
axin2.set_xticklabels([])
axin2.set_yticklabels([])

fig.savefig(
    "plots/alt_transverse_LMG_photon_response.jpeg", bbox_inches="tight", dpi=300
)
