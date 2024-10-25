import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from tqdm import tqdm

import ising
import plot

plot.set_rcParams(size=(9.5, 4.5), lw=2, fs=16)

fig, axes = plt.subplots(1, 2, constrained_layout=True)

W = 1.0
J = 1.0
lams_upperlim = 1
wxs_upperlim = 2.1
lam0s = np.linspace(0, lams_upperlim, 400)
wxs = np.linspace(0, wxs_upperlim, 400)

mxs = np.empty((len(lam0s), len(wxs)))
mzs = np.empty((len(lam0s), len(wxs)))
for i, lam in enumerate(tqdm(lam0s)):
    for j, wx in enumerate(wxs):
        mx = ising.variational_mx(J, wx, W, lam)
        mxs[i, j] = mx
        mzs[i, j] = ising.mz_exact(J, wx - 4 * lam**2 * mx / W)

ax = axes[0]

cm = ax.pcolormesh(lam0s**2, wxs, np.abs(mzs.T), cmap="Reds", vmin=0, vmax=1)
cax = ax.inset_axes([0.675, 0.85, 0.25, 0.03], transform=ax.transAxes)
cbar = fig.colorbar(cm, cax=cax, orientation="horizontal")
cbar.set_label(label=r"$|m_z|$", labelpad=8)
cbar.ax.xaxis.set_label_position("top")
# cax.tick_params(labelsize=12)

ax.set_xlabel(r"$\lambda^2 / (\Omega J)$")
ax.set_ylabel(r"$\omega_x / J$")

ax.set_xlim(0, lams_upperlim)
ax.set_ylim(0, wxs_upperlim)

ax.text(
    0.05,
    0.4,
    r"$z$FMN",
    c="w",
    fontsize=16,
    horizontalalignment="left",
    verticalalignment="top",
    transform=ax.transAxes,
)
ax.text(
    0.95,
    0.4,
    r"$x$FMS",
    c="k",
    fontsize=16,
    horizontalalignment="right",
    verticalalignment="top",
    transform=ax.transAxes,
)
ax.text(
    0.05,
    0.95,
    "(a)",
    fontsize=16,
    horizontalalignment="left",
    verticalalignment="top",
    transform=ax.transAxes,
)

axin = axes[1]

cmin = axin.pcolormesh(lam0s**2, wxs, np.abs(mxs.T), cmap="Blues", vmin=0, vmax=1)
caxin = axin.inset_axes([0.075, 0.11, 0.25, 0.03], transform=axin.transAxes)
cbarin = fig.colorbar(cmin, cax=caxin, orientation="horizontal")
cbarin.set_label(label=r"$|m_x|$", labelpad=8)
cbarin.ax.xaxis.set_label_position("top")
# cax.tick_params(labelsize=12)

axin.set_yticklabels([])
axin.set_xlabel(r"$\lambda^2 / (\Omega J)$")

axin.set_xlim(0, lams_upperlim)
axin.set_ylim(0, wxs_upperlim)


axin.text(
    0.05,
    0.4,
    r"$z$FMN",
    c="k",
    fontsize=16,
    horizontalalignment="left",
    verticalalignment="top",
    transform=axin.transAxes,
)
axin.text(
    0.95,
    0.4,
    r"$x$FMS",
    c="w",
    fontsize=16,
    horizontalalignment="right",
    verticalalignment="top",
    transform=axin.transAxes,
)
axin.text(
    0.05,
    0.95,
    "(b)",
    c="w",
    fontsize=16,
    horizontalalignment="left",
    verticalalignment="top",
    transform=axin.transAxes,
)

fig.savefig("plots/phase_diagram_ising_thesis.jpeg", bbox_inches="tight", dpi=300)
