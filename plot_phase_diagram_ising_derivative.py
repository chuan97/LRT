import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

import ising
import plot

plot.set_rcParams(size=(5.75, 10), lw=2, fs=16)

fig, axes = plt.subplots(2, 1, constrained_layout=True)
ax = axes[0]

W = 1.0
J = 1.0
lam0s = np.linspace(0, 1, 200)
wxs = np.linspace(0.05, 2.5, 200)

mxs = np.empty((len(lam0s), len(wxs)))
mzs = np.empty((len(lam0s), len(wxs)))
for i, lam in enumerate(tqdm(lam0s)):
    for j, wx in enumerate(wxs):
        mx = ising.variational_mx(J, wx, W, lam)
        mxs[i, j] = mx
        mzs[i, j] = ising.mz_exact(J, wx - 4 * lam**2 * mx / W)

mzs_norm_diff = np.gradient(mxs, lam0s[1] - lam0s[0], wxs[1] - wxs[0])
cm = ax.pcolormesh(
    lam0s**2, wxs, np.abs(mzs_norm_diff[0].T), cmap="Reds", norm=mpl.colors.LogNorm()
)
cbar = fig.colorbar(cm, aspect=40, label=r"$\partial_\lambda m_x$")

ax.set_xlabel(r"$\lambda^2 / (\Omega J)$")
ax.set_ylabel(r"$\omega_x / J$")

ax = axes[1]

cm = ax.pcolormesh(
    lam0s**2, wxs, np.abs(mzs_norm_diff[1].T), cmap="Reds", norm=mpl.colors.LogNorm()
)
cbar = fig.colorbar(cm, aspect=40, label=r"$\partial_{\omega_x} m_x$")

ax.set_xlabel(r"$\lambda^2 / (\Omega J)$")
ax.set_ylabel(r"$\omega_x / J$")

fig.savefig(
    "plots/phase_diagram_ising_mxs_deriv_log.jpeg", bbox_inches="tight", dpi=300
)
