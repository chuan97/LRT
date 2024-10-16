import matplotlib.pyplot as plt
import numpy as np

import ising
import plot

plot.set_rcParams(size=(5.75, 4.5), lw=2, fs=16)

fig, axes = plt.subplots(1, 1, constrained_layout=True)
ax = axes

W = 1.0
J = 1.0
lam0s = np.linspace(0, 1, 200)
wxs = np.linspace(0, 3, 200)

mxs = np.empty((len(lam0s), len(wxs)))
mzs = np.empty((len(lam0s), len(wxs)))
for i, lam in enumerate(lam0s):
    for j, wx in enumerate(wxs):
        mx = ising.variational_mx(J, wx, W, lam)
        mxs[i, j] = mx
        mzs[i, j] = ising.mz_exact(J, wx - 4 * lam**2 * mx / W)

cm = ax.pcolormesh(lam0s**2, wxs, np.abs(mzs.T), cmap="Reds")
cbar = fig.colorbar(cm, aspect=40, label=r"$m_z$")

ax.set_xlabel(r"$\lambda^2 / (\Omega J)$")
ax.set_ylabel(r"$\omega_x / J$")

fig.savefig("plots/phase_diagram_ising_mzs.jpeg", bbox_inches="tight", dpi=300)
