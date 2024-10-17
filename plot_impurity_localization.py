import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from scipy.integrate import quad
from tqdm import tqdm

import ising
import plot


def kernel(k, x, J, wx):
    return 2 * J * np.sin(k) / ising.f_ek(k, J, wx) * np.cos(k * x) / (2 * np.pi)


def kernel_im(k, x, J, wx):
    return 2 * J * np.sin(k) / ising.f_ek(k, J, wx) * np.sin(k * x) / (2 * np.pi)


plot.set_rcParams(size=(9.5, 4.5), lw=2, fs=16)

fig, axes = plt.subplots(1, 2, constrained_layout=True)

ax = axes[0]

J = 0.25
W = 1
wxs = [0, 0.05, 0.1, 0.2]
lam = 0.2

N = 150
xs = np.arange(0, N, 1) - N // 2

cmap = plt.get_cmap("RdPu")
colors = cmap(np.linspace(0.3, 1.0, len(wxs)))

for i, wx in enumerate(wxs):
    if lam == 0:
        mx = ising.mx_free(J, wx)
    else:
        mx = ising.variational_mx(J, wx, W, lam)

    cx_re = np.array(
        [
            quad(kernel, 0, 2 * np.pi, args=(x, J, wx - 4 * lam**2 * mx / W))[0]
            for x in xs
        ]
    )
    cx_im = np.array(
        [
            quad(kernel_im, 0, 2 * np.pi, args=(x, J, wx - 4 * lam**2 * mx / W))[0]
            for x in xs
        ]
    )

    cx = cx_re + cx_im

    ax.plot(xs / N, np.abs(cx), label=wx, c=colors[i])

# ax.axvline(1, lw=1, c="k", zorder=0)
# ax.axhline(0.5, lw=1, c="k", zorder=0)
# ax.legend(title=r"$\omega_x$", loc="upper right")
ax.set_ylabel(r"$|\eta_j|$")
ax.set_xlabel(r"$j / N$")
ax.set_yscale("log")
ax.set_ylim(1e-19, 1e1)

ax.text(
    0.05,
    0.85,
    rf"$\lambda/\Omega = {round(lam, 1)}$",
    fontsize=16,
    horizontalalignment="left",
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


ax = axes[1]

lam = 0.6

for i, wx in enumerate(wxs):
    if lam == 0:
        mx = ising.mx_free(J, wx)
    else:
        mx = ising.variational_mx(J, wx, W, lam)

    cx_re = np.array(
        [
            quad(kernel, 0, 2 * np.pi, args=(x, J, wx - 4 * lam**2 * mx / W))[0]
            for x in xs
        ]
    )
    cx_im = np.array(
        [
            quad(kernel_im, 0, 2 * np.pi, args=(x, J, wx - 4 * lam**2 * mx / W))[0]
            for x in xs
        ]
    )

    cx = cx_re + cx_im

    ax.plot(xs / N, np.abs(cx), label=wx, c=colors[i])

# ax.axvline(1, lw=1, c="k", zorder=0)
# ax.axhline(0.5, lw=1, c="k", zorder=0)
ax.legend(title=r"$\omega_x$", loc="upper right", frameon=False)
# ax.set_ylabel(r"$|\eta_j|$")
ax.set_xlabel(r"$j / N$")
ax.set_yscale("log")
ax.set_yticklabels([])
ax.set_ylim(1e-19, 1e1)

ax.text(
    0.05,
    0.85,
    rf"$\lambda/\Omega = {round(lam, 1)}$",
    fontsize=16,
    horizontalalignment="left",
    verticalalignment="top",
    transform=ax.transAxes,
)
ax.text(
    0.05,
    0.95,
    "(b)",
    fontsize=16,
    horizontalalignment="left",
    verticalalignment="top",
    transform=ax.transAxes,
)


fig.savefig("plots/impurity.pdf", bbox_inches="tight", dpi=300)
