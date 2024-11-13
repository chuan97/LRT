import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

import plot
import qhe

plot.set_rcParams(size=(9.5, 4), lw=2, fs=16)

fig, axes = plt.subplots(1, 2, constrained_layout=True)
ax = axes[0]

wc = 1.0
w0 = 1.0
wps = np.linspace(0, 1, 100)
delta = 1e-2

cxx = np.empty_like(wps)
cxy = np.empty_like(wps)
cyy = np.empty_like(wps)

for i, wp in enumerate(wps):
    cxy[i] = qhe.sigma_xy(w0, wc, wp, delta)

ax.axhline(0, c="k", ls=":")
ax.plot(wps / w0, cxy - 1, label=w0)
ax.annotate("", xy=(0, 0), xytext=(0, -1e-4), arrowprops=dict(arrowstyle="<->"))
ax.text(0.02, -0.5e-4, r"$\delta^2/\omega_{\rm c}^2$")

ax.ticklabel_format(
    axis="y", style="sci", scilimits=(0, 0), useOffset=True, useMathText=True
)
ax.set_ylabel(r"$\sigma_{xy} / \left(\frac{e^2 h}{\nu}\right) - 1$")
ax.set_xlabel(r"$\omega_{\rm p} / \omega_0$")

ax.text(
    0.05,
    0.05,
    "(a)",
    fontsize=16,
    horizontalalignment="left",
    verticalalignment="bottom",
    transform=ax.transAxes,
)

ax = axes[1]

delta = 0

rxx = np.array([qhe.rho_xx(w0, wc, wp, delta) for wp in wps])

ax.plot(wps, rxx, label=r"$\rho_{xx}$", c="tab:orange")
ax.axhline(1, c="k", ls=":", zorder=0)

ax.set_ylabel(r"$\rho_{xx} \sigma_{\rm D}$")
ax.set_xlabel(r"$\omega_{\rm p} / \omega_0$")

ax.text(
    0.05,
    0.95,
    "(b)",
    fontsize=16,
    horizontalalignment="left",
    verticalalignment="top",
    transform=ax.transAxes,
)


fig.savefig("plots/dc_conductivity.pdf", bbox_inches="tight", dpi=300)
