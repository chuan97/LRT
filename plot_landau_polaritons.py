import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

import plot
import qhe

plot.set_rcParams(size=(9.5, 5), lw=2, fs=16)
fig, axes = plt.subplots(1, 2, constrained_layout=True)
ax = axes[0]

wcs = np.linspace(0, 2)
w0 = 1
wp = 0.1

up = np.empty_like(wcs)
lp = np.empty_like(wcs)
for i, wc in enumerate(wcs):
    lp[i], up[i] = qhe.landau_polaritons(w0, wc, wp)

ax.plot(wcs, up, c="k")
ax.plot(wcs, lp, c="k")
ax.plot(wcs, wcs, ls=":")
ax.axhline(w0, ls=":")

ax = axes[1]

wcs = np.linspace(0, 2)
w0 = 1
wp = 0.3

ws_upperlimit = 2

up = np.empty_like(wcs)
lp = np.empty_like(wcs)
for i, wc in enumerate(wcs):
    lp[i], up[i] = qhe.landau_polaritons(w0, wc, wp)

ax.plot(wcs, up, c="k")
ax.plot(wcs, lp, c="k")
ax.plot(wcs, wcs, ls=":")
ax.axhline(w0, ls=":")

fig.savefig("plots/landau_polaritons.pdf", bbox_inches="tight", dpi=300)
