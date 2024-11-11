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
delta = 1e-1


cxx = np.empty_like(wcs)
cxy = np.empty_like(wcs)
cyy = np.empty_like(wcs)

for i, wc in enumerate(wcs):
    cxx[i] = qhe.sigma_xx(w0, wc, wp, delta)
    cxy[i] = qhe.sigma_xy(w0, wc, wp, delta)
    cyy[i] = qhe.sigma_yy(w0, wc, wp, delta)

ax.plot(wcs, cxx, label=r"$\sigma_{xx}$")
ax.plot(wcs, cxy - 1, label=r"$\sigma_{xy} - 1$")
ax.plot(wcs, cyy, label=r"$\sigma_{yy}$")

ax.set_xlabel(r"$\omega_{\rm c}$")
ax.legend()

ax = axes[1]

wcs = np.linspace(0, 2)
w0 = 1
wp = 0.3
delta = 1e-1


cxx = np.empty_like(wcs)
cxy = np.empty_like(wcs)
cyy = np.empty_like(wcs)

for i, wc in enumerate(wcs):
    cxx[i] = qhe.sigma_xx(w0, wc, wp, delta)
    cxy[i] = qhe.sigma_xy(w0, wc, wp, delta)
    cyy[i] = qhe.sigma_yy(w0, wc, wp, delta)

ax.plot(wcs, cxx, label=r"$\sigma_{xx}$")
ax.plot(wcs, cxy - 1, label=r"$\sigma_{xy} - 1$")
ax.plot(wcs, cyy, label=r"$\sigma_{yy}$")

ax.set_xlabel(r"$\omega_{\rm c}$")
ax.legend()

fig.savefig("plots/dc_conductivity.pdf", bbox_inches="tight", dpi=300)
