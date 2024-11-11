import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

import plot
import qhe

plot.set_rcParams(size=(9.5, 5), lw=2, fs=16)

fig, axes = plt.subplots(1, 2, constrained_layout=True)
ax = axes[0]

w0s = np.linspace(0, 2)
wc = 2
wp = 0.3
delta = 3e-2


cxx = np.empty_like(w0s)
cxy = np.empty_like(w0s)
cyy = np.empty_like(w0s)

for i, w0 in enumerate(w0s):
    cxx[i] = qhe.sigma_xx(w0, wc, wp, delta)
    cxy[i] = qhe.sigma_xy(w0, wc, wp, delta)
    cyy[i] = qhe.sigma_yy(w0, wc, wp, delta)

ax.plot(w0s, cxx, label=r"$\sigma_{xx}$")
ax.plot(w0s, cxy - 1, label=r"$\sigma_{xy} - 1$")
ax.plot(w0s, cyy, label=r"$\sigma_{yy}$")

ax.set_xlabel(r"$\omega_{0}$")
ax.legend()

ax = axes[1]

w0s = np.linspace(0, 2)
wc = 4
wp = 0.3
delta = 3e-2


cxx = np.empty_like(w0s)
cxy = np.empty_like(w0s)
cyy = np.empty_like(w0s)

for i, w0 in enumerate(w0s):
    cxx[i] = qhe.sigma_xx(w0, wc, wp, delta)
    cxy[i] = qhe.sigma_xy(w0, wc, wp, delta)
    cyy[i] = qhe.sigma_yy(w0, wc, wp, delta)

ax.plot(w0s, cxx, label=r"$\sigma_{xx}$")
ax.plot(w0s, cxy - 1, label=r"$\sigma_{xy} - 1$")
ax.plot(w0s, cyy, label=r"$\sigma_{yy}$")

ax.set_xlabel(r"$\omega_{0}$")
ax.legend()

fig.savefig("plots/dc_conductivity_alt.pdf", bbox_inches="tight", dpi=300)
