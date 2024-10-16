# %%
import matplotlib.pyplot as plt
import numpy as np

import ising
import plot

plot.set_rcParams(size=(5.75, 4.5), lw=2, fs=16)

# %%

fig, axes = plt.subplots(1, 1, constrained_layout=True)
ax = axes

W = 1.0
J = 1.0
lam = 0.7
wxs = np.linspace(0, 3, 200)

mxs = np.empty(len(wxs))
mzs = np.empty(len(wxs))
for j, wx in enumerate(wxs):
    if lam == 0:
        mx = ising.mx_free(J, wx)
    else:
        mx = ising.variational_mx(J, wx, W, lam)

    mxs[j] = mx
    mzs[j] = ising.mz_exact(J, wx - 4 * lam**2 * mx / W)

plt.plot(wxs, np.abs(mxs), c="b", label=r"$m_x$")
plt.plot(wxs, np.abs(mzs), c="r", label=r"$m_z$")

ax.set_xlabel(r"$\omega_x / J$")
plt.legend(frameon=False)
plt.title(rf"$\lambda^2/(\Omega J) = {lam**2}$")

plt.show()

# %%

fig, axes = plt.subplots(1, 1, constrained_layout=True)
ax = axes

W = 1.0
J = 1.0
wx = 1.75
lam0s = np.linspace(0.2, 1, 200)

mxs = np.empty(len(wxs))
mzs = np.empty(len(wxs))
for j, lam in enumerate(lam0s):
    if lam < 0.05:
        mx = ising.mx_free(J, wx)
    else:
        mx = ising.variational_mx(J, wx, W, lam)

    mxs[j] = mx
    mzs[j] = ising.mz_exact(J, wx - 4 * lam**2 * mx / W)

plt.plot(lam0s, np.abs(mxs), c="b", label=r"$m_x$")
plt.plot(lam0s, np.abs(mzs), c="r", label=r"$m_z$")

fig, axes = plt.subplots(1, 1, constrained_layout=True)
ax = axes

mxs_der = np.gradient(mxs, np.diff(lam0s)[0])
mzs_der = np.gradient(mzs, np.diff(lam0s)[0])
plt.plot(lam0s, np.abs(mxs_der), c="b", label=r"$m_x$")
# plt.plot(lam0s, np.abs(mzs_der), c="r", label=r"$m_z$")

ax.set_xlabel(r"$\lambda^2/(\Omega J)$")
plt.legend(frameon=False)
plt.title(rf"$\omega_x / J = {wx}$")

plt.show()


# %%
