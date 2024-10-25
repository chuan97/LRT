# %%
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import quad

import ising

# %%

W = 1
wx = 0.1
J = 0.25
lam = 0.5

mx = ising.variational_mx(J, wx, W, lam)

winf = 2 * np.abs((2 * J) - (wx - 4 * lam**2 * mx / W))
ws = np.linspace(0, winf, 400)
eta = 0.0

Is = np.array([ising.I(w + 1j * eta, J, wx - 4 * lam**2 * mx / W) for w in ws])
plt.plot(ws, -4 * lam**2 * W * Is, label=r"$4 \lambda^2 \Omega I(\omega)$")
plt.plot(ws, -(ws**2 - W**2), label=r"$\Omega^2 - \omega^2$")
plt.axvline(winf, c="k", zorder=0)
plt.legend()
plt.show()


# %%

W = 0.5
wx = 0.1
J = 0.25
lam = 0.2

mx = ising.variational_mx(J, wx, W, lam)

winf = 2 * np.abs((2 * J) - (wx - 4 * lam**2 * mx / W))
ws = np.linspace(0, winf, 1000)
eta = 0.0

Is = np.array([ising.I(w + 1j * eta, J, wx - 4 * lam**2 * mx / W) for w in ws])
plt.plot(ws, (ws**2 - W**2) - 4 * lam**2 * W * Is, label=r"$F(\omega)$")
plt.axvline(winf, c="k", zorder=0)
plt.axhline(0, c="k", zorder=0)
plt.legend()
plt.show()
# %%

W = 1
wx = 0.1
J = 0.25
lam = 0.2

mx = ising.variational_mx(J, wx, W, lam)

wsup = 2 * ((2 * J) + (wx - 4 * lam**2 * mx / W))
ws = np.linspace(wsup, 2, 1000)
eta = 0.0

Is = np.array([ising.I(w + 1j * eta, J, wx - 4 * lam**2 * mx / W) for w in ws])
plt.plot(ws, (ws**2 - W**2) - 4 * lam**2 * W * Is, label=r"$F(\omega)$")
plt.axvline(wsup, c="k", zorder=0)
plt.axhline(0, c="k", zorder=0)
plt.legend()
plt.show()

# %%
