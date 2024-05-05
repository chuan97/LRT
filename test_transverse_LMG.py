import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

import transverse_LMG as tLMG
import plot

plot.set_rcParams(size = (10, 4.5), lw = 2, fs = 16)

W = 1.0
wz = 0.01
wx = 0.01
J = 0.25

lam0s = np.linspace(0, 1, 500)

mxs=[]
mzs=[]
for lam in lam0s:
    mx, mz = tLMG.variational_m(wx, wz, J, W, lam)
    mxs.append(mx)
    mzs.append(mz)
mxs = np.array(mxs)
mzs = np.array(mzs)

plt.plot(lam0s, np.abs(mzs), c='r')
plt.plot(lam0s, np.abs(mxs), c='b')
plt.plot(lam0s, mxs**2 + mzs**2, c='g')

mxs = np.array([tLMG.variational_mx(wx, wz, J, W, lam) for lam in lam0s])
mzs = -np.sqrt(1-mxs**2)
plt.plot(lam0s, np.abs(mzs), c='tab:red', ls='--')
plt.plot(lam0s, np.abs(mxs), c='tab:blue', ls='--')
plt.plot(lam0s, mxs**2 + mzs**2, c='tab:green', ls='--')
plt.axhline(1, c='k', zorder=0, lw='1')

plt.xlabel(r'$\lambda / \Omega$')

plt.show()

