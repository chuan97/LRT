import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

import transverse_LMG as tLMG
import plot

plot.set_rcParams(size = (10, 4.5), lw = 2, fs = 16)

W = 1.0
wz = 0.1
J = 0.25

lam0s = np.linspace(0, 1, 500)

mxs = [tLMG.variational_mx(wz, J, W, lam) for lam in lam0s]
plt.plot(lam0s, mxs)

plt.axhline(1, c='k', zorder=0, lw='1')

plt.ylabel(r'$m_x$')
plt.xlabel(r'\lambda / \Omega')

plt.show()


    