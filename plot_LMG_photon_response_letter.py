import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

import LMG
import dicke
import plot
import green
import polaritons

plot.set_rcParams(size = (5.75, 4.5), lw = 2, fs = 16)

fig, axes = plt.subplots(1, 1, constrained_layout=True)
ax = axes

W = 1.0
wz = 1
J = 0.15

lam0s = np.linspace(0.0001, 1, 200)
ws = np.linspace(0, 2, 200)
eta = 0.01

Dm = np.empty((len(lam0s), len(ws)), dtype=complex)
mxs = []
for i, lam in enumerate(lam0s):
    mx = LMG.f_mx(wz, J, W, lam)
    mxs.append(mx)
    for j, w in enumerate(ws):
        chixx0 = dicke.f_chixx0(w + 1j*eta, wz, 2*(lam**2/W + J)*mx)
        Vind = LMG.f_Vind(w + 1j*eta, W, lam, J)
        chixx = green.f_chixx(Vind, chixx0)
        
        Dm[i, j] = green.f_Dm(w + 1j*eta, W, lam, chixx)
        
cm = ax.pcolormesh(lam0s,
                   ws,
                   -Dm.T.imag,
                   cmap='BuPu',
                   norm=mpl.colors.LogNorm()
                   )

cbar = fig.colorbar(cm,
                    pad = -0.0,
                    aspect = 40,
                    label=r'$-{\rm Im}D(\omega) \Omega$')

axin = inset_axes(ax, width="30%", height="20%", loc=1)
axin.plot(lam0s, np.abs(mxs), c='b', label=r'$|m_x|$')
axin.set_xticklabels([])
axin.tick_params(axis='y', which='major', labelsize=12)
# axin.set_xlabel(r'$\lambda / \Omega$', labelpad=-10)
# axin.set_ylabel(r'$|m_x|$')
axin.text(0.05,
          0.75,
          r'$|m_x|$',
          fontsize=12,
          horizontalalignment='left',
          verticalalignment='center',
          transform=axin.transAxes
          )
axin.set_ylim(-0.1, 1.1)

# ---------- two oscillator polaritons -------------
up_twoosc = []
lp_twoosc = []
for i, lam in enumerate(lam0s):
    pm, pp = polaritons.LMG(wz, W, lam, J)
    
    up_twoosc.append(pp)
    lp_twoosc.append(pm)
    
ax.plot(lam0s, up_twoosc, c='gold', label="LMG polaritons", ls='--')
ax.plot(lam0s, lp_twoosc, c='gold', ls='--')

# ax.legend()
# ---------- two oscillator polaritons -------------

ax.set_ylim(0, 2)
ax.set_xlabel(r'$\lambda / \Omega$')
ax.set_ylabel(r'$\omega / \Omega$')
ax.set_title(rf'$\omega_z/\Omega = {wz} \,,\; J / \Omega = {J / W}$', fontsize=14)


fig.savefig('plots/LMG_photon_response_letter.jpeg', bbox_inches='tight', dpi=300)