import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

import dicke
import ising
import plot
import green

plot.set_rcParams(size = (5.75, 9), lw = 2, fs = 16)

fig, axes = plt.subplots(2, 1, constrained_layout=True)
ax = axes[0]

W = 1
wx = 0.0
J = 0.5

lam0s = np.linspace(0.0001, 2.5, 200)
ws = np.linspace(0, 3, 200)
eta = 0.01

Dm = np.empty((len(lam0s), len(ws)), dtype=complex)
mxs = []
for i, lam in enumerate(lam0s):
    mx = 2*ising.variational_mx(2*J, wx, W, lam)
    mxs.append(mx)
    
    for j, w in enumerate(ws):
        chixx0 = ising.f_chixx0(w + 1j*eta, 2*J, wx + 2*lam**2*mx/W)
        Vind = dicke.f_Vind(w + 1j*eta, W, lam, 0)
        chixx = green.f_chixx(Vind, chixx0)
        
        Dm[i, j] = green.f_Dm(w + 1j*eta, W, lam, chixx)
        

vmin = np.amin(-Dm.imag[-Dm.imag > 1e-6])
vmax = np.amax(-Dm.imag[~np.isnan(-Dm.imag)])
print(vmin, vmax)
        
cm = ax.pcolormesh(lam0s,
                   ws,
                   -Dm.T.imag,
                   cmap='BuPu',
                   norm=mpl.colors.LogNorm(vmin=vmin, vmax=vmax)
                   )
# cbar = fig.colorbar(cm,
#                     pad = -0.0,
#                     aspect = 60,
#                     label=r'$-{\rm Im}D(\omega) \Omega$')

# ax.set_xlabel(r'$\lambda / \Omega$')
ax.set_ylabel(r'$\omega / \Omega$')
ax.set_title(rf'$\omega_x/\Omega = {wx} \,,\; J / \Omega = {J}$', fontsize=14)
ax.set_xticklabels([])

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

ax = axes[1]

W = 1
wx = 0.05
J = 0.5

lam0s = np.linspace(0.0001, 2.5, 200)
ws = np.linspace(0, 3, 200)
eta = 0.01

Dm = np.empty((len(lam0s), len(ws)), dtype=complex)
mxs = []
for i, lam in enumerate(lam0s):
    mx = 2*ising.variational_mx(2*J, wx, W, lam)
    mxs.append(mx)
    
    for j, w in enumerate(ws):
        chixx0 = ising.f_chixx0(w + 1j*eta, 2*J, wx + 2*lam**2*mx/W)
        Vind = dicke.f_Vind(w + 1j*eta, W, lam, 0)
        chixx = green.f_chixx(Vind, chixx0)
        
        Dm[i, j] = green.f_Dm(w + 1j*eta, W, lam, chixx)
        
cm = ax.pcolormesh(lam0s,
                   ws,
                   -Dm.T.imag,
                   cmap='BuPu',
                   norm=mpl.colors.LogNorm(vmin=vmin, vmax=vmax)
                   )
cbar = fig.colorbar(cm,
                    ax=axes.ravel().tolist(),
                    pad = 0.05,
                    aspect = 60,
                    label=r'$-{\rm Im}D(\omega) \Omega$')

ax.set_xlabel(r'$\lambda / \Omega$')
ax.set_ylabel(r'$\omega / \Omega$')
# ax.set_yticklabels([])
ax.set_title(rf'$\omega_x/\Omega = {wx} \,,\; J / \Omega = {J}$', fontsize=14)

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

fig.savefig('plots/ising_photon_response_letter.jpeg', bbox_inches='tight', dpi=300)