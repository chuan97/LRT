import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

import dicke
import ising
import plot
import green

plot.set_rcParams(size = (10, 9), lw = 2, fs = 16)

fig, axes = plt.subplots(2, 2, constrained_layout=True)
ax = axes[0, 0]

W = 1
wx = 0.0
J = 0.5

lam0s = np.linspace(0.0001, 2.5, 100)
ws = np.linspace(0, 3, 100)
eta = 0.01

Dm = np.empty((len(lam0s), len(ws)), dtype=complex)
for i, lam in enumerate(lam0s):
    for j, w in enumerate(ws):
        mx = ising.variational_mx(2*J, wx, W, lam)
        
        chixx0 = ising.f_chixx0(w + 1j*eta, 2*J, wx + 2*lam**2*mx/W)
        Vind = dicke.f_Vind(w + 1j*eta, W, lam, 0)
        chixx = green.f_chixx(Vind, chixx0)
        
        Dm[i, j] = green.f_Dm(w + 1j*eta, W, lam, chixx)
        
cm = ax.pcolormesh(lam0s,
                   ws,
                   -Dm.T.imag,
                   cmap='BuPu',
                   norm=mpl.colors.LogNorm()
                   )
# cbar = fig.colorbar(cm,
#                     pad = -0.0,
#                     aspect = 60,
#                     label=r'$-{\rm Im}D(\omega) \Omega$')

# ax.set_xlabel(r'$\lambda / \Omega$')
ax.set_ylabel(r'$\omega / \Omega$')
ax.set_title(rf'$\omega_x/\Omega = {wx} \,,\; J / \Omega = {J}$')



ax = axes[0, 1]

W = 1
wx = 0.075
J = 0.5

lam0s = np.linspace(0.0001, 2.5, 100)
ws = np.linspace(0, 3, 100)
eta = 0.01

Dm = np.empty((len(lam0s), len(ws)), dtype=complex)
for i, lam in enumerate(lam0s):
    for j, w in enumerate(ws):
        mx = ising.variational_mx(2*J, wx, W, lam)
        
        chixx0 = ising.f_chixx0(w + 1j*eta, 2*J, wx + 2*lam**2*mx/W)
        Vind = dicke.f_Vind(w + 1j*eta, W, lam, 0)
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

# ax.set_xlabel(r'$\lambda / \Omega$')
# ax.set_ylabel(r'$\omega / \Omega$')
ax.set_yticklabels([])
ax.set_title(rf'$\omega_x/\Omega = {wx} \,,\; J / \Omega = {J}$')



ax = axes[1, 0]

W = 1.001
wx = 0.0
J = 0.25

lam0s = np.linspace(0.0001, 2, 100)
ws = np.linspace(0, 2, 100)
eta = 0.01

Dm = np.empty((len(lam0s), len(ws)), dtype=complex)
for i, lam in enumerate(lam0s):
    for j, w in enumerate(ws):
        mx = ising.variational_mx(2*J, wx, W, lam)
        
        chixx0 = ising.f_chixx0(w + 1j*eta, 2*J, wx + 2*lam**2*mx/W)
        Vind = dicke.f_Vind(w + 1j*eta, W, lam, 0)
        chixx = green.f_chixx(Vind, chixx0)
        
        Dm[i, j] = green.f_Dm(w + 1j*eta, W, lam, chixx)
        
cm = ax.pcolormesh(lam0s,
                   ws,
                   -Dm.T.imag,
                   cmap='BuPu',
                   norm=mpl.colors.LogNorm()
                   )
# cbar = fig.colorbar(cm,
#                     pad = -0.0,
#                     aspect = 60,
#                     label=r'$-{\rm Im}D(\omega) \Omega$')

ax.set_xlabel(r'$\lambda / \Omega$')
ax.set_ylabel(r'$\omega / \Omega$')
ax.set_title(rf'$\omega_x/\Omega = {wx} \,,\; J / \Omega = {J}$')



ax = axes[1, 1]

W = 1.001
wx = 0.075
J = 0.25

lam0s = np.linspace(0.0001, 2, 100)
ws = np.linspace(0, 2, 100)
eta = 0.01

Dm = np.empty((len(lam0s), len(ws)), dtype=complex)
for i, lam in enumerate(lam0s):
    for j, w in enumerate(ws):
        mx = ising.variational_mx(2*J, wx, W, lam)
        
        chixx0 = ising.f_chixx0(w + 1j*eta, 2*J, wx + 2*lam**2*mx/W)
        Vind = dicke.f_Vind(w + 1j*eta, W, lam, 0)
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

ax.set_xlabel(r'$\lambda / \Omega$')
# ax.set_ylabel(r'$\omega / \Omega$')
ax.set_yticklabels([])
ax.set_title(rf'$\omega_x/\Omega = {wx} \,,\; J / \Omega = {J}$')

fig.savefig('plots/ising_photon_response_paper.jpeg', bbox_inches='tight', dpi=300)