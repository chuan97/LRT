import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

import dicke
import ising
import plot
import green
import poles

plot.set_rcParams(size = (6, 5), lw = 2, fs = 14)

fig, axes = plt.subplots(1, 1, constrained_layout=True)
ax = axes

W = 1
J = 1
B = 0.0

lam0s = np.linspace(0.0001, 2.5, 100)
ws = np.linspace(0, 3, 100)
eta = 0.001

Dm = np.empty((len(lam0s), len(ws)), dtype=complex)
for i, lam in enumerate(lam0s):
    for j, w in enumerate(ws):
        mx = ising.variational_mx(J, B, W, lam)
        
        chixx0 = ising.f_chixx0(w + 1j*eta, J, B + 2*lam**2*mx/W)
        Vind = dicke.f_Vind(w + 1j*eta, W, lam, 0)
        
        Dm[i, j] = green.f_Dm(w + 1j*eta, W, lam, chixx0, Vind)
        
cm = ax.pcolormesh(lam0s,
                   ws,
                   -Dm.T.imag,
                   cmap='BuPu',
                   norm=mpl.colors.LogNorm()
                   )
cbar = fig.colorbar(cm, pad = 0.0, aspect = 40)
ax.axhline(W, c='k', ls=(0, (1, 5)), lw=1)
ax.axhline(2*J, c='k', ls=(0, (1, 5)), lw=1)

# # ---------- two oscillator polaritons -------------
# up_twoosc = []
# lp_twoosc = []
# for i, lam in enumerate(lam0s):
#     mx = ising.variational_mx(J, B, W, lam)
    
#     pm, pp = poles.polaritons(2*np.sqrt(J**2 + (B + 2*lam**2*mx/W)**2), W, 0.5*lam)
#     up_twoosc.append(pp)
#     lp_twoosc.append(pm)
    
# ax.plot(lam0s, up_twoosc, c='b', label="Poles Two Osc. model", lw=1)
# ax.plot(lam0s, lp_twoosc, c='b', lw=1)

# ax.legend()
# # ---------- two oscillator polaritons -------------

# ax.set_ylim(0, 3)
ax.set_xlabel(r'$\lambda / \Omega$')
ax.set_ylabel(r'$\omega / \Omega$')
ax.set_title(r'$-{\rm Im}D(\omega) \Omega$')

ax.text(1.75, 2.5, rf'$J = {J / W} \Omega$')
ax.text(1.75, 2.3, rf'$B = {round(B / W, 2)} \Omega$')


fig.savefig('plots/ising_photon_response.jpeg', bbox_inches='tight', dpi=300)