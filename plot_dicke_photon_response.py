import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

import dicke
import plot
import green
import poles

plot.set_rcParams(size = (6, 5), lw = 2, fs = 14)

fig, axes = plt.subplots(1, 1, constrained_layout=True)
ax = axes

W = 1
wz = 1
z = 1

lam0s = np.linspace(0.0001, 1, 200)
ws = np.linspace(0, 2, 200)
eta = 0.001

Dm = np.empty((len(lam0s), len(ws)), dtype=complex)
for i, lam in enumerate(lam0s):
    for j, w in enumerate(ws):
        mx = dicke.f_mx(wz, z, W, lam)
        
        chixx0 = dicke.f_chixx0(w + 1j*eta, wz, 2*lam**2*mx/W)
        Vind = dicke.f_Vind(w + 1j*eta, W, lam, z)
        
        Dm[i, j] = green.f_Dm(w + 1j*eta, W, lam, chixx0, Vind)
        
cm = ax.pcolormesh(lam0s,
                   ws,
                   -Dm.T.imag,
                   cmap='BuPu',
                   norm=mpl.colors.LogNorm()
                   )
cbar = fig.colorbar(cm, pad = 0.0, aspect = 40)
ax.axhline(W, c='k', ls=(0, (1, 5)), lw=1)
ax.axvline(np.sqrt(W*wz/4),  c='k', ls=(0, (1, 5)), lw=1)

# ---------- two oscillator polaritons -------------
up_twoosc = []
lp_twoosc = []
for i, lam in enumerate(lam0s):
    if z == 0:
        pm, pp = poles.polaritons(wz, W, lam)
    elif z == 1:
        pm, pp = poles.polaritons(np.sqrt(wz*(wz + 4*lam**2/(W*wz))),
                                W,
                                lam*(1 + 4*lam**2/(W*wz))**(-1/4)
                                )
        
    up_twoosc.append(pp)
    lp_twoosc.append(pm)
    
ax.plot(lam0s, up_twoosc, c='b', label="Dicke polaritons with $P^2$ term", lw=1)
ax.plot(lam0s, lp_twoosc, c='b', lw=1)

ax.legend()
# ---------- two oscillator polaritons -------------

ax.set_ylim(0, 2)
ax.set_xlabel(r'$\lambda / \Omega$')
ax.set_ylabel(r'$\omega / \Omega$')
ax.set_title(r'$-{\rm Im}D(\omega) \Omega$')

fig.savefig('plots/dicke_p2_photon_response.jpeg', bbox_inches='tight', dpi=300)