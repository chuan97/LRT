import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

import dicke
import plot
import green
import polaritons

plot.set_rcParams(size = (10, 4.5), lw = 2, fs = 16)

fig, axes = plt.subplots(1, 2, constrained_layout=True)
ax = axes[0]

W = 1
wz = 1
z = 0

lam0s = np.linspace(0.0001, 1, 200)
ws = np.linspace(0, 2, 200)
eta = 0.01

Dm = np.empty((len(lam0s), len(ws)), dtype=complex)
mxs = []
for i, lam in enumerate(lam0s):
    mx = dicke.f_mx(wz, z, W, lam)
    mxs.append(mx)
        
    for j, w in enumerate(ws):
        chixx0 = dicke.f_chixx0(w + 1j*eta, wz, 2*lam**2*mx/W)
        Vind = dicke.f_Vind(w + 1j*eta, W, lam, z)
        chixx = green.f_chixx(Vind, chixx0)
        
        Dm[i, j] = green.f_Dm(w + 1j*eta, W, lam, chixx)
        
vmin = np.amin(-Dm.imag)
vmax = np.amax(-Dm.imag)

cm = ax.pcolormesh(lam0s,
                   ws,
                   -Dm.T.imag,
                   cmap='BuPu',
                   norm=mpl.colors.LogNorm(vmin=vmin, vmax=vmax)
                   )
# cbar = fig.colorbar(cm, pad = 0.0, aspect = 40)
# ax.axhline(W, c='k', ls=(0, (1, 5)), lw=1)
# ax.axvline(np.sqrt(W*wz/4),  c='k', ls=(0, (1, 5)), lw=1)

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
    if z == 0:
        pm, pp = polaritons.dicke(wz, W, lam)
    elif z == 1:
        pm, pp = polaritons.dicke(np.sqrt(wz*(wz + 4*lam**2/(W*wz))),
                                W,
                                lam*(1 + 4*lam**2/(W*wz))**(-1/4)
                                )
        
    up_twoosc.append(pp)
    lp_twoosc.append(pm)
    
ax.plot(lam0s,
        up_twoosc,
        c='gold',
        label=r"$\Omega_\pm$ without $P^2$ term",
        ls='--'
        )
ax.plot(lam0s, lp_twoosc, c='gold', ls='--')

ax.legend(loc='lower left')
# ---------- two oscillator polaritons -------------

ax.set_ylim(0, 2)
ax.set_xlabel(r'$\lambda / \Omega$')
ax.set_ylabel(r'$\omega / \Omega$')
ax.set_title(rf'$\omega_z / \Omega = {wz} \,,\; \zeta = {z}$', fontsize=14)

ax = axes[1]

W = 1
wz = 1
z = 1

lam0s = np.linspace(0.0001, 1, 200)
ws = np.linspace(0, 2, 200)
eta = 0.01

Dm = np.empty((len(lam0s), len(ws)), dtype=complex)
mxs = []
for i, lam in enumerate(lam0s):
    mx = dicke.f_mx(wz, z, W, lam)
    mxs.append(mx)
        
    for j, w in enumerate(ws):
        chixx0 = dicke.f_chixx0(w + 1j*eta, wz, 2*lam**2*mx/W)
        Vind = dicke.f_Vind(w + 1j*eta, W, lam, z)
        chixx = green.f_chixx(Vind, chixx0)
        
        Dm[i, j] = green.f_Dm(w + 1j*eta, W, lam, chixx)
        
cm = ax.pcolormesh(lam0s,
                   ws,
                   -Dm.T.imag,
                   cmap='BuPu',
                   norm=mpl.colors.LogNorm(vmin=vmin, vmax=vmax)
                   )
cbar = fig.colorbar(cm,
                    pad = -0.0,
                    aspect = 40,
                    label=r'$-{\rm Im}D(\omega) \Omega$')
# ax.axhline(W, c='k', ls=(0, (1, 5)), lw=1)
# ax.axvline(np.sqrt(W*wz/4),  c='k', ls=(0, (1, 5)), lw=1)

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
    if z == 0:
        pm, pp = polaritons.dicke(wz, W, lam)
    elif z == 1:
        pm, pp = polaritons.dicke(np.sqrt(wz*(wz + 4*lam**2/W)),
                                W,
                                lam*(1 + 4*lam**2/(W*wz))**(-1/4)
                                )
        
    up_twoosc.append(pp)
    lp_twoosc.append(pm)
    
ax.plot(lam0s,
        up_twoosc,
        c='limegreen',
        label=r"$\Omega_\pm$ with $P^2$ term",
        ls='--'
        )
ax.plot(lam0s, lp_twoosc, c='limegreen', ls='--')

ax.legend()
# ---------- two oscillator polaritons -------------

ax.set_ylim(0, 2)
ax.set_xlabel(r'$\lambda / \Omega$')
# ax.set_ylabel(r'$\omega / \Omega$')
ax.set_yticklabels([])
ax.set_title(rf'$\omega_z / \Omega = {wz} \,,\; \zeta = {z}$', fontsize=14)

fig.savefig('plots/dicke_photon_response_paper.jpeg', bbox_inches='tight', dpi=300)