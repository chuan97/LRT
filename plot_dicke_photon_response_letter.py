import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

import ising
import dicke
import plot
import green
import polaritons

plot.set_rcParams(size = (9.5, 4.5), lw = 2, fs = 16)
Ns = []
fig, axes = plt.subplots(1, 2, constrained_layout=True)

ax = axes[0]

W = 1
wz = 1
z = 0
eta = 0.01

ws_upperlimit = 2
lam_rightlimit = 1
lam0s = np.linspace(0, lam_rightlimit, 200)
ws = np.linspace(0, ws_upperlimit, 200)
idx = 10

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
# cax = ax.inset_axes([0.65, 0.1, 0.3, 0.03], transform=ax.transAxes)
# cbar = fig.colorbar(cm,
#                     cax=cax,
#                     ticks=[1e-2, 1e0, 1e2],
#                     orientation='horizontal'
#                     )
# cbar.set_label(label=r'$-{\rm Im}D(\omega) \Omega$',
#                fontsize=12
#                )
# cbar.ax.xaxis.set_label_position('top')
# cax.tick_params(labelsize=12)

ax.set_ylim(0, ws_upperlimit)
ax.set_xlim(0, lam_rightlimit)
ax.set_xlabel(r'$\lambda / \Omega$')
# ax.set_xticklabels([])
ax.set_ylabel(r'$\omega / \Omega$')
ax.set_xticks(np.arange(0, lam_rightlimit+0.1, 0.2))
ax.set_yticks(np.arange(0, ws_upperlimit+0.1, 1.0))
# ax.set_title(rf'$\omega_x/\Omega = {wx} \,,\; \omega_z/\Omega = {wz} \,,\; J / \Omega = {J / W}$',
#              fontsize=14)

ax.text(0.05,
        0.05,
        rf'$\zeta = {z}$',
        fontsize=12,
        horizontalalignment='left',
        verticalalignment='center',
        transform=ax.transAxes
        )
ax.text(0.05,
        0.85,
        '(a)',
        fontsize=16,
        horizontalalignment='left',
        verticalalignment='top',
        transform=ax.transAxes
        )
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

ax.legend(loc='upper left', fontsize=12, frameon=False)
# ---------- two oscillator polaritons -------------


axin = inset_axes(ax, width="30%", height="20%", loc=1)
axin.plot(lam0s, np.abs(mxs), c='b', label=r'$|m_x|$')
axin.set_xticklabels([])
axin.set_xticks(np.arange(0, lam_rightlimit+0.1, 0.2))
axin.tick_params(axis='y', which='major', labelsize=12)
axin.text(0.05,
          0.275,
          r'$|m_x|$',
          c='b',
          fontsize=12,
          horizontalalignment='left',
          verticalalignment='center',
          transform=axin.transAxes
          )
axin.set_ylim(-0.1, 1.1)
axin.set_xlim(0, lam_rightlimit)


ax = axes[1]

z = 1

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
cax = ax.inset_axes([0.65, 0.1, 0.3, 0.03], transform=ax.transAxes)
cbar = fig.colorbar(cm,
                    cax=cax,
                    ticks=[1e-2, 1e0, 1e2],
                    orientation='horizontal'
                    )
cbar.set_label(label=r'$-{\rm Im}D(\omega) \Omega$',
               fontsize=12
               )
cbar.ax.xaxis.set_label_position('top')
cax.tick_params(labelsize=12)
# ax.plot(lam0s, N*(data['e2'] - data['e1']), c='r', ls='--')
ax.set_ylim(0, ws_upperlimit)
ax.set_xlim(0, lam_rightlimit)
ax.set_xlabel(r'$\lambda / \Omega$')
# ax.set_xticklabels([])
# ax.set_ylabel(r'$\omega / \Omega$')
ax.set_yticklabels([])
ax.set_xticks(np.arange(0, lam_rightlimit+0.1, 0.2))
ax.set_yticks(np.arange(0, ws_upperlimit+0.1, 1.0))
# ax.set_title(rf'$\omega_x/\Omega = {wx} \,,\; \omega_z/\Omega = {wz} \,,\; J / \Omega = {J / W}$',
#              fontsize=14)

ax.text(0.05,
        0.05,
        rf'$\zeta = {z}$',
        fontsize=12,
        horizontalalignment='left',
        verticalalignment='center',
        transform=ax.transAxes
        )
ax.text(0.05,
        0.85,
        '(b)',
        fontsize=16,
        horizontalalignment='left',
        verticalalignment='top',
        transform=ax.transAxes
        )
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

ax.legend(fontsize=12, frameon=False, loc='upper left')
# ---------- two oscillator polaritons -------------

axin = inset_axes(ax, width="30%", height="20%", loc=1)
axin.plot(lam0s, np.abs(mxs), c='b', label=r'$|m_x|$')
axin.set_xticklabels([])
axin.set_xticks(np.arange(0, lam_rightlimit+0.1, 0.2))
axin.tick_params(axis='y', which='major', labelsize=12)
axin.text(0.05,
          0.275,
          r'$|m_x|$',
          c='b',
          fontsize=12,
          horizontalalignment='left',
          verticalalignment='center',
          transform=axin.transAxes
          )
axin.set_ylim(-0.1, 1.1)
axin.set_xlim(0, lam_rightlimit)

fig.savefig('plots/dicke_photon_response_letter.jpeg',
            bbox_inches='tight',
            dpi=300
            )