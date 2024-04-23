import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

import LMG
import dicke
import plot
import green
import polaritons

plot.set_rcParams(size = (10, 4.5), lw = 2, fs = 16)

fig, axes = plt.subplots(1, 2, constrained_layout=True)
ax = axes[0]

W = 1.0
wz = 1
J = 0.15

lam0s = np.linspace(0.0001, 1, 200)
ws = np.linspace(0, 2, 200)
eta = 0.01

Dm = np.empty((len(lam0s), len(ws)), dtype=complex)
for i, lam in enumerate(lam0s):
    for j, w in enumerate(ws):
        mx = LMG.f_mx(wz, J, W, lam)
        
        chixx0 = dicke.f_chixx0(w + 1j*eta, wz, 2*(lam**2/W + J)*mx)
        Vind = LMG.f_Vind(w + 1j*eta, W, lam, J)
        
        Dm[i, j] = green.f_Dm(w + 1j*eta, W, lam, chixx0, Vind)
        
cm = ax.pcolormesh(lam0s,
                   ws,
                   -Dm.T.imag,
                   cmap='BuPu',
                   norm=mpl.colors.LogNorm()
                   )
# cbar = fig.colorbar(cm,
#                     pad = 0.0,
#                     aspect = 40,
#                     label=r'$-{\rm Im}D(\omega) \Omega$'
#                     )
# ax.axhline(W, c='k', ls=(0, (1, 5)), lw=1)
# ax.axhline(np.sqrt(wz*(wz - 4*J)), c='k', ls=(0, (1, 5)), lw=1)
# ax.axvline(np.sqrt(W*(wz/4 - J)),  c='k', ls=(0, (1, 5)), lw=1)

# # ---------- two oscillator polaritons -------------
# up_twoosc = []
# lp_twoosc = []
# for i, lam in enumerate(lam0s):
#     if wz > 4*(lam**2/W + J):
#         pm, pp = poles.polaritons(np.sqrt(wz*(wz - 4*J)),
#                                 W,
#                                 lam*(1 - 4*J/wz)**(-1/4)
#                                 )
#     else:
#         pm, pp = None, None
#     up_twoosc.append(pp)
#     lp_twoosc.append(pm)
    
# ax.plot(lam0s, up_twoosc, c='b', label="Poles Two Osc. model", lw=1)
# ax.plot(lam0s, lp_twoosc, c='b', lw=1)

# ax.legend()
# # ---------- two oscillator polaritons -------------

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
ax.set_title(rf'$\omega_z/\Omega = {wz} \,,\; J / \Omega = {J / W}$')

ax = axes[1]

W = 1.0
wz = 1
J = 0.3

lam0s = np.linspace(0.0001, 1, 200)
ws = np.linspace(0, 2, 200)
eta = 0.01

Dm = np.empty((len(lam0s), len(ws)), dtype=complex)
for i, lam in enumerate(lam0s):
    for j, w in enumerate(ws):
        mx = LMG.f_mx(wz, J, W, lam)
        
        chixx0 = dicke.f_chixx0(w + 1j*eta, wz, 2*(lam**2/W + J)*mx)
        Vind = LMG.f_Vind(w + 1j*eta, W, lam, J)
        
        Dm[i, j] = green.f_Dm(w + 1j*eta, W, lam, chixx0, Vind)
        
cm = ax.pcolormesh(lam0s,
                   ws,
                   -Dm.T.imag,
                   cmap='BuPu',
                   norm=mpl.colors.LogNorm()
                   )
cbar = fig.colorbar(cm,
                    pad = -0.0,
                    aspect = 60,
                    label=r'$-{\rm Im}D(\omega) \Omega$')
# ax.axhline(W, c='k', ls=(0, (1, 5)), lw=1)
# ax.axhline(np.sqrt(wz*(wz - 4*J)), c='k', ls=(0, (1, 5)), lw=1)
# ax.axvline(np.sqrt(W*(wz/4 - J)),  c='k', ls=(0, (1, 5)), lw=1)

# # ---------- two oscillator polaritons -------------
# up_twoosc = []
# lp_twoosc = []
# for i, lam in enumerate(lam0s):
#     if wz > 4*(lam**2/W + J):
#         pm, pp = poles.polaritons(np.sqrt(wz*(wz - 4*J)),
#                                 W,
#                                 lam*(1 - 4*J/wz)**(-1/4)
#                                 )
#     else:
#         pm, pp = None, None
#     up_twoosc.append(pp)
#     lp_twoosc.append(pm)
    
# ax.plot(lam0s, up_twoosc, c='b', label="Poles Two Osc. model", lw=1)
# ax.plot(lam0s, lp_twoosc, c='b', lw=1)

# ax.legend()
# # ---------- two oscillator polaritons -------------

# ---------- two oscillator polaritons -------------
up_twoosc = []
lp_twoosc = []
for i, lam in enumerate(lam0s):
    pm, pp = polaritons.LMG(wz, W, lam, J)
    
    up_twoosc.append(pp)
    lp_twoosc.append(pm)
    
ax.plot(lam0s, up_twoosc, c='gold', label=r"$\Omega_\pm$ (exact polaritons)", ls='--')
ax.plot(lam0s, lp_twoosc, c='gold', ls='--')

# ax.legend()
# ---------- two oscillator polaritons -------------

ax.set_ylim(0, 2)
ax.set_xlabel(r'$\lambda / \Omega$')
#ax.set_ylabel(r'$\omega / \Omega$')
ax.set_yticklabels([])
ax.set_title(rf'$\omega_z/\Omega = {wz} \,,\; J / \Omega = {J / W}$')

ax.legend()

fig.savefig('plots/LMG_photon_response_paper.jpeg', bbox_inches='tight', dpi=300)