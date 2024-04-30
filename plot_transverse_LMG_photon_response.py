import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

import transverse_LMG as tLMG
import plot
import green
import polaritons

plot.set_rcParams(size = (10, 9), lw = 2, fs = 16)

fig, axes = plt.subplots(2, 2, constrained_layout=True)
ax = axes[0, 0]

W = 1.0
wz = 0.0
wx = 0.0
J = 0.25

lam0s = np.linspace(0.0001, 1, 100)
ws = np.linspace(0, 3, 100)
eta = 0.01

Dm = np.empty((len(lam0s), len(ws)), dtype=complex)
for i, lam in enumerate(lam0s):
    mx = tLMG.variational_mx(wx, wz, J, W, lam)
    mz = -np.sqrt(1-mx**2)
    wztilde = wz - 4*J*mz
    h = wx/2 - 2*lam**2*mx/W
    
    for j, w in enumerate(ws):
        chixx0 = tLMG.f_chixx0(w + 1j*eta, wztilde, h)
        chixz0 = chizx0 = tLMG.f_chixz0(w + 1j*eta, wztilde, h)
        chizz0 = tLMG.f_chizz0(w + 1j*eta, wztilde, h)
        Vindx = tLMG.f_Vindx(w + 1j*eta, W, lam)
        Vindz = tLMG.f_Vindz(w + 1j*eta, J)
        
        chixx = green.f_chixx_twomode(Vindx,
                                      Vindz,
                                      chixx0,
                                      chixz0,
                                      chizx0,
                                      chizz0
                                      )
        
        Dm[i, j] = green.f_Dm(w + 1j*eta, W, lam, chixx)
        # Dm[i, j] = -chixx
        
cm = ax.pcolormesh(lam0s,
                   ws,
                   -Dm.T.imag,
                   cmap='BuPu',
                   norm=mpl.colors.LogNorm()
                   )

ax.set_ylim(0, 3)
ax.set_xlabel(r'$\lambda / \Omega$')
ax.set_ylabel(r'$\omega / \Omega$')
ax.set_title(rf'$\omega_x/\Omega = {wx} \,,\; \omega_z/\Omega = {wz} \,,\; J / \Omega = {J / W}$',
             fontsize=14)

ax = axes[0, 1]

W = 1.0
wz = 0.
wx = 0.2
J = 0.25

lam0s = np.linspace(0.0001, 1, 100)
ws = np.linspace(0, 3, 100)
eta = 0.01

Dm = np.empty((len(lam0s), len(ws)), dtype=complex)
for i, lam in enumerate(lam0s):
    for j, w in enumerate(ws):
        mx = tLMG.variational_mx(wx, wz, J, W, lam)
        mz = -np.sqrt(1-mx**2)
        
        wztilde = wz - 4*J*mz
        h = wx/2 - 2*lam**2*mx/W
        
        chixx0 = tLMG.f_chixx0(w + 1j*eta, wztilde, h)
        chixz0 = chizx0 = tLMG.f_chixz0(w + 1j*eta, wztilde, h)
        chizz0 = tLMG.f_chizz0(w + 1j*eta, wztilde, h)
        Vindx = tLMG.f_Vindx(w + 1j*eta, W, lam)
        Vindz = tLMG.f_Vindz(w + 1j*eta, J)
        
        chixx = green.f_chixx_twomode(Vindx,
                                      Vindz,
                                      chixx0,
                                      chixz0,
                                      chizx0,
                                      chizz0
                                      )
        
        Dm[i, j] = green.f_Dm(w + 1j*eta, W, lam, chixx)
        # Dm[i, j] = -chixx
        
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

ax.set_ylim(0, 3)
ax.set_xlabel(r'$\lambda / \Omega$')
#ax.set_ylabel(r'$\omega / \Omega$')
ax.set_yticklabels([])
ax.set_title(rf'$\omega_x/\Omega = {wx} \,,\; \omega_z/\Omega = {wz} \,,\; J / \Omega = {J / W}$',
             fontsize=14)

ax = axes[1, 0]

W = 1.0
wz = 0.2
wx = 0.
J = 0.25

lam0s = np.linspace(0.0001, 1, 100)
ws = np.linspace(0, 3, 100)
eta = 0.01

Dm = np.empty((len(lam0s), len(ws)), dtype=complex)
for i, lam in enumerate(lam0s):
    mx = tLMG.variational_mx(wx, wz, J, W, lam)
    mz = -np.sqrt(1-mx**2)
    wztilde = wz - 4*J*mz
    h = wx/2 - 2*lam**2*mx/W
    
    for j, w in enumerate(ws):
        chixx0 = tLMG.f_chixx0(w + 1j*eta, wztilde, h)
        chixz0 = chizx0 = tLMG.f_chixz0(w + 1j*eta, wztilde, h)
        chizz0 = tLMG.f_chizz0(w + 1j*eta, wztilde, h)
        Vindx = tLMG.f_Vindx(w + 1j*eta, W, lam)
        Vindz = tLMG.f_Vindz(w + 1j*eta, J)
        
        chixx = green.f_chixx_twomode(Vindx,
                                      Vindz,
                                      chixx0,
                                      chixz0,
                                      chizx0,
                                      chizz0
                                      )
        
        Dm[i, j] = green.f_Dm(w + 1j*eta, W, lam, chixx)
        # Dm[i, j] = -chixx
        
cm = ax.pcolormesh(lam0s,
                   ws,
                   -Dm.T.imag,
                   cmap='BuPu',
                   norm=mpl.colors.LogNorm()
                   )

ax.set_ylim(0, 3)
ax.set_xlabel(r'$\lambda / \Omega$')
ax.set_ylabel(r'$\omega / \Omega$')
ax.set_title(rf'$\omega_x/\Omega = {wx} \,,\; \omega_z/\Omega = {wz} \,,\; J / \Omega = {J / W}$',
             fontsize=14)

ax = axes[1, 1]

W = 1.0
wz = 0.2
wx = 0.2
J = 0.25

lam0s = np.linspace(0.0001, 1, 100)
ws = np.linspace(0, 3, 100)
eta = 0.01

Dm = np.empty((len(lam0s), len(ws)), dtype=complex)
for i, lam in enumerate(lam0s):
    for j, w in enumerate(ws):
        mx = tLMG.variational_mx(wx, wz, J, W, lam)
        mz = -np.sqrt(1-mx**2)
        
        wztilde = wz - 4*J*mz
        h = wx/2 - 2*lam**2*mx/W
        
        chixx0 = tLMG.f_chixx0(w + 1j*eta, wztilde, h)
        chixz0 = chizx0 = tLMG.f_chixz0(w + 1j*eta, wztilde, h)
        chizz0 = tLMG.f_chizz0(w + 1j*eta, wztilde, h)
        Vindx = tLMG.f_Vindx(w + 1j*eta, W, lam)
        Vindz = tLMG.f_Vindz(w + 1j*eta, J)
        
        chixx = green.f_chixx_twomode(Vindx,
                                      Vindz,
                                      chixx0,
                                      chixz0,
                                      chizx0,
                                      chizz0
                                      )
        
        Dm[i, j] = green.f_Dm(w + 1j*eta, W, lam, chixx)
        # Dm[i, j] = -chixx
        
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

ax.set_ylim(0, 3)
ax.set_xlabel(r'$\lambda / \Omega$')
#ax.set_ylabel(r'$\omega / \Omega$')
ax.set_yticklabels([])
ax.set_title(rf'$\omega_x/\Omega = {wx} \,,\; \omega_z/\Omega = {wz} \,,\; J / \Omega = {J / W}$',
             fontsize=14)

fig.savefig('plots/transverse_LMG_photon_response.jpeg', bbox_inches='tight', dpi=300)