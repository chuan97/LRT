import numpy as np
from scipy.optimize import minimize
from scipy.integrate import quad

def e0_ising_kernel(k, J, B):
    return -0.5 * np.sqrt(J**2 + B**2 - 2*J*B*np.cos(k)) / (2 * np.pi)

def e0_ising(J, B):
    return quad(e0_ising_kernel, 0, 2*np.pi, args=(J, B))[0]

def variational_e0(mx, J, B, wc, lam):
    return lam**2*mx**2/wc + e0_ising(J, B + 2*lam**2*mx/wc)

def variational_mx(J, B, wc, lam):
    sol0 = minimize(variational_e0, x0=0.0, args=(J, B, wc, lam))
    sol1 = minimize(variational_e0, x0=0.5, args=(J, B, wc, lam))
    
    if sol0.fun < sol1.fun:
        return sol0.x[0]
    else:
        return sol1.x[0]
    
def mz_exact(J, B):
    if B < J:
        return 0.5*(1 - (B/J)**2)**(1/8)
    else:
        return 0

def G_kernel(k, w, J, B):
    ek = np.sqrt(J**2 + B**2 - 2*J*B*np.cos(k))
    return np.sin(k)**2 / (2 * np.pi * ek * (w**2 - 4*ek**2))

def imG_kernel(k, w, J, B):
    ek = np.sqrt(J**2 + B**2 - 2*J*B*np.cos(k))
    return w * np.sin(k)**2 / (2 * np.pi * ek * (w**2 - 4*ek**2)**2)
    
def f_chixx0(wcomplex, J, B):
    w, eta = wcomplex.real, wcomplex.imag
    
    if w >= 2*np.abs(J - B) and w <= 2*(J + B):
        eps = 0.0001
        disc = np.arccos((J**2 + B**2 - w**2/4)/(2*J*B))
        
        Gre = -J**2 * quad(G_kernel, 0, disc-eps, args=(w, J, B))[0]
        Gre += -J**2 * quad(G_kernel,
                            disc+eps,
                            2*np.pi - disc - eps,
                            args=(w, J, B)
                            )[0]
        Gre += -J**2 * quad(G_kernel, 
                            2*np.pi - disc + eps,
                            2*np.pi,
                            args=(w, J, B)
                            )[0]
        
        Gim = 2 * eta * J**2 * quad(imG_kernel, 0, disc-eps, args=(w, J, B))[0]
        Gim += 2 * eta * J**2 * quad(imG_kernel,
                                     disc+eps,
                                     2*np.pi - disc - eps,
                                     args=(w, J, B)
                                     )[0]
        Gim += 2 * eta * J**2 * quad(imG_kernel,
                                     2*np.pi - disc + eps,
                                     2*np.pi, args=(w, J, B)
                                     )[0]
    else:
        Gre = -J**2 * quad(G_kernel, 0, 2*np.pi, args=(w, J, B))[0]
        Gim = 2 * eta * J**2 * quad(imG_kernel, 0, 2*np.pi, args=(w, J, B))[0]
    
    return Gre + 1j*Gim