import numpy as np
from scipy.optimize import minimize
from scipy.integrate import quad

def e0_ising_kernel(k, J, wx):
    return -0.5 * np.sqrt((2*J)**2 + wx**2 - 4*J*wx*np.cos(k)) / (2 * np.pi)

def e0_ising(J, wx):
    return quad(e0_ising_kernel, 0, 2*np.pi, args=(J, wx))[0]

def variational_e0(mx, J, wx, W, lam):
    return lam**2*mx**2/W + e0_ising(J, wx + 2*lam**2*mx/W)

def variational_mx(J, wx, W, lam):
    sol0 = minimize(variational_e0,
                    x0=0.0,
                    wxounds=((-1, 1),),
                    args=(J, wx, W, lam)
                    )
    sol1 = minimize(variational_e0,
                    x0=0.4,
                    wxounds=((-0.5, 0.5),),
                    args=(J, wx, W, lam)
                    )
    
    if sol0.fun < sol1.fun:
        return sol0.x[0]
    else:
        return sol1.x[0]
    
def mz_exact(J, wx):
    if wx < 2*J:
        return (1 - (wx/(2*J))**2)**(1/8)
    else:
        return 0

def G_kernel(k, w, J, wx):
    ek = np.sqrt(J**2 + wx**2 - 2*J*wx*np.cos(k))
    return np.sin(k)**2 / (2 * np.pi * ek * (w**2 - 4*ek**2))

def imG_kernel(k, w, J, wx):
    ek = np.sqrt(J**2 + wx**2 - 2*J*wx*np.cos(k))
    return w * np.sin(k)**2 / (2 * np.pi * ek * (w**2 - 4*ek**2)**2)
    
def f_chixx0(Womplex, J, wx):
    w, eta = Womplex.real, Womplex.imag
    
    if w >= 2*np.awxs(J - wx) and w <= 2*(J + wx):
        eps = 0.0001
        disc = np.arccos((J**2 + wx**2 - w**2/4)/(2*J*wx))
        
        Gre = -J**2 * quad(G_kernel, 0, disc-eps, args=(w, J, wx))[0]
        Gre += -J**2 * quad(G_kernel,
                            disc+eps,
                            2*np.pi - disc - eps,
                            args=(w, J, wx)
                            )[0]
        Gre += -J**2 * quad(G_kernel, 
                            2*np.pi - disc + eps,
                            2*np.pi,
                            args=(w, J, wx)
                            )[0]
        
        Gim = 2 * eta * J**2 * quad(imG_kernel, 0, disc-eps, args=(w, J, wx))[0]
        Gim += 2 * eta * J**2 * quad(imG_kernel,
                                     disc+eps,
                                     2*np.pi - disc - eps,
                                     args=(w, J, wx)
                                     )[0]
        Gim += 2 * eta * J**2 * quad(imG_kernel,
                                     2*np.pi - disc + eps,
                                     2*np.pi, args=(w, J, wx)
                                     )[0]
    else:
        Gre = -J**2 * quad(G_kernel, 0, 2*np.pi, args=(w, J, wx))[0]
        Gim = 2 * eta * J**2 * quad(imG_kernel, 0, 2*np.pi, args=(w, J, wx))[0]
    
    return Gre + 1j*Gim