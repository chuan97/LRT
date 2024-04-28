import numpy as np

def f_mx(wz, z, W, lam):
    if z != 0 and z != 1:
        print('WARNING: z should be 0 or 1')
        
    lamc = 0.5 * np.sqrt(W * wz)
    
    if lam < lamc or z == 1:
        mx = 0
    else:
        mx = np.sqrt(1 - (lamc/lam)**4)
        
    return mx

def f_chixx0(w, wz, h):
    Eh = np.sqrt(wz**2 + 4*h**2)
    
    return -(wz / Eh)**2 * 2*Eh / (w**2 - Eh**2)

def f_Vind(w, W, lam, zeta):
    return 2 * lam**2 * (W**2 * (zeta - 1) - zeta*w**2) / (W * (W**2 - w**2))