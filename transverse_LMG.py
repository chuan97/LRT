import numpy as np

def f_Vindx(w, W, lam):
    return 2*lam**2*W/(w**2 - W**2)

def f_Vindz(w, J):
    return -2*J

def f_chixx0(w, wz, h):
    Eh = np.sqrt(wz**2 + (2*h)**2)
    
    return - wz**2/Eh**2 * 2*Eh/(w**2 - Eh**2)

def f_chizz0(w, wz, h):
    Eh = np.sqrt(wz**2 + (2*h)**2)
    
    return -(2*h)**2/Eh**2 * 2*Eh/(w**2 - Eh**2)

def f_chixz0(w, wz, h):
    Eh = np.sqrt(wz**2 + (2*h)**2)
    
    return -2*h*wz/Eh**2 * 2*Eh/(w**2 - Eh**2)
