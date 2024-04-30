import numpy as np

def f_Dm(w, W, lam, chixx):
    D0 = f_D0(w, W)
    
    return D0 - lam**2*D0**2*chixx

def f_D0(w, W):
    return 1 / (w - W)

def f_chixx(Vind, chixx0):
    return chixx0 / (1 + Vind*chixx0)

def f_chixx_twomode(Vindx, Vindz, chixx0, chixz0, chizx0, chizz0):
    a = chixx0*chizz0 - chixz0*chizx0
    return (chixx0 + Vindz*a)/(1 + Vindx*chixx0 + Vindz*chizz0 + Vindx*Vindz*a)
