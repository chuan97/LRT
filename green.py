import numpy as np

def f_Dm(w, W, lam, chixx0, Vind):
    chixx = f_chixx(Vind, chixx0)
    D0 = f_D0(w, W)
    
    return D0 - lam**2*D0**2*chixx

def f_D0(w, W):
    return 1 / (w - W)

def f_chixx(Vind, chixx0):
    return chixx0 / (1 + Vind*chixx0)
