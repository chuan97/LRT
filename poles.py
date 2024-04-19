import numpy as np

def polaritons(wz, wc, lam):
    def para(wz, wc, lam):
        innerroot = np.sqrt((wz**2 - wc**2)**2 + 16*lam**2*wz*wc)
        return (np.sqrt(0.5 * (wz**2 + wc**2 - innerroot)),
                np.sqrt(0.5 * (wz**2 + wc**2 + innerroot))
        )
    
    def ferro(wz, wc, lam):
        mu = wz * wc / (4 * lam**2)
        
        innerroot = np.sqrt(((wz/mu)**2 - wc**2)**2 + 4*wc**2*wz**2)
        return (np.sqrt(0.5*((wz/mu)**2 + wc**2 - innerroot)),
                np.sqrt(0.5*((wz/mu)**2 + wc**2 + innerroot))
        )
    
    if 4 * lam**2 < wz * wc:
        # paramagnetic phase
        return para(wz, wc, lam)
    else:
        # ferromagnetic phase
        return ferro(wz, wc, lam)