import numpy as np


def f_mx(wz, J, W, lam):
    if wz > 4 * (lam**2 / W + J):
        mx = 0
    else:
        mx = np.sqrt(1 - (wz / (4 * (lam**2 / W + J))) ** 2)

    return mx


def f_Vind(w, W, lam, J):
    return 2 * (lam**2 * W / (w**2 - W**2) - J)
