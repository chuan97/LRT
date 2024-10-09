import numpy as np
from scipy.optimize import root


def f_mx(wz, J, W, lam):
    if wz > 4 * (lam**2 / W + J):
        mx = 0
    else:
        mx = np.sqrt(1 - (wz / (4 * (lam**2 / W + J))) ** 2)

    return mx


def f_Vind(w, W, lam, J):
    return 2 * (lam**2 * W / (w**2 - W**2) - J)


def f_mx_temp(wz, J, W, lam, beta):
    if wz > 4 * (lam**2 / W + J) * np.tanh(0.5 * beta * wz):
        mx = 0
    else:

        def aux(mx, wz, J, W, lam, beta):
            eps = np.sqrt(wz**2 + 4 * (2 * (lam**2 / W + J) * mx) ** 2)
            return eps - 4 * (lam**2 / W + J) * np.tanh(0.5 * beta * eps)

        mx = root(aux, 1.0, args=(wz, J, W, lam, beta)).x[0]

    return mx
