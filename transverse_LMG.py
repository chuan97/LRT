import numpy as np
from scipy.optimize import minimize


def variational_e0(mx, wx, wz, J, W, lam):
    mz = -np.sqrt(1 - mx**2)
    wztilde = wz - 4 * J * mz
    h = wx / 2 - 2 * lam**2 * mx / W

    return -0.5 * np.sqrt(wztilde**2 + (2 * h) ** 2) + (lam**2 / W - J) * mx**2


def variational_mx(wx, wz, J, W, lam):
    sol0 = minimize(variational_e0, x0=0.0, args=(wx, wz, J, W, lam), bounds=((-1, 1),))
    sol1 = minimize(
        variational_e0, x0=-0.9, args=(wx, wz, J, W, lam), bounds=((-1, 1),)
    )

    if sol0.fun < sol1.fun:
        return sol0.x[0]
    else:
        return sol1.x[0]


def alt_variational_e0(mz, wx, wz, J, W, lam):
    mx = -np.sqrt(1 - mz**2)
    wztilde = wz - 4 * J * mz
    h = wx / 2 - 2 * lam**2 * mx / W

    return -0.5 * np.sqrt(wztilde**2 + (2 * h) ** 2) + lam**2 / W * mx**2 + J * mz**2


def variational_mz(wx, wz, J, W, lam):
    sol0 = minimize(
        alt_variational_e0, x0=-0.9, args=(wx, wz, J, W, lam), bounds=((-1, 1),)
    )
    sol1 = minimize(
        alt_variational_e0, x0=0.0, args=(wx, wz, J, W, lam), bounds=((-1, 1),)
    )

    if sol0.fun < sol1.fun:
        return sol0.x[0]
    else:
        return sol1.x[0]


def alt_alt_variational_e0(m, wx, wz, J, W, lam):
    mx, mz = m

    wztilde = wz - 4 * J * mz
    h = wx / 2 - 2 * lam**2 * mx / W

    return -0.5 * np.sqrt(wztilde**2 + (2 * h) ** 2) + lam**2 / W * mx**2 + J * mz**2


def variational_m(wx, wz, J, W, lam):
    sol0 = minimize(
        alt_alt_variational_e0,
        x0=(0.0, -0.0),
        args=(wx, wz, J, W, lam),
        bounds=((-1, 1), (-1, 1)),
    )
    sol1 = minimize(
        alt_alt_variational_e0,
        x0=(-0.9, -0.9),
        args=(wx, wz, J, W, lam),
        bounds=((-1, 1), (-1, 1)),
    )

    if sol0.fun < sol1.fun:
        return sol0.x
    else:
        return sol1.x


def f_Vindx(w, W, lam):
    return 2 * lam**2 * W / (w**2 - W**2)


def f_Vindz(w, J):
    return -2 * J


def f_chixx0(w, wz, h):
    Eh = np.sqrt(wz**2 + (2 * h) ** 2)

    return -(wz**2) / Eh**2 * 2 * Eh / (w**2 - Eh**2)


def f_chizz0(w, wz, h):
    Eh = np.sqrt(wz**2 + (2 * h) ** 2)

    return -((2 * h) ** 2) / Eh**2 * 2 * Eh / (w**2 - Eh**2)


def f_chixz0(w, wz, h):
    Eh = np.sqrt(wz**2 + (2 * h) ** 2)

    return -2 * h * wz / Eh**2 * 2 * Eh / (w**2 - Eh**2)
