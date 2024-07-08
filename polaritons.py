import numpy as np
from scipy.optimize import root


def normal(w1, w2, lam):
    A = w1**2 + w2**2
    B = np.sqrt((w1**2 - w2**2) ** 2 + 16 * lam**2 * w1 * w2)

    return np.sqrt(0.5 * (A - B)), np.sqrt(0.5 * (A + B))


def super_dicke(wz, wc, lam):
    mu = wz * wc / (4 * lam**2)

    A = (wz / mu) ** 2 + wc**2
    B = np.sqrt(((wz / mu) ** 2 - wc**2) ** 2 + 4 * wc**2 * wz**2)

    return np.sqrt(0.5 * (A - B)), np.sqrt(0.5 * (A + B))


def super_LMG(wz, wc, lam, J):
    mu = wz / (4 * (lam**2 / wc + J))

    A = (wz / mu) ** 2 - 4 * mu * J * wz + wc**2
    B = np.sqrt(
        ((wz / mu) ** 2 - 4 * mu * J * wz - wc**2) ** 2 + 16 * lam**2 * mu * wc * wz
    )

    return np.sqrt(0.5 * (A - B)), np.sqrt(0.5 * (A + B))


def dicke(wz, wc, lam):
    if 4 * lam**2 < wz * wc:
        # paramagnetic phase
        return normal(wz, wc, lam)
    else:
        # ferromagnetic phase
        return super_dicke(wz, wc, lam)


def LMG(wz, wc, lam, J):
    if wz > 4 * (lam**2 / wc + J):
        # paramagnetic phase
        return normal(
            np.sqrt(wz * (wz - 4 * J)), wc, lam * (1 - 4 * J / wz) ** (-1 / 4)
        )
    else:
        # ferromagnetic phase
        return super_LMG(wz, wc, lam, J)


def f_a(wz, wc, lam, J):
    if wz:
        mu = wz / (4 * (lam**2 / wc - J))

        return wz * (1 - mu) * (3 + mu) / (8 * mu * (1 + mu)) - J * (1 - mu) * (
            1 + 3 * mu
        ) / (2 * (1 + mu))
    else:
        return 3 * lam**2 / (2 * wc) - 2 * J


def f_b(wz, wc, lam, J):
    if wz:
        mu = wz / (4 * (lam**2 / wc - J))

        return (1 + mu) * (wz / (2 * mu) + 2 * J)
    else:
        return 2 * lam**2 / wc


def f_c(wz, wc, lam, J):
    if wz:
        mu = wz / (4 * (lam**2 / wc - J))

        return lam * mu * np.sqrt(2 / (1 + mu))
    else:
        return 0.0


def super_LMG_trans(wz, wc, lam, J):
    a = f_a(wz, wc, lam, J)
    b = f_b(wz, wc, lam, J)
    c = f_c(wz, wc, lam, J)

    A = b**2 + 4 * a * b + wc**2
    B = np.sqrt((b**2 + 4 * a * b - wc**2) ** 2 + 16 * b * c**2 * wc)

    if c:
        return np.sqrt(0.5 * (A - B)), np.sqrt(0.5 * (A + B))
    else:
        return np.sqrt(b**2 + 4 * a * b), wc


def LMG_transverse(wz, wc, lam, J):
    if (wz + 4 * J) > 4 * lam**2 / wc:
        # paramagnetic phase
        return normal(wz + 4 * J, wc, lam)
    else:
        # ferromagnetic phase
        return super_LMG_trans(wz, wc, lam, J)


def sqrtb_def_cond(sqrtb, wx, wz, wc, lam, J):
    bb = sqrtb**2
    k = 1 - sqrtb**2

    return (
        8 * sqrtb * (lam**2 / wc - J) * (0.5 - bb)
        - wz * sqrtb
        + 0.5 * wx * np.sqrt(k) * (1 - bb / k)
    )


def coefficients(sqrtb, wx, wz, wc, lam, J):
    k = 1 - sqrtb**2
    if k < 0:
        print("flag", sqrtb, sqrtb**2, k, lam)
    sqrta = 2 * lam / wc * sqrtb * np.sqrt(k)
    bb = sqrtb**2

    a = (
        0.5 * lam / k**2 * sqrta * sqrtb * np.sqrt(k) * (2 * k + bb)
        - 4 * J * bb
        + wx / 4 * sqrtb / np.sqrt(k) * (1 + bb / (2 * k))
    )
    b = (
        wz
        + 2 * lam / k * sqrta * sqrtb * np.sqrt(k)
        + 8 * J * (0.5 - bb)
        + 0.5 * wx * sqrtb / np.sqrt(k)
    )
    c = 2 * lam / k * np.sqrt(k) * (0.5 - bb)

    return a, b, c


def general(wc, a, b, c):
    A = b**2 + 4 * a * b + wc**2
    B = np.sqrt((b**2 + 4 * a * b - wc**2) ** 2 + 16 * b * c**2 * wc)

    if c:
        return np.sqrt(0.5 * (A - B)), np.sqrt(0.5 * (A + B))
    else:
        return np.sqrt(b**2 + 4 * a * b), wc


def LMG_transverse_general(wx, wz, wc, lam, J):
    sqrtb_para = root(sqrtb_def_cond, 0.0, args=(wx, wz, wc, lam, J)).x[0]
    sqrtb_ferro = root(sqrtb_def_cond, np.sqrt(0.5), args=(wx, wz, wc, lam, J)).x[0]

    a, b, c = coefficients(sqrtb_para, wx, wz, wc, lam, J)
    pm_para, pp_para = general(wc, a, b, c)

    a, b, c = coefficients(sqrtb_ferro, wx, wz, wc, lam, J)
    pm_ferro, pp_ferro = general(wc, a, b, c)
    # print(wx, wz, lam, pm_para, pp_para, sqrtb_para, pm_ferro, pp_ferro)
    if np.isnan(pm_para) or (sqrtb_para == 0.0 and wx != 0) or lam ** 2 / wc > J:
        return pm_ferro, pp_ferro
    else:
        return pm_para, pp_para
