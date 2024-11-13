import numpy as np


def landau_polaritons(w0, wc, wp):
    wt = np.sqrt(w0**2 + wp**2)

    A = wt**2 + wc**2
    B = np.sqrt((wt**2 - wc**2) ** 2 + 4 * wc**2 * wp**2)

    return np.sqrt(0.5 * (A - B)), np.sqrt(0.5 * (A + B))


def sigma_xx(w0, wc, wp, delta):
    lp, up = landau_polaritons(w0, wc, wp)

    n1 = wc**2 + wp**2
    n2 = w0**2 * wc**2 / (wc**2 + wp**2) + delta**2
    d1 = lp**2 + delta**2
    d2 = up**2 + delta**2

    return 1 - n1 * n2 / (d1 * d2)


def sigma_xy(w0, wc, wp, delta):
    wt = np.sqrt(w0**2 + wp**2)

    lp, up = landau_polaritons(w0, wc, wp)

    n1 = wc**2
    n2 = w0**2 + delta**2
    d1 = lp**2 + delta**2
    d2 = up**2 + delta**2

    return n1 * n2 / (d1 * d2)


def sigma_yy(w0, wc, wp, delta):
    lp, up = landau_polaritons(w0, wc, wp)

    n1 = wc**2
    n2 = w0**2 + delta**2
    d1 = lp**2 + delta**2
    d2 = up**2 + delta**2

    return 1 - n1 * n2 / (d1 * d2)


def rho_xx(w0, wc, wp, delta):
    return w0**2 + wp**2 + delta**2 / (w0**2 + delta**2)
