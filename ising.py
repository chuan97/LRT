import numpy as np
from scipy.integrate import quad
from scipy.optimize import minimize


def e0_ising_kernel(k, J, wx):
    return -0.5 * np.sqrt((2 * J) ** 2 + wx**2 - 4 * J * wx * np.cos(k))


def e0_ising(J, wx):
    return 1 / (2 * np.pi) * quad(e0_ising_kernel, 0, 2 * np.pi, args=(J, wx))[0]


def variational_e0(mx, J, wx, W, lam):
    return lam**2 * mx**2 / W + e0_ising(J, wx - 4 * lam**2 * mx / W)


def mx_free_kernel(k, J, wx):
    ek = f_ek(k, J, wx)
    return (wx - 2 * J * np.cos(k)) / (2 * np.pi * ek)


def mx_free(J, wx):
    return quad(mx_free_kernel, 0, 2 * np.pi, args=(J, wx))[0]


def variational_mx(J, wx, W, lam):
    sol0 = minimize(variational_e0, x0=0.0, bounds=((-1, 1),), args=(J, wx, W, lam))
    sol1 = minimize(variational_e0, x0=-0.9, bounds=((-1, 1),), args=(J, wx, W, lam))

    if sol0.fun < sol1.fun:
        return sol0.x[0]
    else:
        return sol1.x[0]


def mz_exact(J, wx):
    if wx < 2 * J:
        return (1 - (wx / (2 * J)) ** 2) ** (1 / 8)
    else:
        return 0


def f_ek(k, J, wx):
    return np.sqrt((2 * J) ** 2 + wx**2 - 4 * J * wx * np.cos(k))


def chixx0_kernel(k, w, J, wx, eta):
    ek = f_ek(k, J, wx)
    num = np.sin(k) ** 2 * (w**2 - eta**2 - 4 * ek**2)
    den = ek * ((w**2 - eta**2 - 4 * ek**2) ** 2 + 4 * eta**2 * w**2)

    return num / den


def imchixx0_kernel(k, w, J, wx, eta):
    ek = f_ek(k, J, wx)
    num = np.sin(k) ** 2
    den = ek * ((w**2 - eta**2 - 4 * ek**2) ** 2 + 4 * eta**2 * w**2)

    return num / den


def f_chixx0(w_complex, J, wx):
    w, eta = w_complex.real, w_complex.imag

    re = (
        -4
        * (2 * J) ** 2
        / np.pi
        * quad(chixx0_kernel, 0, 2 * np.pi, args=(w, J, wx, eta))[0]
    )
    im = (
        2
        * eta
        * w
        * 4
        * (2 * J) ** 2
        / np.pi
        * quad(imchixx0_kernel, 0, 2 * np.pi, args=(w, J, wx, eta))[0]
    )

    return re + 1j * im


def altchixx0_kernel(k, w, J, wx, eta):
    ek = f_ek(k, J, wx)
    num = np.sin(k) ** 2 * (2 * ek * (w**2 - 4 * ek**2 + eta**2) - 4 * eta**2 * ek)
    den = ek * 2 * ((w**2 + eta**2 - 4 * ek**2) ** 2 + 16 * eta**2 * ek**2)

    return num / den


def altimchixx0_kernel(k, w, J, wx, eta):
    ek = f_ek(k, J, wx)
    num = np.sin(k) ** 2 * (w**2 + 4 * ek**2 + eta**2)
    den = ek * 2 * ((w**2 + eta**2 - 4 * ek**2) ** 2 + 16 * eta**2 * ek**2)

    return num / den


def f_chixx0_alt(w_complex, J, wx):
    w, eta = w_complex.real, w_complex.imag

    re = (
        -4
        * (2 * J) ** 2
        / np.pi
        * quad(altchixx0_kernel, 0, 2 * np.pi, args=(w, J, wx, eta))[0]
    )
    im = (
        eta
        * 4
        * (2 * J) ** 2
        / np.pi
        * quad(altimchixx0_kernel, 0, 2 * np.pi, args=(w, J, wx, eta))[0]
    )

    return re + 1j * im


def I(w_complex, J, wx):
    w, eta = w_complex.real, w_complex.imag

    re = (
        8 * J**2 / np.pi * quad(altchixx0_kernel, 0, 2 * np.pi, args=(w, J, wx, eta))[0]
    )
    im = (
        -eta
        * 8
        * J**2
        / np.pi
        * quad(altimchixx0_kernel, 0, 2 * np.pi, args=(w, J, wx, eta))[0]
    )

    return re + 1j * im
