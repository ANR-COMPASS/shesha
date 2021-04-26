"""
STARLORD (SeT of Algorithms foR mOdified stRucture function computation)
Set of functions for structure function computation
"""

import numpy as np
from scipy.special import jv  # Bessel function


def dphi_highpass(r, x0, tabx, taby):
    """
    Fonction de structure de phase "haute frequence"
    A renormalise en fonction du r0
    :params:
        r : distance [m]
        x0 : distance interactionneur [m]
        tabx, taby : integrale tabulee obtenue avec la fonction tabulateIj0
    """
    return (r**
            (5. / 3.)) * (1.1183343328701949 - Ij0t83(r * (np.pi / x0), tabx, taby)) * (
                    2 * (2 * np.pi)**(8 / 3.) * 0.0228956)


def dphi_lowpass(r, x0, L0, tabx, taby):
    """
    Fonction de structure de phase "basse frequence"
    A renormalise en fonction du r0
    :params:
        r : distance [m]
        x0 : distance interactionneur [m]
        tabx, taby : integrale tabulee obtenue avec la fonction tabulateIj0
    """
    return rodconan(r, L0) - dphi_highpass(r, x0, tabx, taby)


def Ij0t83(x, tabx, taby):
    """
    Calcul de l'integrale tabulee
    x
   $ t^(-8/3) (1-bessel_j(0,t)) dt
    0

    Pres de 0, le resultat s'approxime par (3/4.)*x^(1./3)*(1-x^2/112.+...)
    """
    res = x.copy()
    ismall = np.where(res < np.exp(-3.0))
    ilarge = np.where(res >= np.exp(-3.0))
    if (ismall[0].size > 0):
        res[ismall] = 0.75 * x[ismall]**(1. / 3) * (1 - x[ismall]**2 / 112.)
    if (ilarge[0].size > 0):
        res[ilarge] = np.interp(x[ilarge], tabx, taby)

    return res


def tabulateIj0():
    """
    Tabulation de l'intesgrale
    Necessaire avant utilisation des fonction dphi_lowpass et dphi_highpass
    """
    n = 10000
    t = np.linspace(-4, 10, n)
    dt = (t[-1] - t[0]) / (n - 1)
    smallx = np.exp(-4.0)
    A = 0.75 * smallx**(1. / 3) * (1 - smallx**2 / 112.)
    X = np.exp(t)
    Y = np.exp(-t * (5. / 3.)) * (1 - jv(0, X))
    Y[1:] = np.cumsum(Y[:-1] + np.diff(Y) / 2.)
    Y[0] = 0.
    Y = Y * dt + A

    return X, Y


def asymp_macdo(x):
    k2 = 1.00563491799858928388289314170833
    k3 = 1.25331413731550012081
    a1 = 0.22222222222222222222
    a2 = -0.08641975308641974829
    a3 = 0.08001828989483310284

    x_1 = 1. / x
    res = k2 - k3 * np.exp(-x) * x**(1. / 3.) * (1.0 + x_1 * (a1 + x_1 *
                                                              (a2 + x_1 * a3)))
    return res


def macdo(x):
    a = 5. / 6.
    x2a = x**(2. * a)
    x22 = x * x / 4.
    s = 0.0

    Ga = [
            0, 12.067619015983075, 5.17183672113560444, 0.795667187867016068,
            0.0628158306210802181, 0.00301515986981185091, 9.72632216068338833e-05,
            2.25320204494595251e-06, 3.93000356676612095e-08, 5.34694362825451923e-10,
            5.83302941264329804e-12
    ]

    Gma = [
            -3.74878707653729304, -2.04479295083852408, -0.360845814853857083,
            -0.0313778969438136685, -0.001622994669507603, -5.56455315259749673e-05,
            -1.35720808599938951e-06, -2.47515152461894642e-08, -3.50257291219662472e-10,
            -3.95770950530691961e-12, -3.65327031259100284e-14
    ]

    x2n = 0.5

    s = Gma[0] * x2a
    s *= x2n

    x2n *= x22

    for n in np.arange(10) + 1:

        s += (Gma[n] * x2a + Ga[n]) * x2n
        x2n *= x22

    return s


def rodconan(r, L0):
    """
    Fonction de structure de phase avec prise en compte de l'echelle externe
    A renormalise en fonction du r0
    """
    res = r * 0.
    k1 = 0.1716613621245709486
    dprf0 = (2 * np.pi / L0) * r
    ilarge = np.where(dprf0 > 4.71239)
    ismall = np.where(dprf0 <= 4.71239)
    if (ilarge[0].size > 0):
        res[ilarge] = asymp_macdo(dprf0[ilarge])
    if (ismall[0].size > 0):
        res[ismall] = -macdo(dprf0[ismall])

    res *= k1 * L0**(5. / 3.)

    return res
