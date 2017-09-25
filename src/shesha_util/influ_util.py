#!/usr/local/bin/python3.6
# encoding: utf-8
'''
Created on 3 aout 2017

@author: fferreira
'''
import numpy as np
import scipy.special as sp

from shesha_constants import DmType, PatternType


def besel_orth(m, n, phi, r):
    """
        TODO: docstring

        :parameters:
            m:
            n:
            phi:
            r:

        :return:
            B:
    """
    # fonction de bessel fourier orthogonale (BFOFS)
    if (m == 0):
        B = sp.jn(0, sp.jn_zeros(0, n)[n - 1] * r)
    elif (m > 0):
        B = sp.jn(m, sp.jn_zeros(m, n)[n - 1] * r) * np.sin(m * phi)
    else:
        B = sp.jn(np.abs(m), sp.jn_zeros(np.abs(m), n)[n -
                                                       1] * r) * np.cos(np.abs(m) * phi)
    return B


def bessel_influence(xx, yy, type_i=PatternType.SQUARE):
    """
        TODO: docstring

        :parameters:
            xx:
            yy:
            type_i: (optional)

        :return:
            influ
    """

    # base sur l article numerical model of the influence function of deformable mirrors based on bessel Fourier orthogonal functions
    # corespond a 3.2pitch

    influ = np.zeros(xx.shape, dtype=np.float32)

    # construction des tableaux :

    # construction des coordonnée cartesienne
    # x = np.arange(size)-middle # -->
    # y = (np.arange(size)-middle)*-1 # -->
    #xx,yy = np.meshgrid(x,y)
    # passage en coordonnée polaire
    r = np.sqrt(xx**2 + yy**2)
    phi = np.arctan2(yy, xx)  # +(np.pi/8.) #petite correction de rotation

    # coef for square IF
    a0 = [
            0.3826, 0.5207, 0.2841, -0.0146, -0.1103, -0.0818, -0.0141, 0.0123, 0.0196,
            0.0037
    ]
    am = [-0.0454, -0.1114, -0.1125, -0.0397, 0.0146, 0.0217, 0.0085, -0.0012, -0.0040]
    a = [-0.0002, -0.0004, -0.0001, 0.0004, 0.0005, 0.0003, 0.0001, 0, 0]

    # search coef for hexagonal IF (m =0, 6, -6 --> 28 term)
    # a0 ->10
    # a6 ->9
    # am6 ->9

    if type_i == PatternType.HEXA or type_i == PatternType.HEXAM4:
        sym = 6

    else:
        sym = 4

    # calcul pour m = 0
    for i in range(len(a0)):
        btemp = (a0[i] * besel_orth(0, i + 1, phi, r))

        influ = influ + btemp
    #print("fin cas m=",0)

    # calcul pour m=6
    for i in range(len(a)):
        influ = influ + (a[i] * besel_orth(sym, i + 1, phi, r))
    #print("fin cas m=",sym)

    # calcul pour m=-6
    for i in range(len(am)):
        influ = influ + (am[i] * besel_orth(-sym, i + 1, phi, r))
    #print("fin cas m=",-sym)

    return influ


def makeRigaut(pitch: float, coupling: float, x=None, y=None):
    """Compute 'Rigaut-like' influence function

    :parameters:
        pitch: (float) : pitch of the DM expressed in pixels
        coupling: (float) : coupling of the actuators
        x: indices of influence function  in relative position x local coordinates (float). 0 = top of the influence function
        y: indices of influence function  in relative position y local coordinates (float). 0 = top of the influence function

    :return:
        influ: (np.ndarray(dims=3,dtype=np.float64)) : cube of the IF for each actuator

    """
    irc = 1.16136 + 2.97422 * coupling + \
        (-13.2381) * coupling**2 + 20.4395 * coupling**3

    p1 = 4.49469 + (7.25509 + (-32.1948 + 17.9493 * coupling) * coupling) * coupling
    p2 = 2.49456 + (-0.65952 + (8.78886 - 6.23701 * coupling) * coupling) * coupling

    tmp_c = 1.0 / np.abs(irc)
    ccc = (coupling - 1. + tmp_c**p1) / (np.log(tmp_c) * tmp_c**p2)

    smallsize = int(np.ceil(2 * irc * pitch + 10))
    if (smallsize % 2 != 0):
        smallsize += 1
    # clip
    if (x is None or y is None):
        return smallsize
    else:
        # normalized coordiantes in local ref frame
        x = np.abs(x) / (irc * pitch)
        y = np.abs(y) / (irc * pitch)

        x[x < 1e-8] = 1e-8
        x[x > 2] = 2.
        y[y < 1e-8] = 1e-8
        y[y > 2] = 2.
        tmp = (1. - x**p1 + ccc * np.log(x) * x**p2) * \
            (1. - y**p1 + ccc * np.log(y) * y**p2)
        tmp = tmp * (x <= 1.0) * (y <= 1.0)
        return tmp


def makeRadialSchwartz(pitch: float, coupling: float, x=None, y=None):
    """Compute radial Schwartz influence function

    :parameters:
        pitch: (float) : pitch of the DM expressed in pixels
        coupling: (float) : coupling of the actuators
        x: indices of influence function  in relative position x local coordinates (float). 0 = top of the influence function
        y: indices of influence function  in relative position y local coordinates (float). 0 = top of the influence function

    :return:
        influ: (np.ndarray(dims=3,dtype=np.float64)) : cube of the IF for each actuator

    """
    k = 6  # order of the Schwartz function
    #
    a = pitch / np.sqrt(k / (np.log(coupling) - k) + 1.)
    smallsize = int(2 * np.ceil(a) + 2)
    if (x is None or y is None):
        return smallsize
    else:
        r = (x * x + y * y) / (a * a)
        ok = np.where(r < 1)
        sc = np.zeros(r.shape)
        sc[ok] = np.exp((k / (r[ok] - 1.0)) + k)
        #influ[:,:,:] = sc[:,:,None] * np.ones(ntotact)[None,None,:]
        return sc


def makeSquareSchwartz(pitch: float, coupling: float, x=None, y=None):
    """Compute Square Schwartz influence function

    :parameters:
        pitch: (float) : pitch of the DM expressed in pixels
        coupling: (float) : coupling of the actuators
        x: indices of influence function  in relative position x local coordinates (float). 0 = top of the influence function
        y: indices of influence function  in relative position y local coordinates (float). 0 = top of the influence function

    :return:
        influ: (np.ndarray(dims=3,dtype=np.float64)) : cube of the IF for each actuator

    """
    k = 6  # order of the Schwartz function
    #
    a = pitch / np.sqrt(k / (np.log(coupling) - k) + 1.)

    if (x is None or y is None):
        smallsize = int(2 * np.ceil(a) + 2)
        return smallsize
    else:
        xx = (x / a)**2
        yy = (y / a)**2
        ok = np.where((xx < 1) * (yy < 1))
        sc = np.zeros(xx.shape)
        sc[ok] = np.exp((k / (xx[ok] - 1)) + k) * \
            np.exp((k / (yy[ok] - 1)) + k)
        return sc


def makeBlacknutt(pitch: float, coupling: float, x=None, y=None):
    """Compute Blacknutt influence function
    Attention, ici on ne peut pas choisir la valeur de coupling.
    La variable a ete laissee dans le code juste pour compatibilité avec les
    autres fonctions, mais elle n'est pas utilisee.

    :parameters:
        pitch: (float): pitch of the DM expressed in pixels
        coupling: (float) : coupling of the actuators
        x: indices of influence function  in relative position x local coordinates (float). 0 = top of the influence function
        y: indices of influence function  in relative position y local coordinates (float). 0 = top of the influence function

    :return:
        influ: (np.ndarray(dims=3,dtype=np.float64)) : cube of the IF for each actuator

    """
    smallsize = int(np.ceil(4 * pitch + 1))
    if (x is None or y is None):
        return smallsize
    else:
        cg = smallsize // 2
        xx = x / float(cg)
        yy = y / float(cg)
        a = np.array([0.355768, 0.487396, 0.144232, 0.012604], dtype=np.float32)
        ok = np.where((np.abs(xx) < 1) * (np.abs(yy) < 1))
        sc = np.zeros(xx.shape)
        sc[ok] = (a[0] + a[1] * np.cos(np.pi * xx[ok]) +
                  a[2] * np.cos(2 * np.pi * xx[ok]) + a[3] * np.cos(3 * np.pi * xx[ok])) *\
            (a[0] + a[1] * np.cos(np.pi * yy[ok]) +
             a[2] * np.cos(2 * np.pi * yy[ok]) + a[3] * np.cos(3 * np.pi * yy[ok]))

        return sc


def makeGaussian(pitch: float, coupling: float, x=None, y=None):
    """Compute Gaussian influence function. Coupling parameter is not taken into account

    :parameters:
        pitch: (float) : pitch of the DM expressed in pixels
        coupling: (float) : coupling of the actuators
        x: indices of influence function  in relative position x local coordinates (float). 0 = top of the influence function
        y: indices of influence function  in relative position y local coordinates (float). 0 = top of the influence function

    :return:
        influ: (np.ndarray(dims=3,dtype=np.float64)) : cube of the IF for each actuator

    """
    irc = 1.16136 + 2.97422 * coupling + \
        (-13.2381) * coupling**2 + 20.4395 * coupling**3
    tmp_c = 1.0 / np.abs(irc)

    smallsize = int(np.ceil(2 * irc * pitch + 10))
    if (smallsize % 2 != 0):
        smallsize += 1

    if (x is None or y is None):
        return smallsize
    else:
        xdg = np.linspace(-1, 1, smallsize, dtype=np.float32)
        x = np.tile(xdg, (smallsize, 1))
        y = x.T
        sig = 0.8
        gauss = 1 / np.cos(np.exp(-(x**2 / sig + y**2 / sig))**2)
        # Force value at zero on array limits
        gauss -= gauss[gauss.shape[0] / 2.].min()
        gauss[gauss < 0.] = 0
        gauss /= gauss.max()  # Normalize
        return gauss


def makeBessel(pitch: float, coupling: float, x: np.ndarray=None, y: np.ndarray=None,
               patternType: bytes=PatternType.SQUARE):
    """Compute Bessel influence function

    :parameters:
        pitch: (float) : pitch of the DM expressed in pixels
        coupling: (float) : coupling of the actuators
        x: indices of influence function  in relative position x local coordinates (float). 0 = top of the influence function
        y: indices of influence function  in relative position y local coordinates (float). 0 = top of the influence function

    :return:
        influ: (np.ndarray(dims=3,dtype=np.float64)) : cube of the IF for each actuator

    """
    smallsize = int(np.ceil(pitch * 3.2))

    if (x is None or y is None):
        return smallsize
    else:
        # size_pitch = smallsize/np.float32(p_dm._pitch) # size of influence fonction in pitch
        # xdg= np.linspace(-1*(size_pitch/3.2),size_pitch/3.2, smallsize,dtype=np.float32)
        # x = np.tile(xdg, (smallsize,1))
        # y = x.T
        influ_u = bessel_influence(x / (1.6 * pitch), y / (1.6 * pitch), patternType)
        influ_u = influ_u * (influ_u >= 0)
        return influ_u
