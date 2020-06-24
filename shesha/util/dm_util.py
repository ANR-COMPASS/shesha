## @package   shesha.util.dm_util
## @brief     Utilities function for DM geometry initialization
## @author    COMPASS Team <https://github.com/ANR-COMPASS>
## @version   4.4.2
## @date      2011/01/28
## @copyright GNU Lesser General Public License
#
#  This file is part of COMPASS <https://anr-compass.github.io/compass/>
#
#  Copyright (C) 2011-2019 COMPASS Team <https://github.com/ANR-COMPASS>
#  All rights reserved.
#  Distributed under GNU - LGPL
#
#  COMPASS is free software: you can redistribute it and/or modify it under the terms of the GNU Lesser
#  General Public License as published by the Free Software Foundation, either version 3 of the License,
#  or any later version.
#
#  COMPASS: End-to-end AO simulation tool using GPU acceleration
#  The COMPASS platform was designed to meet the need of high-performance for the simulation of AO systems.
#
#  The final product includes a software package for simulating all the critical subcomponents of AO,
#  particularly in the context of the ELT and a real-time core based on several control approaches,
#  with performances consistent with its integration into an instrument. Taking advantage of the specific
#  hardware architecture of the GPU, the COMPASS tool allows to achieve adequate execution speeds to
#  conduct large simulation campaigns called to the ELT.
#
#  The COMPASS platform can be used to carry a wide variety of simulations to both testspecific components
#  of AO of the E-ELT (such as wavefront analysis device with a pyramid or elongated Laser star), and
#  various systems configurations such as multi-conjugate AO.
#
#  COMPASS is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the
#  implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
#  See the GNU Lesser General Public License for more details.
#
#  You should have received a copy of the GNU Lesser General Public License along with COMPASS.
#  If not, see <https://www.gnu.org/licenses/lgpl-3.0.txt>.

import numpy as np

import shesha.constants as scons
from shesha.constants import CONST

from . import utilities as util

from typing import List, Union


def dim_dm_support(cent: float, extent: int, ssize: int):
    """ Compute the DM support dimensions

    :parameters:

        cent : (float): center of the pupil

        extent: (float): size of the DM support

        ssize: (int): size of ipupil support
    """
    n1 = np.floor(cent - extent / 2)
    n2 = np.ceil(cent + extent / 2)
    if (n1 < 1):
        n1 = 1
    if (n2 > ssize):
        n2 = ssize

    return int(n1), int(n2)


def dim_dm_patch(pupdiam: int, diam: float, type: bytes, alt: float,
                 xpos_wfs: List[float], ypos_wfs: List[float]):
    """ compute patchDiam for DM

    :parameters:

        pupdiam: (int) : pupil diameter

        diam: (float) : telescope diameter

        type: (bytes) : type of dm

        alt: (float) : altitude of dm

        xpos_wfs: (list) : list of wfs xpos

        ypos_wfs: (list) : list of wfs ypos
    """

    if len(xpos_wfs) == 0:
        norms = 0.  # type: Union[float, List[float]]
    else:
        norms = [
                np.linalg.norm([xpos_wfs[w], ypos_wfs[w]]) for w in range(len(xpos_wfs))
        ]
    if ((type == scons.DmType.PZT) or (type == scons.DmType.TT)):
        pp = (diam / pupdiam)
    elif (type == scons.DmType.KL):
        pp = (pupdiam)
    else:
        raise TypeError("This type of DM doesn't exist ")

    patchDiam = int(pupdiam + 2 * np.max(norms) * CONST.ARCSEC2RAD * np.abs(alt) / (pp))
    return patchDiam


def createSquarePattern(pitch: float, nxact: int):
    """
    Creates a list of M=nxact^2 actuator positions spread over an square grid.
    Coordinates are centred around (0,0).

    :parameters:

        pitch: (float) : distance in pixels between 2 adjacent actus

        nxact: (int) : number of actu across the pupil diameter

    :return:

        xy: (np.ndarray(dims=2,dtype=np.float32)) : xy[M,2] list of coodinates
    """

    xy = np.tile(np.arange(nxact) - (nxact - 1.) / 2., (nxact, 1)).astype(np.float32)
    xy = np.array([xy.flatten(), xy.T.flatten()]) * pitch
    xy = np.float32(xy)
    return xy


def createHexaPattern(pitch: float, supportSize: int):
    """
    Creates a list of M actuator positions spread over an hexagonal grid.
    The number M is the number of points of this grid, it cannot be
    known before the procedure is called.
    Coordinates are centred around (0,0).
    The support that limits the grid is a square [-supportSize/2, supportSize/2].

    :parameters:

        pitch: (float) : distance in pixels between 2 adjacent actus

        supportSize: (int) : size in pixels of the support over which the coordinate list
             should be returned.

    :return:

        xy: (np.ndarray(dims=2,dtype=np.float32)) : xy[2,M] list of coordinates
    """
    V3 = np.sqrt(3)
    nx = int(np.ceil((supportSize / 2.0) / pitch) + 1)
    x = pitch * (np.arange(2 * nx + 1, dtype=np.float32) - nx)
    Nx = x.shape[0]
    ny = int(np.ceil((supportSize / 2.0) / pitch / V3) + 1)
    y = (V3 * pitch) * (np.arange(2 * ny + 1, dtype=np.float32) - ny)
    Ny = y.shape[0]
    x = np.tile(x, (Ny, 1)).flatten()
    y = np.tile(y, (Nx, 1)).T.flatten()
    x = np.append(x, x + pitch / 2.)
    y = np.append(y, y + pitch * V3 / 2.)
    xy = np.float32(np.array([y, x]))
    return xy


def createDoubleHexaPattern(pitch: float, supportSize: int, pupAngleDegree: float):
    """
    Creates a list of M actuator positions spread over an hexagonal grid.
    The number M is the number of points of this grid, it cannot be
    known before the procedure is called.
    Coordinates are centred around (0,0).
    The support of the grid is a square [-supportSize/2,vsupportSize/2].

    :parameters:

        pitch: (float) : distance in pixels between 2 adjacent actus
        supportSize: (int) : size in pixels of the support over which the coordinate list
             should be returned.
        pupAngleDegree: (float) : Rotation angle of the DM

    :return:

        xy: (np.ndarray(dims=2,dtype=np.float32)) : xy[2,M] list of coodinates
    """
    # ici on cree le pattern hexa. Par simplicite on replique le meme code
    # que dans createHexaPattern(). On a donc un reseau "pointe selon X"
    V3 = np.sqrt(3)
    pi = np.pi
    ny = int(np.ceil((supportSize / 2.0) / pitch) + 1)
    y = pitch * (np.arange(2 * ny + 1, dtype=np.float32) - ny)
    nx = int(np.ceil((supportSize / 2.0) / pitch / V3) + 1)
    x = (V3 * pitch) * (np.arange(2 * nx + 1, dtype=np.float32) - nx)
    # LA ligne de code qui change tout: on shifte le reseau de 1 pitch en X.
    x = x + pitch
    x, y = np.meshgrid(x, y, indexing='ij')
    x = x.flatten()
    y = y.flatten()
    x = np.append(x, x + pitch * V3 / 2.)
    y = np.append(y, y + pitch / 2.)

    # classement dans l'ordre kivabien
    u = x + 1e-3 * y
    idx = np.argsort(u)
    x = x[idx]
    y = y[idx]

    # on selection 1/6ieme du reseau, entre -30 et 30 degres
    th = np.arctan2(y, x)
    nn = np.where(((th < pi / 6) & (th > -pi / 6)))
    x = x[nn]
    y = y[nn]

    # on va maintenant repliquer ce reseau 6 fois, et le rotationnant a chaque
    # fois de 60°. Note:
    X = np.array([])
    Y = np.array([])
    for k in range(6):
        xx = np.cos(k * pi / 3) * x + np.sin(k * pi / 3) * y
        yy = -np.sin(k * pi / 3) * x + np.cos(k * pi / 3) * y
        X = np.r_[X, xx]
        Y = np.r_[Y, yy]

    # Rotation matrices pour suivre l'angle pupille
    rot = pupAngleDegree * np.pi / 180.0
    mrot = np.array([[np.cos(rot), -np.sin(rot)], [np.sin(rot), np.cos(rot)]])
    XY = np.dot(mrot, [X, Y])
    return np.float32(XY)


def filterActuWithPupil(actuPos: np.ndarray, pupil: np.ndarray,
                        threshold: float) -> np.ndarray:
    '''
        Select actuators based on their distance to the nearest pupil pixel
        The implementation proposed here is easy but limits it precision to
        an integer roundoff of the threshold

        actuPos: 2 x nActu np.array[float]: actuator position list - pupil pixel units
        pupil: nPup x nPup np.ndarray[bool]: pupil mask
        threshold: float: max allowed distance - pupil pixel units
    '''

    # Gen a dilation mask of expected size
    from scipy.ndimage.morphology import binary_dilation
    k = np.ceil(threshold)
    i, j = np.meshgrid(np.arange(2 * k + 1), np.arange(2 * k + 1), indexing='ij')
    disk = ((i - k)**2 + (j - k)**2)**.5 <= k
    dilatedPupil = binary_dilation(pupil, disk)
    actuIsIn = dilatedPupil[(np.round(actuPos[0]).astype(np.int32),
                             np.round(actuPos[1]).astype(np.int32))]
    return actuPos[:, actuIsIn]


def select_actuators(xc: np.ndarray, yc: np.ndarray, nxact: int, pitch: int, cobs: float,
                     margin_in: float, margin_out: float, N=None):
    """
    Select the "valid" actuators according to the system geometry

    :parameters:

        xc: actuators x positions (origine in center of mirror)

        yc: actuators y positions (origine in center of mirror)

        nxact:

        pitch:

        cobs:

        margin_in:

        margin_out:

        N:

    :return:

        liste_fin: actuator indice selection for xpos/ypos


    """
    # the following determine if an actuator is to be considered or not
    # relative to the pitchmargin parameter.
    dis = np.sqrt(xc**2 + yc**2)

    # test Margin_in
    rad_in = (((nxact - 1) / 2) * cobs - margin_in) * pitch

    if N is None:
        if (margin_out is None):
            margin_out = 1.44
        rad_out = ((nxact - 1.) / 2. + margin_out) * pitch

        valid_actus = np.where((dis <= rad_out) * (dis >= rad_in))[0]

    else:
        valid_actus = np.where(dis >= rad_in)[0]
        indsort = np.argsort(dis[valid_actus])

        if (N > valid_actus.size):
            print('Too many actuators wanted, restricted to ', valid_actus.size)
        else:
            valid_actus = np.sort(indsort[:N])

    return valid_actus


def make_zernike(nzer: int, size: int, diameter: int, xc=-1., yc=-1., ext=0):
    """Compute the zernike modes

    :parameters:

        nzer: (int) : number of modes

        size: (int) : size of the screen

        diameter: (int) : pupil diameter

        xc: (float) : (optional) x-position of the center

        yc: (float) : (optional) y-position of the center

        ext: (int) : (optional) extension

    :return:

        z : (np.ndarray(ndims=3,dtype=np.float64)) : zernikes modes
    """
    m = 0
    n = 0

    if (xc == -1):
        xc = size / 2
    if (yc == -1):
        yc = size / 2

    radius = (diameter + 1.) / 2.
    zr = util.dist(size, xc, yc).astype(np.float32).T / radius
    zmask = np.zeros((zr.shape[0], zr.shape[1], nzer), dtype=np.float32)
    zmaskmod = np.zeros((zr.shape[0], zr.shape[1], nzer), dtype=np.float32)

    zmask[:, :, 0] = (zr <= 1).astype(np.float32)
    zmaskmod[:, :, 0] = (zr <= 1.2).astype(np.float32)

    for i in range(1, nzer):
        zmask[:, :, i] = zmask[:, :, 0]
        zmaskmod[:, :, i] = zmaskmod[:, :, 0]

    zrmod = zr * zmaskmod[:, :, 0]

    zr = zr * zmask[:, :, 0]

    x = np.tile(np.linspace(1, size, size).astype(np.float32), (size, 1))
    zteta = np.arctan2(x - yc, x.T - xc).astype(np.float32)

    z = np.zeros((size, size, nzer), dtype=np.float32)

    for zn in range(nzer):
        n, m = zernumero(zn + 1)

        if ext:
            for i in range((n - m) // 2 + 1):
                z[:, :, zn] = z[:, :, zn] + (-1.) ** i * zrmod ** (n - 2. * i) * float(np.math.factorial(n - i)) / \
                    float(np.math.factorial(i) * np.math.factorial((n + m) / 2 - i) *
                          np.math.factorial((n - m) / 2 - i))
        else:
            for i in range((n - m) // 2 + 1):
                z[:, :, zn] = z[:, :, zn] + (-1.) ** i * zr ** (n - 2. * i) * float(np.math.factorial(n - i)) / \
                    float(np.math.factorial(i) * np.math.factorial((n + m) / 2 - i) *
                          np.math.factorial((n - m) / 2 - i))

        if ((zn + 1) % 2 == 1):
            if (m == 0):
                z[:, :, zn] = z[:, :, zn] * np.sqrt(n + 1.)
            else:
                z[:, :, zn] = z[:, :, zn] * \
                    np.sqrt(2. * (n + 1)) * np.sin(m * zteta)
        else:
            if (m == 0):
                z[:, :, zn] = z[:, :, zn] * np.sqrt(n + 1.)
            else:
                z[:, :, zn] = z[:, :, zn] * \
                    np.sqrt(2. * (n + 1)) * np.cos(m * zteta)

    if (ext):
        return z * zmaskmod
    else:
        return z * zmask


def zernumero(zn: int):
    """
    Returns the radial degree and the azimuthal number of zernike
    number zn, according to Noll numbering (Noll, JOSA, 1976)

    :parameters:

        zn: (int) : zernike number

    :returns:

        rd: (int) : radial degrees

        an: (int) : azimuthal numbers

    """
    j = 0
    for n in range(101):
        for m in range(n + 1):
            if ((n - m) % 2 == 0):
                j = j + 1
                if (j == zn):
                    return n, m
                if (m != 0):
                    j = j + 1
                    if (j == zn):
                        return n, m
