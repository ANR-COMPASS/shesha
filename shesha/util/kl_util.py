## @package   shesha.util.kl_util
## @brief     Functions for DM KL initialization
## @author    COMPASS Team <https://github.com/ANR-COMPASS>
## @version   5.5.0
## @date      2022/01/24
## @copyright GNU Lesser General Public License
#
#  This file is part of COMPASS <https://anr-compass.github.io/compass/>
#
#  Copyright (C) 2011-2023 COMPASS Team <https://github.com/ANR-COMPASS>
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

import shesha.constants as scons
import numpy as np
from scipy import interpolate

from typing import Tuple


def make_radii(cobs: float, nr: int) -> float:
    """ TODO: docstring

        Args:

            cobs: (float) : central obstruction

            nr : (int) :
    """
    d = (1. - cobs * cobs) / nr
    rad2 = cobs**2 + d / 16. + d * (np.arange(nr, dtype=np.float32))
    radp = np.sqrt(rad2)
    return radp


def make_kernels(cobs: float, nr: int, radp: np.ndarray, kl_type: bytes,
                 outscl: float = 3.) -> np.ndarray:
    """
    This routine generates the kernel used to find the KL modes.
    The  kernel constructed here should be simply a discretization
    of the continuous kernel. It needs rescaling before it is treated
    as a matrix for finding  the eigen-values. The outer scale
    should be in units of the diameter of the telescope.

    TODO:

    Args:

        cobs : (float): central obstruction

        nr : (int) :

        radp : (float) :

        kl_type : (bytes) : "kolmo" or "karman"

        outscl : (float) : outter scale for Von Karman spectrum

    :return:

        kers :
    """
    nth = 5 * nr
    kers = np.zeros((nth, nr, nr), dtype=np.float32)
    cth = np.cos((np.arange(nth, dtype=np.float32)) * (2. * np.pi / nth))
    dth = 2. * np.pi / nth
    fnorm = -1. / (2 * np.pi * (1. - cobs**2)) * 0.5
    # the 0.5 is to give  the r**2 kernel, not the r kernel
    for i in range(nr):
        for j in range(i + 1):
            te = 0.5 * np.sqrt(radp[i]**2 + radp[j]**2 - (2 * radp[i] * radp[j]) * cth)
            # te in units of the diameter, not the radius
            if (kl_type == scons.KLType.KOLMO):

                te = 6.88 * te**(5. / 3.)

            elif (kl_type == scons.KLType.KARMAN):

                te = 6.88 * te**(5. / 3.) * (1 - 1.485 * (te / outscl)**
                                             (1. / 3.) + 5.383 * (te / outscl)**
                                             (2) - 6.281 * (te / outscl)**(7. / 3.))

            else:

                raise TypeError("kl type unknown")

            f = np.fft.fft(te, axis=-1)
            kelt = fnorm * dth * np.float32(f.real)
            kers[:, i, j] = kers[:, j, i] = kelt
    return kers


def piston_orth(nr: int) -> np.ndarray:
    """ TODO: docstring

        Args:

            nr:

        :return:

            s:
    """
    s = np.zeros((nr, nr), dtype=np.float32)  # type: np.ndarray[np.float32]
    for j in range(nr - 1):
        rnm = 1. / np.sqrt(np.float32((j + 1) * (j + 2)))
        s[0:j + 1, j] = rnm
        s[j + 1, j] = -1 * (j + 1) * rnm

    rnm = 1. / np.sqrt(nr)
    s[:, nr - 1] = rnm
    return s


def make_azimuth(nord: int, npp: int) -> np.ndarray:
    """ TODO: docstring

        Args:

            nord:

            npp:

        :return:

            azbas:
    """

    azbas = np.zeros((npp, np.int32(1 + nord)), dtype=np.float32)
    th = np.arange(npp, dtype=np.float32) * (2. * np.pi / npp)

    azbas[:, 0] = 1.0
    for i in np.arange(1, nord, 2):
        azbas[:, np.int32(i)] = np.cos((np.int32(i) // 2 + 1) * th)
    for i in np.arange(2, nord, 2):
        azbas[:, np.int32(i)] = np.sin((np.int32(i) // 2) * th)

    return azbas


def radii(nr: int, npp: int, cobs: float) -> np.ndarray:
    """
    This routine generates an nr x npp array with npp copies of the
    radial coordinate array. Radial coordinate span the range from
    r=cobs to r=1 with successive annuli having equal areas (ie, the
    area between cobs and 1 is divided into nr equal rings, and the
    points are positioned at the half-area mark on each ring). There
    are no points on the border.

    TODO:

        Args:

            nr:

            npp:

            cobs: (float) : central obstruction

        :return:

            r
    """

    r2 = cobs**2 + (np.arange(nr, dtype=np.float32) + 0.) / nr * (1.0 - cobs**2)
    rs = np.sqrt(r2)
    r = np.transpose(np.tile(rs, (npp, 1)))

    return r


#__________________________________________________________________________
#__________________________________________________________________________


def polang(r: np.ndarray) -> np.ndarray:
    """
    This routine generates an array with the same dimensions as r,
    but containing the azimuthal values for a polar coordinate system.

    TODO:

        Args:

            r:

        :return:

            p:
    """
    s = r.shape
    nr = s[0]
    np1 = s[1]
    phi1 = np.arange(np1, dtype=np.float32) / float(np1) * 2. * np.pi
    p1, p2 = np.meshgrid(np.ones(nr), phi1)
    p = np.transpose(p2)

    return p


#__________________________________________________________________________
#__________________________________________________________________________


def setpincs(ax: np.ndarray, ay: np.ndarray, px: np.ndarray, py: np.ndarray,
             cobs: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    This routine determines a set of squares for interpolating
    from cartesian to polar coordinates, using only those points
    with cobs < r < 1
    SEE ALSO : pcgeom

    TODO:

        Args:

            ax:

            ay:

            px:

            py:

            cobs: (float) : central obstruction

        :return:

            pincx:

            pincy:

            pincw
    """

    s = ax.shape
    nc = s[0]
    # s = px.shape# not used
    # nr = s[0]# not used
    # npp = s[1]# not used
    dcar = (ax[nc - 1, 0] - ax[0, 0]) / (nc - 1)
    ofcar = ax[0, 0]
    rlx = (px - ofcar) / dcar
    rly = (py - ofcar) / dcar
    lx = np.int32(rlx)
    ly = np.int32(rly)
    shx = rlx - lx
    shy = rly - ly

    pincx = np.zeros((4, lx.shape[0], lx.shape[1]))
    pincx[[1, 2], :, :] = lx + 1
    pincx[[0, 3], :, :] = lx

    pincy = np.zeros((4, ly.shape[0], ly.shape[1]))
    pincy[[0, 1], :, :] = ly
    pincy[[2, 3], :, :] = ly + 1

    pincw = np.zeros((4, shx.shape[0], shx.shape[1]))
    pincw[0, :, :] = (1 - shx) * (1 - shy)
    pincw[1, :, :] = shx * (1 - shy)
    pincw[2, :, :] = shx * shy
    pincw[3, :, :] = (1 - shx) * shy

    axy = ax**2 + ay**2
    axyinap = np.clip(axy, cobs**2. + 1.e-3, 0.999)
    # sizeaxyinap=axyinap.shape[1]# not used

    # pincw = pincw*axyinap[pincx+(pincy-1)*sizeaxyinap] --->

    for z in range(pincw.shape[0]):
        for i in range(pincw.shape[1]):
            for j in range(pincw.shape[2]):
                pincw[z, i, j] = pincw[z, i, j] * axyinap[np.int32(pincx[z, i, j]),
                                                          np.int32(pincy[z, i, j])]

    pincw = pincw * np.tile(1.0 / np.sum(pincw, axis=0), (4, 1, 1))

    return pincx, pincy, pincw


def pcgeom(nr, npp, cobs, ncp, ncmar):
    """
    This routine builds a geom_struct. px and py are the x and y
    coordinates of points in the polar arrays.  cr and cp are the
    r and phi coordinates of points in the cartesian grids. ncmar
    allows the possibility that there is a margin of ncmar points
    in the cartesian arrays outside the region of interest

    TODO:

        Args:

            nr:

            npp:

            cobs: (float) : central obstruction

            ncp:

            ncmar:

        :returns:

            ncp:

            ncmar:

            px:

            py:

            cr:

            cp:

            pincx:

            pincy:

            pincw:

            ap:
    """
    nused = ncp - 2 * ncmar
    ff = 0.5 * nused
    hw = np.float32(ncp - 1) / 2.

    r = radii(nr, npp, cobs)
    p = polang(r)

    px0 = r * np.cos(p)
    py0 = r * np.sin(p)
    px = ff * px0 + hw
    py = ff * py0 + hw
    ax = np.reshape(
            np.arange(int(ncp)**2, dtype=np.float32) + 1, (int(ncp), int(ncp)), order='F')
    ax = np.float32(ax - 1) % ncp - 0.5 * (ncp - 1)
    ax = ax / (0.5 * nused)
    ay = np.transpose(ax)

    pincx, pincy, pincw = setpincs(ax, ay, px0, py0, cobs)

    dpi = 2 * np.pi
    cr2 = (ax**2 + ay**2)
    ap = np.clip(cr2, cobs**2 + 1.e-3, 0.999)
    #cr = (cr2 - cobs**2) / (1 - cobs**2) * nr - 0.5;
    cr = (cr2 - cobs**2) / (1 - cobs**2) * nr
    cp = (np.arctan2(ay, ax) + dpi) % dpi
    cp = (npp / dpi) * cp

    cr = np.clip(cr, 1.e-3, nr - 1.001)
    # fudge -----, but one of the less bad ones
    cp = np.clip(cp, 1.e-3, npp - 1.001)
    # fudge -----  this is the line which
    # gives that step in the cartesian grid
    # at phi = 0.
    return ncp, ncmar, px, py, cr, cp, pincx, pincy, pincw, ap


def set_pctr(dim: int, nr, npp, nkl: int, cobs: float, nord, ncmar=None, ncp=None):
    """
    This routine calls pcgeom to build a geom_struct with the
    right initializations. bas is a gkl_basis_struct built with
    the gkl_bas routine.
    TODO:

    Args:

        dim:

        nr:

        npp:

        nkl:

        cobs:

        nord:

        ncmar: (optional)

        ncp: (optional)

    :returns:

        ncp

        ncmar

        px

        py

        cr

        cp

        pincx

        pincy

        pincw

        ap
    """
    ncp = dim
    if (ncmar == None):
        ncmar = 2
    if (ncp == None):
        ncp = 128
    ncp, ncmar, px, py, cr, cp, pincx, pincy, pincw, ap = pcgeom(
            nr, npp, cobs, ncp, ncmar)
    return ncp, ncmar, px, py, cr, cp, pincx, pincy, pincw, ap


def gkl_fcom(kers: np.ndarray, cobs: float, nf: int):
    """
    This routine does the work : finding the eigenvalues and
    corresponding eigenvectors. Sort them and select the right
    one. It returns the KL modes : in polar coordinates : rabas
    as well as the associated variance : evals. It also returns
    a bunch of indices used to recover the modes in cartesian
    coordinates (nord, npo and ordd).

    Args:

        kerns : (np.ndarray[ndim= ,dtype=np.float32]) :

        cobs : (float) : central obstruction

        nf : (int) :
    """
    nkl = nf
    st = kers.shape
    nr = st[1]
    nt = st[0]
    nxt = 0
    fktom = (1. - cobs**2) / nr

    evs = np.zeros((nr, nt), dtype=np.float32)
    # ff isnt used - the normalisation for
    # the eigenvectors is straightforward:
    # integral of surface**2 divided by area = 1,
    # and the cos**2 term gives a factor
    # half, so multiply zero order by
    # sqrt(n) and the rest by sqrt (2n)

    # zero order is a special case...
    # need to deflate to eliminate infinite eigenvalue - actually want
    # evals/evecs of zom - b where b is big and negative
    zom = kers[0, :, :]
    s = piston_orth(nr)

    ts = np.transpose(s)
    # b1 = ((ts(,+)*zom(+,))(,+)*s(+,))(1:nr-1, 1:nr-1)
    btemp = (ts.dot(zom).dot(s))[0:nr - 1, 0:nr - 1]

    #newev = SVdec(fktom*b1,v0,vt)
    v0, newev, vt = np.linalg.svd(
            fktom * btemp, full_matrices=True
    )  # type: np.ndarray[np.float32], np.ndarray[np.float32],np.ndarray[np.float32]

    v1 = np.zeros((nr, nr), dtype=np.float32)
    v1[0:nr - 1, 0:nr - 1] = v0
    v1[nr - 1, nr - 1] = 1

    vs = s.dot(v1)
    newev = np.concatenate((newev, [0]))
    # print(np.size(newev))
    evs[:, nxt] = np.float32(newev)
    kers[nxt, :, :] = np.sqrt(nr) * vs

    nxt = 1
    while True:
        vs, newev, vt = np.linalg.svd(fktom * kers[nxt, :, :], full_matrices=True)
        # newev = SVdec(fktom*kers(,,nxt),vs,vt)
        evs[:, nxt] = np.float32(newev)
        kers[nxt, :, :] = np.sqrt(2. * nr) * vs
        mxn = max(np.float32(newev))
        egtmxn = np.floor(evs[:, 0:nxt + 1] > mxn)
        nxt = nxt + 1
        if ((2 * np.sum(egtmxn) - np.sum(egtmxn[:, 0])) >= nkl):
            break

    nus = nxt - 1
    kers = kers[0:nus + 1, :, :]

    #evs = reform (evs [:, 1:nus], nr*(nus))

    evs = np.reshape(evs[:, 0:nus + 1], nr * (nus + 1), order='F')
    a = np.argsort(-1. * evs)[0:nkl]

    # every eigenvalue occurs twice except
    # those for the zeroth order mode. This
    # could be done without the loops, but
    # it isn't the stricking point anyway...

    no = 0
    ni = 0
    #oind = array(long,nf+1)
    oind = np.zeros(nkl + 1, dtype=np.int32)

    while True:
        if (a[ni] < nr):
            oind[no] = a[ni]
            no = no + 1
        else:
            oind[no] = a[ni]
            oind[no + 1] = a[ni]
            no = no + 2

        ni = ni + 1
        if (no >= (nkl)):
            break

    oind = oind[0:nkl]
    tord = (oind) // nr + 1

    odd = np.arange(nkl, dtype=np.int32) % 2
    pio = (oind) % nr + 1

    evals = evs[oind]
    ordd = 2 * (tord - 1) - np.floor((tord > 1) & (odd)) + 1

    nord = max(ordd)

    rabas = np.zeros((nr, nkl), dtype=np.float32)
    sizenpo = int(nord)
    npo = np.zeros(sizenpo, dtype=np.int32)

    for i in range(nkl):
        npo[np.int32(ordd[i]) - 1] = npo[np.int32(ordd[i]) - 1] + 1
        rabas[:, i] = kers[tord[i] - 1, :, pio[i] - 1]

    return evals, nord, npo, ordd, rabas


#-------------------------------------------------------------------------
# function for calculate DM_kl on python


def gkl_sfi(p_dm, i):
    #DOCUMENT
    #This routine returns the i'th function from the generalised KL
    #basis bas. bas must be generated first with gkl_bas.
    nr = p_dm._nr
    npp = p_dm._npp
    ordd = p_dm._ord
    rabas = p_dm._rabas
    azbas = p_dm._azbas
    nkl = p_dm.nkl

    if (i > nkl - 1):
        raise TypeError("kl funct order it's so big")

    else:

        ordi = np.int32(ordd[i])
        rabasi = rabas[:, i]
        azbasi = np.transpose(azbas)
        azbasi = azbasi[ordi, :]

        sf1 = np.zeros((nr, npp), dtype=np.float64)
        for j in range(npp):
            sf1[:, j] = rabasi

        sf2 = np.zeros((npp, nr), dtype=np.float64)
        for j in range(nr):
            sf2[:, j] = azbasi

        sf = sf1 * np.transpose(sf2)

        return sf


def pol2car(pol, p_dm, mask=0):
    # DOCUMENT cart=pol2car(cpgeom, pol, mask=)
    # This routine is used for polar to cartesian conversion.
    # pol is built with gkl_bas and cpgeom with pcgeom.
    # However, points not in the aperture are actually treated
    # as though they were at the first or last radial polar value
    # -- a small fudge, but not serious  ?*******
    #cd = interpolate.interp2d(cr, cp,pol)
    ncp = p_dm._ncp
    cr = p_dm._cr
    cp = p_dm._cp
    nr = p_dm._nr
    npp = p_dm._npp

    r = np.arange(nr, dtype=np.float64)
    phi = np.arange(npp, dtype=np.float64)
    tab_phi, tab_r = np.meshgrid(phi, r)
    tab_x = (tab_r / (nr)) * np.cos((tab_phi / (npp)) * 2 * np.pi)
    tab_y = (tab_r / (nr)) * np.sin((tab_phi / (npp)) * 2 * np.pi)

    newx = np.linspace(-1, 1, ncp)
    newy = np.linspace(-1, 1, ncp)
    tx, ty = np.meshgrid(newx, newy)

    cd = interpolate.griddata((tab_r.flatten(), tab_phi.flatten()), pol.flatten(),
                              (cr, cp), method='cubic')
    cdf = interpolate.griddata((tab_r.flatten("F"), tab_phi.flatten("F")),
                               pol.flatten("F"), (cr, cp), method='cubic')
    cdxy = interpolate.griddata((tab_y.flatten(), tab_x.flatten()), pol.flatten(),
                                (tx, ty), method='cubic')

    if (mask == 1):
        ap = p_dm.ap
        cd = cd * (ap)
        cdf = cdf * (ap)
        cdxy = cdxy * (ap)

    return cd, cdf, cdxy


def kl_view(p_dm, mask=1):

    nkl = p_dm.nkl
    ncp = p_dm._ncp

    tab_kl = np.zeros((nkl, ncp, ncp), dtype=np.float64)
    tab_klf = np.zeros((nkl, ncp, ncp), dtype=np.float64)
    tab_klxy = np.zeros((nkl, ncp, ncp), dtype=np.float64)

    for i in range(nkl):

        tab_kl[i, :, :], tab_klf[i, :, :], tab_klxy[i, :, :] = pol2car(
                gkl_sfi(p_dm, i), p_dm, mask)

    return tab_kl, tab_klf, tab_klxy
