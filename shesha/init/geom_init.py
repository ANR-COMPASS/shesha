## @package   shesha.init.geom_init
## @brief     Initialization of the system geometry and of the Telescope object
## @author    COMPASS Team <https://github.com/ANR-COMPASS>
## @version   4.4.0
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

import shesha.config as conf
import shesha.constants as scons
from shesha.constants import CONST

import shesha.util.make_pupil as mkP
import shesha.util.utilities as util
from shesha.sutra_wrap import carmaWrap_context, Telescope
from shesha.constants import ApertureType
import numpy as np


def tel_init(context: carmaWrap_context, p_geom: conf.Param_geom, p_tel: conf.Param_tel,
             r0=None, ittime=None, p_wfss=None, dm=None):
    """
        Initialize the overall geometry of the AO system, including pupil and WFS

    :parameters:
        context: (carmaWrap_context) : context
        p_geom: (Param_geom) : geom settings
        p_tel: (Param_tel) : telescope settings
        r0: (float) : atmos r0 @ 0.5 microns
        ittime: (float) : 1/loop frequency [s]
        p_wfss: (list of Param_wfs) : wfs settings
        dm: (list of Param_dm) : (optional) dms settings [=None]
    :return:
        telescope: (Telescope): Telescope object

    """
    if p_wfss is not None:
        # WFS geometry
        nsensors = len(p_wfss)

        any_sh = [o.type for o in p_wfss].count(scons.WFSType.SH) > 0
        # dm = None
        if (p_wfss[0].dms_seen is None and dm is not None):
            for i in range(nsensors):
                if (not p_wfss[i].openloop):
                    p_wfss[i].set_dms_seen(np.arange(len(dm), dtype=np.int32))

        # first get the wfs with max # of subaps
        # we'll derive the geometry from the requirements in terms of sampling
        if (any_sh):
            indmax = np.argsort([o.nxsub for o in p_wfss
                                 if o.type == scons.WFSType.SH])[-1]
        else:
            indmax = np.argsort([o.nxsub for o in p_wfss])[-1]

        print("*-----------------------")
        print("Computing geometry of WFS", indmax)

        init_wfs_geom(p_wfss[indmax], r0, p_tel, p_geom, ittime, verbose=1)
        # #do the same for other wfs
        for i in range(nsensors):
            if (i != indmax):
                print("*-----------------------")
                print("Computing geometry of WFS", i)
                init_wfs_geom(p_wfss[i], r0, p_tel, p_geom, ittime, verbose=1)

    else:
        geom_init(p_geom, p_tel)

    telescope = Telescope(context, p_geom._spupil.shape[0],
                          np.where(p_geom._spupil > 0)[0].size,
                          (p_geom._spupil * p_geom._apodizer).astype(np.float32),
                          p_geom._mpupil.shape[0], p_geom._mpupil)

    if (p_geom._phase_ab_M1 is not None):
        telescope.set_phase_ab_M1(p_geom._phase_ab_M1)
        telescope.set_phase_ab_M1_m(p_geom._phase_ab_M1_m)

    return telescope


def init_wfs_geom(p_wfs: conf.Param_wfs, r0: float, p_tel: conf.Param_tel,
                  p_geom: conf.Param_geom, ittime: float, verbose=1):
    """Compute the geometry of WFSs: valid subaps, positions of the subaps,
    flux per subap, etc...

    :parameters:
        p_wfs: (Param_wfs) : wfs settings

        r0: (float) : atmos r0 @ 0.5 microns

        p_tel: (Param_tel) : telescope settings

        geom: (Param_geom) : geom settings

        ittime: (float) : 1/loop frequency [s]

        verbose: (int) : (optional) display informations if 0


    """

    if p_geom.pupdiam:
        if p_wfs.type == scons.WFSType.SH or p_wfs.type == scons.WFSType.PYRHR or p_wfs.type == scons.WFSType.PYRLR:
            pdiam = p_geom.pupdiam // p_wfs.nxsub
            if (p_geom.pupdiam % p_wfs.nxsub > 0):
                pdiam += 1

    else:
        pdiam = -1

    p_wfs._pdiam = pdiam

    init_wfs_size(p_wfs, r0, p_tel, verbose)

    if not p_geom.is_init:
        # this is the wfs with largest # of subaps
        # the overall geometry is deduced from it
        #if not p_geom.pupdiam:
        if p_geom.pupdiam != 0 and p_geom.pupdiam != p_wfs._pdiam * p_wfs.nxsub:
            print("WARNING: pupdiam set value not correct")
        p_geom.pupdiam = p_wfs._pdiam * p_wfs.nxsub
        print("pupdiam used: ", p_geom.pupdiam)
        if p_wfs.type == scons.WFSType.PYRHR or p_wfs.type == scons.WFSType.PYRLR:
            geom_init(p_geom, p_tel, padding=p_wfs._nrebin)
        elif (p_wfs.type == scons.WFSType.SH):
            geom_init(p_geom, p_tel)
        else:
            raise RuntimeError("This WFS can not be used")

    if (p_wfs.type == scons.WFSType.PYRHR or p_wfs.type == scons.WFSType.PYRLR):
        init_pyrhr_geom(p_wfs, r0, p_tel, p_geom, ittime, verbose=1)
    elif (p_wfs.type == scons.WFSType.SH):
        init_sh_geom(p_wfs, r0, p_tel, p_geom, ittime, verbose=1)
    else:
        raise RuntimeError("This WFS can not be used")


def init_wfs_size(p_wfs: conf.Param_wfs, r0: float, p_tel: conf.Param_tel, verbose=1):
    """Compute all the parameters usefull for further WFS image computation (array sizes)

    :parameters:
        p_wfs: (Param_wfs) : wfs settings

        r0: (float) : atmos r0 @ 0.5 microns

        p_tel: (Param_tel) : telescope settings

        verbose: (int) : (optional) display informations if 0

    Scheme to determine arrays sizes
    sh :
    k = 6
    p = k * d/r0
    n = int(2*d*v/lambda/CONST.RAD2ARCSEC)+1
    N = fft_goodsize(k*n/v*lambda/r0*CONST.RAD2ARCSEC)
    u = k * lambda / r0 * CONST.RAD2ARCSEC / N
    n = v/u - int(v/u) > 0.5 ? int(v/u)+1 : int(v/u)
    v = n * u
    Nt = v * Npix

    pyr :
    Fs = field stop radius in arcsec
    N size of big array for FFT in pixels
    P pupil diameter in pixels
    D diameter of telescope in m
    Nssp : number of pyr measurement points in the pupil

    Rf = Fs . N . D / lambda / P
    ideally we choose : Fs = lambda / D . Nssp / 2

    if we want good sampling of r0 (avoid aliasing of speckles)
    P > D / r0 . m
    with m = 2 or 3

    to get reasonable space between pupil images : N > P.(2 + 3S)
    with S close to 1
    N must be a power of 2

    to ease computation lets assume that Nssp is a divider of P
    scaling factor between high res pupil images in pyramid model
    and actual pupil size on camera images would be P / Nssp

    """

    r0 = r0 * (p_wfs.Lambda * 2)**(6. / 5)

    if (r0 != 0):
        if (verbose):
            print("r0 for WFS :", "%3.2f" % r0, " m")
        # seeing = CONST.RAD2ARCSEC * (p_wfs.lambda * 1.e-6) / r0
        if (verbose):
            print("seeing for WFS : ",
                  "%3.2f" % (CONST.RAD2ARCSEC * (p_wfs.Lambda * 1.e-6) / r0), "\"")

    if (p_wfs._pdiam <= 0):
        # this case is usualy for the wfs with max # of subaps
        # we look for the best compromise between pixsize and fov
        if (p_wfs.type == scons.WFSType.SH):

            subapdiam = p_tel.diam / float(p_wfs.nxsub)  # diam of subap
            k = 6
            pdiam = int(k * subapdiam / r0)  # number of phase points per subap
            if (pdiam < 16):
                pdiam = 16

            # Must be even to keep ssp and actuators grids aligned in the pupil
            if ((pdiam * p_wfs.nxsub) % 2):
                pdiam += 1

            nrebin = int(2 * subapdiam * p_wfs.pixsize /
                         (p_wfs.Lambda * 1.e-6) / CONST.RAD2ARCSEC) + 1
            nrebin = max(2, nrebin)
            # first atempt on a rebin factor

            # since we clipped pdiam we have to be carreful in nfft computation
            Nfft = util.fft_goodsize(
                    int(pdiam / subapdiam * nrebin / p_wfs.pixsize * CONST.RAD2ARCSEC *
                        (p_wfs.Lambda * 1.e-6)))

        elif (p_wfs.type == scons.WFSType.PYRHR or p_wfs.type == scons.WFSType.PYRLR):
            # while (pdiam % p_wfs.npix != 0) pdiam+=1;
            k = 3 if p_wfs.type == scons.WFSType.PYRHR else 2
            pdiam = int(p_tel.diam / r0 * k)
            while (pdiam % p_wfs.nxsub != 0):
                pdiam += 1  # we choose to have a multiple of p_wfs.nxsub
            pdiam = pdiam // p_wfs.nxsub
            if (pdiam < 8):
                pdiam = 8

        # quantum pixel size
    else:
        pdiam = p_wfs._pdiam
        # this case is for a wfs with fixed # of phase points
        Nfft = util.fft_goodsize(2 * pdiam)
    # size of the support in fourier domain

    # qpixsize = pdiam * \
    #     (p_wfs.Lambda * 1.e-6) / subapdiam * CONST.RAD2ARCSEC / Nfft
    # # quantum pixel size

    if (p_wfs.type == scons.WFSType.SH):
        subapdiam = p_tel.diam / float(p_wfs.nxsub)  # diam of subap

        # size of the support in fourier domain

        # qpixsize = k * (p_wfs.Lambda*1.e-6) / r0 * CONST.RAD2ARCSEC / Nfft
        qpixsize = (pdiam * (p_wfs.Lambda * 1.e-6) / subapdiam * CONST.RAD2ARCSEC) / Nfft

        # # actual rebin factor
        if (p_wfs.pixsize / qpixsize - int(p_wfs.pixsize / qpixsize) > 0.5):
            nrebin = int(p_wfs.pixsize / qpixsize) + 1
        else:
            nrebin = int(p_wfs.pixsize / qpixsize)

        # actual pixel size
        pixsize = nrebin * qpixsize

        if (pixsize * p_wfs.npix > qpixsize * Nfft):
            Ntot = util.fft_goodsize(int(pixsize * p_wfs.npix / qpixsize) + 1)
        else:
            Ntot = Nfft

        if (Ntot % 2 != Nfft % 2):
            Ntot += 1

    elif (p_wfs.type == scons.WFSType.PYRHR or p_wfs.type == scons.WFSType.PYRLR):
        pdiam = pdiam * p_wfs.nxsub
        m = 3
        # fft_goodsize( m * pdiam)
        Nfft = int(2**np.ceil(np.log2(m * pdiam)))

        nrebin = pdiam // p_wfs.nxsub
        while (Nfft % nrebin != 0):
            nrebin += 1  # we choose to have a divisor of Nfft
            pdiam = nrebin * p_wfs.nxsub
            Nfft = int(2**np.ceil(np.log2(m * pdiam)))

        qpixsize = (pdiam *
                    (p_wfs.Lambda * 1.e-6) / p_tel.diam * CONST.RAD2ARCSEC) / Nfft

        Ntot = Nfft // pdiam * p_wfs.nxsub
        pixsize = qpixsize * nrebin
        pdiam = pdiam // p_wfs.nxsub

    p_wfs._pdiam = pdiam
    p_wfs.pixsize = pixsize
    p_wfs._qpixsize = qpixsize
    p_wfs._Nfft = Nfft
    p_wfs._Ntot = Ntot
    p_wfs._nrebin = nrebin
    p_wfs._subapd = p_tel.diam / p_wfs.nxsub

    if (verbose):
        if (p_wfs.type == scons.WFSType.SH):
            print("quantum pixsize : ", "%5.4f" % qpixsize, "\"")
            print("simulated FoV : ", "%3.2f" % (Ntot * qpixsize), "\" x ",
                  "%3.2f" % (Ntot * qpixsize), "\"")
            print("actual pixsize : ", "%5.4f" % pixsize)
            print("actual FoV : ", "%3.2f" % (pixsize * p_wfs.npix), "\" x ",
                  "%3.2f" % (pixsize * p_wfs.npix), "\"")
            print("number of phase points : ", p_wfs._pdiam)
            print("size of fft support : ", Nfft)
            print("size of HR spot support : ", Ntot)

        elif (p_wfs.type == scons.WFSType.PYRHR or p_wfs.type == scons.WFSType.PYRLR):
            print("quantum pixsize in pyr image : ", "%5.4f" % qpixsize, "\"")
            print("simulated FoV : ", "%3.2f" % (Nfft * qpixsize), "\" x ",
                  "%3.2f" % (Nfft * qpixsize), "\"")
            print("number of phase points : ", p_wfs._pdiam * p_wfs.nxsub)
            print("size of fft support : ", Nfft)


def compute_nphotons(wfs_type, ittime, optthroughput, diam, cobs=0, nxsub=0, zerop=0,
                     gsmag=0, lgsreturnperwatt=0, laserpower=0, verbose=1):
    ''' Determines the number of photons TBC

    :parameters:
        wfs_type: (scons.WFSType) : wfs type: SH or PYRHR.

        ittime: (float) : 1/loop frequency [s].

        optthroughput: (float) : wfs global throughput.

        diam: (float) : telescope diameter.

        cobs: (float) : (optional for SH)  telescope central obstruction.

        nxsub: (int) : (optional for PYRHR)  linear number of subaps.

        zerop: (float) : (optional for LGS)  detector zero point expressed in ph/m**2/s in the bandwidth of the WFS.

        gsmag: (float) : (optional for LGS)  magnitude of guide star.

        lgsreturnperwatt: (float) : (optional for NGS) return per watt factor (high season : 10 ph/cm2/s/W).

        laserpower: (float) : (optional for NGS) laser power in W.

        verbose: (bool) : (optional) display informations if True.

    for PYRHR WFS: nphotons = compute_nphotons(scons.WFSType.PYRHR, ittime,
                                optthroughput, diam, cobs=?, zerop=?, gsmag=?)
    for NGS SH WFS: nphotons = compute_nphotons(scons.WFSType.SH, ittime,
                                optthroughput, diam, nxsub=?, zerop=?, gsmag=?)
    for LGS SH WFS: nphotons = compute_nphotons(scons.WFSType.SH, ittime,
                                optthroughput, diam, nxsub=?,
                                lgsreturnperwatt=?, laserpower=?)
    '''
    surface = 0
    nphotons = 0
    if (wfs_type == scons.WFSType.PYRHR or wfs_type == scons.WFSType.PYRLR):
        surface = np.pi / 4. * (1 - cobs**2.) * diam**2.
    elif (wfs_type == scons.WFSType.SH):
        # from the guide star
        if (laserpower == 0):
            if (zerop == 0):
                zerop = 1e11
            surface = (diam / nxsub)**2.
            # include throughput to WFS for unobstructed
            # subaperture per iteration
        else:  # we are dealing with a LGS
            nphotons = lgsreturnperwatt * laserpower * \
                optthroughput * (diam / nxsub) ** 2. * 1e4 * ittime
            # detected by WFS
            # ... for given power include throughput to WFS
            # for unobstructed subaperture per iteration
            if (verbose):
                print("nphotons : ", nphotons)
            return nphotons
    else:
        raise RuntimeError("WFS unknown")

    nphotons = zerop * 10. ** (-0.4 * gsmag) * ittime * \
            optthroughput * surface

    if (verbose):
        print("nphotons : ", nphotons)
    return nphotons


def init_pyrhr_geom(p_wfs: conf.Param_wfs, r0: float, p_tel: conf.Param_tel,
                    p_geom: conf.Param_geom, ittime: float, verbose: bool = True):
    """Compute the geometry of PYRHR WFSs: valid subaps, positions of the subaps,
    flux per subap, etc...

    :parameters:
        p_wfs: (Param_wfs) : wfs settings

        r0: (float) : atmos r0 @ 0.5 microns

        p_tel: (Param_tel) : telescope settings

        geom: (Param_geom) : geom settings

        ittime: (float) : 1/loop frequency [s]

        verbose: (bool) : (optional) display informations if True

    """

    p_wfs.npix = p_wfs._pdiam
    p_wfs._Ntot = p_geom._n

    # Creating field stop mask
    fsradius_pixels = int(p_wfs.fssize / p_wfs._qpixsize / 2.)
    if (p_wfs.fstop == scons.FieldStopType.ROUND):
        focmask = util.dist(p_wfs._Nfft, xc=p_wfs._Nfft / 2. - 0.5,
                            yc=p_wfs._Nfft / 2. - 0.5) < (fsradius_pixels)
    elif (p_wfs.fstop == scons.FieldStopType.SQUARE):
        X = np.indices((p_wfs._Nfft, p_wfs._Nfft)) + 1  # TODO: +1 ??
        x = X[1] - (p_wfs._Nfft + 1.) / 2.
        y = X[0] - (p_wfs._Nfft + 1.) / 2.
        focmask = (np.abs(x) <= (fsradius_pixels)) * \
            (np.abs(y) <= (fsradius_pixels))
    else:
        msg = "PYRHR wfs fstop must be FieldStopType.[ROUND|SQUARE]"
        raise ValueError(msg)

    pyr_focmask = focmask * 1.0  # np.fft.fftshift(focmask*1.0)
    p_wfs._submask = np.fft.fftshift(pyr_focmask)

    # Creating pyramid mask
    pyrsize = p_wfs._Nfft
    cobs = p_tel.cobs
    rpup = p_geom.pupdiam / 2.0
    dpup = p_geom.pupdiam
    nrebin = p_wfs._nrebin
    fracsub = p_wfs.fracsub
    if p_wfs.pyr_pup_sep == -1:
        pup_sep = int(p_wfs.nxsub)
    else:
        pup_sep = p_wfs.pyr_pup_sep
    # number of pix between two centers two pupil images

    y = np.tile(np.arange(pyrsize) - pyrsize / 2, (pyrsize, 1))
    x = y.T

    Pangle = pup_sep * nrebin  # Pyramid angle in HR pixels
    if p_wfs.nPupils == 0:
        p_wfs.nPupils = 4
    # Centers is a nPupils x 2 array describing the position of the quadrant centers
    centers = Pangle / np.sin(
            np.pi / p_wfs.nPupils) * np.c_[np.cos(
                    (2 * np.arange(p_wfs.nPupils) + 1) * np.pi / p_wfs.nPupils),
                                           np.sin((2 * np.arange(p_wfs.nPupils) + 1) *
                                                  np.pi / p_wfs.nPupils)]
    # In case nPupils == 4, we put the centers back in the normal A-B-C-D ordering scheme, for misalignment processing.
    if p_wfs.nPupils == 4:
        centers = np.round(centers[[1, 0, 2, 3], :]).astype(np.int32)
    # Add in the misalignments of quadrant centers, in LR px units
    if p_wfs.pyr_misalignments is not None:
        mis = np.asarray(p_wfs.pyr_misalignments) * nrebin
    else:
        mis = np.zeros((p_wfs.nPupils, 2), dtype=np.float32)
    # Pyramid as minimal intersect of 4 tilting planes
    pyr = 2 * np.pi / pyrsize * np.min(
            np.asarray([(c[0] + m[0]) * x + (c[1] + m[1]) * y
                        for c, m in zip(centers, mis)]), axis=0)
    p_wfs._halfxy = np.fft.fftshift(pyr.T)

    # Valid pixels identification
    # Generate buffer with pupil at center
    pup = np.zeros((pyrsize, pyrsize), dtype=np.int32)
    pup[pyrsize // 2 - p_geom._n // 2:pyrsize // 2 + p_geom._n // 2,
        pyrsize // 2 - p_geom._n // 2:pyrsize // 2 + p_geom._n // 2] = \
            p_geom._mpupil

    # Shift the mask to obtain the geometrically valid pixels
    for qIdx in range(p_wfs.nPupils):
        quadOnCenter = np.roll(pup, tuple((centers[qIdx] + mis[qIdx]).astype(np.int32)),
                               (0, 1))
        mskRebin = util.rebin(quadOnCenter.copy(),
                              [pyrsize // nrebin, pyrsize // nrebin])
        if qIdx == 0:
            stackedSubap = np.roll(mskRebin >= fracsub,
                                   tuple((-centers[qIdx] / nrebin).astype(np.int32)),
                                   (0, 1))
            mskRebTot = (mskRebin >= fracsub).astype(np.int32)
        else:
            stackedSubap += np.roll(mskRebin >= fracsub,
                                    tuple((-centers[qIdx] / nrebin).astype(np.int32)),
                                    (0, 1))
            mskRebTot += (mskRebin >= fracsub).astype(np.int32) * (qIdx + 1)

    validRow, validCol = [], []
    # If n == 4, we need to tweak -again- the order to feed
    # compass XY controllers in the appropriate order
    if p_wfs.nPupils == 4:
        centers = centers[[2, 1, 3, 0], :]
        # mis = mis[[2, 1, 3, 0], :] # We don't need mis anymore - but if so keep the order of centers
    for qIdx in range(p_wfs.nPupils):
        tmpWh = np.where(
                np.roll(stackedSubap, tuple((centers[qIdx] / nrebin).astype(np.int32)),
                        (0, 1)))
        validRow += [tmpWh[0].astype(np.int32)]
        validCol += [tmpWh[1].astype(np.int32)]
    nvalid = validRow[0].size
    validRow = np.asarray(validRow).flatten()
    validCol = np.asarray(validCol).flatten()
    stack, index = np.unique(np.c_[validRow, validCol], axis=0, return_index=True)

    p_wfs._nvalid = nvalid
    p_wfs._validsubsx = validRow[np.sort(index)]
    p_wfs._validsubsy = validCol[np.sort(index)]
    p_wfs._hrmap = mskRebTot.astype(np.int32)

    if (p_wfs.pyr_pos is None):
        pixsize = (np.pi * p_wfs._qpixsize) / (3600 * 180)
        #            scale_fact = 2 * np.pi / npup * \
        #                (p_wfs.Lambda / p_tel.diam / 4.848) / pixsize * p_wfs.pyr_ampl
        #             Proposition de Flo
        scale_fact = 2 * np.pi / pyrsize * \
            (p_wfs.Lambda * 1e-6 / p_tel.diam) / pixsize * p_wfs.pyr_ampl
        cx = scale_fact * \
            np.sin((np.arange(p_wfs.pyr_npts)) * 2. * np.pi / p_wfs.pyr_npts)
        cy = scale_fact * \
            np.cos((np.arange(p_wfs.pyr_npts)) * 2. * np.pi / p_wfs.pyr_npts)
        # mod_npts = p_wfs.pyr_npts #UNUSED
    else:
        if (verbose):
            print("Using user-defined positions [arcsec] for the pyramid modulation")
        cx = p_wfs.pyr_pos[:, 0] / p_wfs._qpixsize
        cy = p_wfs.pyr_pos[:, 1] / p_wfs._qpixsize
        # mod_npts=cx.shape[0] #UNUSED

    p_wfs._pyr_cx = cx.copy()
    p_wfs._pyr_cy = cy.copy()

    # telSurf = np.pi / 4. * (1 - p_tel.cobs**2.) * p_tel.diam**2.
    # p_wfs._nphotons = p_wfs.zerop * \
    #     10. ** (-0.4 * p_wfs.gsmag) * ittime * \
    #     p_wfs.optthroughput * telSurf
    p_wfs._nphotons = compute_nphotons(scons.WFSType.PYRHR, ittime, p_wfs.optthroughput,
                                       p_tel.diam, cobs=p_tel.cobs, zerop=p_wfs.zerop,
                                       gsmag=p_wfs.gsmag, verbose=verbose)

    # spatial filtering by the pixel extent:
    # *2/2 intended. min should be 0.40 = sinc(0.5)^2.
    y = np.tile(np.arange(pyrsize) - pyrsize // 2, (pyrsize, 1))
    x = y.T
    x = x * 1. / pyrsize
    y = y * 1. / pyrsize
    sincar = np.fft.fftshift(np.sinc(x) * np.sinc(y))

    # sincar = np.roll(np.pi*x*np.pi*y,x.shape[1],axis=1)
    p_wfs._sincar = sincar.astype(np.float32)

    #pup = p_geom._mpupil
    a = pyrsize // nrebin
    b = p_geom._n // nrebin
    pupvalid = stackedSubap[a // 2 - b // 2:a // 2 + b // 2, a // 2 - b // 2:a // 2 +
                            b // 2]
    p_wfs._isvalid = pupvalid.T.astype(np.int32)

    validsubsx = np.where(pupvalid)[0].astype(np.int32)
    validsubsy = np.where(pupvalid)[1].astype(np.int32)

    istart = np.arange(p_wfs.nxsub + 2) * p_wfs.npix
    jstart = np.copy(istart)

    # sorting out valid subaps
    fluxPerSub = np.zeros((p_wfs.nxsub + 2, p_wfs.nxsub + 2), dtype=np.float32)

    for i in range(p_wfs.nxsub + 2):
        indi = istart[i]
        for j in range(p_wfs.nxsub + 2):
            indj = jstart[j]
            fluxPerSub[i, j] = np.sum(
                    p_geom._mpupil[indi:indi + p_wfs.npix, indj:indj + p_wfs.npix])

    fluxPerSub = fluxPerSub / p_wfs.nxsub**2.

    p_wfs._fluxPerSub = fluxPerSub.copy()

    phasemap = np.zeros((p_wfs.npix * p_wfs.npix, p_wfs._nvalid), dtype=np.int32)
    X = np.indices((p_geom._n, p_geom._n))  # we need c-like indice
    tmp = X[1] + X[0] * p_geom._n

    pyrtmp = np.zeros((p_geom._n, p_geom._n), dtype=np.int32)

    for i in range(len(validsubsx)):
        indi = istart[validsubsy[i]]  # +2-1 (yorick->python
        indj = jstart[validsubsx[i]]
        phasemap[:, i] = tmp[indi:indi + p_wfs.npix, indj:indj + p_wfs.npix].flatten("C")
        pyrtmp[indi:indi + p_wfs.npix, indj:indj + p_wfs.npix] = i

    p_wfs._phasemap = phasemap

    p_wfs._pyr_offsets = pyrtmp  # pshift


def init_sh_geom(p_wfs: conf.Param_wfs, r0: float, p_tel: conf.Param_tel,
                 p_geom: conf.Param_geom, ittime: float, verbose: bool = True):
    """Compute the geometry of SH WFSs: valid subaps, positions of the subaps,
    flux per subap, etc...

    :parameters:
        p_wfs: (Param_wfs) : wfs settings

        r0: (float) : atmos r0 @ 0.5 microns

        p_tel: (Param_tel) : telescope settings

        geom: (Param_geom) : geom settings

        ittime: (float) : 1/loop frequency [s]

        verbose: (bool) : (optional) display informations if True

    """
    p_wfs.nPupils = 1
    # this is the i,j index of lower left pixel of subap in _spupil
    istart = ((np.linspace(0, p_geom.pupdiam, p_wfs.nxsub + 1))[:-1]).astype(np.int64)
    # Translation in _mupil useful for raytracing
    istart += 2
    jstart = np.copy(istart)

    # sorting out valid subaps
    fluxPerSub = np.zeros((p_wfs.nxsub, p_wfs.nxsub), dtype=np.float32)

    for i in range(p_wfs.nxsub):
        indi = istart[i]  # +2-1 (yorick->python)
        for j in range(p_wfs.nxsub):
            indj = jstart[j]  # +2-1 (yorick->python)
            fluxPerSub[i, j] = np.sum(
                    p_geom._mpupil[indi:indi + p_wfs._pdiam, indj:indj + p_wfs._pdiam])
            # fluxPerSub[i,j] = np.where(p_geom._mpupil[indi:indi+pdiam,indj:indj+pdiam] > 0)[0].size

    fluxPerSub = fluxPerSub / p_wfs._pdiam**2.

    pupvalid = (fluxPerSub >= p_wfs.fracsub) * 1
    p_wfs._isvalid = pupvalid.astype(np.int32)
    p_wfs._nvalid = int(np.sum(p_wfs._isvalid))
    p_wfs._fluxPerSub = fluxPerSub.copy()
    validx = np.where(p_wfs._isvalid.T)[1].astype(np.int32)
    validy = np.where(p_wfs._isvalid.T)[0].astype(np.int32)
    p_wfs._validsubsx = validx
    p_wfs._validsubsy = validy
    p_wfs._validpuppixx = validx * p_wfs._pdiam + 2
    p_wfs._validpuppixy = validy * p_wfs._pdiam + 2

    # this defines how we cut the phase into subaps
    phasemap = np.zeros((p_wfs._pdiam * p_wfs._pdiam, p_wfs._nvalid), dtype=np.int32)

    X = np.indices((p_geom._n, p_geom._n))
    tmp = X[1] + X[0] * p_geom._n

    n = p_wfs._nvalid
    # for i in range(p_wfs._nvalid):
    for i in range(n):
        indi = istart[p_wfs._validsubsy[i]]  # +2-1 (yorick->python)
        indj = jstart[p_wfs._validsubsx[i]]
        phasemap[:, i] = tmp[indi:indi + p_wfs._pdiam, indj:indj +
                             p_wfs._pdiam].flatten()
    p_wfs._phasemap = phasemap
    p_wfs._validsubsx *= p_wfs.npix
    p_wfs._validsubsy *= p_wfs.npix

    # this is a phase shift of 1/2 pix in x and y
    halfxy = np.linspace(0, 2 * np.pi, p_wfs._Nfft + 1)[0:p_wfs._pdiam] / 2.
    halfxy = np.tile(halfxy, (p_wfs._pdiam, 1))
    halfxy += halfxy.T
    # p_wfs._halfxy = <float*>(halfxy*0.).data #dont work: half*0 is temp
    # python obj

    if (p_wfs.npix % 2 == 1 and p_wfs._nrebin % 2 == 1):
        # p_wfs._halfxy = <float*>(halfxy*0.)
        halfxy = np.zeros((p_wfs._pdiam, p_wfs._pdiam), dtype=np.float32)
        p_wfs._halfxy = halfxy.astype(np.float32)
    else:
        p_wfs._halfxy = halfxy.astype(np.float32)

    # this defines how we create a larger fov if required
    if (p_wfs._Ntot != p_wfs._Nfft):
        indi = int((p_wfs._Ntot - p_wfs._Nfft) / 2.)  # +1 -1 (yorick>python)
        indj = int(indi + p_wfs._Nfft)
        X = np.indices((p_wfs._Nfft, p_wfs._Nfft)) + 1  # TODO: +1 ??
        x = X[1]
        y = X[0]
        # hrpix
        tmp = np.zeros((p_wfs._Ntot, p_wfs._Ntot))
        tmp[indi:indj, indi:indj] = np.roll(x + (y - 1) * p_wfs._Nfft, p_wfs._Nfft // 2,
                                            axis=0)
        tmp[indi:indj, indi:indj] = np.roll(tmp[indi:indj, indi:indj], p_wfs._Nfft // 2,
                                            axis=1)
        # hrmap=roll(hrpix)
        tmp = np.roll(tmp, p_wfs._Ntot // 2, axis=0)
        tmp = np.roll(tmp, p_wfs._Ntot // 2, axis=1)

        tmp = np.where(tmp.flatten())[0]

        p_wfs._hrmap = np.copy(tmp.astype(np.int32))
        p_wfs._hrmap = np.reshape(p_wfs._hrmap, (p_wfs._hrmap.shape[0], 1))
        # must be set even if unused

    else:
        tmp = np.zeros((2, 2))
        p_wfs._hrmap = np.copy(tmp.astype(np.int32))
        # must be set even if unused

    if (p_wfs._nrebin * p_wfs.npix % 2 != p_wfs._Ntot % 2):
        # +2-1 yorick>python
        indi = int((p_wfs._Ntot - p_wfs._nrebin * p_wfs.npix) / 2.) + 1
    else:
        indi = int((p_wfs._Ntot - p_wfs._nrebin * p_wfs.npix) / 2.) + 0
    indj = int(indi + p_wfs._nrebin * p_wfs.npix)

    X = np.indices((p_wfs._nrebin * p_wfs.npix, p_wfs._nrebin * p_wfs.npix))
    x = (X[1] / p_wfs._nrebin).astype(np.int64)
    y = (X[0] / p_wfs._nrebin).astype(np.int64)

    # binindices
    binindices = np.zeros((p_wfs._Ntot, p_wfs._Ntot))
    binindices[indi:indj, indi:indj] = x + y * p_wfs.npix + 1

    binmap = np.zeros((p_wfs._nrebin * p_wfs._nrebin, p_wfs.npix * p_wfs.npix))

    X = np.indices((p_wfs._Ntot, p_wfs._Ntot))
    tmp = X[1] + X[0] * p_wfs._Ntot

    if (p_wfs.gsalt <= 0):
        binindices = np.roll(binindices, binindices.shape[0] // 2, axis=0)
        binindices = np.roll(binindices, binindices.shape[1] // 2, axis=1)

    for i in range(p_wfs.npix * p_wfs.npix):
        binmap[:, i] = tmp[np.where(binindices == i + 1)]
    # binmap=np.reshape(binmap.flatten("F"),(binmap.shape[0],binmap.shape[1]))
    p_wfs._binmap = np.copy(binmap.astype(np.int32))

    dr0 = p_tel.diam / r0 * \
        (0.5 / p_wfs.Lambda) ** 1.2 / \
        np.cos(p_geom.zenithangle * CONST.DEG2RAD) ** 0.6
    fwhmseeing = p_wfs.Lambda / \
        (p_tel.diam / np.sqrt(p_wfs.nxsub ** 2. + (dr0 / 1.5) ** 2.)) / 4.848
    kernelfwhm = np.sqrt(fwhmseeing**2. + p_wfs.kernel**2.)

    tmp = util.makegaussian(p_wfs._Ntot, kernelfwhm / p_wfs._qpixsize, p_wfs._Ntot // 2,
                            p_wfs._Ntot // 2).astype(np.float32)

    tmp = np.roll(tmp, tmp.shape[0] // 2, axis=0)
    tmp = np.roll(tmp, tmp.shape[1] // 2, axis=1)

    tmp[0, 0] = 1.  # this insures that even with fwhm=0, the kernel is a dirac
    tmp = tmp / np.sum(tmp)
    tmp = np.fft.fft2(tmp).astype(np.complex64) / (p_wfs._Ntot * p_wfs._Ntot)
    p_wfs._ftkernel = np.copy(tmp).astype(np.complex64)

    # dealing with photometry
    # telSurf = np.pi / 4. * (1 - p_tel.cobs**2.) * p_tel.diam**2.

    # from the guide star
    if (p_wfs.gsalt == 0):
        if (p_wfs.zerop == 0):
            p_wfs.zerop = 1e11
        # p_wfs._nphotons = p_wfs.zerop * 10 ** (-0.4 * p_wfs.gsmag) * \
        #     p_wfs.optthroughput * \
        #     (p_tel.diam / p_wfs.nxsub) ** 2. * ittime
        # include throughput to WFS
        # for unobstructed subaperture
        # per iteration
        p_wfs._nphotons = compute_nphotons(scons.WFSType.SH, ittime, p_wfs.optthroughput,
                                           p_tel.diam, nxsub=p_wfs.nxsub,
                                           zerop=p_wfs.zerop, gsmag=p_wfs.gsmag,
                                           verbose=verbose)

    else:  # we are dealing with a LGS
        # p_wfs._nphotons = p_wfs.lgsreturnperwatt * \
        #     p_wfs.laserpower * \
        #     p_wfs.optthroughput * \
        #     (p_tel.diam / p_wfs.nxsub) ** 2. * 1e4 * \
        #     ittime
        # detected by WFS
        # ... for given power
        # include throughput to WFS
        # for unobstructed subaperture
        # per iteration
        p_wfs._nphotons = compute_nphotons(scons.WFSType.SH, ittime, p_wfs.optthroughput,
                                           p_tel.diam, nxsub=p_wfs.nxsub,
                                           lgsreturnperwatt=p_wfs.lgsreturnperwatt,
                                           laserpower=p_wfs.laserpower, verbose=verbose)


def geom_init(p_geom: conf.Param_geom, p_tel: conf.Param_tel, padding=2):
    """
        Initialize the system geometry

    :parameters:
        p_geom: (Param_geom) : geometry settings
        p_tel: (Param_tel) : telescope settings
        padding: (optional) : padding factor for PYRHR geometry
    """

    # First power of 2 greater than pupdiam
    p_geom.ssize = int(2**np.ceil(np.log2(p_geom.pupdiam) + 1))
    # Using images centered on 1/2 pixels
    p_geom.cent = p_geom.ssize / 2 - 0.5

    p_geom._p1 = int(np.ceil(p_geom.cent - p_geom.pupdiam / 2.))
    p_geom._p2 = int(np.floor(p_geom.cent + p_geom.pupdiam / 2.))

    p_geom.pupdiam = p_geom._p2 - p_geom._p1 + 1

    p_geom._n = p_geom.pupdiam + 2 * padding
    p_geom._n1 = p_geom._p1 - padding
    p_geom._n2 = p_geom._p2 + padding

    cent = p_geom.pupdiam / 2. - 0.5

    # Useful pupil
    p_geom._spupil = mkP.make_pupil(p_geom.pupdiam, p_geom.pupdiam, p_tel, cent,
                                    cent).astype(np.float32)

    # large pupil (used for image formation)
    p_geom._ipupil = util.pad_array(p_geom._spupil, p_geom.ssize).astype(np.float32)

    # useful pupil + 4 pixels
    p_geom._mpupil = util.pad_array(p_geom._spupil, p_geom._n).astype(np.float32)

    if (p_tel.std_piston and p_tel.std_tt):
        p_geom._phase_ab_M1 = mkP.make_phase_ab(p_geom.pupdiam, p_geom.pupdiam, p_tel,
                                                p_geom._spupil, cent,
                                                cent).astype(np.float32)
        p_geom._phase_ab_M1_m = util.pad_array(p_geom._phase_ab_M1,
                                               p_geom._n).astype(np.float32)
    #TODO: apodizer
    """
    if p_geom.apod:
        if p_geom.apodFile is None or p_geom.apodFile == '':
            apod_filename = shesha_savepath + \
                "apodizer/SP_HARMONI_I4_C6_N1024.npy"
        p_geom._apodizer = makeA.make_apodizer(
                p_geom.pupdiam, p_geom.pupdiam,
                apod_filename.encode(), 180. / 12.).astype(np.float32)
    else:
    """
    p_geom._apodizer = np.ones(p_geom._spupil.shape, dtype=np.int32)
    p_geom._pixsize = p_tel.diam / p_geom.pupdiam
    p_geom.is_init = True


def geom_init_generic(p_geom, pupdiam, t_spiders=0.01, spiders_type="six", xc=0, yc=0,
                      real=0, cobs=0):
    """Initialize the system geometry

    :parameters:
        pupdiam: (long) : linear size of total pupil

        t_spiders: (float) : secondary supports ratio.

        spiders_type: (str) :  secondary supports type: "four" or "six".

        xc: (int)

        yc: (int)

        real: (int)

        cobs: (float) : central obstruction ratio.
    """
    # Initialize the system pupil
    # first poxer of 2 greater than pupdiam
    p_geom.ssize = int(2**np.ceil(np.log2(pupdiam) + 1))
    # using images centered on 1/2 pixels
    p_geom.cent = p_geom.ssize / 2 + 0.5
    # valid pupil geometry
    pupdiam = int(pupdiam)
    p_geom._p1 = int(np.ceil(p_geom.cent - pupdiam / 2.))
    p_geom._p2 = int(np.floor(p_geom.cent + pupdiam / 2.))
    p_geom.pupdiam = p_geom._p2 - p_geom._p1 + 1
    p_geom._n = p_geom.pupdiam + 4
    p_geom._n1 = p_geom._p1 - 2
    p_geom._n2 = p_geom._p2 + 2

    # useful pupil
    p_geom._spupil = mkP.make_pupil_generic(pupdiam, pupdiam, t_spiders, spiders_type,
                                            xc, yc, real, cobs)

    # large pupil (used for image formation)
    p_geom._ipupil = pad_array(p_geom._spupil, p_geom.ssize).astype(np.float32)

    # useful pupil + 4 pixels
    p_geom._mpupil = pad_array(p_geom._spupil, p_geom._n).astype(np.float32)


def pad_array(A, N):
    S = A.shape
    D1 = (N - S[0]) // 2
    D2 = (N - S[1]) // 2
    padded = np.zeros((N, N))
    padded[D1:D1 + S[0], D2:D2 + S[1]] = A
    return padded
