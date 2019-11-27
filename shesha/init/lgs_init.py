## @package   shesha.init.lgs_init
## @brief     Initialization of a LGS in a Wfs object
## @author    COMPASS Team <https://github.com/ANR-COMPASS>
## @version   4.3.2
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


import os
try:
    shesha_db = os.environ['SHESHA_DB_ROOT']
except KeyError as err:
    import warnings
    shesha_db = os.environ['SHESHA_ROOT'] + "/data/"
    # warnings.warn("'SHESHA_DB_ROOT' not defined, using default one: " + shesha_db)
finally:
    shesha_savepath = shesha_db
# print("shesha_savepath:", shesha_savepath)

import shesha.config as conf
import shesha.constants as scons
from shesha.util import utilities as util
import numpy as np

from shesha.sutra_wrap import Sensors
import scipy.ndimage.interpolation as sci


def make_lgs_prof1d(p_wfs: conf.Param_wfs, p_tel: conf.Param_tel, prof: np.ndarray,
                    h: np.ndarray, beam: float, center=""):
    """same as prep_lgs_prof but cpu only. original routine from rico

    :parameters:
        p_tel: (Param_tel) : telescope settings

        prof: (np.ndarray[dtype=np.float32]) : Na profile intensity, in arbitrary units

        h: (np.ndarray[dtype=np.float32]) : altitude, in meters. h MUST be an array with EQUALLY spaced elements.

        beam: (float) : size in arcsec of the laser beam

        center: (string) : either "image" or "fourier" depending on where the centre should be.
    """

    p_wfs._prof1d = prof
    p_wfs._profcum = np.zeros(h.shape[0] + 1, dtype=np.float32)
    p_wfs._profcum[1:] = prof.cumsum()

    subapdiam = p_tel.diam / p_wfs.nxsub  # diam of subap
    if (p_wfs.nxsub > 1):
        xsubs = np.linspace((subapdiam - p_tel.diam) / 2, (p_tel.diam - subapdiam) / 2,
                            p_wfs.nxsub).astype(np.float32)
    else:
        xsubs = np.zeros(1, dtype=np.float32)
    ysubs = xsubs.copy().astype(np.float32)

    # cdef int nP=prof.shape[0] #UNUSED
    hG = np.sum(h * prof) / np.sum(prof)
    x = np.arange(p_wfs._Ntot).astype(np.float32) - p_wfs._Ntot / 2
    # x expressed in pixels. (0,0) is in the fourier-center.
    x = x * p_wfs._qpixsize  # x expressed in arcseconds
    # cdef float dx=x[1]-x[0] #UNUSED
    # cdef float dh=h[1]-h[0] #UNUSED

    if (p_wfs.nxsub > 1):
        dOffAxis = np.sqrt((xsubs[p_wfs._validsubsy // p_wfs.npix] - p_wfs.lltx)**2 +
                           (ysubs[p_wfs._validsubsx // p_wfs.npix] - p_wfs.llty)**2)
    else:
        dOffAxis = np.sqrt((xsubs - p_wfs.lltx)**2 + (ysubs - p_wfs.llty)**2)

    profi = np.zeros((p_wfs._Ntot, p_wfs._nvalid), dtype=np.float32)

    subsdone = np.ones(p_wfs._nvalid, dtype=np.int32)
    dif2do = np.zeros(p_wfs._nvalid, dtype=np.int32)

    while (np.any(subsdone)):
        tmp = dOffAxis[np.where(subsdone)][0]
        inds = np.where(dOffAxis == tmp)[0]
        # height, translated in arcsec due to perspective effect
        zhc = (h - hG) * (206265. * tmp / hG**2)
        dzhc = zhc[1] - zhc[0]

        if (p_wfs._qpixsize > dzhc):
            avg_zhc = np.zeros(zhc.size + 1, dtype=np.float32)
            avg_zhc[0] = zhc[0]
            avg_zhc[avg_zhc.size - 1] = zhc[zhc.size - 1]
            avg_zhc[1:-1] = 0.5 * (zhc[1:] + zhc[:-1])
            avg_x = np.zeros(x.size + 1, dtype=np.float32)
            avg_x[0] = x[0]
            avg_x[avg_x.size - 1] = x[x.size - 1]
            avg_x[1:-1] = 0.5 * (x[1:] + x[:-1])

            for i in range(inds.size):
                profi[:, inds[i]] = np.diff(np.interp(avg_x, avg_zhc,
                                                      p_wfs._profcum)).astype(np.float32)

        else:
            for i in range(inds.size):
                profi[:, inds[i]] = np.interp(x, zhc, prof)
        subsdone[inds] = 0

    w = beam / 2.35482005
    if (w == 0):
        # TODO what is n
        n = 1
        g = np.zeros(n, dtype=np.float32)
        if (center == "image"):
            g[n / 2 - 1] = 0.5
            g[n / 2] = 0.5
        else:
            g[n / 2] = 1

    else:
        if (center == "image"):
            if ((p_wfs.npix * p_wfs._nrebin) % 2 != p_wfs._Nfft % 2):
                g = np.exp(-(x + p_wfs._qpixsize)**2 / (2 * w**2.))
            else:
                g = np.exp(-(x + p_wfs._qpixsize / 2)**2 / (2 * w**2.))

        else:
            g = np.exp(-x**2 / (2 * w**2.))

    p_wfs._ftbeam = np.fft.fft(g).astype(np.complex64)
    p_wfs._beam = g.astype(np.float32)
    # convolved profile in 1D.

    g_extended = np.tile(g, (p_wfs._nvalid, 1)).T

    p1d = np.fft.ifft(
            np.fft.fft(profi, axis=0) * np.fft.fft(g_extended, axis=0),
            axis=0).real.astype(np.float32)
    p1d = p1d * p1d.shape[0]
    p1d = np.roll(p1d, int(p_wfs._Ntot / 2. - 0.5), axis=0)
    p1d = np.abs(p1d)

    im = np.zeros((p1d.shape[1], p1d.shape[0], p1d.shape[0]), dtype=np.float32)
    for i in range(p1d.shape[0]):
        im[:, i, :] = g[i] * p1d.T

    if (ysubs.size > 1):
        azimuth = np.arctan2(ysubs[p_wfs._validsubsy // p_wfs.npix] - p_wfs.llty,
                             xsubs[p_wfs._validsubsx // p_wfs.npix] - p_wfs.lltx)
    else:
        azimuth = np.arctan2(ysubs - p_wfs.llty, xsubs - p_wfs.lltx)

    p_wfs._azimuth = azimuth

    if (center == "image"):
        xcent = p_wfs._Ntot / 2. - 0.5
        ycent = xcent
    else:
        xcent = p_wfs._Ntot / 2.  #+ 1
        ycent = xcent

    if (ysubs.size > 0):
        # TODO rotate
        # im = util.rotate3d(im, azimuth * 180 / np.pi, xcent, ycent) --> ça marche pas !!
        # max_im = np.max(im, axis=(1, 2))
        # im = (im.T / max_im).T
        for k in range(im.shape[0]):
            img = im[k, :, :] / im[k, :, :].max()
            im[k, :, :] = sci.rotate(img, azimuth[k] * 180 / np.pi, reshape=False)

    else:
        # im = util.rotate(im, azimuth * 180 / np.pi, xcent, ycent)
        # im = im / np.max(im)
        im = sci.rotate(img, azimuth * 180 / np.pi, reshape=False)

    p_wfs._lgskern = im.T


def prep_lgs_prof(p_wfs: conf.Param_wfs, nsensors: int, p_tel: conf.Param_tel,
                  sensors: Sensors, center="", imat=0):
    """The function returns an image array(double,n,n) of a laser beacon elongated by perpective
    effect. It is obtaind by convolution of a gaussian of width "lgsWidth" arcseconds, with the
    line of the sodium profile "prof". The altitude of the profile is the array "h".

        :parameters:
            p_wfs: (Param_wfs) : WFS settings

            nsensors: (int) : wfs index

            p_tel: (Param_tel) : telescope settings

            Sensors: (Sensors) : WFS object

            center: (string) : either "image" or "fourier" depending on where the centre should be.

    Computation of LGS spot from the sodium profile:
    Everything is done here in 1D, because the Na profile is the result of the convolution of a function
    P(x,y) = profile(x) . dirac(y)
    by a gaussian function, for which variables x and y can be split :
    exp(-(x^2+y^2)/2.s^2)  =  exp(-x^2/2.s^2) * exp(-y^2/2.s^2)
    The convolution is (symbol $ denotes integral)
    C(X,Y) = $$ exp(-x^2/2.s^2) * exp(-y^2/2.s^2) * profile(x-X) * dirac(y-Y)  dx  dy
    First one performs the integration along y
    C(X,Y) = exp(-Y^2/2.s^2)  $ exp(-x^2/2.s^2) * profile(x-X)  dx
    which shows that the profile can be computed by
    - convolving the 1-D profile
    - multiplying it in the 2nd dimension by a gaussian function

    If one has to undersample the inital profile, then some structures may be "lost". In this case,
    it's better to try to "save" those structures by re-sampling the integral of the profile, and
    then derivating it afterwards.
    Now, if the initial profile is a coarse one, and that one has to oversample it, then a
    simple re-sampling of the profile is adequate.
    """
    if (p_wfs.proftype is None or p_wfs.proftype == ""):
        p_wfs.set_proftype(scons.ProfType.GAUSS1)

    profilename = scons.ProfType.FILES[p_wfs.proftype]

    profile_path = shesha_savepath + profilename
    print("reading Na profile from", profile_path)
    prof = np.load(profile_path)
    make_lgs_prof1d(p_wfs, p_tel, np.mean(prof[1:, :], axis=0), prof[0, :],
                    p_wfs.beamsize, center="image")
    p_wfs.set_altna(prof[0, :].astype(np.float32))
    p_wfs.set_profna(np.mean(prof[1:, :], axis=0).astype(np.float32))

    p_wfs._prof1d = p_wfs._profna
    p_wfs._profcum = np.zeros(p_wfs._profna.size + 1, dtype=np.float32)
    p_wfs._profcum[1:] = p_wfs._profna.cumsum()
    subapdiam = p_tel.diam / p_wfs.nxsub  # diam of subap

    if (p_wfs.nxsub > 1):
        xsubs = np.linspace((subapdiam - p_tel.diam) / 2, (p_tel.diam - subapdiam) / 2,
                            p_wfs.nxsub).astype(np.float32)
    else:
        xsubs = np.zeros(1, dtype=np.float32)
    ysubs = xsubs.copy().astype(np.float32)

    # center of gravity of the profile
    hG = np.sum(p_wfs._altna * p_wfs._profna) / np.sum(p_wfs._profna)
    x = np.arange(p_wfs._Ntot).astype(np.float32) - p_wfs._Ntot / 2
    # x expressed in pixels. (0,0) is in the fourier-center

    x = x * p_wfs._qpixsize  # x expressed in arcseconds
    # cdef float dx=x[1]-x[0] #UNUSED
    dh = p_wfs._altna[1] - p_wfs._altna[0]

    if (p_wfs.nxsub > 1):
        dOffAxis = np.sqrt((xsubs[p_wfs._validsubsx // p_wfs.npix] - p_wfs.lltx)**2 +
                           (ysubs[p_wfs._validsubsy // p_wfs.npix] - p_wfs.llty)**2)
    else:
        dOffAxis = np.sqrt((xsubs - p_wfs.lltx)**2 + (ysubs - p_wfs.llty)**2)

    if (imat > 0):
        dOffAxis *= 0.

    w = p_wfs.beamsize / 2.35482005  # TODO: FIXME
    if (w == 0):
        # TODO what is n
        n = 1
        g = np.zeros(n, dtype=np.float32)
        if (center == "image"):
            g[n / 2 - 1] = 0.5
            g[n / 2] = 0.5
        else:
            g[n / 2] = 1

    else:
        if (center == "image"):
            g = np.exp(-(x + p_wfs._qpixsize / 2)**2 / (2 * w**2.))
        else:
            g = np.exp(-x**2 / (2 * w**2.))

    p_wfs._ftbeam = np.fft.fft(g, axis=0).astype(np.complex64)
    p_wfs._beam = g
    # convolved profile in 1D.

    if (xsubs.size > 1):
        azimuth = np.arctan2(ysubs[p_wfs._validsubsy // p_wfs.npix] - p_wfs.llty,
                             xsubs[p_wfs._validsubsx // p_wfs.npix] - p_wfs.lltx)
    else:
        azimuth = np.arctan2(ysubs - p_wfs.llty, xsubs - p_wfs.lltx)

    p_wfs._azimuth = azimuth

    sensors.d_wfs[nsensors].d_gs.d_lgs.lgs_init(
            p_wfs._prof1d.size, hG, p_wfs._altna[0], dh, p_wfs._qpixsize, dOffAxis,
            p_wfs._prof1d, p_wfs._profcum, p_wfs._beam, p_wfs._ftbeam, p_wfs._azimuth)
