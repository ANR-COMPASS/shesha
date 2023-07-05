## @package   shesha.init.wfs_init
## @brief     Initialization of a Sensors object
## @author    COMPASS Team <https://github.com/ANR-COMPASS>
## @version   5.4.4
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

import shesha.config as conf
import shesha.constants as scons
from shesha.constants import CONST

from . import lgs_init as LGS
from shesha.sutra_wrap import carmaWrap_context, Sensors, Telescope
import numpy as np


def wfs_init(context: carmaWrap_context, telescope: Telescope, p_wfss: list,
             p_tel: conf.Param_tel, p_geom: conf.Param_geom, p_dms=None, p_atmos=None):
    """
    Create and initialise  a Sensors object

    Args:
        context : (carmaWrap_context)
        telescope: (Telescope) : Telescope object
        p_wfss: (list of Param_wfs) : wfs settings
        p_tel: (Param_tel) : telescope settings
        p_geom: (Param_geom) : geom settings
        p_dms : (list of Param_dm) : (optional) dms settings
        p_atmos: (Param_atmos) : (optional) atmos settings
    :return:
        g_wfs: (Sensors): Sensors object
    """
    # create sensor object on gpu
    # and init sensor gs object on gpu
    nsensors = len(p_wfss)
    # arrays needed to call Sensors constructor
    t_wfs = [
            'shlo' if (o.type == scons.WFSType.SH and o.is_low_order) else
            'pyrhr' if o.type == scons.WFSType.PYRLR else o.type for o in p_wfss
    ]

    # cdef np.ndarray t_wfs  = np.array([o.type  for o in
    # wfs],dtype=np.str)
    nxsub = np.array([o.nxsub for o in p_wfss], dtype=np.int64)
    nvalid = np.array([o._nvalid for o in p_wfss], dtype=np.int64)
    nPupils = np.array([o.nPupils for o in p_wfss], dtype=np.int64)
    nphase = np.array([o._pdiam for o in p_wfss], dtype=np.int64)
    pdiam = np.array([o._subapd for o in p_wfss], dtype=np.float32)
    npix = np.array([o.npix for o in p_wfss], dtype=np.int64)
    nrebin = np.array([o._nrebin for o in p_wfss], dtype=np.int64)
    nfft = np.array([o._Nfft for o in p_wfss], dtype=np.int64)
    ntota = np.array([o._Ntot for o in p_wfss], dtype=np.int64)
    nphot = np.array([o._nphotons for o in p_wfss], dtype=np.float32)
    nphot4imat = np.array([o.nphotons4imat for o in p_wfss], dtype=np.float32)
    lgs = np.array([o.gsalt > 0 for o in p_wfss], dtype=np.int32)
    fakecam = np.array([o.fakecam for o in p_wfss], dtype=bool)
    maxFlux = np.array([o.max_flux_per_pix for o in p_wfss], dtype=np.int32)
    max_pix_value = np.array([o.max_pix_value for o in p_wfss], dtype=np.int32)

    # arrays needed to call initgs
    xpos = np.array([o.xpos for o in p_wfss], dtype=np.float32)
    ypos = np.array([o.ypos for o in p_wfss], dtype=np.float32)
    Lambda = np.array([o.Lambda for o in p_wfss], dtype=np.float32)
    zerop = p_wfss[0].zerop
    size = np.zeros(nsensors, dtype=np.int64) + p_geom._n
    seed = np.arange(nsensors, dtype=np.int64) + 1234
    npup = (np.zeros((nsensors)) + p_geom._n).astype(np.int64)

    G = np.array([o.G for o in p_wfss], dtype=np.float32)
    thetaML = np.array([o.thetaML for o in p_wfss], dtype=np.float32)
    dx = np.array([o.dx for o in p_wfss], dtype=np.float32)
    dy = np.array([o.dy for o in p_wfss], dtype=np.float32)

    roket_flag = any([w.roket for w in p_wfss])

    if (p_wfss[0].type == scons.WFSType.SH):
        g_wfs = Sensors(context, telescope, t_wfs, nsensors, nxsub, nvalid, nPupils,
                        npix, nphase, nrebin, nfft, ntota, npup, pdiam, nphot,
                        nphot4imat, lgs, fakecam, maxFlux, max_pix_value,
                        context.active_device, roket_flag)

        mag = np.array([o.gsmag for o in p_wfss], dtype=np.float32)
        noise = np.array([o.noise for o in p_wfss], dtype=np.float32)

        g_wfs.initgs(xpos, ypos, Lambda, mag, zerop, size, noise, seed, G, thetaML, dx,
                     dy)

    elif (p_wfss[0].type == scons.WFSType.PYRHR or
          p_wfss[0].type == scons.WFSType.PYRLR):
        npup = np.array([o.pyr_npts for o in p_wfss])
        npix = np.array([o._validsubsx.size for o in p_wfss])
        G = np.array([o.G for o in p_wfss], dtype=np.float32)
        thetaML = np.array([o.thetaML for o in p_wfss], dtype=np.float32)
        dx = np.array([o.dx for o in p_wfss], dtype=np.float32)
        dy = np.array([o.dy for o in p_wfss], dtype=np.float32)

        g_wfs = Sensors(context, telescope, t_wfs, nsensors, nxsub, nvalid, nPupils,
                        npix, nphase, nrebin, nfft, ntota, npup, pdiam, nphot,
                        nphot4imat, lgs, fakecam, maxFlux, max_pix_value,
                        context.active_device, roket_flag)

        mag = np.array([o.gsmag for o in p_wfss], dtype=np.float32)
        noise = np.array([o.noise for o in p_wfss], dtype=np.float32)
        g_wfs.initgs(xpos, ypos, Lambda, mag, zerop, size, noise, seed, G, thetaML, dx,
                     dy)

    else:
        raise Exception("WFS type unknown")

    # fill sensor object with data

    for i in range(nsensors):
        p_wfs = p_wfss[i]
        wfs = g_wfs.d_wfs[i]
        fluxPerSub = p_wfs._fluxPerSub.T[np.where(p_wfs._isvalid.T > 0)].copy()
        if p_wfs.type == scons.WFSType.PYRHR or p_wfs.type == scons.WFSType.PYRLR:
            halfxy = np.exp(1j * p_wfs._halfxy).astype(np.complex64).T.copy()
            if (p_wfs._pyr_weights is None):
                p_wfs.set_pyr_weights(np.ones(p_wfs._pyr_cx.size))
            wfs.compute_pyrfocalplane = p_wfs.pyr_compute_focalplane
            wfs.load_arrays(halfxy, p_wfs._pyr_cx, p_wfs._pyr_cy, p_wfs._pyr_weights,
                            p_wfs._sincar, p_wfs._submask, p_wfs._validsubsx,
                            p_wfs._validsubsy, p_wfs._phasemap, fluxPerSub, 
                            p_wfs._ttprojmat)
        else:
            wfs.load_arrays(p_wfs._phasemap, p_wfs._hrmap, p_wfs._binmap, p_wfs._halfxy,
                            fluxPerSub, p_wfs._validsubsx, p_wfs._validsubsy,
                            p_wfs._validpuppixx, p_wfs._validpuppixy, p_wfs._ttprojmat,
                            p_wfs._ftkernel)
            if (p_wfs._submask is not None):
                g_wfs.set_field_stop(i, p_wfs._submask, p_wfs._submask.shape[0])

    # lgs case
    for i in range(nsensors):
        if (p_wfss[i].gsalt > 0):
            # lgs mode requested
            # init sensor lgs object with necessary data
            LGS.prep_lgs_prof(p_wfss[i], i, p_tel, g_wfs)

    type_target = "atmos"  # FIXME

    for i in range(len(p_wfss)):
        p_wfs = p_wfss[i]
        if p_wfs.gsalt > 0:
            gsalt = 1. / p_wfs.gsalt
        else:
            gsalt = 0

        if p_wfs.atmos_seen is not None and p_atmos is not None:
            for j in range(p_atmos.nscreens):
                xoff = (gsalt * p_atmos.alt[j] * p_tel.diam / 2. +
                        p_wfs.xpos * CONST.ARCSEC2RAD * p_atmos.alt[j]) / \
                    p_atmos.pupixsize
                yoff = (gsalt * p_atmos.alt[j] * p_tel.diam / 2. +
                        p_wfs.ypos * CONST.ARCSEC2RAD * p_atmos.alt[j]) / \
                    p_atmos.pupixsize
                xoff = xoff + (p_atmos.dim_screens[j] - p_geom._n) / 2.
                yoff = yoff + (p_atmos.dim_screens[j] - p_geom._n) / 2.
                g_wfs.d_wfs[i].d_gs.add_layer(type_target, j, xoff, yoff)

        if (not p_wfs.open_loop and p_dms is not None):
            if (p_wfs.dms_seen is None):
                p_wfs.dms_seen = np.arange(len(p_dms)).astype(np.int32)
            for j in range(p_wfs.dms_seen.size):
                k = p_wfs.dms_seen[j]
                dims = p_dms[k]._n2 - p_dms[k]._n1 + 1
                dim = p_geom._mpupil.shape[0]
                if (dim < dims):
                    dim = dims
                xoff = (gsalt * p_dms[k].alt * p_tel.diam / 2. + \
                        p_wfs.xpos * CONST.ARCSEC2RAD * p_dms[k].alt ) * \
                        p_geom.pupdiam / p_tel.diam
                yoff = (gsalt * p_dms[k].alt * p_tel.diam / 2. + \
                        p_wfs.ypos * CONST.ARCSEC2RAD * p_dms[k].alt ) * \
                        p_geom.pupdiam / p_tel.diam
                xoff = xoff + (dim - p_geom._n) / 2
                yoff = yoff + (dim - p_geom._n) / 2
                g_wfs.d_wfs[i].d_gs.add_layer(p_dms[k].type, k, xoff, yoff)

    return g_wfs
