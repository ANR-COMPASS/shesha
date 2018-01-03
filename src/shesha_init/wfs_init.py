'''
Initialization of a Sensors object
'''

import shesha_config as conf
import shesha_constants as scons
from shesha_constants import CONST

from . import lgs_init as LGS
from sutra_bind.wrap import naga_context, Sensors, Telescope
import numpy as np


def wfs_init(context: naga_context, telescope: Telescope, p_wfss: list,
             p_tel: conf.Param_tel, p_geom: conf.Param_geom, p_dms=None, p_atmos=None):
    """
    Create and initialise  a Sensors object

    :parameters:
        context : (naga_context)
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
    t_wfs = [o.type for o in p_wfss]
    # cdef np.ndarray t_wfs  = np.array([o.type  for o in
    # wfs],dtype=np.str)
    nxsub = np.array([o.nxsub for o in p_wfss], dtype=np.int64)
    nvalid = np.array([o._nvalid for o in p_wfss], dtype=np.int64)
    nphase = np.array([o._pdiam for o in p_wfss], dtype=np.int64)
    pdiam = np.array([o._subapd for o in p_wfss], dtype=np.float32)
    npix = np.array([o.npix for o in p_wfss], dtype=np.int64)
    nrebin = np.array([o._nrebin for o in p_wfss], dtype=np.int64)
    nfft = np.array([o._Nfft for o in p_wfss], dtype=np.int64)
    ntota = np.array([o._Ntot for o in p_wfss], dtype=np.int64)
    nphot = np.array([o._nphotons for o in p_wfss], dtype=np.float32)
    nphot4imat = np.array([o.nphotons4imat for o in p_wfss], dtype=np.float32)
    lgs = np.array([o.gsalt > 0 for o in p_wfss], dtype=np.int32)

    # arrays needed to call sensors_initgs
    xpos = np.array([o.xpos for o in p_wfss], dtype=np.float32)
    ypos = np.array([o.ypos for o in p_wfss], dtype=np.float32)
    Lambda = np.array([o.Lambda for o in p_wfss], dtype=np.float32)
    zerop = p_wfss[0].zerop
    size = np.zeros(nsensors, dtype=np.int64) + p_geom._n
    seed = np.array([], dtype=np.int64)
    npup = (np.zeros((nsensors)) + p_geom._n).astype(np.int64)

    G = np.array([o.G for o in p_wfss], dtype=np.float32)
    thetaML = np.array([o.thetaML for o in p_wfss], dtype=np.float32)
    dx = np.array([o.dx for o in p_wfss], dtype=np.float32)
    dy = np.array([o.dy for o in p_wfss], dtype=np.float32)

    error_budget_flag = any([w.error_budget for w in p_wfss])

    if (p_wfss[0].type == scons.WFSType.SH):
        g_wfs = Sensors(context, nsensors, telescope, t_wfs, npup, nxsub, nvalid, nphase,
                        pdiam, npix, nrebin, nfft, ntota, nphot, nphot4imat, lgs,
                        error_budget=error_budget_flag)

        mag = np.array([o.gsmag for o in p_wfss], dtype=np.float32)
        noise = np.array([o.noise for o in p_wfss], dtype=np.float32)

        g_wfs.init_gs(xpos, ypos, Lambda, mag, zerop, size, noise, seed, G, thetaML, dx,
                      dy)

    elif (p_wfss[0].type == scons.WFSType.PYRHR):
        npup = np.array([o.pyr_npts for o in p_wfss])
        G = np.array([o.G for o in p_wfss], dtype=np.float32)
        thetaML = np.array([o.thetaML for o in p_wfss], dtype=np.float32)
        dx = np.array([o.dx for o in p_wfss], dtype=np.float32)
        dy = np.array([o.dy for o in p_wfss], dtype=np.float32)

        g_wfs = Sensors(context, nsensors, telescope, t_wfs, npup, nxsub, nvalid, nphase,
                        pdiam, npix, nrebin, nfft, ntota, nphot, nphot4imat, lgs,
                        error_budget=error_budget_flag)

        mag = np.array([o.gsmag for o in p_wfss], dtype=np.float32)
        noise = np.array([o.noise for o in p_wfss], dtype=np.float32)
        g_wfs.init_gs(xpos, ypos, Lambda, mag, zerop, size, noise, seed, G, thetaML, dx,
                      dy)

    else:
        raise Exception("WFS type unknown")

    # fill sensor object with data
    for i in range(nsensors):
        wfs_initarr(g_wfs, i, p_wfss[i])

    # lgs case
    for i in range(nsensors):
        if (p_wfss[i].gsalt > 0):
            # lgs mode requested
            # init sensor lgs object with necessary data
            LGS.prep_lgs_prof(p_wfss[i], i, p_tel, g_wfs)

    type_target = b"atmos"  # FIXME

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
                g_wfs.add_layer(i, type_target, p_atmos.alt[j], xoff, yoff)

        if (not p_wfs.openloop and p_dms is not None):
            if (p_wfs.dms_seen is None):
                p_wfs.dms_seen = np.arange(len(p_dms)).astype(np.int32)
            for j in range(p_wfs.dms_seen.size):
                k = p_wfs.dms_seen[j]
                dims = p_dms[k]._n2 - p_dms[k]._n1 + 1
                dim = p_geom._mpupil.shape[0]
                if (dim < dims):
                    dim = dims
                xoff = p_wfs.xpos * CONST.ARCSEC2RAD * \
                    p_dms[k].alt / p_tel.diam * p_geom.pupdiam
                yoff = p_wfs.ypos * CONST.ARCSEC2RAD * \
                    p_dms[k].alt / p_tel.diam * p_geom.pupdiam
                xoff = xoff + (dim - p_geom._n) / 2
                yoff = yoff + (dim - p_geom._n) / 2
                g_wfs.add_layer(i, p_dms[k].type, p_dms[k].alt, xoff, yoff)

    return g_wfs


def wfs_initarr(wfs: Sensors, i: int, p_wfs: conf.Param_wfs):
    """ Wrapper for the cython function Sensors.sensors_initarrays

    :parameters:
        wfs: (Sensors) : Sensors object
        i: (int) : wfs index
        p_wfs: (Param_wfs): wfs parameters
    """
    fluxPerSub = p_wfs._fluxPerSub.T[np.where(p_wfs._isvalid > 0)].copy()
    if p_wfs.type == scons.WFSType.PYRHR:
        halfxy = np.exp(1j * p_wfs._halfxy).astype(np.complex64).T.copy()
    else:
        halfxy = p_wfs._halfxy
    wfs.init_arrays(i, p_wfs._phasemap, p_wfs._hrmap, halfxy, fluxPerSub,
                    p_wfs._validsubsx, p_wfs._validsubsy, p_wfs._istart + 1,
                    p_wfs._jstart + 1, p_wfs._binmap, p_wfs._ftkernel, p_wfs._pyr_cx,
                    p_wfs._pyr_cy, p_wfs._sincar, p_wfs._submask)
