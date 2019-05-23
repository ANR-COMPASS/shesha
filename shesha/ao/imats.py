""" @package shesha.ao.imats

Computation implementations of interaction matrix

"""

import numpy as np  # type: ignore
import time
from typing import List  # Mypy checker
from tqdm import tqdm

import shesha.config as conf
import shesha.constants as scons
import shesha.init.lgs_init as lgs
import shesha.util.hdf5_util as h5u

from shesha.sutra_wrap import Sensors, Dms, Rtc_FFF as Rtc
from shesha.constants import CONST

from astropy.io import fits


def imat_geom(wfs: Sensors, dms: Dms, p_wfss: List[conf.Param_wfs],
              p_dms: List[conf.Param_dm], p_controller: conf.Param_controller,
              meth: int = 0) -> np.ndarray:
    """ Compute the interaction matrix with a geometric method

    :parameters:

        wfs: (Sensors) : Sensors object

        dms: (Dms) : Dms object

        p_wfss: (list of Param_wfs) : wfs settings

        p_dms: (list of Param_dm) : dms settings

        p_controller: (Param_controller) : controller settings

        meth: (int) : (optional) method type (0 or 1)
    """

    nwfs = p_controller.nwfs.size
    ndm = p_controller.ndm.size
    imat_size1 = 0
    imat_size2 = 0

    for dm in dms.d_dms:
        dm.reset_shape()

    for nw in range(nwfs):
        nm = p_controller.nwfs[nw]
        imat_size1 += p_wfss[nm]._nvalid * 2

    for nmc in range(ndm):
        nm = p_controller.ndm[nmc]
        imat_size2 += p_dms[nm]._ntotact

    imat_cpu = np.zeros((imat_size1, imat_size2), dtype=np.float32)
    ind = 0
    cc = 0
    print("Doing imat geom...")
    for nmc in range(ndm):
        nm = p_controller.ndm[nmc]
        dms.d_dms[nm].reset_shape()
        for i in tqdm(range(p_dms[nm]._ntotact), desc="DM%d" % nmc):
            dms.d_dms[nm].comp_oneactu(i, p_dms[nm].push4imat)
            nslps = 0
            for nw in range(nwfs):
                n = p_controller.nwfs[nw]
                wfs.d_wfs[n].d_gs.raytrace(dms, rst=1)
                wfs.d_wfs[n].slopes_geom(meth)
                imat_cpu[nslps:nslps + p_wfss[n]._nvalid * 2, ind] = np.array(
                        wfs.d_wfs[n].d_slopes)
                nslps += p_wfss[n]._nvalid * 2
            imat_cpu[:, ind] = imat_cpu[:, ind] / p_dms[nm].push4imat
            ind = ind + 1
            cc = cc + 1
            dms.d_dms[nm].reset_shape()

    return imat_cpu


def imat_init(ncontrol: int, rtc: Rtc, dms: Dms, p_dms: list, wfs: Sensors, p_wfss: list,
              p_tel: conf.Param_tel, p_controller: conf.Param_controller, M2V=None,
              dataBase: dict = {}, use_DB: bool = False) -> None:
    """ Initialize and compute the interaction matrix on the GPU

    :parameters:

        ncontrol: (int) : controller's index

        rtc: (Rtc) : Rtc object

        dms: (Dms) : Dms object

        p_dms: (Param_dms) : dms settings

        wfs: (Sensors) : Sensors object

        p_wfss: (list of Param_wfs) : wfs settings

        p_tel: (Param_tel) : telescope settings

        p_controller: (Param_controller) : controller settings

        M2V:(np.array) :  KL_matrix

        dataBase:(dict): (optional) dict containing paths to files to load

        use_DB:(bool) : (optional) use dataBase flag
    """
    # first check if wfs is using lgs
    # if so, load new lgs spot, just for imat
    for i in range(len(p_wfss)):
        if (p_wfss[i].gsalt > 0):
            # TODO: check that
            save_profile = p_wfss[i].proftype
            p_wfss[i].proftype = scons.ProfType.GAUSS1
            lgs.prep_lgs_prof(p_wfss[i], i, p_tel, wfs, imat=1)

    if "imat" in dataBase:
        imat = h5u.load_imat_from_dataBase(dataBase)
        rtc.d_control[ncontrol].set_imat(imat)
    else:
        t0 = time.time()
        if M2V is not None:
            p_controller._M2V = M2V.copy()
            rtc.do_imat_basis(ncontrol, dms, M2V.shape[1], M2V, p_controller.klpush)
        else:
            rtc.do_imat(ncontrol, dms)
        print("done in %f s" % (time.time() - t0))
        imat = np.array(rtc.d_control[ncontrol].d_imat)
        if use_DB:
            h5u.save_imat_in_dataBase(imat)
    p_controller.set_imat(imat)

    # Restore original profile in lgs spots
    for i in range(len(p_wfss)):
        if (p_wfss[i].gsalt > 0):
            p_wfss[i].proftype = save_profile
            lgs.prep_lgs_prof(p_wfss[i], i, p_tel, wfs)


#write imat_ts:
#   loop over ts directions
#   change WFS offset to direction
#   do imat geom


def imat_geom_ts_multiple_direction(wfs: Sensors, dms: Dms, p_wfss: List[conf.Param_wfs],
                                    p_dms: List[conf.Param_dm], p_geom: conf.Param_geom,
                                    ind_TS: int, p_tel: conf.Param_tel, x, y,
                                    meth: int = 0) -> np.ndarray:
    """ Compute the interaction matrix with a geometric method for multiple truth sensors (with different direction)

    :parameters:

        wfs: (Sensors) : Sensors object

        dms: (Dms) : Dms object

        p_wfss: (list of Param_wfs) : wfs settings

        ind_TS: (int) : index of the truth sensor in the wfs settings list

        p_dms: (list of Param_dm) : dms settings

        ind_DMs: (list of int) : indices of used DMs

        p_controller: (Param_controller) : controller settings

        meth: (int) : (optional) method type (0 or 1)
    """
    p_wfs = p_wfss[ind_TS]
    imat_size2 = 0
    print("DMS_SEEN: ", p_wfs.dms_seen)
    for nm in p_wfs.dms_seen:
        imat_size2 += p_dms[nm]._ntotact
    imat_cpu = np.ndarray((0, imat_size2))

    for i in range(x.size):
        xpos = x[i]
        ypos = y[i]
        for j in range(p_wfs.dms_seen.size):
            k = p_wfs.dms_seen[j]
            dims = p_dms[k]._n2 - p_dms[k]._n1 + 1
            dim = p_geom._mpupil.shape[0]
            if (dim < dims):
                dim = dims
            xoff = xpos * CONST.ARCSEC2RAD * \
                p_dms[k].alt / p_tel.diam * p_geom.pupdiam
            yoff = ypos * CONST.ARCSEC2RAD * \
                p_dms[k].alt / p_tel.diam * p_geom.pupdiam
            xoff = xoff + (dim - p_geom._n) / 2
            yoff = yoff + (dim - p_geom._n) / 2
            wfs.d_wfs[ind_TS].d_gs.add_layer("dm", k, xoff, yoff)
        imat_cpu = np.concatenate(
                (imat_cpu,
                 imat_geom_ts(wfs, dms, p_wfss, ind_TS, p_dms, p_wfs.dms_seen, meth)),
                axis=0)
    return imat_cpu


def imat_geom_ts(wfs: Sensors, dms: Dms, p_wfss: conf.Param_wfs, ind_TS: int,
                 p_dms: List[conf.Param_dm], ind_DMs: List[int],
                 meth: int = 0) -> np.ndarray:
    """ Compute the interaction matrix with a geometric method for a single truth sensor

    :parameters:

        wfs: (Sensors) : Sensors object

        dms: (Dms) : Dms object

        p_wfss: (list of Param_wfs) : wfs settings

        ind_TS: (int) : index of the truth sensor in the wfs settings list

        p_dms: (list of Param_dm) : dms settings

        ind_DMs: (list of int) : indices of used DMs

        p_controller: (Param_controller) : controller settings

        meth: (int) : (optional) method type (0 or 1)
    """

    #nwfs = 1 #p_controller.nwfs.size # as parameter list of indices for wfs if several ts (only 1 ts for now)
    ndm = len(ind_DMs)  #p_controller.ndm.size # as parameter list of indices of used dms
    imat_size1 = p_wfss[ind_TS]._nvalid * 2  # as parameter (nvalid)
    imat_size2 = 0

    # for nw in range(nwfs):
    #     nm = p_controller.nwfs[nw]
    #     imat_size1 += p_wfss[nm]._nvalid * 2

    for dm in dms.d_dms:
        dm.reset_shape()

    imat_size2 = 0
    for nm in ind_DMs:
        imat_size2 += p_dms[nm]._ntotact

    imat_cpu = np.zeros((imat_size1, imat_size2), dtype=np.float64)
    ind = 0
    cc = 0
    print("Doing imat geom...")
    for nm in ind_DMs:
        dms.d_dms[nm].reset_shape()
        for i in tqdm(range(p_dms[nm]._ntotact), desc="DM%d" % nm):
            dms.d_dms[nm].comp_oneactu(i, p_dms[nm].push4imat)
            wfs.d_wfs[ind_TS].d_gs.raytrace("dm")
            wfs.slopes_geom(ind_TS, meth)
            imat_cpu[:, ind] = np.array(wfs.d_wfs[ind_TS].d_slopes)
            imat_cpu[:, ind] = imat_cpu[:, ind] / p_dms[nm].push4imat
            ind = ind + 1
            cc = cc + 1
            dms.d_dms[nm].reset_shape()

    return imat_cpu
