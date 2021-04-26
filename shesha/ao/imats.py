## @package   shesha.ao.imats
## @brief     Computation implementations of interaction matrix
## @author    COMPASS Team <https://github.com/ANR-COMPASS>
## @version   5.1.0
## @date      2020/05/18
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

import numpy as np  # type: ignore
import time
from typing import List  # Mypy checker
from tqdm import tqdm, trange

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

    Args:

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

    Args:

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


def imat_geom_ts_multiple_direction(wfs: Sensors, dms: Dms, p_ts: conf.Param_wfs,
                                    p_dms: List[conf.Param_dm], p_geom: conf.Param_geom,
                                    ind_TS: int, ind_dmseen: List, p_tel: conf.Param_tel,
                                    x, y, meth: int = 0) -> np.ndarray:
    """ Compute the interaction matrix with a geometric method for multiple truth sensors (with different direction)

    Args:
        wfs: (Sensors) : Sensors object

        dms: (Dms) : Dms object

        p_ts: (Param_wfs) : truth sensor settings

        ind_TS: (int) : index of the truth sensor in Sensors (wfs)

        p_dms: (list of Param_dm) : dms settings

        ind_DMs: (list of int) : indices of used DMs

        p_controller: (Param_controller) : controller settings

    Kwargs:
        meth: (int) : (optional) method type (0 or 1)
    """
    imat_size2 = 0
    print("DMS_SEEN: ", ind_dmseen)
    for nm in ind_dmseen:
        imat_size2 += p_dms[nm]._ntotact
    imat_cpu = np.ndarray((0, imat_size2))

    for i in trange(x.size, desc="TS pos"):
        xpos = x[i]
        ypos = y[i]
        for k in ind_dmseen:
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
            wfs.d_wfs[ind_TS].d_gs.remove_layer(p_dms[k].type, k)
            wfs.d_wfs[ind_TS].d_gs.add_layer(p_dms[k].type, k, xoff, yoff)
        imat_cpu = np.concatenate(
                (imat_cpu, imat_geom_ts(wfs, dms, p_ts, ind_TS, p_dms, ind_dmseen,
                                        meth)), axis=0)

    return imat_cpu


def imat_geom_ts(wfs: Sensors, dms: Dms, p_ts: conf.Param_wfs, ind_TS: int,
                 p_dms: List[conf.Param_dm], ind_DMs: List[int],
                 meth: int = 0) -> np.ndarray:
    """ Compute the interaction matrix with a geometric method for a single truth sensor

    Args:
        wfs: (Sensors) : Sensors object

        dms: (Dms) : Dms object

        p_ts: (Param_wfs) : truth sensor settings

        ind_TS: (int) : index of the truth sensor in Sensors (wfs)

        p_dms: (list of Param_dm) : dms settings

        ind_DMs: (list of int) : indices of used DMs

        p_controller: (Param_controller) : controller settings

    Kwargs:
        meth: (int) : (optional) method type (0 or 1)
    """

    #nwfs = 1 #p_controller.nwfs.size # as parameter list of indices for wfs if several ts (only 1 ts for now)
    ndm = len(ind_DMs)  #p_controller.ndm.size # as parameter list of indices of used dms
    imat_size1 = p_ts._nvalid * 2  # as parameter (nvalid)
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
    for nm in tqdm(ind_DMs, desc="imat geom DM"):
        dms.d_dms[nm].reset_shape()
        for i in trange(p_dms[nm]._ntotact, desc="imat geom actu"):
            dms.d_dms[nm].comp_oneactu(i, p_dms[nm].push4imat)
            wfs.d_wfs[ind_TS].d_gs.raytrace(dms, rst=1)
            wfs.d_wfs[ind_TS].slopes_geom(meth)
            imat_cpu[:, ind] = np.array(wfs.d_wfs[ind_TS].d_slopes)
            imat_cpu[:, ind] = imat_cpu[:, ind] / p_dms[nm].push4imat
            ind = ind + 1
            cc = cc + 1
            dms.d_dms[nm].reset_shape()

    return imat_cpu


def get_metaD(sup, p_wfs, TS_xpos=None, TS_ypos=None, ind_TS=-1, n_control=0):
    """Create an interaction matrix for the current simulation given TS position

    Args:
        sup : (CompassSupervisor) : current COMPASS simulation

        p_ts: (Param_wfs) : truth sensor settings

        TS_xpos : np.ndarray : TS position (x axis)

        TS_ypos : np.ndarray : TS position (y axis)

        ind_TS: (int) : index of the truth sensor in Sensors (wfs)

        n_control : (int) : index of the controller

    :return:
        metaD :  np.ndarray :interaction matrix
    """
    if (TS_xpos is None):
        TS_xpos = np.array([t.xpos for t in sup.config.p_wfs_ts])
    elif (isinstance(TS_xpos, list)):
        TS_xpos = np.array(TS_xpos)
    elif (isinstance(TS_xpos, int) or isinstance(TS_xpos, float)):
        TS_xpos = np.array([TS_xpos]).astype(np.float32)
    if (TS_xpos.size < 1):
        TS_xpos = np.zeros((1))

    if (TS_ypos is None):
        TS_ypos = np.array([t.ypos for t in sup.config.p_wfs_ts])
    elif (isinstance(TS_ypos, list)):
        TS_ypos = np.array(TS_ypos)
    elif (isinstance(TS_ypos, int) or isinstance(TS_ypos, float)):
        TS_ypos = np.array([TS_ypos]).astype(np.float32)
    if (TS_ypos.size < 1):
        TS_ypos = np.zeros((1))

    return imat_geom_ts_multiple_direction(sup.wfs._wfs, sup.dms._dms, p_wfs,
                                           sup.config.p_dms, sup.config.p_geom, ind_TS,
                                           sup.config.p_controllers[n_control].ndm,
                                           sup.config.p_tel, TS_xpos, TS_ypos)
