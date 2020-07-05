## @package   shesha.ao.tomo
## @brief     Computation of tomographic reconstructor
## @author    COMPASS Team <https://github.com/ANR-COMPASS>
## @version   5.0.0
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

import numpy as np

import shesha.config as conf
import shesha.constants as scons
from shesha.constants import CONST
from shesha.sutra_wrap import Sensors, Dms, Rtc_FFF as Rtc, Atmos

import typing
from typing import List


def do_tomo_matrices(ncontrol: int, rtc: Rtc, p_wfss: List[conf.Param_wfs], dms: Dms,
                     atmos: Atmos, wfs: Sensors, p_controller: conf.Param_controller,
                     p_geom: conf.Param_geom, p_dms: list, p_tel: conf.Param_tel,
                     p_atmos: conf.Param_atmos):
    """ Compute Cmm and Cphim matrices for the MV controller on GPU

    :parameters:

        ncontrol: (int): controller index

        rtc: (Rtc) : rtc object

        p_wfss: (list of Param_wfs) : wfs settings

        dms: (Dms) : Dms object

        atmos: (Atmos) : Atmos object

        wfs: (Sensors) : Sensors object

        p_controller: (Param_controller): controller settings

        p_geom: (Param_geom) : geom settings

        p_dms: (list of Param_dms) : dms settings

        p_tel: (Param_tel) : telescope settings

        p_atmos: (Param_atmos) : atmos settings
    """
    nvalidperwfs = np.array([o._nvalid for o in p_wfss], dtype=np.int64)
    # Bring bottom left corner of valid subapertures in ipupil
    ipup = p_geom._ipupil
    spup = p_geom._spupil
    s2ipup = (ipup.shape[0] - spup.shape[0]) / 2.
    # Total number of subapertures
    nvalid = sum([nvalidperwfs[j] for j in p_controller.nwfs])
    ind = 0
    # X-position of the bottom left corner of each valid subaperture
    X = np.zeros(nvalid, dtype=np.float64)
    # Y-position of the bottom left corner of each subaperture
    Y = np.zeros(nvalid, dtype=np.float64)

    for k in p_controller.nwfs:
        posx = p_wfss[k]._validpuppixx + s2ipup
        # X-position of the bottom left corner of each valid subaperture
        # bring the origin in the center of ipupil and 0-index it
        posx = posx - ipup.shape[0] / 2 - 1
        posy = p_wfss[k]._validpuppixy + s2ipup
        posy = posy.T - ipup.shape[0] / 2 - 1
        p2m = (p_tel.diam / p_wfss[k].nxsub) / \
            p_wfss[k]._pdiam  # Size of one pixel in meters
        posx *= p2m  # Positions in meters
        posy *= p2m
        X[ind:ind + p_wfss[k]._nvalid] = posx
        Y[ind:ind + p_wfss[k]._nvalid] = posy
        ind += p_wfss[k]._nvalid

    # Get the total number of pzt DM and actuators to control
    nactu = 0
    npzt = 0
    for k in p_controller.ndm:
        if (p_dms[k].type == scons.DmType.PZT):
            nactu += p_dms[k]._ntotact
            npzt += 1

    Xactu = np.zeros(nactu, dtype=np.float64)  # X-position actuators in ipupil
    Yactu = np.zeros(nactu, dtype=np.float64)  # Y-position actuators in ipupil
    k2 = np.zeros(npzt, dtype=np.float64)  # k2 factors for computation
    pitch = np.zeros(npzt, dtype=np.float64)
    alt_DM = np.zeros(npzt, dtype=np.float64)
    ind = 0
    indk = 0
    for k in p_controller.ndm:
        if (p_dms[k].type == scons.DmType.PZT):
            p2m = p_tel.diam / p_geom.pupdiam
            # Conversion in meters in the center of ipupil
            actu_x = (p_dms[k]._xpos - ipup.shape[0] / 2) * p2m
            actu_y = (p_dms[k]._ypos - ipup.shape[0] / 2) * p2m
            pitch[indk] = actu_x[1] - actu_x[0]
            k2[indk] = p_wfss[0].Lambda / 2. / np.pi / p_dms[k].unitpervolt
            alt_DM[indk] = p_dms[k].alt
            Xactu[ind:ind + p_dms[k]._ntotact] = actu_x
            Yactu[ind:ind + p_dms[k]._ntotact] = actu_y

            ind += p_dms[k]._ntotact
            indk += 1

    # Select a DM for each layer of atmos
    NlayersDM = np.zeros(npzt, dtype=np.int64)  # Useless for now
    indlayersDM = selectDMforLayers(p_atmos, p_controller, p_dms)
    # print("indlayer = ",indlayersDM)

    # Get FoV
    # conf.RAD2ARCSEC = 180.0/np.pi * 3600.
    wfs_distance = np.zeros(len(p_controller.nwfs), dtype=np.float64)
    ind = 0
    for k in p_controller.nwfs:
        wfs_distance[ind] = np.sqrt(p_wfss[k].xpos**2 + p_wfss[k].ypos**2)
        ind += 1
    FoV = np.max(wfs_distance) / CONST.RAD2ARCSEC

    # WFS postions in rad
    alphaX = np.zeros(len(p_controller.nwfs))
    alphaY = np.zeros(len(p_controller.nwfs))

    ind = 0
    for k in p_controller.nwfs:
        alphaX[ind] = p_wfss[k].xpos / CONST.RAD2ARCSEC
        alphaY[ind] = p_wfss[k].ypos / CONST.RAD2ARCSEC
        ind += 1

    L0_d = np.copy(p_atmos.L0).astype(np.float64)
    frac_d = np.copy(p_atmos.frac * (p_atmos.r0**(-5.0 / 3.0))).astype(np.float64)

    print("Computing Cphim...")
    rtc.d_control[ncontrol].compute_Cphim(atmos, wfs, dms, L0_d, frac_d, alphaX, alphaY,
                                          X, Y, Xactu, Yactu, p_tel.diam, k2, NlayersDM,
                                          indlayersDM, FoV, pitch,
                                          alt_DM.astype(np.float64))
    print("Done")

    print("Computing Cmm...")
    rtc.d_control[ncontrol].compute_Cmm(atmos, wfs, L0_d, frac_d, alphaX, alphaY,
                                        p_tel.diam, p_tel.cobs)
    print("Done")

    Nact = np.zeros([nactu, nactu], dtype=np.float32)
    F = np.zeros([nactu, nactu], dtype=np.float32)
    ind = 0
    for k in range(len(p_controller.ndm)):
        if (p_dms[k].type == "pzt"):
            Nact[ind:ind + p_dms[k]._ntotact, ind:ind +
                 p_dms[k]._ntotact] = create_nact_geom(p_dms[k])
            F[ind:ind + p_dms[k]._ntotact, ind:ind +
              p_dms[k]._ntotact] = create_piston_filter(p_dms[k])
            ind += p_dms[k]._ntotact
    rtc.d_control[ncontrol].filter_cphim(F, Nact)


def selectDMforLayers(p_atmos: conf.Param_atmos, p_controller: conf.Param_controller,
                      p_dms: list):
    """ For each atmos layer, select the DM which have to handle it in the Cphim computation for MV controller

    :parameters:

        p_atmos : (Param_atmos) : atmos parameters

        p_controller : (Param_controller) : controller parameters

        p_dms :(list of Param_dm) : dms parameters

    :return:

        indlayersDM : (np.array(dtype=np.int32)) : for each atmos layer, the Dm number corresponding to it
    """
    indlayersDM = np.zeros(p_atmos.nscreens, dtype=np.int64)
    for i in range(p_atmos.nscreens):
        mindif = 1e6
        for j in p_controller.ndm:
            alt_diff = np.abs(p_dms[j].alt - p_atmos.alt[i])
            if (alt_diff < mindif):
                indlayersDM[i] = j
                mindif = alt_diff

    return indlayersDM


def create_nact_geom(p_dm: conf.Param_dm):
    """ Compute the DM coupling matrix

    :param:

        p_dm : (Param_dm) : dm parameters

    :return:

        Nact : (np.array(dtype=np.float64)) : the DM coupling matrix
    """
    nactu = p_dm._ntotact
    Nact = np.zeros([nactu, nactu], dtype=np.float32)
    coupling = p_dm.coupling
    dim = p_dm._n2 - p_dm._n1 + 1
    mask = np.zeros([dim, dim], dtype=np.float32)
    shape = np.zeros([dim, dim], dtype=np.float32)

    for i in range(len(p_dm._i1)):
        mask[p_dm._j1[i]][p_dm._i1[i]] = 1

    mask_act = np.where(mask)

    pitch = int(p_dm._pitch)

    for i in range(nactu):
        shape *= 0
        # Diagonal
        shape[p_dm._j1[i]][p_dm._i1[i]] = 1
        # Left, right, above and under the current actuator
        shape[p_dm._j1[i]][p_dm._i1[i] - pitch] = coupling
        shape[p_dm._j1[i] - pitch][p_dm._i1[i]] = coupling
        shape[p_dm._j1[i]][p_dm._i1[i] + pitch] = coupling
        shape[p_dm._j1[i] + pitch][p_dm._i1[i]] = coupling
        # Diagonals of the current actuators
        shape[p_dm._j1[i] - pitch][p_dm._i1[i] - pitch] = coupling**2
        shape[p_dm._j1[i] - pitch][p_dm._i1[i] + pitch] = coupling**2
        shape[p_dm._j1[i] + pitch][p_dm._i1[i] + pitch] = coupling**2
        shape[p_dm._j1[i] + pitch][p_dm._i1[i] - pitch] = coupling**2

        Nact[:, i] = shape[mask_act]

    return Nact


def create_piston_filter(p_dm: conf.Param_dm):
    """ Create the piston filter matrix

    :parameters:

        p_dm: (Param_dm): dm settings
    """
    nactu = p_dm._ntotact
    F = np.ones([nactu, nactu], dtype=np.float32)
    F = F * (-1.0 / nactu)
    for i in range(nactu):
        F[i][i] = 1 - 1.0 / nactu
    return F
