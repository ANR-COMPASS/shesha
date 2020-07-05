## @package   shesha.init.atmos_init
## @brief     Initialization of a Atmos object
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

import shesha.config as conf
from shesha.constants import CONST
import shesha.util.iterkolmo as itK
import shesha.util.hdf5_util as h5u
from shesha.sutra_wrap import carmaWrap_context, Atmos
from tqdm import tqdm
import numpy as np


def atmos_init(context: carmaWrap_context, p_atmos: conf.Param_atmos,
               p_tel: conf.Param_tel, p_geom: conf.Param_geom, ittime=None, p_wfss=None,
               p_targets=None, dataBase={}, use_DB=False):
    """
    Initializes an Atmos object

    :parameters:
        context: (carmaWrap_context): GPU device context
        p_atmos: (Param_atmos): Atmosphere parameters
        p_tel: (Param_tel): Telescope parameters
        p_geom: (Param_geom): Geometry parameters
        ittime: (float): (optional) exposition time [s]
        p_wfss: (list of Param_wfs): (optional) WFS parameters
        p_targets: (list of Param_target): (optional) target parameters
        dataBase: (dict): (optional) dictionary for data base
        use_DB: (bool): (optional) flag for using the dataBase system
    :return:
        atm : (Atmos): Atmos object
    """
    if not p_geom.is_init:
        raise RuntimeError("Cannot init atmosphere with uninitialized p_geom.")

    # Deleted carmaWrap_context : get the singleton

    if p_atmos.r0 is None:
        p_atmos.r0 = 0.

    if ittime is None:
        ittime = 1.
    # Adjust layers alt using zenith angle
    p_atmos.alt = p_atmos.alt / np.cos(p_geom.zenithangle * CONST.DEG2RAD)
    # Pixel size in meters
    p_atmos.pupixsize = p_tel.diam / p_geom.pupdiam

    # Off-axis wavefront sensors and targets
    # Note : p_wfss is a list of single-WFS configs
    #        but p_target groups several targets
    #        hence different xpos, ypos syntaxes
    norms = [0.]
    if p_wfss is not None:
        norms += [(w.xpos**2 + w.ypos**2)**0.5 for w in p_wfss]
    if p_targets is not None:
        norms += [(p_target.xpos**2 + p_target.ypos**2)**0.5 for p_target in p_targets]

    max_size = max(norms)

    # Meta-pupil diameter for all layers depending on altitude
    patch_diam = (p_geom._n + 2 * (max_size * CONST.ARCSEC2RAD * p_atmos.alt) /
                  p_atmos.pupixsize + 4).astype(np.int64)
    p_atmos.dim_screens = (patch_diam + patch_diam % 2)

    # Phase screen speeds
    lin_delta = p_geom.pupdiam / p_tel.diam * p_atmos.windspeed * \
        np.cos(CONST.DEG2RAD * p_geom.zenithangle) * ittime
    p_atmos._deltax = lin_delta * np.sin(CONST.DEG2RAD * p_atmos.winddir + np.pi)
    p_atmos._deltay = lin_delta * np.cos(CONST.DEG2RAD * p_atmos.winddir + np.pi)

    # Fraction normalization
    p_atmos.frac /= np.sum(p_atmos.frac)

    if p_atmos.L0 is None:
        # Set almost infinite L0
        p_atmos.L0 = np.ones(p_atmos.nscreens, dtype=np.float32) * 1e5
    L0_pix = p_atmos.L0 * p_geom.pupdiam / p_tel.diam

    if p_atmos.seeds is None:
        p_atmos.seeds = np.arange(p_atmos.nscreens, dtype=np.int64) + 1234

    r0_layers = p_atmos.r0 / (p_atmos.frac**(3. / 5.) * p_atmos.pupixsize)
    stencil_size = itK.stencil_size_array(p_atmos.dim_screens)

    atm = Atmos(context, p_atmos.nscreens, p_atmos.r0, r0_layers, p_atmos.dim_screens,
                stencil_size, p_atmos.alt, p_atmos.windspeed, p_atmos.winddir,
                p_atmos._deltax, p_atmos._deltay, context.active_device)

    print("Creating turbulent layers :")
    for i in tqdm(range(p_atmos.nscreens)):
        if "A" in dataBase:
            A, B, istx, isty = h5u.load_AB_from_dataBase(dataBase, i)
        else:
            A, B, istx, isty = itK.AB(p_atmos.dim_screens[i], L0_pix[i],
                                      p_atmos._deltax[i], p_atmos._deltay[i], 0)
            if use_DB:
                h5u.save_AB_in_database(i, A, B, istx, isty)

        atm.init_screen(i, A, B, istx, isty, p_atmos.seeds[i])

    return atm
