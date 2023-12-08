## @package   shesha.util.rtc_util
## @brief     Some utilities functions for RTC
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

import numpy as np


def create_interp_mat(dimx: int, dimy: int):
    """TODO doc

    Args:

        dimx: (int) :

        dimy: (int) :
    """
    n = max(dimx, dimy)
    tmp2, tmp1 = np.indices((n, n))
    tmp1 = tmp1[:dimx, :dimy] - (dimx // 2)
    tmp2 = tmp2[:dimx, :dimy] - (dimy // 2)

    tmp = np.zeros((tmp1.size, 6), np.int32)
    tmp[:, 0] = (tmp1**2).flatten()
    tmp[:, 1] = (tmp2**2).flatten()
    tmp[:, 2] = (tmp1 * tmp2).flatten()
    tmp[:, 3] = tmp1.flatten()
    tmp[:, 4] = tmp2.flatten()
    tmp[:, 5] = 1

    return np.dot(np.linalg.inv(np.dot(tmp.T, tmp)), tmp.T).T


def centroid_gain(E, F):
    """ Returns the mean centroid gain

    Args:

        E : (np.array(dtype=np.float32)) : measurements from WFS

        F : (np.array(dtype=np.float32)) : geometric measurements

    :return:

        cgain : (float) : mean centroid gain between the sets of WFS measurements and geometric ones
    """
    if E.ndim == 1:
        cgains = np.polyfit(E, F, 1)[0]
    elif E.ndim == 2:
        cgains = np.zeros(E.shape[1])
        for k in range(E.shape[1]):
            cgains[k] = np.polyfit(E[:, k], F[:, k], 1)[0]
        cgains = np.mean(cgains)
    else:
        raise ValueError("E and F must 1D or 2D arrays")

    return cgains
