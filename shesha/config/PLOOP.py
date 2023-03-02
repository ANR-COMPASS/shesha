## @package   shesha.config.PLOOP
## @brief     Param_loop class definition
## @author    COMPASS Team <https://github.com/ANR-COMPASS>
## @version   5.4.0
## @date      2022/01/24
## @copyright GNU Lesser General Public License
#
#  This file is part of COMPASS <https://anr-compass.github.io/compass/>
#
#  Copyright (C) 2011-2022 COMPASS Team <https://github.com/ANR-COMPASS>
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
from . import config_setter_utils as csu


#################################################
# P-Class (parametres) Param_loop
#################################################
class Param_loop:

    def __init__(self):
        self.__niter = 0
        self.__ittime = 0.
        self.__devices = np.array([0], dtype=np.int32)

    def get_devices(self):
        """ Get the list of GPU devices used

        :return: (np.ndarray[ndim=1, dtype=np.int32_t]) : list of GPU devices
        """
        return self.__devices

    def set_devices(self, devices):
        """ Set the list of GPU devices used

        Args:
            devices: (np.ndarray[ndim=1, dtype=np.int32_t]) : list of GPU devices
        """
        self.__devices = csu.enforce_array(devices, len(devices), dtype=np.int32,
                                           scalar_expand=False)

    devices = property(get_devices, set_devices)

    def get_niter(self):
        """ Get the number of iteration

        :return: (long) : number of iteration
        """
        return self.__niter

    def set_niter(self, n):
        """ Set the number of iteration

        Args:
            n: (long) : number of iteration
        """
        self.__niter = csu.enforce_int(n)

    niter = property(get_niter, set_niter)

    def get_ittime(self):
        """ Get iteration time

        :return: (float) :iteration time
        """
        return self.__ittime

    def set_ittime(self, t):
        """ Set iteration time

        Args:
            t: (float) :iteration time
        """
        self.__ittime = csu.enforce_float(t)

    ittime = property(get_ittime, set_ittime)
