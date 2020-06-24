## @package   shesha.config.PTARGET
## @brief     Param_target class definition
## @author    COMPASS Team <https://github.com/ANR-COMPASS>
## @version   4.4.2
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

import numpy as np
from . import config_setter_utils as csu

#################################################
# P-Class (parametres) Param_target
#################################################


class Param_target:

    def __init__(self):
        self.__apod = False
        """ boolean for apodizer"""
        self.__Lambda = None
        """ observation wavelength for each target"""
        self.__xpos = None
        """ x positions on sky (in arcsec) for each target"""
        self.__ypos = None
        """ y positions on sky (in arcsec) for each target"""
        self.__mag = None
        """ magnitude for each target"""
        self.__zerop = 1.
        """ target flux for magnitude 0"""
        self.__dms_seen = None
        """ index of dms seen by the target"""

    def get_apod(self):
        """ Get apodizer flag

        :return: (bool) : apod
        """
        return self.__apod

    def set_apod(self, l):
        """ Set apodizer flag

        :param l: (bool) : apod
        """
        self.__apod = csu.enforce_or_cast_bool(l)

    apod = property(get_apod, set_apod)

    def get_Lambda(self):
        """ Get the wavelength of targets

        :return: (np.ndarray[ndim=2, dtype=np.float32]) : wavelength of targets
        """
        return self.__Lambda

    def set_Lambda(self, n):
        """ Set the wavelength of targets

        :param n: (np.ndarray[ndim=2, dtype=np.float32]) : wavelength of targets
        """
        self.__Lambda = csu.enforce_float(n)

    Lambda = property(get_Lambda, set_Lambda)

    def get_xpos(self):
        """ Get the X-position of targets in the field [arcsec]

        :return: (np.ndarray[ndim=2, dtype=np.float32]) : X position of targets [arcsec]
        """
        return self.__xpos

    def set_xpos(self, n):
        """ Set the X-position of targets in the field [arcsec]

        :param n: (np.ndarray[ndim=2, dtype=np.float32]) : X position of targets [arcsec]
        """
        self.__xpos = csu.enforce_float(n)

    xpos = property(get_xpos, set_xpos)

    def get_ypos(self):
        """ Get the Y-position of targets in the field [arcsec]

        :return: (np.ndarray[ndim=2, dtype=np.float32]): Y position of targets [arcsec]
        """
        return self.__ypos

    def set_ypos(self, n):
        """ Set the Y-position of targets in the field [arcsec]

        :param n: (np.ndarray[ndim=2, dtype=np.float32]): Y position of targets [arcsec]
        """
        self.__ypos = csu.enforce_float(n)

    ypos = property(get_ypos, set_ypos)

    def get_mag(self):
        """ Get the magnitudes of targets

        :return: (np.ndarray[ndim=2, dtype=np.float32]) : magnitudes
        """
        return self.__mag

    def set_mag(self, n):
        """ Set the magnitudes of targets

        :param n: (np.ndarray[ndim=2, dtype=np.float32]) : magnitudes
        """
        self.__mag = csu.enforce_float(n)

    mag = property(get_mag, set_mag)

    def get_zerop(self):
        """ Get the zero point of targets

        :return: (float) : zero point of targets
        """
        return self.__zerop

    def set_zerop(self, n):
        """ Set the zero point of targets

        :param n: (float) : zero point of targets
        """
        self.__zerop = csu.enforce_float(n)

    zerop = property(get_zerop, set_zerop)

    def get_dms_seen(self):
        """ Get the dms_seen by the targets

        :return: (np.ndarray[ndim=2, dtype=np.int32]) : index of dms seen
        """
        return self.__dms_seen

    def set_dms_seen(self, n):
        """ Set the dms_seen by the targets

        :param n: (np.ndarray[ndim=2, dtype=np.int32]) : index of dms seen
        """
        if (isinstance(n, list)):
            n = np.array(n)
        self.__dms_seen = csu.enforce_array(n, size=n.size, dtype=np.int32,
                                            scalar_expand=True)

    dms_seen = property(get_dms_seen, set_dms_seen)
