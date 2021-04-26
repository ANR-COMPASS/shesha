## @package   shesha.config.PTEL
## @brief     Param_tel class definition
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

import numpy as np
from . import config_setter_utils as csu
import shesha.constants as const


#################################################
# P-Class (parametres) Param_tel
#################################################
class Param_tel:

    def __init__(self):
        """ telescope diameter (in meters)."""
        self.__diam = 0.0
        """ central obstruction ratio."""
        self.__cobs = 0.0
        """ EELT aperture type: "Nominal", "BP1", "BP3", "BP5" (for back-up plan with 1, 3, or 5 missing annulus)."""
        self.__type_ap = const.ApertureType.GENERIC
        """ secondary supports ratio."""
        self.__t_spiders = -1.
        """ secondary supports type: "four" or "six"."""
        self.__spiders_type = None
        """ rotation angle of pupil."""
        self.__pupangle = 0.0
        """ number of missing segments for EELT pupil (max is 20)."""
        self.__nbrmissing = 0
        """ Gap between segments [meters]"""
        self.__gap = 0.0
        """ std of reflectivity errors for EELT segments (fraction)."""
        self.__referr = 0.0
        """ std of piston errors for EELT segments  """
        self.__std_piston = 0.0
        """ std of tip-tilt errors for EELT segments. """
        self.__std_tt = 0.0
        """ Vector for define segments numbers need. """
        self.__vect_seg = None

    def get_diam(self):
        """ Get the telescope diameter

        :return: (float) : telescope diameter (in meters)
        """
        return self.__diam

    def set_diam(self, d):
        """ Set the telescope diameter

        :param d: (float) : telescope diameter (in meters)
        """
        self.__diam = csu.enforce_float(d)

    diam = property(get_diam, set_diam)

    def get_cobs(self):
        """ Get the central obstruction ratio

        :return: (float) : central obstruction ratio
        """
        return self.__cobs

    def set_cobs(self, c):
        """ Set the central obstruction ratio

        :param c: (float) : central obstruction ratio
        """
        self.__cobs = csu.enforce_float(c)

    cobs = property(get_cobs, set_cobs)

    def get_type_ap(self):
        """ Get the EELT aperture type

        :return: (str) : EELT aperture type
        """
        return self.__type_ap

    def set_type_ap(self, t):
        """ Set the EELT aperture type

        :param t: (str) : EELT aperture type
        """
        self.__type_ap = const.check_enum(const.ApertureType, t)

    type_ap = property(get_type_ap, set_type_ap)

    def get_t_spiders(self):
        """ Get the secondary supports ratio

        :return: (float) : secondary supports ratio
        """
        return self.__t_spiders

    def set_t_spiders(self, spider):
        """ Set the secondary supports ratio

        :param spider: (float) : secondary supports ratio
        """
        self.__t_spiders = csu.enforce_float(spider)

    t_spiders = property(get_t_spiders, set_t_spiders)

    def get_spiders_type(self):
        """ Get the secondary supports type

        :return: (str) : secondary supports type
        """
        return self.__spiders_type

    def set_spiders_type(self, spider):
        """ Set the secondary supports type

        :param spider: (str) : secondary supports type
        """
        self.__spiders_type = const.check_enum(const.SpiderType, spider)

    spiders_type = property(get_spiders_type, set_spiders_type)

    def get_pupangle(self):
        """ Get the rotation angle of pupil

        :return: (float) : rotation angle of pupil
        """
        return self.__pupangle

    def set_pupangle(self, p):
        """ Set the rotation angle of pupil

        :param p: (float) : rotation angle of pupil
        """
        self.__pupangle = csu.enforce_float(p)

    pupangle = property(get_pupangle, set_pupangle)

    def get_nbrmissing(self):
        """ Get the number of missing segments for EELT pupil

        :return: (long) : number of missing segments for EELT pupil (max is 20)
        """
        return self.__nbrmissing

    def set_nbrmissing(self, nb):
        """ Set the number of missing segments for EELT pupil

        :param nb: (long) : number of missing segments for EELT pupil (max is 20)
        """
        self.__nbrmissing = csu.enforce_int(nb)

    nbrmissing = property(get_nbrmissing, set_nbrmissing)

    def get_gap(self):
        """ Get the Gap between segments

        :return: (float) : Gap between segments (meters)
        """
        return self.__gap

    def set_gap(self, gap):
        """ Set the Gap between segments

        :param gap: (float) : Gap between segments (meters)
        """
        self.__gap = csu.enforce_float(gap)

    gap = property(get_gap, set_gap)

    def get_referr(self):
        """ Get the std of reflectivity errors for EELT segments

        :return: (float) : std of reflectivity errors for EELT segments (fraction)
        """
        return self.__referr

    def set_referr(self, ref):
        """ Set the std of reflectivity errors for EELT segments

        :param ref: (float) : std of reflectivity errors for EELT segments (fraction)
        """
        self.__referr = csu.enforce_float(ref)

    referr = property(get_referr, set_referr)

    def get_std_piston(self):
        """ Get the std of piston errors for EELT segments

        :return: (float) : std of piston errors for EELT segments
        """
        return self.__std_piston

    def set_std_piston(self, piston):
        """ Set the std of piston errors for EELT segments

        :param piston: (float) : std of piston errors for EELT segments
        """
        self.__std_piston = csu.enforce_float(piston)

    std_piston = property(get_std_piston, set_std_piston)

    def get_std_tt(self):
        """ Get the std of tip-tilt errors for EELT segments

        :return: (float) : std of tip-tilt errors for EELT segments
        """
        return self.__std_tt

    def set_std_tt(self, tt):
        """ Set the std of tip-tilt errors for EELT segments

        :param tt: (float) : std of tip-tilt errors for EELT segments
        """
        self.__std_tt = csu.enforce_float(tt)

    std_tt = property(get_std_tt, set_std_tt)

    def get_vect_seg(self):
        """ Get the segment number for construct ELT pupil"

        :return: (list of int32) : segment numbers
        """
        return self.__vect_seg

    def set_vect_seg(self, vect):
        """ Set the segment number for construct ELT pupil"

        :param vect: (list of int32) : segment numbers
        """
        self.__vect_seg = csu.enforce_array(vect, len(vect), dtype=np.int32,
                                            scalar_expand=False)

    vect_seg = property(get_vect_seg, set_vect_seg)
