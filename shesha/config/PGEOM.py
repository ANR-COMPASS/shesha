## @package   shesha.config.PGEOM
## @brief     Param_geom class definition
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

from . import config_setter_utils as csu
import numpy as np


#################################################
# P-Class (parametres) Param_geom
#################################################
class Param_geom:

    def __init__(self):
        """ Private members were initialized yet """
        self.__is_init = False
        """ linear size of full image (in pixels)."""
        self.__ssize = 0
        """ observations zenith angle (in deg)."""
        self.__zenithangle = 0.
        """ boolean for apodizer"""
        self.__apod = False
        """ File to load an apodizer from """
        self.__apod_file = None
        """ linear size of total pupil (in pixels)."""
        self.__pupdiam = 0
        """ central point of the simulation."""
        self.__cent = 0.
        """ Pixel size of the simulation [meters]."""
        self.__pixsize = 0.

        # Internals
        self.__ipupil = None  # total pupil (include full guard band)
        self.__mpupil = None  # medium pupil (part of the guard band)
        self.__spupil = None  # small pupil (without guard band)
        self.__phase_ab_M1 = None  # Phase aberration in the pupil (small size)
        # Phase aberration in the pupil (medium size)
        self.__phase_ab_M1_m = None
        self.__apodizer = None  # apodizer (same size as small pupil)
        self.__p1 = 0  # min x,y for valid points in mpupil
        self.__p2 = 0  # max x,y for valid points in mpupil
        self.__n = 0  # linear size of mpupil
        self.__n1 = 0  # min x,y for valid points in ipupil
        self.__n2 = 0  # max x,y for valid points in ipupil

    def get_is_init(self):
        """ Get the is_init flag

        :return: (bool) : is_init flag
        """
        return self.__is_init

    def set_is_init(self, i):
        """ set the is_init flag

        :param i: (bool) : is_init flag
        """
        self.__is_init = csu.enforce_or_cast_bool(i)

    is_init = property(get_is_init, set_is_init)

    def get_ipupil(self):
        """ Get the pupil in the biggest support

        :return: (np.ndarray[ndim=2, dtype=np.float32]) : pupil
        """
        return self.__ipupil

    def set_ipupil(self, s):
        """ Set the pupil in the biggest support

        :param s: (np.ndarray[ndim=2, dtype=np.float32]) : pupil
        """
        self.__ipupil = csu.enforce_arrayMultiDim(s.copy(), s.shape, dtype=np.float32)

    _ipupil = property(get_ipupil, set_ipupil)

    def get_mpupil(self):
        """ Get the pupil in the middle support

        :return: (np.ndarray[ndim=2, dtype=np.float32]) : pupil
        """
        return self.__mpupil

    def set_mpupil(self, s):
        """ Set the pupil in the middle support

        :param s: (np.ndarray[ndim=2, dtype=np.float32]) : pupil
        """
        self.__mpupil = csu.enforce_arrayMultiDim(s.copy(), s.shape, dtype=np.float32)

    _mpupil = property(get_mpupil, set_mpupil)

    def get_spupil(self):
        """ Get the pupil in the smallest support

        :return: (np.ndarray[ndim=2, dtype=np.float32]) : pupil
        """
        return self.__spupil

    def set_spupil(self, s):
        """ Set the pupil in the smallest support

        :param s: (np.ndarray[ndim=2, dtype=np.float32]) : pupil
        """
        self.__spupil = csu.enforce_arrayMultiDim(s.copy(), s.shape, dtype=np.float32)

    _spupil = property(get_spupil, set_spupil)

    def get_phase_ab_M1(self):
        """ Get the phase aberration of the M1 defined in spupil support

        :return: (np.ndarray[ndim=2, dtype=np.float32]) : phase aberrations
        """
        return self.__phase_ab_M1

    def set_phase_ab_M1(self, s):
        """ Set the phase aberration of the M1 defined in spupil support

        :param s: (np.ndarray[ndim=2, dtype=np.float32]) : phase aberrations
        """
        self.__phase_ab_M1 = csu.enforce_arrayMultiDim(s.copy(), self.__spupil.shape,
                                                       dtype=np.float32)

    _phase_ab_M1 = property(get_phase_ab_M1, set_phase_ab_M1)

    def get_phase_ab_M1_m(self):
        """ Get the phase aberration of the M1 defined in mpupil support

        :return: (np.ndarray[ndim=2, dtype=np.float32]) : phase aberrations
        """
        return self.__phase_ab_M1_m

    def set_phase_ab_M1_m(self, s):
        """ Set the phase aberration of the M1 defined in mpupil support

        :param s: (np.ndarray[ndim=2, dtype=np.float32]) : phase aberrations
        """
        self.__phase_ab_M1_m = csu.enforce_arrayMultiDim(s.copy(), self.__mpupil.shape,
                                                         dtype=np.float32)

    _phase_ab_M1_m = property(get_phase_ab_M1_m, set_phase_ab_M1_m)

    def get_apodizer(self):
        """ Get the apodizer defined in spupil support

        :return: (np.ndarray[ndim=2, dtype=np.float32]) : apodizer
        """
        return self.__apodizer

    def set_apodizer(self, s):
        """ Set the apodizer defined in spupil support

        :param s: (np.ndarray[ndim=2, dtype=np.float32]) : apodizer
        """
        self.__apodizer = csu.enforce_arrayMultiDim(s.copy(), self.__spupil.shape,
                                                    dtype=np.float32)

    _apodizer = property(get_apodizer, set_apodizer)

    def get_ssize(self):
        """ Get linear size of full image

        :return: (long) : linear size of full image (in pixels).
        """
        return self.__ssize

    def set_ssize(self, s):
        """ Set linear size of full image

        :param s: (long) : linear size of full image (in pixels).
        """
        self.__ssize = csu.enforce_int(s)

    ssize = property(get_ssize, set_ssize)

    def get_n(self):
        """ Get the linear size of mpupil

        :return: (long) : coordinate (same in x and y) [pixel]
        """
        return self.__n

    def set_n(self, s):
        """ Set the linear size of mpupil

        :param s: (long) : coordinate (same in x and y) [pixel]
        """
        self.__n = csu.enforce_int(s)

    _n = property(get_n, set_n)

    def get_n1(self):
        """ Get the bottom-left corner coordinates of the pupil in the ipupil support

        :return: (long) : coordinate (same in x and y) [pixel]
        """
        return self.__n1

    def set_n1(self, s):
        """ Set the bottom-left corner coordinates of the pupil in the ipupil support

        :param s: (long) : coordinate (same in x and y) [pixel]
        """
        self.__n1 = csu.enforce_int(s)

    _n1 = property(get_n1, set_n1)

    def get_n2(self):
        """ Get the upper-right corner coordinates of the pupil in the ipupil support

        :return: (long) : coordinate (same in x and y) [pixel]
        """
        return self.__n2

    def set_n2(self, s):
        """ Set the upper-right corner coordinates of the pupil in the ipupil support

        :param s: (long) : coordinate (same in x and y) [pixel]
        """
        self.__n2 = csu.enforce_int(s)

    _n2 = property(get_n2, set_n2)

    def get_p2(self):
        """ Get the upper-right corner coordinates of the pupil in the mpupil support

        :return: (long) : coordinate (same in x and y) [pixel]
        """
        return self.__p2

    def set_p2(self, s):
        """ Set the upper-right corner coordinates of the pupil in the mpupil support

        :param s: (long) : coordinate (same in x and y) [pixel]
        """
        self.__p2 = csu.enforce_int(s)

    _p2 = property(get_p2, set_p2)

    def get_p1(self):
        """ Get the bottom-left corner coordinates of the pupil in the mpupil support

        :return: (long) : coordinate (same in x and y) [pixel]
        """
        return self.__p1

    def set_p1(self, s):
        """ Set the bottom-left corner coordinates of the pupil in the mpupil support

        :param s: (long) : coordinate (same in x and y) [pixel]
        """
        self.__p1 = csu.enforce_int(s)

    _p1 = property(get_p1, set_p1)

    def get_zenithangle(self):
        """ Get observations zenith angle

        :return: (float) : observations zenith angle (in deg).
        """
        return self.__zenithangle

    def set_zenithangle(self, z):
        """ Set observations zenith angle

        :param z: (float) : observations zenith angle (in deg).
        """
        self.__zenithangle = csu.enforce_float(z)

    zenithangle = property(get_zenithangle, set_zenithangle)

    def get_pupdiam(self):
        """ Get the linear size of total pupil

        :return: (long) : linear size of total pupil (in pixels).
        """
        return self.__pupdiam

    def set_pupdiam(self, p):
        """ Set the linear size of total pupil

        :param p: (long) : linear size of total pupil (in pixels).
        """
        self.__pupdiam = csu.enforce_int(p)

    pupdiam = property(get_pupdiam, set_pupdiam)

    def get_cent(self):
        """ Get the central point of the simulation

        :return: (float) : central point of the simulation.
        """
        return self.__cent

    def set_cent(self, c):
        """ Set the central point of the simulation

        :param c: (float) : central point of the simulation.
        """
        self.__cent = csu.enforce_float(c)

    cent = property(get_cent, set_cent)

    def get_apod(self):
        """ Gells if the apodizer is used
            The apodizer is used if a is not 0

        :return: (int) boolean for apodizer
        """
        return self.__apod

    def set_apod(self, a):
        """ Tells if the apodizer is used
            The apodizer is used if a is not 0

        :param a: (int) boolean for apodizer
        """
        self.__apod = csu.enforce_or_cast_bool(a)

    apod = property(get_apod, set_apod)

    def get_apod_file(self):
        """ Get the path of apodizer file

        :return: (str) : apodizer file name
        """
        return self.__apod_file

    def set_apod_file(self, f):
        """ Set the path of apodizer file

        :param filename: (str) : apodizer file name
        """
        self.__apod_file = f

    apod_file = property(get_apod_file, set_apod_file)

    def get_pixsize(self):
        """ Get the pixsizeral point of the simulation

        :return: (float) : pixsizeral point of the simulation.
        """
        return self.__pixsize

    def set_pixsize(self, c):
        """ Set the pixel size of the simulation

        :param c: (float) : pixel size of the simulation.
        """
        self.__pixsize = csu.enforce_float(c)

    _pixsize = property(get_pixsize, set_pixsize)
