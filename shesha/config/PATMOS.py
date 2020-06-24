## @package   shesha.config.PATMOS
## @brief     Param_atmos class definition
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
# P-Class (parametres) Param_atmos
#################################################


class Param_atmos:

    def __init__(self):
        """ Number of turbulent layers."""
        self.__nscreens = 0
        """ Global r0."""
        self.__r0 = None
        """ Pupil pixel size (in meters)."""
        self.__pupixsize = None
        """ L0 per layers in meters."""
        self.__L0 = None
        """ Linear size of phase screens."""
        self.__dim_screens = None
        """ Altitudes of each layer."""
        self.__alt = None
        """ Wind directions of each layer."""
        self.__winddir = None
        """ Wind speeds of each layer."""
        self.__windspeed = None
        """ Fraction of r0 for each layer."""
        self.__frac = None
        """ x translation speed (in pix / iteration) for each layer."""
        self.__deltax = None
        """ y translation speed (in pix / iteration) for each layer."""
        self.__deltay = None
        """ RNG Seeds for each layer."""
        self.__seeds = None

    def get_nscreens(self):
        """ Set the number of turbulent layers

        :return: (long) number of screens.
        """
        return self.__nscreens

    def set_nscreens(self, n):
        """ Set the number of turbulent layers

        :param n: (long) number of screens.
        """
        self.__nscreens = csu.enforce_int(n)

    nscreens = property(get_nscreens, set_nscreens)

    def get_r0(self):
        """ Get the global r0

        :return: (float) : global r0
        """
        return self.__r0

    def set_r0(self, r):
        """ Set the global r0

        :param r: (float) : global r0
        """
        self.__r0 = csu.enforce_float(r)

    r0 = property(get_r0, set_r0)

    def get_pupixsize(self):
        """ Get the pupil pixel size

        :return: (float) : pupil pixel size
        """
        return self.__pupixsize

    def set_pupixsize(self, xsize):
        """ Set the pupil pixel size

        :param xsize: (float) : pupil pixel size
        """
        self.__pupixsize = csu.enforce_float(xsize)

    pupixsize = property(get_pupixsize, set_pupixsize)

    def get_L0(self):
        """ Get the L0 per layers

        :return: (lit of float) : L0 for each layers
        """
        return self.__L0

    def set_L0(self, l):
        """ Set the L0 per layers

        :param l: (lit of float) : L0 for each layers
        """
        self.__L0 = csu.enforce_array(l, size=self.nscreens, dtype=np.float32,
                                      scalar_expand=True)

    L0 = property(get_L0, set_L0)

    def get_dim_screens(self):
        """ Get the size of the phase screens

        :return: (lit of float) : phase screens sizes
        """
        return self.__dim_screens

    def set_dim_screens(self, l):
        """ Set the size of the phase screens

        :param l: (lit of float) : phase screens sizes
        """
        self.__dim_screens = csu.enforce_array(l, size=self.nscreens, dtype=np.int64,
                                               scalar_expand=False)

    dim_screens = property(get_dim_screens, set_dim_screens)

    def get_alt(self):
        """ Get the altitudes of each layer

        :return: (lit of float) : altitudes
        """
        return self.__alt

    def set_alt(self, h):
        """ Set the altitudes of each layer

        :param h: (lit of float) : altitudes
        """
        self.__alt = csu.enforce_array(h, size=self.nscreens, dtype=np.float32,
                                       scalar_expand=False)

    alt = property(get_alt, set_alt)

    def get_winddir(self):
        """ Get the wind direction for each layer

        :return: (lit of float) : wind directions
        """
        return self.__winddir

    def set_winddir(self, l):
        """ Set the wind direction for each layer

        :param l: (lit of float) : wind directions
        """
        self.__winddir = csu.enforce_array(l, size=self.nscreens, dtype=np.float32,
                                           scalar_expand=True)

    winddir = property(get_winddir, set_winddir)

    def get_windspeed(self):
        """ Get the the wind speed for each layer

        :return: (list of float) : wind speeds
        """
        return self.__windspeed

    def set_windspeed(self, l):
        """ Set the the wind speed for each layer

        :param l: (list of float) : wind speeds
        """
        self.__windspeed = csu.enforce_array(l, size=self.nscreens, dtype=np.float32,
                                             scalar_expand=True)

    windspeed = property(get_windspeed, set_windspeed)

    def get_frac(self):
        """ Get the fraction of r0 for each layers

        :return: (lit of float) : fraction of r0
        """
        return self.__frac

    def set_frac(self, l):
        """ Set the fraction of r0 for each layers

        :param l: (lit of float) : fraction of r0
        """
        self.__frac = csu.enforce_array(l, size=self.nscreens, dtype=np.float32,
                                        scalar_expand=True)

    frac = property(get_frac, set_frac)

    def get_deltax(self):
        """ Get the translation speed on axis x for each layer

        :return: (lit of float) : translation speed
        """
        return self.__deltax

    def set_deltax(self, l):
        """ Set the translation speed on axis x for each layer

        :param l: (lit of float) : translation speed
        """
        self.__deltax = csu.enforce_array(l, size=self.nscreens, dtype=np.float32,
                                          scalar_expand=True)

    _deltax = property(get_deltax, set_deltax)

    def get_deltay(self):
        """ Get the translation speed on axis y for each layer

        :return: (lit of float) : translation speed
        """
        return self.__deltay

    def set_deltay(self, l):
        """ Set the translation speed on axis y for each layer

        :param l: (lit of float) : translation speed
        """
        self.__deltay = csu.enforce_array(l, size=self.nscreens, dtype=np.float32,
                                          scalar_expand=True)

    _deltay = property(get_deltay, set_deltay)

    def get_seeds(self):
        """ Get the seed for each layer

        :return: (lit of int) : seed
        """
        return self.__seeds

    def set_seeds(self, l):
        """ Set the seed for each layer

        :param l: (lit of int) : seed
        """
        self.__seeds = csu.enforce_array(l, size=self.nscreens, dtype=np.int64,
                                         scalar_expand=True)

    seeds = property(get_seeds, set_seeds)
