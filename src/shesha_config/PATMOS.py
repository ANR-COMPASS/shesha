'''
Param_atmos class definition
Parameters for atmosphere
'''
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

    def set_nscreens(self, n):
        """ Set the number of turbulent layers

        :param n: (long) number of screens.
        """
        self.__nscreens = csu.enforce_int(n)

    nscreens = property(lambda x: x.__nscreens, set_nscreens)

    def set_r0(self, r):
        """ Set the global r0

        :param r: (float) : global r0
        """
        self.__r0 = csu.enforce_float(r)

    r0 = property(lambda x: x.__r0, set_r0)

    def set_pupixsize(self, xsize):
        """ Set the pupil pixel size

        :param xsize: (float) : pupil pixel size
        """
        self.__pupixsize = csu.enforce_float(xsize)

    pupixsize = property(lambda x: x.__pupixsize, set_pupixsize)

    def set_L0(self, l):
        """ Set the L0 per layers

        :param l: (lit of float) : L0 for each layers
        """
        self.__L0 = csu.enforce_array(l, size=self.nscreens, dtype=np.float32,
                                      scalar_expand=True)

    L0 = property(lambda x: x.__L0, set_L0)

    def set_dim_screens(self, l):
        """ Set the size of the phase screens

        :param l: (lit of float) : phase screens sizes
        """
        self.__dim_screens = csu.enforce_array(l, size=self.nscreens, dtype=np.int64,
                                               scalar_expand=False)

    dim_screens = property(lambda x: x.__dim_screens, set_dim_screens)

    def set_alt(self, l):
        """ Set the altitudes of each layer

        :param l: (lit of float) : altitudes
        """
        self.__alt = csu.enforce_array(l, size=self.nscreens, dtype=np.float32,
                                       scalar_expand=False)

    alt = property(lambda x: x.__alt, set_alt)

    def set_winddir(self, l):
        """ Set the wind direction for each layer

        :param l: (lit of float) : wind directions
        """
        self.__winddir = csu.enforce_array(l, size=self.nscreens, dtype=np.float32,
                                           scalar_expand=True)

    winddir = property(lambda x: x.__winddir, set_winddir)

    def set_windspeed(self, l):
        """ Set the the wind speed for each layer

        :param l: (list of float) : wind speeds
        """
        self.__windspeed = csu.enforce_array(l, size=self.nscreens, dtype=np.float32,
                                             scalar_expand=True)

    windspeed = property(lambda x: x.__windspeed, set_windspeed)

    def set_frac(self, l):
        """ Set the fraction of r0 for each layers

        :param l: (lit of float) : fraction of r0
        """
        self.__frac = csu.enforce_array(l, size=self.nscreens, dtype=np.float32,
                                        scalar_expand=True)

    frac = property(lambda x: x.__frac, set_frac)

    def set_deltax(self, l):
        """ Set the translation speed on axis x for each layer

        :param l: (lit of float) : translation speed
        """
        self.__deltax = csu.enforce_array(l, size=self.nscreens, dtype=np.float32,
                                          scalar_expand=True)

    _deltax = property(lambda x: x.__deltax, set_deltax)

    def set_deltay(self, l):
        """ Set the translation speed on axis y for each layer

        :param l: (lit of float) : translation speed
        """
        self.__deltay = csu.enforce_array(l, size=self.nscreens, dtype=np.float32,
                                          scalar_expand=True)

    _deltay = property(lambda x: x.__deltay, set_deltay)

    def set_seeds(self, l):
        """ Set the seed for each layer

        :param l: (lit of int) : seed
        """
        self.__seeds = csu.enforce_array(l, size=self.nscreens, dtype=np.int64,
                                         scalar_expand=True)

    seeds = property(lambda x: x.__seeds, set_seeds)
