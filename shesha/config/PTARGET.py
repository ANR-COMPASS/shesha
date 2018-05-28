'''
Param_target class definition
Parameters for target
'''
''

import numpy as np
from . import config_setter_utils as csu

#################################################
# P-Class (parametres) Param_target
#################################################


class Param_target:

    def __init__(self):
        self.__ntargets = 0
        """ number of targets"""
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

    def set_ntargets(self, n):
        """ Set the number of targets

        :param n: (long) number of targets
        """
        self.__ntargets = csu.enforce_int(n)

    ntargets = property(lambda x: x.__ntargets, set_ntargets)

    def set_apod(self, l):
        """ Set apodizer flag

        :param l: (bool) : apod
        """
        self.__apod = csu.enforce_or_cast_bool(l)

    apod = property(lambda x: x.__apod, set_apod)

    def set_Lambda(self, n):
        """ Set the wavelength of targets

        :param n: (np.ndarray[ndim=2, dtype=np.float32]) : wavelength of targets
        """
        self.__Lambda = csu.enforce_array(n, size=self.ntargets, dtype=np.float32,
                                          scalar_expand=True)

    Lambda = property(lambda x: x.__Lambda, set_Lambda)

    def set_xpos(self, n):
        """ Set the X-position of targets in the field [arcsec]

        :param n: (np.ndarray[ndim=2, dtype=np.float32]) : X position of targets [arcsec]
        """
        self.__xpos = csu.enforce_array(n, size=self.ntargets, dtype=np.float32,
                                        scalar_expand=True)

    xpos = property(lambda x: x.__xpos, set_xpos)

    def set_ypos(self, n):
        """ Set the Y-position of targets in the field [arcsec]

        :param n: (np.ndarray[ndim=2, dtype=np.float32]): Y position of targets [arcsec]
        """
        self.__ypos = csu.enforce_array(n, size=self.ntargets, dtype=np.float32,
                                        scalar_expand=True)

    ypos = property(lambda x: x.__ypos, set_ypos)

    def set_mag(self, n):
        """ Set the magnitudes of targets

        :param n: (np.ndarray[ndim=2, dtype=np.float32]) : magnitudes
        """
        self.__mag = csu.enforce_array(n, size=self.ntargets, dtype=np.float32,
                                       scalar_expand=True)

    mag = property(lambda x: x.__mag, set_mag)

    def set_zerop(self, n):
        """ Set the zero point of targets

        :param n: (float) : zero point of targets
        """
        self.__zerop = csu.enforce_float(n)

    zerop = property(lambda x: x.__zerop, set_zerop)

    def set_dms_seen(self, n):
        """ Set the dms_seen by the targets

        :param n: (np.ndarray[ndim=2, dtype=np.int32]) : index of dms seen
        """
        self.__dms_seen = csu.enforce_array(n, size=n.size, dtype=np.int32,
                                            scalar_expand=True)

    dms_seen = property(lambda x: x.__dms_seen, set_dms_seen)
