'''
Param_tel class definition
Parameters for telescope definition
'''

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
        """ std of reflectivity errors for EELT segments (fraction)."""
        self.__referr = 0.0
        """ std of piston errors for EELT segments  """
        self.__std_piston = 0.0
        """ std of tip-tilt errors for EELT segments. """
        self.__std_tt = 0.0
        """ Vector for define segments numbers need. """
        self.__vect_seg = None

    def set_diam(self, d):
        """ Set the telescope diameter

        :param d: (float) : telescope diameter (in meters)
        """
        self.__diam = csu.enforce_float(d)

    diam = property(lambda x: x.__diam, set_diam)

    def set_cobs(self, c):
        """ Set the central obstruction ratio

        :param c: (float) : central obstruction ratio
        """
        self.__cobs = csu.enforce_float(c)

    cobs = property(lambda x: x.__cobs, set_cobs)

    def set_type_ap(self, t):
        """ Set the EELT aperture type

        :param t: (str) : EELT aperture type
        """
        self.__type_ap = const.check_enum(const.ApertureType, t)

    type_ap = property(lambda x: x.__type_ap, set_type_ap)

    def set_t_spiders(self, spider):
        """ Set the secondary supports ratio

        :param spider: (float) : secondary supports ratio
        """
        self.__t_spiders = csu.enforce_float(spider)

    t_spiders = property(lambda x: x.__t_spiders, set_t_spiders)

    def set_spiders_type(self, spider):
        """ Set the secondary supports type

        :param spider: (str) : secondary supports type
        """
        self.__spiders_type = const.check_enum(const.SpiderType, spider)

    spiders_type = property(lambda x: x.__spiders_type, set_spiders_type)

    def set_pupangle(self, p):
        """ Set the rotation angle of pupil

        :param p: (float) : rotation angle of pupil
        """
        self.__pupangle = csu.enforce_float(p)

    pupangle = property(lambda x: x.__pupangle, set_pupangle)

    def set_nbrmissing(self, nb):
        """ Set the number of missing segments for EELT pupil

        :param nb: (long) : number of missing segments for EELT pupil (max is 20)
        """
        self.__nbrmissing = csu.enforce_int(nb)

    nbrmissing = property(lambda x: x.__nbrmissing, set_nbrmissing)

    def set_referr(self, ref):
        """ Set the std of reflectivity errors for EELT segments

        :param ref: (float) : std of reflectivity errors for EELT segments (fraction)
        """
        self.__referr = csu.enforce_float(ref)

    referr = property(lambda x: x.__referr, set_referr)

    def set_std_piston(self, piston):
        """ Set the std of piston errors for EELT segments

        :param piston: (float) : std of piston errors for EELT segments
        """
        self.__std_piston = csu.enforce_float(piston)

    std_piston = property(lambda x: x.__std_piston, set_std_piston)

    def set_std_tt(self, tt):
        """ Set the std of tip-tilt errors for EELT segments

        :param tt: (float) : std of tip-tilt errors for EELT segments
        """
        self.__std_tt = csu.enforce_float(tt)

    std_tt = property(lambda x: x.__std_tt, set_std_tt)

    def set_vect_seg(self, vect):
        """ Set the segment number for construct ELT pupil"

        :param vect: (list of int32) : segment numbers
        """
        self.__vect_seg = csu.enforce_array(vect,
                                            len(vect), dtype=np.int32,
                                            scalar_expand=False)

    vect_seg = property(lambda x: x.__vect_seg, set_vect_seg)
