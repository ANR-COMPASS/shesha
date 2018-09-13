'''
Param_centroider class definition
Parameters for centroider
'''

import numpy as np
from . import config_setter_utils as csu
import shesha.constants as scons

#################################################
# P-Class (parametres) Param_centroider
#################################################


class Param_centroider:

    def __init__(self):
        self.__nwfs = None
        """ index of wfs in y_wfs structure on which we want to do centroiding"""
        self.__type = None
        """ type of centroiding cog, tcog, bpcog, wcog, corr"""
        self.__nslope = 0
        """ Number of output slopes"""
        self.__type_fct = scons.CentroiderFctType.GAUSS
        """ type of ref function gauss, file, model"""
        self.__weights = None
        """ optional reference function(s) used for centroiding"""
        self.__nmax = None
        """ number of brightest pixels"""
        self.__thresh = 1.e-4
        """Threshold"""
        self.__width = None
        """ width of the Gaussian"""
        self.__sizex = None
        """ x-size for inter mat (corr)"""
        self.__sizey = None
        """ x-size for inter mat (corr)"""
        self.__interpmat = None
        """ optional reference function(s) used for corr centroiding"""
        self.__method = 1
        """ optional method used in the pyrhr centroider (
                    0: nosinus global
                    1: sinus global
                    2: nosinus local
                    3: sinus local)"""
        self.__pyrscale = 0
        """ pyrscale = (p_wfs.Lambda * 1e-6 / sim.config.p_tel.diam) * p_wfs.pyr_ampl * CONST.RAD2ARCSEC
        """

    def set_nwfs(self, n):
        """ Set the index of the WFS handled by the centroider

        :param n: (long) : WFS index
        """
        self.__nwfs = csu.enforce_int(n)

    nwfs = property(lambda x: x.__nwfs, set_nwfs)

    def set_nslope(self, n):
        """ Set the number of slope

        :param n: (long) :number of slope
        """
        self.__nslope = csu.enforce_int(n)

    _nslope = property(lambda x: x.__nslope, set_nslope)

    def set_type(self, t):
        """ Set the centroider type

        :param t: (string) : type
        """
        self.__type = scons.check_enum(scons.CentroiderType, t)

    type = property(lambda x: x.__type, set_type)

    def set_type_fct(self, t):
        """ TODO: docstring

        :param t: (string) : type
        """
        self.__type_fct = scons.check_enum(scons.CentroiderFctType, t)

    type_fct = property(lambda x: x.__type_fct, set_type_fct)

    def set_weights(self, w):
        """ Set the weights used by a wcog cetroider

        :param w: (np.ndarray[ndim=1, dtype=np.float32]) : weights
        """
        self.__weights = csu.enforce_arrayMultiDim(w, w.shape, dtype=np.float32)

    weights = property(lambda x: x.__weights, set_weights)

    def set_nmax(self, n):
        """ Set the nmax pixels used by a bpcog centroider

        :param n: (int) : nmax
        """
        self.__nmax = csu.enforce_int(n)

    nmax = property(lambda x: x.__nmax, set_nmax)

    def set_thresh(self, t):
        """ Set the threshold used by a tcog centroider

        :param t: (float) : thresh
        """
        self.__thresh = csu.enforce_float(t)

    thresh = property(lambda x: x.__thresh, set_thresh)

    def set_width(self, t):
        """ Set the width of the gaussian used by a corr centroider

        :param t: (float) : width
        """
        self.__width = csu.enforce_float(t)

    width = property(lambda x: x.__width, set_width)

    def set_sizex(self, n):
        """ Set the x size of inter mat for corr centroider

        :param n: (int) : sizex
        """
        self.__sizex = csu.enforce_int(n)

    sizex = property(lambda x: x.__sizex, set_sizex)

    def set_sizey(self, n):
        """ Set the y size of interp mat for corr centroider

        :param n: (int) : sizey
        """
        self.__sizey = csu.enforce_int(n)

    sizey = property(lambda x: x.__sizey, set_sizey)

    def set_interpmat(self, imap):
        """ Set the interp mat for corr centroider

        :param imap: (np.ndarray[ndim=2, dtype=np.float32]) : sizey
        """
        self.__interpmat = csu.enforce_arrayMultiDim(imap, imap.shape, dtype=np.float32)

    interpmat = property(lambda x: x.__interpmat, set_interpmat)

    def set_method(self, n):
        """ Set the method used by a pyr centroider:
                    0: nosinus global
                    1: sinus global
                    2: nosinus local
                    3: sinus local

        :param n: (int) : method
        """
        self.__method = csu.enforce_int(n)

    method = property(lambda x: x.__method, set_method)

    def set_pyrscale(self, t):
        """ TODO
        Set the ... (p_wfs.Lambda * 1e-6 / sim.config.p_tel.diam) * p_wfs.pyr_ampl * CONST.RAD2ARCSEC

        :param t: (float) : pyrscale
        """
        self.__pyrscale = csu.enforce_float(t)

    pyrscale = property(lambda x: x.__pyrscale, set_pyrscale)
