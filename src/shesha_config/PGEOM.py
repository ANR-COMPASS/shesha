#!/usr/local/bin/python2.7
# encoding: utf-8
'''
Created on 13 juil. 2017

@author: vdeo
'''
from . import config_setter_utils as csu
import numpy as np


#################################################
# P-Class (parametres) Param_geom
#################################################
class Param_geom:

    def __init__(self):
        """ Private members were initialized yet """
        self.__is_init = False
        """linear size of full image (in pixels)."""
        self.__ssize = 0
        """observations zenith angle (in deg)."""
        self.__zenithangle = 0.
        """boolean for apodizer"""
        self.__apod = False
        """ File to load an apodizer from """
        self.__apod_file = None
        """linear size of total pupil (in pixels)."""
        self.__pupdiam = 0
        """central point of the simulation."""
        self.__cent = 0.

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

    def set_is_init(self, i):
        """set the is_init flag

        :param i: (bool) : is_init flag
        """
        self.__is_init = csu.enforce_or_cast_bool(i)

    is_init = property(lambda x: x.__is_init, set_is_init)

    def set_ipupil(self, s):
        """Set the pupil in the biggest support

         :param s: (np.ndarray[ndim=2, dtype=np.float32]) : pupil """
        self.__ipupil = csu.enforce_arrayMultiDim(s.copy(), s.shape, dtype=np.float32)

    _ipupil = property(lambda x: x.__ipupil, set_ipupil)

    def set_mpupil(self, s):
        """Set the pupil in the middle support

         :param s: (np.ndarray[ndim=2, dtype=np.float32]) : pupil """
        self.__mpupil = csu.enforce_arrayMultiDim(s.copy(), s.shape, dtype=np.float32)

    _mpupil = property(lambda x: x.__mpupil, set_mpupil)

    def set_spupil(self, s):
        """Set the pupil in the smallest support

         :param s: (np.ndarray[ndim=2, dtype=np.float32]) : pupil """
        self.__spupil = csu.enforce_arrayMultiDim(s.copy(), s.shape, dtype=np.float32)

    _spupil = property(lambda x: x.__spupil, set_spupil)

    def set_phase_ab_M1(self, s):
        """Set the phase aberration of the M1 defined in spupil support

         :param s: (np.ndarray[ndim=2, dtype=np.float32]) : phase aberrations """
        self.__phase_ab_M1 = csu.enforce_arrayMultiDim(s.copy(), self.__spupil.shape,
                                                       dtype=np.float32)

    _phase_ab_M1 = property(lambda x: x.__phase_ab_M1, set_phase_ab_M1)

    def set_phase_ab_M1_m(self, s):
        """Set the phase aberration of the M1 defined in mpupil support

         :param s: (np.ndarray[ndim=2, dtype=np.float32]) : phase aberrations """
        self.__phase_ab_M1_m = csu.enforce_arrayMultiDim(s.copy(), self.__mpupil.shape,
                                                         dtype=np.float32)

    _phase_ab_M1_m = property(lambda x: x.__phase_ab_M1_m, set_phase_ab_M1_m)

    def set_apodizer(self, s):
        """Set the apodizer defined in spupil support

         :param s: (np.ndarray[ndim=2, dtype=np.float32]) : apodizer"""
        self.__apodizer = csu.enforce_arrayMultiDim(s.copy(), self.__spupil.shape,
                                                    dtype=np.float32)

    _apodizer = property(lambda x: x.__apodizer, set_apodizer)

    def set_ssize(self, s):
        """Set linear size of full image

         :param s: (long) : linear size of full image (in pixels)."""
        self.__ssize = csu.enforce_int(s)

    ssize = property(lambda x: x.__ssize, set_ssize)

    def set_n(self, s):
        """Set the linear size of mpupil

         :param s: (long) : coordinate (same in x and y) [pixel]"""
        self.__n = csu.enforce_int(s)

    _n = property(lambda x: x.__n, set_n)

    def set_n1(self, s):
        """Set the bottom-left corner coordinates of the pupil in the ipupil support

         :param s: (long) : coordinate (same in x and y) [pixel]"""
        self.__n1 = csu.enforce_int(s)

    _n1 = property(lambda x: x.__n1, set_n1)

    def set_n2(self, s):
        """Set the upper-right corner coordinates of the pupil in the ipupil support

         :param s: (long) : coordinate (same in x and y) [pixel]"""
        self.__n2 = csu.enforce_int(s)

    _n2 = property(lambda x: x.__n2, set_n2)

    def set_p2(self, s):
        """Set the upper-right corner coordinates of the pupil in the mpupil support

         :param s: (long) : coordinate (same in x and y) [pixel]"""
        self.__p2 = csu.enforce_int(s)

    _p2 = property(lambda x: x.__p2, set_p2)

    def set_p1(self, s):
        """Set the bottom-left corner coordinates of the pupil in the mpupil support

         :param s: (long) : coordinate (same in x and y) [pixel]"""
        self.__p1 = csu.enforce_int(s)

    _p1 = property(lambda x: x.__p1, set_p1)

    def set_zenithangle(self, z):
        """Set observations zenith angle

         :param z: (float) : observations zenith angle (in deg)."""
        self.__zenithangle = csu.enforce_float(z)

    zenithangle = property(lambda x: x.__zenithangle, set_zenithangle)

    def set_pupdiam(self, p):
        """Set the linear size of total pupil

        :param p: (long) : linear size of total pupil (in pixels)."""
        self.__pupdiam = csu.enforce_int(p)

    pupdiam = property(lambda x: x.__pupdiam, set_pupdiam)

    def set_cent(self, c):
        """Set the central point of the simulation

         :param c: (float) : central point of the simulation."""
        self.__cent = csu.enforce_float(c)

    cent = property(lambda x: x.__cent, set_cent)

    def set_apod(self, a):
        """
            Tells if the apodizer is used
            The apodizer is used if a is not 0
        :param a: (int) boolean for apodizer
        """
        self.__apod = csu.enforce_or_cast_bool(a)

    apod = property(lambda x: x.__apod, set_apod)

    def set_apod_file(self, f):
        """set the path of apodizer file

        :param filename: (str) : apodizer file name
        """
        self.__apod_file = f

    apod_file = property(lambda x: x.__apod_file, set_apod_file)
