#!/usr/local/bin/python2.7
# encoding: utf-8
'''
Created on 13 juil. 2017

@author: vdeo
'''

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

    def set_devices(self, devices):
        """
            Set the list of GPU devices used

        :parameters:
            devices: (np.ndarray[ndim=1, dtype=np.int32_t]) : list of GPU devices
        """
        self.__devices = csu.enforce_array(devices,
                                           len(devices), dtype=np.int32,
                                           scalar_expand=False)

    devices = property(lambda x: x.__devices, set_devices)

    def set_niter(self, n):
        """
            Set the number of iteration

        :parameters:
            n: (long) : number of iteration
        """
        self.__niter = csu.enforce_int(n)

    niter = property(lambda x: x.__niter, set_niter)

    def set_ittime(self, t):
        """
            Set iteration time

        :parameters:
            t: (float) :iteration time
        """
        self.__ittime = csu.enforce_float(t)

    ittime = property(lambda x: x.__ittime, set_ittime)
