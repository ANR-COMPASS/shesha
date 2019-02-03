'''
Param_loop class definition
Parameters for AO loop
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

    def get_devices(self):
        """ Get the list of GPU devices used

        :return: (np.ndarray[ndim=1, dtype=np.int32_t]) : list of GPU devices
        """
        return self.__devices

    def set_devices(self, devices):
        """ Set the list of GPU devices used

        :parameters:
            devices: (np.ndarray[ndim=1, dtype=np.int32_t]) : list of GPU devices
        """
        self.__devices = csu.enforce_array(devices, len(devices), dtype=np.int32,
                                           scalar_expand=False)

    devices = property(get_devices, set_devices)

    def get_niter(self):
        """ Get the number of iteration

        :return: (long) : number of iteration
        """
        return self.__niter

    def set_niter(self, n):
        """ Set the number of iteration

        :parameters:
            n: (long) : number of iteration
        """
        self.__niter = csu.enforce_int(n)

    niter = property(get_niter, set_niter)

    def get_ittime(self):
        """ Get iteration time

        :return: (float) :iteration time
        """
        return self.__ittime

    def set_ittime(self, t):
        """ Set iteration time

        :parameters:
            t: (float) :iteration time
        """
        self.__ittime = csu.enforce_float(t)

    ittime = property(get_ittime, set_ittime)
