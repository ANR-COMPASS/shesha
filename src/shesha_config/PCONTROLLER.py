'''
Param_controller class definition
Parameters for controller
'''

import numpy as np
from . import config_setter_utils as csu
import shesha_constants as scons

#################################################
# P-Class (parametres) Param_controller
#################################################


class Param_controller:

    def __init__(self):
        self.__type = None
        """ type of controller"""
        self.__kl_imat = False
        """ set imat kl on-off"""
        self.__klgain = None
        """ gain for kl mod in imat kl """
        self.__nwfs = None
        """ index of wfss in controller"""
        self.__nvalid = None
        """ number of valid subaps per wfs"""
        self.__ndm = None
        """ index of dms in controller"""
        self.__nactu = None
        """ number of controled actuator per dm"""
        self.__imat = None
        """ full interaction matrix"""
        self.__cmat = None
        """ full control matrix"""
        self.__maxcond = None
        """ max condition number"""
        self.__TTcond = None
        """ tiptilt condition number for cmat filtering with mv controller"""
        self.__delay = None
        """ loop delay [frames]"""
        self.__gain = None
        """ loop gain """
        self.__nkl = None
        self.__cured_ndivs = None
        """ subdivision levels in cured"""
        self.__modopti = False
        """ Flag for modal optimization"""
        self.__nrec = 2048
        """ Number of sample of open loop slopes for modal optimization computation"""
        self.__nmodes = None
        """ Number of modes for M2V matrix (modal optimization)"""
        self.__gmin = 0.
        """ Minimum gain for modal optimization"""
        self.__gmax = 1.
        """ Maximum gain for modal optimization"""
        self.__ngain = 15
        """ Number of tested gains"""

    def set_kl_imat(self, n):
        """Set type imat, for imat on kl set at 1

        :param k: (int) : imat kl
        """
        self.__kl_imat = csu.enforce_or_cast_bool(n)

    kl_imat = property(lambda x: x.__kl_imat, set_kl_imat)

    def set_type(self, t):
        """ Set the controller type

        :param t: (string) : type
        """
        self.__type = scons.check_enum(scons.ControllerType, t)

    type = property(lambda x: x.__type, set_type)

    def set_klgain(self, g):
        """ Set klgain for imatkl size = number of kl mode

        :param g: (np.ndarray[ndim=1, dtype=np.float32]) : g
        """
        self.__klgain = csu.enforce_array(g, len(g), dtype=np.float32)

    klgain = property(lambda x: x.__klgain, set_klgain)

    def set_nkl(self, n):
        """ Set the number of KL modes used in imat_kl and used for computation of covmat in case of minimum variance controller

        :param n: (long) : number of KL modes
        """
        self.__nkl = csu.enforce_int(n)

    nkl = property(lambda x: x.__nkl, set_nkl)

    def set_nwfs(self, l):
        """ Set the indices of wfs

        :param l: (np.ndarray[ndim=1, dtype=np.int32]) : indices of wfs
        """
        self.__nwfs = csu.enforce_array(l, len(l), dtype=np.int32, scalar_expand=False)

    nwfs = property(lambda x: x.__nwfs, set_nwfs)

    def set_ndm(self, l):
        """ Set the indices of dms

        :param l: (np.ndarray[ndim=1, dtype=np.int32]) : indices of dms
        """
        self.__ndm = csu.enforce_array(l, len(l), dtype=np.int32, scalar_expand=False)

    ndm = property(lambda x: x.__ndm, set_ndm)

    def set_nactu(self, l):
        """ Set the indices of dms

        :param l: (np.ndarray[ndim=1, dtype=np.int32]) : indices of dms
        """
        self.__nactu = csu.enforce_array(l, len(l), dtype=np.int32, scalar_expand=False)

    nactu = property(lambda x: x.__nactu, set_nactu)

    def set_nvalid(self, l):
        """ Set the number of valid subaps

        :param l: (list of int) : number of valid subaps
        """
        self.__nvalid = csu.enforce_array(l, len(l), dtype=np.int32, scalar_expand=False)

    nvalid = property(lambda x: x.__nvalid, set_nvalid)

    def set_maxcond(self, m):
        """ Set the max condition number

        :param m: (float) : max condition number
        """
        self.__maxcond = csu.enforce_float(m)

    maxcond = property(lambda x: x.__maxcond, set_maxcond)

    def set_TTcond(self, m):
        """ Set the tiptilt condition number for cmat filtering with mv controller

        :param m: (float) : tiptilt condition number
        """
        self.__TTcond = csu.enforce_float(m)

    TTcond = property(lambda x: x.__TTcond, set_TTcond)

    def set_delay(self, d):
        """ Set the loop delay expressed in frames

        :param d: (float) :delay [frames]
        """
        self.__delay = csu.enforce_float(d)

    delay = property(lambda x: x.__delay, set_delay)

    def set_gain(self, g):
        """ Set the loop gain

        :param g: (float) : loop gain
        """
        self.__gain = csu.enforce_float(g)

    gain = property(lambda x: x.__gain, set_gain)

    def set_cured_ndivs(self, n):
        """ Set the subdivision levels in cured

        :param c: (long) : subdivision levels in cured
        """
        self.__cured_ndivs = csu.enforce_int(n)

    cured_ndivs = property(lambda x: x.__cured_ndivs, set_cured_ndivs)

    def set_modopti(self, n):
        """ Set the flag for modal optimization

        :param n: (int) : flag for modal optimization
        """
        self.__modopti = csu.enforce_or_cast_bool(n)

    modopti = property(lambda x: x.__modopti, set_modopti)

    def set_nrec(self, n):
        """ Set the number of sample of open loop slopes for modal optimization computation

        :param n: (int) : number of sample
        """
        self.__nrec = csu.enforce_int(n)

    nrec = property(lambda x: x.__nrec, set_nrec)

    def set_nmodes(self, n):
        """ Set the number of modes for M2V matrix (modal optimization)

        :param n: (int) : number of modes
        """
        self.__nmodes = csu.enforce_int(n)

    nmodes = property(lambda x: x.__nmodes, set_nmodes)

    def set_gmin(self, g):
        """ Set the minimum gain for modal optimization

        :param g: (float) : minimum gain for modal optimization
        """
        self.__gmin = csu.enforce_float(g)

    gmin = property(lambda x: x.__gmin, set_gmin)

    def set_gmax(self, g):
        """ Set the maximum gain for modal optimization

        :param g: (float) : maximum gain for modal optimization
        """
        self.__gmax = csu.enforce_float(g)

    gmax = property(lambda x: x.__gmax, set_gmax)

    def set_ngain(self, n):
        """ Set the number of tested gains

        :param n: (int) : number of tested gains
        """
        self.__ngain = csu.enforce_int(n)

    ngain = property(lambda x: x.__ngain, set_ngain)

    def set_imat(self, imat):
        """ Set the full interaction matrix

        :param imat: (np.ndarray[ndim=2,dtype=np.float32_t]) : full interaction matrix
        """
        self.__imat = csu.enforce_arrayMultiDim(imat, (2 * self.nvalid.sum(),
                                                       self.nactu.sum()),
                                                dtype=np.float32)

    _imat = property(lambda x: x.__imat, set_imat)

    def set_cmat(self, cmat):
        """ Set the full control matrix

        :param cmat: (np.ndarray[ndim=2,dtype=np.float32_t]) : full control matrix
        """
        self.__cmat = csu.enforce_arrayMultiDim(cmat, (self.nactu.sum(),
                                                       2 * self.nvalid.sum()),
                                                dtype=np.float32)

    _cmat = property(lambda x: x.__cmat, set_cmat)
