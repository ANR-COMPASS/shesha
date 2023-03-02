## @package   shesha.config.PCONTROLLER
## @brief     Param_controller class definition
## @author    COMPASS Team <https://github.com/ANR-COMPASS>
## @version   5.4.0
## @date      2022/01/24
## @copyright GNU Lesser General Public License
#
#  This file is part of COMPASS <https://anr-compass.github.io/compass/>
#
#  Copyright (C) 2011-2022 COMPASS Team <https://github.com/ANR-COMPASS>
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
import shesha.constants as scons

#################################################
# P-Class (parametres) Param_controller
#################################################


class Param_controller:

    def __init__(self):
        self.__type = None
        """ type of controller"""
        self.__command_law = None
        """ type of command law type for generic controller only"""
        self.__nwfs = None
        """ index of wfss in controller"""
        self.__nvalid = 0
        """ number of valid subaps"""
        self.__nslope = 0
        """ number of slope to handle"""
        self.__nslope_buffer = 1
        """ number of previous slopes to use in control"""
        self.__ndm = None
        """ index of dms in controller"""
        self.__nactu = 0
        """ number of controled actuator"""
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
        ''' MODAL OPTIMIZATION (style Gendron Lena 1994)'''
        self.__modopti = False
        """ Flag for modal optimization"""
        self.__nrec = 2048
        """ Number of sample of open loop slopes for modal optimization computation"""
        self.__nmodes = None
        """ Number of modes for M2V matrix (modal optimization)"""
        self.__nmode_buffer = 0
        """ Number of previous modal vectors to use for control"""
        self.__gmin = 0.
        """ Minimum gain for modal optimization"""
        self.__gmax = 1.
        """ Maximum gain for modal optimization"""
        self.__ngain = 15
        """ Number of tested gains"""
        ''' KL (actually BTT) BASIS INITIALIZATION '''
        self.__do_kl_imat = False
        """ set imat kl on-off"""
        self.__klpush = None
        """ Modal pushes values """
        self.__klgain = None
        """ Gain applied to modes at cMat inversion """
        self.__nstates = 0
        """ Number of states for generic linear controller """
        self.__nstate_buffer = 0
        """ Number of state vectors to use for control"""
        ''' MODAL OPTIMIZATION CLOSE'''
        self.__close_opti = False
        """ Flag for modal optimization with close """
        self.__mgain_init = 1.0
        """ Initial values of the modal gains """
        self.__lfdownup = (0.01, 0.01)
        """ Modal gain correction learning factor """
        self.__close_learning_factor = 0.3
        """ Autocorrelation learning factor """
        self.__close_target = 0.0
        """ Update framerate """
        self.__close_update_index = 1
        """ Target value """
        self.__n_iir_in = 0
        """ number of input taps to iir filter """
        self.__n_iir_out = 0
        """ number of output taps to iir filter """
        self.__polc = 0
        """ flag to do polc in generic linear controller """
        self.__modal = 0
        """ flag to use a modal control in generic linenar controller """
        self.__kernconv4imat = 1
        """ Flag to use kernel convolution when computing imat """
        self.__calpix_name = "compass_calPix"
        self.__loopdata_name = "compass_loopData"


    def get_calpix_name(self):
        """ Get the topic name of calpix stream

        :return: (string) : type
        """
        return self.__calpix_name

    def set_calpix_name(self, calpix_name):
        """ Set the calpix topic name type

        :param t: (string) : type
        """
        self.__calpix_name = calpix_name

    calpix_name = property(get_calpix_name, set_calpix_name)

    def get_loopdata_name(self):
        """ Get the topic name of calpix stream

        :return: (string) : type
        """
        return self.__loopdata_name

    def set_loopdata_name(self, loopdata_name):
        """ Set the loop data topic name type

        :param t: (string) : type
        """
        self.__loopdata_name = loopdata_name

    loopdata_name = property(get_loopdata_name, set_loopdata_name)

    def get_type(self):
        """ Get the controller type

        :return: (string) : type
        """
        return self.__type

    def set_type(self, t):
        """ Set the controller type

        :param t: (string) : type
        """
        self.__type = scons.check_enum(scons.ControllerType, t)

    type = property(get_type, set_type)

    def get_command_law(self):
        """ Get the command law type for generic controller only

        :return: (string) : Command law type
        """
        return self.__command_law

    def set_command_law(self, t):
        """ Set the command law type for generic controller only

        :param t: (string) : Command law type
        """
        self.__command_law = scons.check_enum(scons.CommandLawType, t)

    command_law = property(get_command_law, set_command_law)

    def get_do_kl_imat(self):
        """Get type imat, for imat on kl set at 1

        :return: (int) : imat kl
        """
        return self.__do_kl_imat

    def set_do_kl_imat(self, n):
        """Set type imat, for imat on kl set at 1

        :param k: (int) : imat kl
        """
        self.__do_kl_imat = csu.enforce_or_cast_bool(n)

    do_kl_imat = property(get_do_kl_imat, set_do_kl_imat)

    def get_klpush(self):
        """ Get klgain for imatkl size = number of kl mode

        :return: (np.ndarray[ndim=1, dtype=np.float32]) : g
        """
        return self.__klpush

    def set_klpush(self, g):
        """ Set klgain for imatkl size = number of kl mode

        :param g: (np.ndarray[ndim=1, dtype=np.float32]) : g
        """
        self.__klpush = csu.enforce_array(g, len(g), dtype=np.float32)

    klpush = property(get_klpush, set_klpush)

    def get_klgain(self):
        """ Get klgain for imatkl size = number of kl mode

        :return: (np.ndarray[ndim=1, dtype=np.float32]) : g
        """
        return self.__klgain

    def set_klgain(self, g):
        """ Set klgain for imatkl size = number of kl mode

        :param g: (np.ndarray[ndim=1, dtype=np.float32]) : g
        """
        self.__klgain = csu.enforce_array(g, len(g), dtype=np.float32)

    klgain = property(get_klgain, set_klgain)

    def get_nkl(self):
        """ Get the number of KL modes used in imat_kl and used for computation of covmat in case of minimum variance controller

        :return: (long) : number of KL modes
        """
        return self.__nkl

    def set_nkl(self, n):
        """ Set the number of KL modes used in imat_kl and used for computation of covmat in case of minimum variance controller

        :param n: (long) : number of KL modes
        """
        self.__nkl = csu.enforce_int(n)

    nkl = property(get_nkl, set_nkl)

    def get_nwfs(self):
        """ Get the indices of wfs

        :return: (np.ndarray[ndim=1, dtype=np.int32]) : indices of wfs
        """
        return self.__nwfs

    def set_nwfs(self, l):
        """ Set the indices of wfs

        :param l: (np.ndarray[ndim=1, dtype=np.int32]) : indices of wfs
        """
        self.__nwfs = csu.enforce_array(l, len(l), dtype=np.int32, scalar_expand=False)

    nwfs = property(get_nwfs, set_nwfs)

    def get_ndm(self):
        """ Get the indices of dms

        :return: (np.ndarray[ndim=1, dtype=np.int32]) : indices of dms
        """
        return self.__ndm

    def set_ndm(self, l):
        """ Set the indices of dms

        :param l: (np.ndarray[ndim=1, dtype=np.int32]) : indices of dms
        """
        self.__ndm = csu.enforce_array(l, len(l), dtype=np.int32, scalar_expand=False)

    ndm = property(get_ndm, set_ndm)

    def get_nactu(self):
        """ Get the number of actuators

        :return: (int) : number of actus
        """
        return self.__nactu

    def set_nactu(self, l):
        """ Set the number of actuators

        :param l: (int) : number of actus
        """
        self.__nactu = csu.enforce_int(l)

    nactu = property(get_nactu, set_nactu)

    def get_nslope(self):
        """ Get the number of slopes

        :return: (int) : number of slopes
        """
        return self.__nslope

    def set_nslope(self, l):
        """ Set the number of slopes

        :param l: (int) : number of slopes
        """
        self.__nslope = csu.enforce_int(l)

    nslope = property(get_nslope, set_nslope)

    def get_nslope_buffer(self):
        """ Get the number of slope buffers

        :return: (int) : number of slopes buffers
        """
        return self.__nslope_buffer

    def set_nslope_buffer(self, l):
        """ Set the number of slope buffers

        :param l: (int) : number of slope buffers
        """
        self.__nslope_buffer = csu.enforce_int(l)

    nslope_buffer = property(get_nslope_buffer, set_nslope_buffer)

    def get_nvalid(self):
        """ Get the number of valid subaps

        :return: (list of int) : number of valid subaps
        """
        return self.__nvalid

    def set_nvalid(self, l):
        """ Set the number of valid subaps

        :param l: (list of int) : number of valid subaps
        """
        self.__nvalid = csu.enforce_int(l)

    nvalid = property(get_nvalid, set_nvalid)

    def get_maxcond(self):
        """ Get the max condition number

        :return: (float) : max condition number
        """
        return self.__maxcond

    def set_maxcond(self, m):
        """ Set the max condition number

        :param m: (float) : max condition number
        """
        self.__maxcond = csu.enforce_float(m)

    maxcond = property(get_maxcond, set_maxcond)

    def get_TTcond(self):
        """ Get the tiptilt condition number for cmat filtering with mv controller

        :return: (float) : tiptilt condition number
        """
        return self.__TTcond

    def set_TTcond(self, m):
        """ Set the tiptilt condition number for cmat filtering with mv controller

        :param m: (float) : tiptilt condition number
        """
        self.__TTcond = csu.enforce_float(m)

    TTcond = property(get_TTcond, set_TTcond)

    def get_delay(self):
        """ Get the loop delay expressed in frames

        :return: (float) :delay [frames]
        """
        return self.__delay

    def set_delay(self, d):
        """ Set the loop delay expressed in frames

        :param d: (float) :delay [frames]
        """
        self.__delay = csu.enforce_float(d)

    delay = property(get_delay, set_delay)

    def get_gain(self):
        """ Get the loop gain

        :return: (float) : loop gain
        """
        return self.__gain

    def set_gain(self, g):
        """ Set the loop gain

        :param g: (float) : loop gain
        """
        self.__gain = csu.enforce_float(g)

    gain = property(get_gain, set_gain)

    def get_cured_ndivs(self):
        """ Get the subdivision levels in cured

        :return: (long) : subdivision levels in cured
        """
        return self.__cured_ndivs

    def set_cured_ndivs(self, n):
        """ Set the subdivision levels in cured

        :param c: (long) : subdivision levels in cured
        """
        self.__cured_ndivs = csu.enforce_int(n)

    cured_ndivs = property(get_cured_ndivs, set_cured_ndivs)

    def get_modopti(self):
        """ Get the flag for modal optimization

        :return: (int) : flag for modal optimization
        """
        return self.__modopti

    def set_modopti(self, n):
        """ Set the flag for modal optimization

        :param n: (int) : flag for modal optimization
        """
        self.__modopti = csu.enforce_or_cast_bool(n)

    modopti = property(get_modopti, set_modopti)

    def get_nrec(self):
        """ Get the number of sample of open loop slopes for modal optimization computation

        :return: (int) : number of sample
        """
        return self.__nrec

    def set_nrec(self, n):
        """ Set the number of sample of open loop slopes for modal optimization computation

        :param n: (int) : number of sample
        """
        self.__nrec = csu.enforce_int(n)

    nrec = property(get_nrec, set_nrec)

    def get_nmodes(self):
        """ Get the number of modes for M2V matrix (modal optimization)

        :return: (int) : number of modes
        """
        return self.__nmodes

    def set_nmodes(self, n):
        """ Set the number of modes for M2V matrix (modal optimization)

        :param n: (int) : number of modes
        """
        self.__nmodes = csu.enforce_int(n)

    nmodes = property(get_nmodes, set_nmodes)

    def get_nmode_buffer(self):
        """ Get the number of mode buffers

        :return: (int) : number of mode buffers
        """
        return self.__nmode_buffer

    def set_nmode_buffer(self, n):
        """ Set the number of mode buffers

        :param n: (int) : number of modes buffers
        """
        self.__nmode_buffer = csu.enforce_int(n)

    nmode_buffer = property(get_nmode_buffer, set_nmode_buffer)

    def get_gmin(self):
        """ Get the minimum gain for modal optimization

        :return: (float) : minimum gain for modal optimization
        """
        return self.__gmin

    def set_gmin(self, g):
        """ Set the minimum gain for modal optimization

        :param g: (float) : minimum gain for modal optimization
        """
        self.__gmin = csu.enforce_float(g)

    gmin = property(get_gmin, set_gmin)

    def get_gmax(self):
        """ Get the maximum gain for modal optimization

        :return: (float) : maximum gain for modal optimization
        """
        return self.__gmax

    def set_gmax(self, g):
        """ Set the maximum gain for modal optimization

        :param g: (float) : maximum gain for modal optimization
        """
        self.__gmax = csu.enforce_float(g)

    gmax = property(get_gmax, set_gmax)

    def get_ngain(self):
        """ Get the number of tested gains

        :return: (int) : number of tested gains
        """
        return self.__ngain

    def set_ngain(self, n):
        """ Set the number of tested gains

        :param n: (int) : number of tested gains
        """
        self.__ngain = csu.enforce_int(n)

    ngain = property(get_ngain, set_ngain)

    def get_imat(self):
        """ Get the full interaction matrix

        :return: (np.ndarray[ndim=2,dtype=np.float32_t]) : full interaction matrix
        """
        return self.__imat

    def set_imat(self, imat):
        """ Set the full interaction matrix

        :param imat: (np.ndarray[ndim=2,dtype=np.float32_t]) : full interaction matrix
        """
        self.__imat = csu.enforce_arrayMultiDim(
                imat,
                (self.nslope, -1),  # Allow nModes or nActu as second dimension
                dtype=np.float32)

    _imat = property(get_imat, set_imat)

    def get_cmat(self):
        """ Get the full control matrix

        :return: (np.ndarray[ndim=2,dtype=np.float32_t]) : full control matrix
        """
        return self.__cmat

    def set_cmat(self, cmat):
        """ Set the full control matrix

        :param cmat: (np.ndarray[ndim=2,dtype=np.float32_t]) : full control matrix
        """
        self.__cmat = csu.enforce_arrayMultiDim(cmat, (self.nactu, self.nslope),
                                                dtype=np.float32)

    _cmat = property(get_cmat, set_cmat)

    def get_nstates(self):
        """ Get the number of states

        :return: (int) : number of states
        """
        return self.__nstates

    def set_nstates(self, l):
        """ Set the number of states

        :param l: (int) : number of states
        """
        self.__nstates = csu.enforce_int(l)

    nstates = property(get_nstates, set_nstates)

    def get_nstate_buffer(self):
        """ Get the number of state buffer

        :return: (int) : number of state buffer
        """
        return self.__nstate_buffer

    def set_nstate_buffer(self, l):
        """ Set the number of state buffer

        :param l: (int) : number of state buffer
        """
        self.__nstate_buffer = csu.enforce_int(l)

    nstate_buffer = property(get_nstate_buffer, set_nstate_buffer)

    def get_close_opti(self):
        """ Get flag for CLOSE modal optimization

        :return: (bool) : CLOSE flag
        """
        return self.__close_opti

    def set_close_opti(self, close_opti):
        """ Set the flag for CLOSE modal optimization

        :param close_opti: (bool) : CLOSE flag
        """
        self.__close_opti = close_opti

    close_opti = property(get_close_opti, set_close_opti)

    def get_mgain_init(self):
        """ Get the initial value of modal gains

        :return: (float) : initial value for modal gains
        """
        return self.__mgain_init

    def set_mgain_init(self, mgain_init):
        """ Set the initial value of modal gains

        :param mgain_init: (float) : init valuo of modal gain
        """
        self.__mgain_init = csu.enforce_float(mgain_init)

    mgain_init = property(get_mgain_init, set_mgain_init)

    def get_lfdownup(self):
        """ Get the autocorrelation learning factors

        :return: (tuple) : learning factors for autocorrelation
        """
        return self.__lfdownup

    def set_lfdownup(self, qminus, qplus):
        """ Set the autocorrelation learning factor

        :param qminus: (float) : learning factor when higher than target
        :param qplus: (float) : learning factor when lower than target
        """
        self.__lfdownup = (csu.enforce_float(qminus), csu.enforce_float(qplus))

    lfdownup = property(get_lfdownup, set_lfdownup)

    def get_close_learning_factor(self):
        """ Get the modal gain learning factor

        :return: (float) : learning factor for modal gain
        """
        return self.__close_learning_factor

    def set_close_learning_factor(self, p):
        """ Set the modal gain optimization learning factor

        :param p: (float) : learning factor
        """
        self.__close_learning_factor = csu.enforce_float(p)

    lf = property(get_close_learning_factor, set_close_learning_factor)

    def get_close_target(self):
        """ Get the autocorrelation target

        :return: (float) : CLOSE autocorrelation target
        """
        return self.__close_target

    def set_close_target(self, t):
        """ Set the autocorrelation target

        :param t: (float) : close target
        """
        self.__close_target = csu.enforce_float(t)

    close_target = property(get_close_target, set_close_target)

    def get_close_update_index(self):
        """ Get the modal gains update rate

        :return: (int) : CLOSE update index
        """
        return self.__close_update_index

    def set_close_update_index(self, idx):
        """ Set the modal gains update rate

        :param idx: (int) : close update index
        """
        self.__close_update_index = csu.enforce_int(idx)

    close_update_index = property(get_close_update_index, set_close_update_index)

    def get_n_iir_in(self):
        """ Get the number of inputs used in iir filter

        :return: (int) : number of iir inputs
        """
        return self.__n_iir_in

    def set_n_iir_in(self, n):
        """ Set the number of inputs used in iir filter

        :param : (int) : number of iir inputs
        """
        self.__n_iir_in = csu.enforce_int(n)

    n_iir_in = property(get_n_iir_in, set_n_iir_in)

    def get_n_iir_out(self):
        """ Get the number of outputs used in iir filter

        :return: (int) : number of iir outputs
        """
        return self.__n_iir_out

    def set_n_iir_out(self, n):
        """ Set the number of outputs used in iir filter

        :param : (int) : number of iir outputs
        """
        self.__n_iir_out = csu.enforce_int(n)

    n_iir_out = property(get_n_iir_out, set_n_iir_out)

    def get_polc(self):
        """ Get POLC flag (True means using POL slopes)

        :return: (bool) : POLC flag
        """
        return self.__polc

    def set_polc(self, p):
        """ Set POLC flag (True means using POL slopes)

        :param : (bool) : POLC flag
        """
        self.__polc = csu.enforce_or_cast_bool(p)

    polc = property(get_polc, set_polc)

    def get_modal(self):
        """ Get flag to use modal control \n(allows MVM from modes to actu)

        :return: (bool) : modal flag
        """
        return self.__modal

    def set_modal(self, m):
        """ Set flag to use modal control \n(allows MVM from modes to actu)

        :param : (bool) : modal flag
        """
        self.__modal = csu.enforce_or_cast_bool(m)

    modal = property(get_modal, set_modal)

    def get_kernconv4imat(self):
        """Get kernconv4imat, i.e. a flag for using kernel convolution to have better
        sensitivity on SH spot movements for imat computation

        :return: (int) : kernconv4imat
        """
        return self.__kernconv4imat

    def set_kernconv4imat(self, n):
        """Set kernconv4imat, i.e. a flag for using kernel convolution to have better
        sensitivity on SH spot movements for imat computation

        :param k: (int) : kernconv4imat
        """
        self.__kernconv4imat = csu.enforce_or_cast_bool(n)

    kernconv4imat = property(get_kernconv4imat, set_kernconv4imat)
