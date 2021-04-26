## @package   shesha.config.PDMS
## @brief     Param_dm class definition
## @author    COMPASS Team <https://github.com/ANR-COMPASS>
## @version   5.1.0
## @date      2020/05/18
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
import shesha.constants as scons
from . import config_setter_utils as csu


#################################################
# P-Class (parametres) Param_dm
#################################################
class Param_dm:

    def __init__(self):

        # DM properties
        self.__nact = 0  # linear number of actuators across the pupil diameter
        self.__alt = 0.0  # DM conjugation altitude
        self.__thresh = 0.0  # Threshold on response for selection
        self.__keep_all_actu = False  # if True, don't mask actu by pupil
        self.__coupling = 0.2  # Actuator coupling (< .3)
        self.__gain = 1.0  # Actuator gains
        self.__pupoffset = np.array([0, 0])
        self.__dim_screen = 0  # Phase screen dimension
        # Global offset in pupil (x,y) of the whole actuator pattern

        self.__unitpervolt = 0.01
        # Influence function sensitivity in unit/volt. Optional [0.01]
        # Stackarray: mic/volt, Tip-tilt: arcsec/volt.
        self.__push4imat = 1.  # nominal voltage for imat

        # Margins for actuator selection
        self.__margin_out = None  # outer margin (pitches) from pupil diameter
        # inner margin (pitches) from central obstruction
        self.__margin_in = 0.
        self.__pzt_extent = 5.  # Extent of pzt DM (pitches)
        self.__segmented_mirror = False  # Crop influence functions where spiders are.

        # KL DM
        self.__nfunc = 0
        self.__nkl = 0  # Number of KL for KL dm
        self.__outscl = None  # Outer scale in units of telescope diam for Karman KL
        self.__nr = None  # number of radial points
        self.__npp = None  # number of elements
        self.__ord = None  # the radial orders of the basis
        self.__rabas = None  # the radial array of the basis
        self.__azbas = None  # the azimuthal array of the basis
        self.__ncp = None  # dim of grid
        self.__cr = None  # radial coord in cartesien grid
        self.__cp = None  # phi coord in cartesien grid
        self.__ap = None
        self.__nfunc = 0

        # Hidden variable safe-typed in shesha_constants
        self.__type = None  # Private storage of type
        self.__type_pattern = None  # Private storage of type_pattern
        self.__influ_type = scons.InfluType.DEFAULT  # Private storage of influ_type
        self.__type_kl = scons.KLType.KOLMO  # Private storage for KL type

        # HDF5 storage management
        self.__file_influ_fits = None  # Filename for influ hdf5 file
        self.__center_name = None  # Center name in hdf5
        self.__cube_name = None  # Influence function cube name in hdf5
        self.__x_name = None  # x coord name in hdf5
        self.__y_name = None  # y coord name in hdf5
        self.__influ_res = None  # influence resolution name in hdf5
        self.__diam_dm = None  # diam of the tel pupil projected on the dm plane
        self.__diam_dm_proj = None  # diam of the dm pupil projected on the tel pupil plane

        # PXD cleanup
        # internal kwrd
        self.__pitch = None
        """ inter-actuator space in pixels"""
        self.__ntotact = None
        """ total number of actuators over the full area of the pupil"""
        self.__influsize = None
        """ influ function support size"""
        self.__n1 = None
        """ position of leftmost pixel in largest support"""
        self.__n2 = None
        """ position of rightmost pixel in largest support"""
        self.__puppixoffset = None
        self.__influ = None
        """ influence functions"""
        self.__xpos = None
        """ x positions of influ functions"""
        self.__ypos = None
        """ y positions of influ functions"""
        self.__i1 = None
        self.__j1 = None
        self.__influpos = None
        self.__ninflu = None
        """ Influence functions"""
        self.__influstart = None  # np.ndarray - Influence function handling

        # Registration
        self.__G = 1.0
        """ Magnifying factor"""
        self.__theta = 0.0
        """ WFS rotation angle in the pupil"""
        self.__dx = 0.0
        """ X axis misalignment in meters"""
        self.__dy = 0.0
        """ Y axis misalignment in meters"""

    def get_ap(self):
        """ Get ap TODO!!!

        :return: (float) : TODO
        """
        return self.__ap

    def set_ap(self, ap):
        """ Set ap TODO!!!

        :param ap: (float) : TODO
        """
        self.__ap = csu.enforce_arrayMultiDim(ap, (ap.shape[0], ap.shape[1]),
                                              dtype=np.float32)

    ap = property(get_ap, set_ap)

    def get_nfunc(self):
        """ Get nfunc TODO !!!

        :return: (int) : TODO
        """
        return self.__nfunc

    def set_nfunc(self, nfunc):
        """ Set nfunc TODO !!!

        :param nfunc: (int) : TODO
        """
        self.__nfunc = csu.enforce_int(nfunc)

    nfunc = property(get_nfunc, set_nfunc)

    def get_pzt_extent(self):
        """ Get extent of pzt dm in pich unit default = 5

        :return: (int) : extent pzt dm
        """
        return self.__pzt_extent

    def set_pzt_extent(self, p):
        """ Set extent of pzt dm in pich unit default = 5

        :param p: (int) : extent pzt dm
        """
        self.__pzt_extent = csu.enforce_int(p)

    pzt_extent = property(get_pzt_extent, set_pzt_extent)

    def get_segmented_mirror(self):
        return self.__segmented_mirror

    def set_segmented_mirror(self, b):
        """ Define mirror influence functions to be cropped by the spiders
        (more generally, pupil edges)

        :param p: (bool) : segment the mirror
        """
        self.__segmented_mirror = csu.enforce_or_cast_bool(b)

    segmented_mirror = property(get_segmented_mirror, set_segmented_mirror)

    def get_influ_type(self):
        """ Get the influence function type for pzt DM

        :return: (str) : centroider type
        """
        return self.__influ_type

    def set_influ_type(self, t):
        """ Set the influence function type for pzt DM

        :param t: (str) : centroider type
        """
        self.__influ_type = scons.check_enum(scons.InfluType, t)

    influ_type = property(get_influ_type, set_influ_type)

    def get_influpos(self):
        """ Get the influence functions pixels that contributes to each DM pixel

        :return: (np.ndarray[ndim=1, drype=np.int32]) : influpos
        """
        return self.__influpos

    def set_influpos(self, ip):
        """ Set the influence functions pixels that contributes to each DM pixel

        :param ip: (np.ndarray[ndim=1, drype=np.int32]) : influpos
        """
        self.__influpos = csu.enforce_array(ip, ip.size, dtype=np.int32)

    _influpos = property(get_influpos, set_influpos)

    def get_ninflu(self):
        """ Get the number of influence functions pixels that contributes
        to each DM pixel

        :return: (np.ndarray[ndim=1, drype=np.int32]) : ninflu
        """
        return self.__ninflu

    def set_ninflu(self, n):
        """ Set the number of influence functions pixels that contributes
        to each DM pixel

        :param n: (np.ndarray[ndim=1, drype=np.int32]) : ninflu
        """
        self.__ninflu = csu.enforce_array(n, n.size, dtype=np.int32)

    _ninflu = property(get_ninflu, set_ninflu)

    def get_influstart(self):
        """ Get the index where to start a new DM pixel shape in the array influpos
        to each DM pixel

        :return: (np.ndarray[ndim=1, drype=np.int32]) : influstart
        """
        return self.__influstart

    def set_influstart(self, n):
        """ Set the index where to start a new DM pixel shape in the array influpos
        to each DM pixel

        :param n: (np.ndarray[ndim=1, drype=np.int32]) : influstart
        """
        self.__influstart = csu.enforce_array(n, n.size, dtype=np.int32)

    _influstart = property(get_influstart, set_influstart)

    def get_gain(self):
        """ Get the gain to apply to the actuators of the dm

        :return: (float) : gain
        """
        return self.__gain

    def set_gain(self, g):
        """ Set the gain to apply to the actuators of the dm

        :param g: (float) : gain
        """
        self.__gain = csu.enforce_float(g)

    gain = property(get_gain, set_gain)

    def _get_dim_screen(self):
        """ Get the phase screen dimension

        :return: (long) : phase screen dimension
        """
        return self.__dim_screen

    def _set_dim_screen(self, n):
        """ Set the phase screen dimension

        :param n: (long) : phase screen dimension
        """
        self.__dim_screen = csu.enforce_int(n)

    _dim_screen = property(_get_dim_screen, _set_dim_screen)

    def get_nkl(self):
        """ Get the number of KL modes used for computation of covmat in case of minimum variance controller

        :return: (long) : number of KL modes
        """
        return self.__nkl

    def set_nkl(self, n):
        """ Set the number of KL modes used for computation of covmat in case of minimum variance controller

        :param n: (long) : number of KL modes
        """
        self.__nkl = csu.enforce_int(n)

    nkl = property(get_nkl, set_nkl)

    def get_type_kl(self):
        """ Get the type of KL used for computation

        :return: (string) : KL types : kolmo or karman
        """
        return self.__type_kl

    def set_type_kl(self, t):
        """ Set the type of KL used for computation

        :param t: (string) : KL types : kolmo or karman
        """
        self.__type_kl = scons.check_enum(scons.KLType, t)

    type_kl = property(get_type_kl, set_type_kl)

    def get_type(self):
        """ Get the dm type

        :return: (str) : type of dm
        """
        return self.__type

    def set_type(self, t):
        """ set the dm type

        :param t: (str) : type of dm
        """
        self.__type = scons.check_enum(scons.DmType, t)

    type = property(get_type, set_type)

    def get_type_pattern(self):
        """ Get the pattern type

        :return: (str) : type of pattern
        """
        return self.__type_pattern

    def set_type_pattern(self, t):
        """ set the pattern type

        :param t: (str) : type of pattern
        """
        self.__type_pattern = scons.check_enum(scons.PatternType, t)

    type_pattern = property(get_type_pattern, set_type_pattern)

    def get_file_influ_fits(self):
        """ Get the name of hdf5 influence file

        :return: (str) : Hdf5 file influence name
        """
        return self.__file_influ_fits

    def set_file_influ_fits(self, f):
        """ set the name of hdf5 influence file

        :param filename: (str) : Hdf5 file influence name
        """
        self.__file_influ_fits = f

    file_influ_fits = property(get_file_influ_fits, set_file_influ_fits)

    def get_center_name(self):
        """ Get the name of hdf5 influence file

        :return: (str) : Hdf5 file influence name
        """
        return self.__center_name

    def set_center_name(self, f):
        """ set the name of hdf5 influence file

        :param filename: (str) : Hdf5 file influence name
        """
        self.__center_name = f

    center_name = property(get_center_name, set_center_name)

    def get_cube_name(self):
        """ Get the name of influence cube in hdf5

        :return: (str) : name of influence cube
        """
        return self.__cube_name

    def set_cube_name(self, cubename):
        """ set the name of influence cube in hdf5

        :param cubename: (str) : name of influence cube
        """
        self.__cube_name = cubename

    cube_name = property(get_cube_name, set_cube_name)

    def get_x_name(self):
        """ Get the name of x coord of influence fonction in file

        :return: (str) : name of x coord of influence
        """
        return self.__x_name

    def set_x_name(self, xname):
        """ set the name of x coord of influence fonction in file

        :param t: (str) : name of x coord of influence
        """
        self.__x_name = xname

    x_name = property(get_x_name, set_x_name)

    def get_y_name(self):
        """ Get the name of y coord of influence fonction in file

        :return: (str) : name of y coord of influence
        """
        return self.__y_name

    def set_y_name(self, yname):
        """ set the name of y coord of influence fonction in file

        :param yname: (str) : name of y coord of influence
        """
        self.__y_name = yname

    y_name = property(get_y_name, set_y_name)

    def get_influ_res(self):
        """ Get the name of influence fonction resolution in file

        :return: (str) : name of resoltion (meter/pixel) of influence
        """
        return self.__influ_res

    def set_influ_res(self, res):
        """ set the name of influence fonction resolution in file

        :param res: (str) : name of resoltion (meter/pixel) of influence
        """
        self.__influ_res = res

    influ_res = property(get_influ_res, set_influ_res)

    def get_diam_dm(self):
        """ Get the diameter of the tel pupil projected on the dm plane

        :return: (float) : diameter (meters) of the tel pupil projected on the dm plane
        """
        return self.__diam_dm

    def set_diam_dm(self, di):
        """ Set the diameter of the tel pupil projected on the dm plane

        :param di: (float) : diameter (meters) of the tel pupil projected on the dm plane
        """
        self.__diam_dm = di

    diam_dm = property(get_diam_dm, set_diam_dm)

    def get_diam_dm_proj(self):
        """ Get the diameter of the dm pupil projected on the tel pupil plane

        :return: (float) : diameter (meters) of the dm pupil projected on the tel pupil plane
        """
        return self.__diam_dm_proj

    def set_diam_dm_proj(self, dp):
        """ Set the diameter of the dm pupil projected on the tel pupil plane

        :param dp: (float) : diameter (meters) of the dm pupil projected on the tel pupil plane
        """
        self.__diam_dm_proj = dp

    diam_dm_proj = property(get_diam_dm_proj, set_diam_dm_proj)

    def get_nact(self):
        """ Get the number of actuator

        :return: (long) : number of actuators in the dm
        """
        return self.__nact

    def set_nact(self, n):
        """ set the number of actuator

        :param n: (long) : number of actuators in the dm
        """
        self.__nact = csu.enforce_int(n)

    nact = property(get_nact, set_nact)

    def get_margin_out(self):
        """ Get the margin for outside actuator select

        :return: (float) : unit is actuator pitch (+) for extra (-) for intra
        """
        return self.__margin_out

    def set_margin_out(self, n):
        """ set the margin for outside actuator select

        :param n: (float) : unit is actuator pitch (+) for extra (-) for intra
        """
        self.__margin_out = csu.enforce_float(n)

    margin_out = property(get_margin_out, set_margin_out)

    def get_margin_in(self):
        """ Get the margin for inside actuator select (central obstruction)

        :return: (float) : unit is actuator pitch (+) for extra (-) for intra
        """
        return self.__margin_in

    def set_margin_in(self, n):
        """ set the margin for inside actuator select (central obstruction)

        :param n: (float) : unit is actuator pitch (+) for extra (-) for intra
        """
        self.__margin_in = csu.enforce_float(n)

    margin_in = property(get_margin_in, set_margin_in)

    def get_alt(self):
        """ Get the conjugaison altitude

        :return: (float) : conjugaison altitude (im m)
        """
        return self.__alt

    def set_alt(self, a):
        """ set the conjugaison altitude

        :param a: (float) : conjugaison altitude (im m)
        """
        self.__alt = csu.enforce_float(a)

    alt = property(get_alt, set_alt)

    def get_thresh(self):
        """ Get the threshold on response for selection

        :return: (float) : threshold on response for selection (<1)
        """
        return self.__thresh

    def set_thresh(self, t):
        """ set the threshold on response for selection

        :param t: (float) : threshold on response for selection (<1)
        """
        self.__thresh = csu.enforce_float(t)

    thresh = property(get_thresh, set_thresh)

    def get_keep_all_actu(self):
        """ Get the flag for keeping all actuators

        :return: (bool) : keep all actuator flag (boolean)
        """
        return self.__keep_all_actu

    def set_keep_all_actu(self, k):
        """ set the flag for keeping all actuators

        :param k: (f) : keep all actuator flag (boolean)
        """
        self.__keep_all_actu = csu.enforce_or_cast_bool(k)

    keep_all_actu = property(get_keep_all_actu, set_keep_all_actu)

    def get_coupling(self):
        """ Get the actuators coupling

        :return: (float) : actuators coupling (<0.3)
        """
        return self.__coupling

    def set_coupling(self, c):
        """ set the actuators coupling

        :param c: (float) : actuators coupling (<0.3)
        """
        self.__coupling = csu.enforce_float(c)

    coupling = property(get_coupling, set_coupling)

    def get_unitpervolt(self):
        """ Get the Influence function sensitivity

        :return: (float) : Influence function sensitivity in unit/volt
        """
        return self.__unitpervolt

    def set_unitpervolt(self, u):
        """ set the Influence function sensitivity

        :param u: (float) : Influence function sensitivity in unit/volt
        """
        self.__unitpervolt = csu.enforce_float(u)

    unitpervolt = property(get_unitpervolt, set_unitpervolt)

    def get_push4imat(self):
        """ Get the nominal voltage for imat

        :return: (float) : nominal voltage for imat
        """
        return self.__push4imat

    def set_push4imat(self, p):
        """ set the nominal voltage for imat

        :param p: (float) : nominal voltage for imat
        """
        self.__push4imat = csu.enforce_float(p)

    push4imat = property(get_push4imat, set_push4imat)

    def get_ntotact(self):
        """ Get the total number of actuators

        :return: (long) : total number of actuators
        """
        return self.__ntotact

    def set_ntotact(self, n):
        """ set the total number of actuators

        :param n: (long) : total number of actuators
        """
        self.__ntotact = csu.enforce_int(n)

    _ntotact = property(get_ntotact, set_ntotact)

    def get_pitch(self):
        """ Get the actuators pitch [pixels]

        :return: (float) : actuators pitch [pixels]
        """
        return self.__pitch

    def set_pitch(self, p):
        """ set the actuators pitch [pixels]

        :param p: (float) : actuators pitch [pixels]
        """
        self.__pitch = csu.enforce_float(p)

    _pitch = property(get_pitch, set_pitch)

    def get_influsize(self):
        """ Get the actuators influsize [pixels]

        :return: (int) : actuators influsize [pixels]
        """
        return self.__influsize

    def set_influsize(self, s):
        """ set the actuators influsize [pixels]

        :param s: (int) : actuators influsize [pixels]
        """
        self.__influsize = csu.enforce_int(s)

    _influsize = property(get_influsize, set_influsize)

    def get_n1(self):
        """ Get the position of bottom left pixel in the largest support

        :return: (int) : actuators n1 [pixels]
        """
        return self.__n1

    def set_n1(self, n):
        """ set the position of bottom left pixel in the largest support

        :param n: (int) : actuators n1 [pixels]
        """
        self.__n1 = csu.enforce_int(n)

    _n1 = property(get_n1, set_n1)

    def get_n2(self):
        """ Get the position of bottom right pixel in the largest support

        :return: (int) : actuators n2 [pixels]
        """
        return self.__n2

    def set_n2(self, n):
        """ set the position of bottom right pixel in the largest support

        :param n: (int) : actuators n2 [pixels]
        """
        self.__n2 = csu.enforce_int(n)

    _n2 = property(get_n2, set_n2)

    def get_xpos(self):
        """ Get the x positions of influ functions (lower left corner)

        :return: (np.ndarray[ndim=1,dtype=np.float32_t]) : x positions of influ functions
        """
        return self.__xpos

    def set_xpos(self, xpos):
        """ Set the x positions of influ functions (lower left corner)

        :param xpos: (np.ndarray[ndim=1,dtype=np.float32_t]) : x positions of influ functions
        """
        self.__xpos = csu.enforce_array(xpos, self.__ntotact, dtype=np.float32)

    _xpos = property(get_xpos, set_xpos)

    def get_ypos(self):
        """ Get the y positions of influ functions (lower left corner)

        :return: (np.ndarray[ndim=1,dtype=np.float32_t]) : y positions of influ functions
        """
        return self.__ypos

    def set_ypos(self, ypos):
        """ Set the y positions of influ functions (lower left corner)

        :param ypos: (np.ndarray[ndim=1,dtype=np.float32_t]) : y positions of influ functions
        """
        self.__ypos = csu.enforce_array(ypos, self.__ntotact, dtype=np.float32)

    _ypos = property(get_ypos, set_ypos)

    def get_i1(self):
        """ Get the X-position of the bottom left corner of each influence function

        :return: (np.ndarray[ndim=1,dtype=np.int32_t]) :
        """
        return self.__i1

    def set_i1(self, i1):
        """ Set the X-position of the bottom left corner of each influence function

        :param i1: (np.ndarray[ndim=1,dtype=np.int32_t]) :
        """
        self.__i1 = csu.enforce_array(i1, self.__ntotact, dtype=np.int32)

    _i1 = property(get_i1, set_i1)

    def get_j1(self):
        """ Get the Y-position of the bottom left corner of each influence function

        :return: (np.ndarray[ndim=1,dtype=np.int32_t]) :
        """
        return self.__j1

    def set_j1(self, j1):
        """ Set the Y-position of the bottom left corner of each influence function

        :param j1: (np.ndarray[ndim=1,dtype=np.int32_t]) :
        """
        self.__j1 = csu.enforce_array(j1, self.__ntotact, dtype=np.int32)

    _j1 = property(get_j1, set_j1)

    def get_influ(self):
        """ Get the influence function

        :return: (np.ndarray[ndim=3,dtype=np.float32_t]) : influence function
        """
        return self.__influ

    def set_influ(self, influ):
        """ Set the influence function

        :param influ: (np.ndarray[ndim=3,dtype=np.float32_t]) : influence function
        """
        self.__influ = csu.enforce_arrayMultiDim(influ,
                                                 (self.__influsize, self.__influsize,
                                                  self._ntotact), dtype=np.float32)

    _influ = property(get_influ, set_influ)

    def get_pupoffset(self):
        """ Get the pupil offset in meters

        :return: (np.ndarray[ndim=1,dtype=np.float32_t]) : offsets [m]
        """
        return self.__pupoffset

    def set_pupoffset(self, off):
        """ Set the pupil offset in meters

        :param off: (np.ndarray[ndim=1,dtype=np.float32_t]) : offsets [m]
        """
        self.__pupoffset = csu.enforce_array(off, 2, dtype=np.float32)

    pupoffset = property(get_pupoffset, set_pupoffset)

    def get_puppixoffset(self):
        """ Get the pupil offset in pixels

        :return: (np.ndarray[ndim=1,dtype=np.float32_t]) : offsets [pixels]
        """
        return self.__puppixoffset

    def set_puppixoffset(self, off):
        """ Set the pupil offset in pixels

        :param off: (np.ndarray[ndim=1,dtype=np.float32_t]) : offsets [pixels]
        """
        self.__puppixoffset = csu.enforce_array(off, 2, dtype=np.float32)

    _puppixoffset = property(get_puppixoffset, set_puppixoffset)

    def get_outscl(self):
        """ Get the outer scale for KL with Von Karman spectrum

        :return: (float) : outer scale [m]
        """
        return self.__outscl

    def set_outscl(self, L0):
        """ Set the outer scale for KL with Von Karman spectrum

        :param L0: (float) : outer scale [m]
        """
        self.__outscl = csu.enforce_float(L0)

    outscl = property(get_outscl, set_outscl)

    def get_nr(self):
        """ Get the number of radial points for KL

        :return: (int) : number of radial points
        """
        return self.__nr

    def set_nr(self, n):
        """ Set the number of radial points for KL

        :param n: (int) : number of radial points
        """
        self.__nr = csu.enforce_int(n)

    _nr = property(get_nr, set_nr)

    def get_npp(self):
        """ Get the number of elements (?) for KL

        :return: (int) : number of elements
        """
        return self.__npp

    def set_npp(self, n):
        """ Set the number of elements (?) for KL

        :param n: (int) : number of elements
        """
        self.__npp = csu.enforce_int(n)

    _npp = property(get_npp, set_npp)

    def get_ncp(self):
        """ Get the dimension of grid (?)

        :return: (int) : dimension
        """
        return self.__ncp

    def set_ncp(self, n):
        """ Set the dimension of grid (?)

        :param n: (int) : dimension
        """
        self.__ncp = csu.enforce_int(n)

    _ncp = property(get_ncp, set_ncp)

    def get_ord(self):
        """ Get the radial orders of the basis

        :return: (int) : radial order of the basis
        """
        return self.__ord

    def set_ord(self, n):
        """ Set the radial orders of the basis

        :param n: (int) : radial order of the basis
        """
        self.__ord = csu.enforce_array(n, n.size, dtype=np.int32)

    _ord = property(get_ord, set_ord)

    def get_rabas(self):
        """ Get the radial array of the KL basis

        :return: (np.ndarray[ndim=1,dtype=np.float32_t]) : radial array
        """
        return self.__rabas

    def set_rabas(self, r):
        """ Set the radial array of the KL basis

        :param r: (np.ndarray[ndim=1,dtype=np.float32_t]) : radial array
        """
        self.__rabas = csu.enforce_arrayMultiDim(r, r.shape, dtype=np.float32)

    _rabas = property(get_rabas, set_rabas)

    def get_azbas(self):
        """ Get the azimuthal array of the KL basis

        :return: (np.ndarray[ndim=1,dtype=np.float32_t]) : azimuthal array
        """
        return self.__azbas

    def set_azbas(self, r):
        """ Set the azimuthal array of the KL basis

        :param r: (np.ndarray[ndim=1,dtype=np.float32_t]) : azimuthal array
        """
        self.__azbas = csu.enforce_arrayMultiDim(r, r.shape, dtype=np.float32)

    _azbas = property(get_azbas, set_azbas)

    def get_cr(self):
        """ Get the radial coordinates in carthesian grid

        :return: (np.ndarray[ndim=1,dtype=np.float32_t]) : radial coordinates in carthesian grid
        """
        return self.__cr

    def set_cr(self, r):
        """ Set the radial coordinates in carthesian grid

        :param r: (np.ndarray[ndim=1,dtype=np.float32_t]) : radial coordinates in carthesian grid
        """
        self.__cr = csu.enforce_arrayMultiDim(r, r.shape, dtype=np.float32)

    _cr = property(get_cr, set_cr)

    def get_cp(self):
        """ Get the phi coordinates in carthesian grid

        :return: (np.ndarray[ndim=1,dtype=np.float32_t]) : phi coordinates in carthesian grid
        """
        return self.__cp

    def set_cp(self, r):
        """ Set the phi coordinates in carthesian grid

        :param r: (np.ndarray[ndim=1,dtype=np.float32_t]) : phi coordinates in carthesian grid
        """
        self.__cp = csu.enforce_arrayMultiDim(r, r.shape, dtype=np.float32)

    _cp = property(get_cp, set_cp)

    def get_G(self):
        """ Get the magnifying factor

        :return: (float) : magnifying factor
        """
        return self.__G

    def set_G(self, G):
        """ Set the magnifying factor

        :param G: (float) : magnifying factor
        """
        self.__G = csu.enforce_float(G)

    G = property(get_G, set_G)

    def get_theta(self):
        """ Get the rotation angle in the pupil

        :return: (float) : rotation angle (rad)
        """
        return self.__theta

    def set_theta(self, theta):
        """ Set the rotation angle in the pupil

        :param theta: (float) : rotation angle (rad)
        """
        self.__theta = csu.enforce_float(theta)

    theta = property(get_theta, set_theta)

    def get_dx(self):
        """ Get the X axis misalignment

        :return: (float) : dx (pix)
        """
        return self.__dx

    def set_dx(self, dx):
        """ Set the X axis misalignment

        :param dx: (float) : dx (pix)
        """
        self.__dx = csu.enforce_float(dx)

    dx = property(get_dx, set_dx)

    def get_dy(self):
        """ Get the Y axis misalignment

        :return: (float) : dy (pix)
        """
        return self.__dy

    def set_dy(self, dy):
        """ Set the Y axis misalignment

        :param dy: (float) : dy (pix)
        """
        self.__dy = csu.enforce_float(dy)

    dy = property(get_dy, set_dy)
