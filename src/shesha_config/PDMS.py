'''
Param_dm class definition
Parameters for DM
'''

import numpy as np
import shesha_constants as scons
from . import config_setter_utils as csu


#################################################
# P-Class (parametres) Param_dm
#################################################
class Param_dm:

    def __init__(self):

        # DM properties
        self.__nact = 0  # DM number of actuators
        self.__alt = 0.0  # DM conjugation altitude
        self.__thresh = 0.0  # Threshold on response for selection
        self.__coupling = 0.2  # Actuator coupling (< .3)
        self.__gain = 1.0  # Actuator gains
        self.__pupoffset = np.array([0, 0])
        # Global offset in pupil (x,y) of the whole actuator pattern

        self.__unitpervolt = 0.01
        # Influence function sensitivity in unit/volt. Optional [0.01]
        # Stackarray: mic/volt, Tip-tilt: arcsec/volt.
        self.__push4imat = 1.  # nominal voltage for imat

        # Margins for actuator selection
        self.__margin_out = -1.  # outer margin (pitches) from pupil diameter
        # inner margin (pitches) from central obstruction
        self.__margin_in = 0.
        self.__pzt_extent = 5.  # Extent of pzt DM (pitches)

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
        self.__influType = scons.InfluType.DEFAULT  # Private storage of influType
        self.__type_kl = scons.KLType.KOLMO  # Private storage for KL type

        # HDF5 storage management
        self.__file_influ_hdf5 = None  # Filename for influ hdf5 file
        self.__center_name = None  # Center name in hdf5
        self.__cube_name = None  # Influence function cube name in hdf5
        self.__x_name = None  # x coord name in hdf5
        self.__y_name = None  # y coord name in hdf5
        self.__influ_res = None  # influence resolution name in hdf5
        self.__diam_dm = None  # name for dm diameter
        self.__diam_dm_proj = None  # name for dm diameter in pupil plane

        # PXD cleanup
        # internal kwrd
        self.__pitch = None
        """ inter-actuator space in pixels"""
        self.__ntotact = None
        """ total number of actuators"""
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

    def set_ap(self, ap):
        """ Set ap TODO!!!

        :param ap: (float) : TODO
        """
        self.__ap = csu.enforce_float(ap)

    ap = property(lambda x: x.__ap, set_ap)

    def set_nfunc(self, nfunc):
        """ Set nfunc TODO !!!

        :param nfunc: (int) : TODO
        """
        self.__nfunc = csu.enforce_int(nfunc)

    nfunc = property(lambda x: x.__nfunc, set_nfunc)

    def set_pzt_extent(self, p):
        """ Set extent of pzt dm in pich unit default = 5

        :param p: (int) : extent pzt dm
        """
        self.__pzt_extent = csu.enforce_int(p)

    pzt_extent = property(lambda x: x.__pzt_extent, set_pzt_extent)

    def set_influType(self, t):
        """ Set the influence function type for pzt DM

        :param t: (str) : centroider type
        """
        self.__influType = scons.check_enum(scons.InfluType, t)

    influType = property(lambda x: x.__influType, set_influType)

    def set_influpos(self, ip):
        """ Set the influence functions pixels that contributes to each DM pixel

        :param ip: (np.ndarray[ndim=1, drype=np.int32]) : influpos
        """
        self.__influpos = csu.enforce_array(ip, ip.size, dtype=np.int32)

    _influpos = property(lambda x: x.__influpos, set_influpos)

    def set_ninflu(self, n):
        """ Set the number of influence functions pixels that contributes
        to each DM pixel

        :param n: (np.ndarray[ndim=1, drype=np.int32]) : ninflu
        """
        self.__ninflu = csu.enforce_array(n, n.size, dtype=np.int32)

    _ninflu = property(lambda x: x.__ninflu, set_ninflu)

    def set_influstart(self, n):
        """ Set the index where to start a new DM pixel shape in the array influpos
        to each DM pixel

        :param n: (np.ndarray[ndim=1, drype=np.int32]) : influstart
        """
        self.__influstart = csu.enforce_array(n, n.size, dtype=np.int32)

    _influstart = property(lambda x: x.__influstart, set_influstart)

    def set_gain(self, g):
        """ Set the gain to apply to the actuators of the dm

        :param g: (float) : gain
        """
        self.__gain = csu.enforce_float(g)

    gain = property(lambda x: x.__gain, set_gain)

    def set_nkl(self, n):
        """ Set the number of KL modes used for computation of covmat in case of minimum variance controller

        :param n: (long) : number of KL modes
        """
        self.__nkl = csu.enforce_int(n)

    nkl = property(lambda x: x.__nkl, set_nkl)

    def set_type_kl(self, t):
        """ Set the type of KL used for computation

        :param t: (string) : KL types : kolmo or karman
        """
        self.__type_kl = scons.check_enum(scons.KLType, t)

    type_kl = property(lambda x: x.__type_kl, set_type_kl)

    def set_type(self, t):
        """ set the dm type

        :param t: (str) : type of dm
        """
        self.__type = scons.check_enum(scons.DmType, t)

    type = property(lambda x: x.__type, set_type)

    def set_type_pattern(self, t):
        """ set the pattern type

        :param t: (str) : type of pattern
        """
        self.__type_pattern = scons.check_enum(scons.PatternType, t)

    type_pattern = property(lambda x: x.__type_pattern, set_type_pattern)

    def set_file_influ_hdf5(self, f):
        """ set the name of hdf5 influence file

        :param filename: (str) : Hdf5 file influence name
        """
        self.__file_influ_hdf5 = f

    file_influ_hdf5 = property(lambda x: x.__file_influ_hdf5, set_file_influ_hdf5)

    def set_center_name(self, f):
        """ set the name of hdf5 influence file

        :param filename: (str) : Hdf5 file influence name
        """
        self.__center_name = f

    center_name = property(lambda x: x.__center_name, set_center_name)

    def set_cube_name(self, cubename):
        """ set the name of influence cube in hdf5

        :param cubename: (str) : name of influence cube
        """
        self.__cube_name = cubename

    cube_name = property(lambda x: x.__cube_name, set_cube_name)

    def set_x_name(self, xname):
        """ set the name of x coord of influence fonction in file

        :param t: (str) : name of x coord of influence
        """
        self.__x_name = xname

    x_name = property(lambda x: x.__x_name, set_x_name)

    def set_y_name(self, yname):
        """ set the name of y coord of influence fonction in file

        :param yname: (str) : name of y coord of influence
        """
        self.__y_name = yname

    y_name = property(lambda x: x.__y_name, set_y_name)

    def set_influ_res(self, res):
        """ set the name of influence fonction resolution in file

        :param res: (str) : name of resoltion (meter/pixel) of influence
        """
        self.__influ_res = res

    influ_res = property(lambda x: x.__influ_res, set_influ_res)

    def set_diam_dm(self, di):
        """ set the name of dm diameter in file

        :param di: (str) : name of diameter (meter) dm
        """
        self.__diam_dm = di

    diam_dm = property(lambda x: x.__diam_dm, set_diam_dm)

    def set_diam_dm_proj(self, dp):
        """ set the name of dm diameter projet on puille in file

        :param dp: (str) : name of diameter (meter in pupil plan) dm
        """
        self.__diam_dm_proj = dp

    diam_dm_proj = property(lambda x: x.__diam_dm_proj, set_diam_dm_proj)

    def set_nact(self, n):
        """ set the number of actuator

        :param n: (long) : number of actuators in the dm
        """
        self.__nact = csu.enforce_int(n)

    nact = property(lambda x: x.__nact, set_nact)

    def set_margin_out(self, n):
        """ set the margin for outside actuator select

        :param n: (float) : unit is actuator pitch (+) for extra (-) for intra
        """
        self.__margin_out = csu.enforce_float(n)

    margin_out = property(lambda x: x.__margin_out, set_margin_out)

    def set_margin_in(self, n):
        """ set the margin for inside actuator select (central obstruction)

        :param n: (float) : unit is actuator pitch (+) for extra (-) for intra
        """
        self.__margin_in = csu.enforce_float(n)

    margin_in = property(lambda x: x.__margin_in, set_margin_in)

    def set_alt(self, a):
        """ set the conjugaison altitude

        :param a: (float) : conjugaison altitude (im m)
        """
        self.__alt = csu.enforce_float(a)

    alt = property(lambda x: x.__alt, set_alt)

    def set_thresh(self, t):
        """ set the threshold on response for selection

        :param t: (float) : threshold on response for selection (<1)
        """
        self.__thresh = csu.enforce_float(t)

    thresh = property(lambda x: x.__thresh, set_thresh)

    def set_coupling(self, c):
        """ set the actuators coupling

        :param c: (float) : actuators coupling (<0.3)
        """
        self.__coupling = csu.enforce_float(c)

    coupling = property(lambda x: x.__coupling, set_coupling)

    def set_unitpervolt(self, u):
        """ set the Influence function sensitivity

        :param u: (float) : Influence function sensitivity in unit/volt
        """
        self.__unitpervolt = csu.enforce_float(u)

    unitpervolt = property(lambda x: x.__unitpervolt, set_unitpervolt)

    def set_push4imat(self, p):
        """ set the nominal voltage for imat

        :param p: (float) : nominal voltage for imat
        """
        self.__push4imat = csu.enforce_float(p)

    push4imat = property(lambda x: x.__push4imat, set_push4imat)

    def set_ntotact(self, n):
        """ set the total number of actuators

        :param n: (long) : total number of actuators
        """
        self.__ntotact = csu.enforce_int(n)

    _ntotact = property(lambda x: x.__ntotact, set_ntotact)

    def set_pitch(self, p):
        """ set the actuators pitch [pixels]

        :param p: (float) : actuators pitch [pixels]
        """
        self.__pitch = csu.enforce_float(p)

    _pitch = property(lambda x: x.__pitch, set_pitch)

    def set_influsize(self, s):
        """ set the actuators influsize [pixels]

        :param s: (int) : actuators influsize [pixels]
        """
        self.__influsize = csu.enforce_int(s)

    _influsize = property(lambda x: x.__influsize, set_influsize)

    def set_n1(self, n):
        """ set the position of bottom left pixel in the largest support

        :param n: (int) : actuators n1 [pixels]
        """
        self.__n1 = csu.enforce_int(n)

    _n1 = property(lambda x: x.__n1, set_n1)

    def set_n2(self, n):
        """ set the position of bottom right pixel in the largest support

        :param n: (int) : actuators n2 [pixels]
        """
        self.__n2 = csu.enforce_int(n)

    _n2 = property(lambda x: x.__n2, set_n2)

    def set_xpos(self, xpos):
        """ Set the x positions of influ functions (lower left corner)

        :param xpos: (np.ndarray[ndim=1,dtype=np.float32_t]) : x positions of influ functions
        """
        self.__xpos = csu.enforce_array(xpos, self.__ntotact, dtype=np.float32)

    _xpos = property(lambda x: x.__xpos, set_xpos)

    def set_ypos(self, ypos):
        """ Set the y positions of influ functions (lower left corner)

        :param ypos: (np.ndarray[ndim=1,dtype=np.float32_t]) : y positions of influ functions
        """
        self.__ypos = csu.enforce_array(ypos, self.__ntotact, dtype=np.float32)

    _ypos = property(lambda x: x.__ypos, set_ypos)

    def set_i1(self, i1):
        """ Set the X-position of the bottom left corner of each influence function

        :param i1: (np.ndarray[ndim=1,dtype=np.int32_t]) :
        """
        self.__i1 = csu.enforce_array(i1, self.__ntotact, dtype=np.int32)

    _i1 = property(lambda x: x.__i1, set_i1)

    def set_j1(self, j1):
        """ Set the Y-position of the bottom left corner of each influence function

        :param j1: (np.ndarray[ndim=1,dtype=np.int32_t]) :
        """
        self.__j1 = csu.enforce_array(j1, self.__ntotact, dtype=np.int32)

    _j1 = property(lambda x: x.__j1, set_j1)

    def set_influ(self, influ):
        """ Set the influence function

        :param influ: (np.ndarray[ndim=3,dtype=np.float32_t]) : influence function
        """
        self.__influ = csu.enforce_arrayMultiDim(influ,
                                                 (self.__influsize, self.__influsize,
                                                  self._ntotact), dtype=np.float32)

    _influ = property(lambda x: x.__influ, set_influ)

    def set_pupoffset(self, off):
        """ Set the pupil offset in meters

        :param off: (np.ndarray[ndim=1,dtype=np.float32_t]) : offsets [m]
        """
        self.__pupoffset = csu.enforce_array(off, 2, dtype=np.float32)

    pupoffset = property(lambda x: x.__pupoffset, set_pupoffset)

    def set_puppixoffset(self, off):
        """ Set the pupil offset in pixels

        :param off: (np.ndarray[ndim=1,dtype=np.float32_t]) : offsets [pixels]
        """
        self.__puppixoffset = csu.enforce_array(off, 2, dtype=np.float32)

    _puppixoffset = property(lambda x: x.__puppixoffset, set_puppixoffset)

    def set_outscl(self, L0):
        """ Set the outer scale for KL with Von Karman spectrum

        :param L0: (float) : outer scale [m]
        """
        self.__outscl = csu.enforce_float(L0)

    outscl = property(lambda x: x.__outscl, set_outscl)

    def set_nr(self, n):
        """ Set the number of radial points for KL

        :param n: (int) : number of radial points
        """
        self.__nr = csu.enforce_int(n)

    _nr = property(lambda x: x.__nr, set_nr)

    def set_npp(self, n):
        """ Set the number of elements (?) for KL

        :param n: (int) : number of elements
        """
        self.__npp = csu.enforce_int(n)

    _npp = property(lambda x: x.__npp, set_npp)

    def set_ncp(self, n):
        """ Set the dimension of grid (?)

        :param n: (int) : dimension
        """
        self.__ncp = csu.enforce_int(n)

    _ncp = property(lambda x: x.__ncp, set_ncp)

    def set_ord(self, n):
        """ Set the radial orders of the basis

        :param n: (int) : radial order of the basis
        """
        self.__ord = csu.enforce_array(n, n.size, dtype=np.int32)

    _ord = property(lambda x: x.__ord, set_ord)

    def set_rabas(self, r):
        """ Set the radial array of the KL basis

        :param r: (np.ndarray[ndim=1,dtype=np.float32_t]) : radial array
        """
        self.__rabas = csu.enforce_arrayMultiDim(r, r.shape, dtype=np.float32)

    _rabas = property(lambda x: x.__rabas, set_rabas)

    def set_azbas(self, r):
        """ Set the azimuthal array of the KL basis

        :param r: (np.ndarray[ndim=1,dtype=np.float32_t]) : azimuthal array
        """
        self.__azbas = csu.enforce_arrayMultiDim(r, r.shape, dtype=np.float32)

    _azbas = property(lambda x: x.__azbas, set_azbas)

    def set_cr(self, r):
        """ Set the radial coordinates in carthesian grid

        :param r: (np.ndarray[ndim=1,dtype=np.float32_t]) : radial coordinates in carthesian grid
        """
        self.__cr = csu.enforce_arrayMultiDim(r, r.shape, dtype=np.float32)

    _cr = property(lambda x: x.__cr, set_cr)

    def set_cp(self, r):
        """ Set the phi coordinates in carthesian grid

        :param r: (np.ndarray[ndim=1,dtype=np.float32_t]) : phi coordinates in carthesian grid
        """
        self.__cp = csu.enforce_arrayMultiDim(r, r.shape, dtype=np.float32)

    _cp = property(lambda x: x.__cp, set_cp)
