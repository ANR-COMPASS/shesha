'''
Param_wfs class definition
Parameters for WFS
'''

import numpy as np
from . import config_setter_utils as csu
import shesha_constants as scons

#################################################
# P-Class (parametres) Param_wfs
#################################################


class Param_wfs:

    def __init__(self, error_budget=False):
        self.__type = None
        """ type of wfs : "sh" or "pyr"."""
        self.__nxsub = 0
        """ linear number of subaps."""
        self.__npix = 0
        """ number of pixels per subap."""
        self.__pixsize = 0
        """ pixel size (in arcsec) for a subap."""
        self.__Lambda = 0
        """ observation wavelength (in um) for a subap."""
        self.__optthroughput = 0
        """ wfs global throughput."""
        self.__fracsub = 0
        """ minimal illumination fraction for valid subaps."""
        self.__openloop = False
        """ 1 if in "open-loop" mode (i.e. does not see dm)."""
        self.__fssize = 0
        """ size of field stop in arcsec."""
        self.__fstop = None
        """ Fields of the wfs diaphragm shape : "square" or "none" """

        self.__atmos_seen = 0
        """ 1 if the WFS sees the atmosphere layers"""
        self.__dms_seen = None
        """ index of dms seen by the WFS"""
        self.__error_budget = error_budget
        """ If True, enable error budget analysis for the simulation"""

        # target kwrd
        self.__xpos = 0
        """ guide star x position on sky (in arcsec)."""
        self.__ypos = 0
        """ guide star x position on sky (in arcsec)."""
        self.__gsalt = 0
        """ altitude of guide star (in m) 0 if ngs."""
        self.__gsmag = 0
        """ magnitude of guide star."""
        self.__zerop = 0
        """ detector zero point expressed in ph/m**2/s in the bandwidth of the WFS"""
        self.__noise = 0
        """ desired noise : < 0 = no noise / 0 = photon only / > 0 photon + ron."""

        self.__kernel = 0  #
        self.__ftkernel = None  # (float*)

        # lgs only
        self.__lgsreturnperwatt = 0
        """ return per watt factor (high season : 10 ph/cm2/s/W)."""
        self.__laserpower = 0
        """ laser power in W."""
        self.__lltx = 0
        """ x position (in meters) of llt."""
        self.__llty = 0
        """ y position (in meters) of llt."""
        self.__proftype = None
        """ type of sodium profile "gauss", "exp", etc ..."""
        self.__beamsize = None
        """ laser beam fwhm on-sky (in arcsec)."""

        # misalignment

        self.__G = 1.0
        """ Magnifying factor"""
        self.__thetaML = 0.0
        """ WFS rotation angle in the pupil"""
        self.__dx = 0.0
        """ X axis misalignment in pixels"""
        self.__dy = 0.0
        """ Y axis misalignment in pixels"""
        # internal kwrd
        self.__pdiam = 0
        """ pupil diam for a subap (in pixel)"""
        self.__Nfft = 0
        """ array size for fft for a subap (in pixel)"""
        self.__Ntot = 0
        """ total size of hr image for a subap (in pixel)"""
        self.__nrebin = 0
        """ rebin factor from hr to binned image for a subap"""
        self.__nvalid = 0
        """ number of valid subaps"""

        self.__nphotons = 0
        """ number of photons per subap"""
        self.__nphotons4imat = 1.e5
        """ number of photons per subap used for doing imat"""
        self.__subapd = 0
        """ subap diameter (m)"""
        self.__fluxPerSub = None
        """ fraction of nphotons per subap"""
        self.__qpixsize = 0
        """ quantum pixel size for the simulation"""

        self.__istart = None
        """ (int*) x start indexes for cutting phase screens"""
        self.__jstart = None
        """ (int*) y start indexes for cutting phase screens"""
        # cdef np.ndarray _validsubs    # (i,j) indices of valid subaps
        self.__validsubsx = None
        """ (int*) indices of valid subaps along axis x"""
        self.__validsubsy = None
        """ (int*) indices of valid subaps along axis y"""
        self.__isvalid = None
        """ (int*) array of 0/1 for valid subaps"""
        self.__phasemap = None
        """ (int*) array of pixels transform from phase screen into subaps phase screens"""
        self.__hrmap = None
        """ (int*) array of pixels transform from minimal FoV image to (in case type is sh or geo)"""
        self.__sincar = None
        """ (float*) array of pixels transform from minimal FoV image to (in case type is "pyr" or "roof")"""
        # full FoV image (array of 0 if the same)
        self.__binmap = None
        """ (int*) array of pixels transform from full FoV hr images to binned images"""
        self.__halfxy = None
        """ (float*) phase offset for 1/2 pixel shift in (x,y)"""

        self.__submask = None
        """ (float*) fieldstop for each subap"""

        self.__lgskern = None
        """ lgs kernels for each subap"""
        self.__profna = None
        """ sodium profile"""
        self.__altna = None
        """ corresponding altitude"""
        self.__prof1d = None
        """ hr profile"""
        self.__profcum = None
        """ hr profile cumulated"""
        self.__beam = None
        """ 1d beam function"""
        self.__ftbeam = None
        """ 1d beam function fft"""
        self.__azimuth = None
        """ angles of rotation for each spot"""

        # pyramid-nly kwrds
        self.__pyr_ampl = 0
        """ pyramid wfs modulation amplitude radius [arcsec]."""
        self.__pyr_npts = 0
        """ total number of point along modulation circle [unitless]."""
        self.__pyr_pos = None
        """ positions for modulation, overwrites ampl and npts [arcsec]"""
        self.__pyr_loc = None
        """ Location of modulation, before/after the field stop.
        valid value are "before" or "after" (default "after")."""
        self.__pyrtype = None
        """ Type of pyramid, either 0 for "Pyramid" or 1 for "RoofPrism"."""
        self.__pyr_pup_sep = -1
        """ Pyramid pupil separation. (default: long(wfs.nxsub))"""
        self.__pyr_misalignments = None
        """ Pyramid quadrant misalignments: by how much pupil subimages
        are off from their expected positions (in rebinned pixels)"""

        # pyramid internal kwrds
        self._pyr_offsets = None  # (float*)
        self.__pyr_cx = None  # (float*)
        self.__pyr_cy = None  # (float*)

    def set_type(self, type):
        """ Set the type of wfs

        :param t: (str) : type of wfs ("sh" or "pyr")
        """
        self.__type = scons.check_enum(scons.WFSType, type)

    type = property(lambda x: x.__type, set_type)

    def set_nxsub(self, n):
        """ Set the linear number of subaps

        :param n: (long) : linear number of subaps
        """
        self.__nxsub = csu.enforce_int(n)

    nxsub = property(lambda x: x.__nxsub, set_nxsub)

    def set_npix(self, n):
        """ Set the number of pixels per subap

        :param n: (long) : number of pixels per subap
        """
        self.__npix = csu.enforce_int(n)

    npix = property(lambda x: x.__npix, set_npix)

    def set_pixsize(self, p):
        """ Set the pixel size

        :param p: (float) : pixel size (in arcsec) for a subap
        """
        self.__pixsize = csu.enforce_float(p)

    pixsize = property(lambda x: x.__pixsize, set_pixsize)

    def set_Lambda(self, L):
        """ Set the observation wavelength

        :param L: (float) : observation wavelength (in um) for a subap
        """
        self.__Lambda = L

    Lambda = property(lambda x: x.__Lambda, set_Lambda)

    def set_optthroughput(self, o):
        """ Set the wfs global throughput

        :param o: (float) : wfs global throughput
        """
        self.__optthroughput = csu.enforce_float(o)

    optthroughput = property(lambda x: x.__optthroughput, set_optthroughput)

    def set_fracsub(self, f):
        """ Set the minimal illumination fraction for valid subaps

        :param f: (float) : minimal illumination fraction for valid subaps
        """
        self.__fracsub = csu.enforce_float(f)

    fracsub = property(lambda x: x.__fracsub, set_fracsub)

    def set_openloop(self, o):
        """ Set the loop state (open or closed)

        :param o: (long) : 1 if in "open-loop" mode (i.e. does not see dm)
        """
        self.__openloop = csu.enforce_or_cast_bool(o)

    openloop = property(lambda x: x.__openloop, set_openloop)

    def set_fssize(self, f):
        """ Set the size of field stop

        :param f: (float) : size of field stop in arcsec
        """
        self.__fssize = csu.enforce_float(f)

    fssize = property(lambda x: x.__fssize, set_fssize)

    def set_fstop(self, f):
        """ Set the size of field stop

        :param f: (str) : size of field stop in arcsec
        """
        self.__fstop = scons.check_enum(scons.FieldStopType, f)

    fstop = property(lambda x: x.__fstop, set_fstop)

    def set_atmos_seen(self, i):
        """ Tells if the wfs sees the atmosphere layers

        :param i: (bool) :True if the WFS sees the atmosphere layers
        """
        self.__atmos_seen = csu.enforce_or_cast_bool(i)

    atmos_seen = property(lambda x: x.__atmos_seen, set_atmos_seen)

    def set_xpos(self, x):
        """ Set the guide star x position on sky

        :param x: (float) : guide star x position on sky (in arcsec)
        """
        self.__xpos = csu.enforce_float(x)

    xpos = property(lambda x: x.__xpos, set_xpos)

    def set_ypos(self, y):
        """ Set the guide star y position on sky

        :param y: (float) : guide star y position on sky (in arcsec)
        """
        self.__ypos = csu.enforce_float(y)

    ypos = property(lambda x: x.__ypos, set_ypos)

    def set_G(self, G):
        """ Set the magnifying factor

        :param G: (float) : magnifying factor
        """
        self.__G = csu.enforce_float(G)

    G = property(lambda x: x.__G, set_G)

    def set_thetaML(self, thetaML):
        """ Set the rotation angle in the pupil

        :param thetaML: (float) : rotation angle (rad)
        """
        self.__thetaML = csu.enforce_float(thetaML)

    thetaML = property(lambda x: x.__thetaML, set_thetaML)

    def set_dx(self, dx):
        """ Set the X axis misalignment

        :param dx: (float) : dx (pix)
        """
        self.__dx = csu.enforce_float(dx)

    dx = property(lambda x: x.__dx, set_dx)

    def set_dy(self, dy):
        """ Set the Y axis misalignment

        :param dy: (float) : dy (pix)
        """
        self.__dy = csu.enforce_float(dy)

    dy = property(lambda x: x.__dy, set_dy)

    def set_gsalt(self, g):
        """ Set the altitude of guide star

        :param g: (float) : altitude of guide star (in m) 0 if ngs
        """
        self.__gsalt = csu.enforce_float(g)

    gsalt = property(lambda x: x.__gsalt, set_gsalt)

    def set_gsmag(self, g):
        """ Set the magnitude of guide star

        :param g: (float) : magnitude of guide star
        """
        self.__gsmag = csu.enforce_float(g)

    gsmag = property(lambda x: x.__gsmag, set_gsmag)

    def set_zerop(self, z):
        """ Set the detector zero point

        :param z: (float) : detector zero point
        """
        self.__zerop = csu.enforce_float(z)

    zerop = property(lambda x: x.__zerop, set_zerop)

    def set_noise(self, n):
        """ Set the desired noise

        :param n: (float) : desired noise : < 0 = no noise / 0 = photon only / > 0 photon + ron
        """
        self.__noise = csu.enforce_float(n)

    noise = property(lambda x: x.__noise, set_noise)

    def set_nphotons4imat(self, nphot):
        """ Set the desired numner of photons used for doing imat

        :param nphot: (float) : desired number of photons
        """
        self.__nphotons4imat = csu.enforce_float(nphot)

    nphotons4imat = property(lambda x: x.__nphotons4imat, set_nphotons4imat)

    def set_kernel(self, k):
        """ Set the attribute kernel

        :param k: (float) :
        """
        self.__kernel = csu.enforce_float(k)

    kernel = property(lambda x: x.__kernel, set_kernel)

    def set_laserpower(self, l):
        """ Set the laser power

        :param l: (float) : laser power in W
        """
        self.__laserpower = csu.enforce_float(l)

    laserpower = property(lambda x: x.__laserpower, set_laserpower)

    def set_lltx(self, l):
        """ Set the x position of llt

        :param l: (float) : x position (in meters) of llt
        """
        self.__lltx = csu.enforce_float(l)

    lltx = property(lambda x: x.__lltx, set_lltx)

    def set_llty(self, l):
        """ Set the y position of llt

        :param l: (float) : y position (in meters) of llt
        """
        self.__llty = csu.enforce_float(l)

    llty = property(lambda x: x.__llty, set_llty)

    def set_proftype(self, p):
        """ Set the type of sodium profile

        :param p: (str) : type of sodium profile "gauss", "exp", etc ...
        """
        self.__proftype = scons.check_enum(scons.ProfType, p)

    proftype = property(lambda x: x.__proftype, set_proftype)

    def set_beamsize(self, b):
        """ Set the laser beam fwhm on-sky

        :param b: (float) : laser beam fwhm on-sky (in arcsec)
        """
        self.__beamsize = csu.enforce_float(b)

    beamsize = property(lambda x: x.__beamsize, set_beamsize)

    def set_pyr_ampl(self, p):
        """ Set the pyramid wfs modulation amplitude radius

        :param p: (float) : pyramid wfs modulation amplitude radius (in arsec)
        """
        self.__pyr_ampl = csu.enforce_float(p)

    pyr_ampl = property(lambda x: x.__pyr_ampl, set_pyr_ampl)

    def set_pyr_npts(self, p):
        """ Set the total number of point along modulation circle

        :param p: (long) : total number of point along modulation circle
        """
        self.__pyr_npts = csu.enforce_int(p)

    pyr_npts = property(lambda x: x.__pyr_npts, set_pyr_npts)

    def set_pyr_loc(self, p):
        """ Set the location of modulation

        :param p: (str) : location of modulation, before/after the field stop.
                          valid value are "before" or "after" (default "after")
        """
        self.__pyr_loc = bytes(p.encode('UTF-8'))

    pyr_loc = property(lambda x: x.__pyr_loc, set_pyr_loc)

    def set_pyrtype(self, p):
        """ Set the type of pyramid,

        :param p: (str) : type of pyramid, either 0 for "Pyramid" or 1 for "RoofPrism"
        """
        self.__pyrtype = bytes(p.encode('UTF-8'))

    pyrtype = property(lambda x: x.__pyrtype, set_pyrtype)

    def set_pyr_cx(self, cx):
        """ Set the x position of modulation points for pyramid sensor

        :param cx: (np.ndarray[ndim=1,dtype=np.floatt32_t) : x positions
        """
        self.__pyr_cx = csu.enforce_array(cx.copy(), self.__pyr_npts, dtype=np.float32)

    _pyr_cx = property(lambda x: x.__pyr_cx, set_pyr_cx)

    def set_pyr_cy(self, cy):
        """ Set the y position of modulation points for pyramid sensor

        :param cy: (np.ndarray[ndim=1,dtype=np.floatt32_t) : y positions
        """
        self.__pyr_cy = csu.enforce_array(cy.copy(), self.__pyr_npts, dtype=np.float32)

    _pyr_cy = property(lambda x: x.__pyr_cy, set_pyr_cy)

    def set_dms_seen(self, dms_seen):
        """ Set the index of dms seen by the WFS

        :param dms_seen: (np.ndarray[ndim=1,dtype=np.int32_t) : index of dms seen by the WFS
        """
        self.__dms_seen = csu.enforce_array(dms_seen.copy(), dms_seen.size,
                                            dtype=np.int32)

    dms_seen = property(lambda x: x.__dms_seen, set_dms_seen)

    def set_lgsreturnperwatt(self, lpw):
        """ Set the return per watt factor

        :param lpw: (float) : return per watt factor (high season : 10 ph/cm2/s/W)
        """
        self.__lgsreturnperwatt = csu.enforce_float(lpw)

    lgsreturnperwatt = property(lambda x: x.__lgsreturnperwatt, set_lgsreturnperwatt)

    def set_altna(self, a):
        """ Set the corresponding altitude

        :param a: (np.ndarray[ndim=1,dtype=np.float32]) : corresponding altitude
        """
        self.__altna = csu.enforce_array(a.copy(), a.size, dtype=np.float32)

    _altna = property(lambda x: x.__altna, set_altna)

    def set_profna(self, p):
        """ Set the sodium profile

        :param p: (np.ndarray[ndim=1,dtype=np.float32]) : sodium profile
        """
        self.__profna = csu.enforce_array(p.copy(), p.size, dtype=np.float32)

    _profna = property(lambda x: x.__profna, set_profna)

    def set_error_budget(self, error_budget):
        """ Set the error budget flag : if True, enable error budget analysis
        for this simulation

        :param error_budget: (bool) : error budget flag
        """
        self.__error_budget = csu.enforce_or_cast_bool(error_budget)

    error_budget = property(lambda x: x.__error_budget, set_error_budget)

    def set_pyr_pup_sep(self, pyr_pup_sep):
        """ Set the pyramid pupil separation. (default: long(wfs.nxsub))

        :param pyr_pup_sep: (long) : pyramid pupil separation wanted
        """
        self.__pyr_pup_sep = csu.enforce_int(pyr_pup_sep)

    pyr_pup_sep = property(lambda x: x.__pyr_pup_sep, set_pyr_pup_sep)

    def set_pyr_misalignments(self, misalignments: np.ndarray) -> None:

        self.__pyr_misalignments = csu.enforce_arrayMultiDim(misalignments.copy(),
                                                             (4, 2), dtype=np.float32)

    pyr_misalignments = property(lambda x: x.__pyr_misalignments, set_pyr_misalignments)

    def set_nvalid(self, n):
        """ Set the number of valid subapertures

        :param n: (long) : number of valid subapertures
        """
        self.__nvalid = csu.enforce_int(n)

    _nvalid = property(lambda x: x.__nvalid, set_nvalid)

    def set_validsubsx(self, vx):
        """ Set the valid subapertures along X-axis

        :param vx: (np.array(dim=1, dtype=np.int32)) : validsubsx
        """
        if self.__type == scons.WFSType.PYRHR:
            self.__validsubsx = csu.enforce_array(vx, 4 * self.__nvalid, dtype=np.int32)
        else:
            self.__validsubsx = csu.enforce_array(vx, self.__nvalid, dtype=np.int32)

    _validsubsx = property(lambda x: x.__validsubsx, set_validsubsx)

    def set_validsubsy(self, vy):
        """ Set the valid subapertures along Y-axis

        :param vy: (np.array(dim=1, dtype=np.int32)) : validsubsy
        """
        if self.__type == scons.WFSType.PYRHR:
            self.__validsubsy = csu.enforce_array(vy, 4 * self.__nvalid, dtype=np.int32)
        else:
            self.__validsubsy = csu.enforce_array(vy, self.__nvalid, dtype=np.int32)

    _validsubsy = property(lambda x: x.__validsubsy, set_validsubsy)

    def get_validsub(self):
        """ Return both validsubsx and validsubsy

        :return: (tuple) : (self._validsubsx, self._validsubsy)
        """
        return self._validsubsx, self._validsubsy

    def set_Nfft(self, n):
        """ Set the size of FFT support for a subap

        :param n: (long) : size of FFT support
        """
        self.__Nfft = csu.enforce_int(n)

    _Nfft = property(lambda x: x.__Nfft, set_Nfft)

    def set_Ntot(self, n):
        """ Set the size of hr image for a subap

        :param n: (long) : size of hr image for a subap
        """
        self.__Ntot = csu.enforce_int(n)

    _Ntot = property(lambda x: x.__Ntot, set_Ntot)

    def set_nrebin(self, n):
        """ Set the rebin factor from hr to binned image for a subap

        :param n: (long) : rebin factor
        """
        self.__nrebin = csu.enforce_int(n)

    _nrebin = property(lambda x: x.__nrebin, set_nrebin)

    def set_pdiam(self, n):
        """ Set the subap diameter in pixels

        :param n: (long) : subap diam in pixels
        """
        self.__pdiam = csu.enforce_int(n)

    _pdiam = property(lambda x: x.__pdiam, set_pdiam)

    def set_nphotons(self, n):
        """ Set number of photons per subap

        :param n: (float) : number of photons per subap
        """
        self.__nphotons = csu.enforce_float(n)

    _nphotons = property(lambda x: x.__nphotons, set_nphotons)

    def set_qpixsize(self, n):
        """ Set the quantum pixel size for the simulation

        :param n: (float) : quantum pixel size
        """
        self.__qpixsize = csu.enforce_float(n)

    _qpixsize = property(lambda x: x.__qpixsize, set_qpixsize)

    def set_subapd(self, n):
        """ Set the subap diameter (m)

        :param n: (float) : subap diameter (m)
        """
        self.__subapd = csu.enforce_float(n)

    _subapd = property(lambda x: x.__subapd, set_subapd)

    def set_fluxPerSub(self, data):
        """ Set the subap diameter (m)

        :param data: (np.array(ndim=2, dtype=np.float32)) : subap diameter (m)
        """
        if self.__type == scons.WFSType.PYRHR:
            self.__fluxPerSub = csu.enforce_arrayMultiDim(data.copy(),
                                                          (self.__nxsub + 2,
                                                           self.__nxsub + 2),
                                                          dtype=np.float32)
        else:
            self.__fluxPerSub = csu.enforce_arrayMultiDim(data.copy(), (self.__nxsub,
                                                                        self.__nxsub),
                                                          dtype=np.float32)

    _fluxPerSub = property(lambda x: x.__fluxPerSub, set_fluxPerSub)

    def set_ftkernel(self, data):
        """ TODO : docstring
        """
        self.__ftkernel = csu.enforce_arrayMultiDim(data.copy(), data.shape,
                                                    dtype=np.complex64)

    _ftkernel = property(lambda x: x.__ftkernel, set_ftkernel)

    def set_sincar(self, data):
        """ TODO : docstring
        """
        self.__sincar = csu.enforce_arrayMultiDim(data.copy(), data.shape,
                                                  dtype=np.float32)

    _sincar = property(lambda x: x.__sincar, set_sincar)

    def set_halfxy(self, data):
        """ TODO : docstring
        """
        self.__halfxy = csu.enforce_arrayMultiDim(data.copy(), data.shape,
                                                  dtype=np.float32)

    _halfxy = property(lambda x: x.__halfxy, set_halfxy)

    def set_submask(self, data):
        """ TODO : docstring
        """
        self.__submask = csu.enforce_arrayMultiDim(data.copy(), data.shape,
                                                   dtype=np.float32)

    _submask = property(lambda x: x.__submask, set_submask)

    def set_lgskern(self, data):
        """ TODO : docstring
        """
        self.__lgskern = csu.enforce_arrayMultiDim(data.copy(), data.shape,
                                                   dtype=np.float32)

    _lgskern = property(lambda x: x.__lgskern, set_lgskern)

    def set_azimuth(self, data):
        """ TODO : docstring
        """
        self.__azimuth = csu.enforce_arrayMultiDim(data.copy(), data.shape,
                                                   dtype=np.float32)

    _azimuth = property(lambda x: x.__azimuth, set_azimuth)

    def set_prof1d(self, data):
        """ TODO : docstring
        """
        self.__prof1d = csu.enforce_arrayMultiDim(data.copy(), data.shape,
                                                  dtype=np.float32)

    _prof1d = property(lambda x: x.__prof1d, set_prof1d)

    def set_profcum(self, data):
        """ TODO : docstring
        """
        self.__profcum = csu.enforce_array(data.copy(), data.size, dtype=np.float32)

    _profcum = property(lambda x: x.__profcum, set_profcum)

    def set_beam(self, data):
        """ TODO : docstring
        """
        self.__beam = csu.enforce_array(data.copy(), data.size, dtype=np.float32)

    _beam = property(lambda x: x.__beam, set_beam)

    def set_ftbeam(self, data):
        """ TODO : docstring
        """
        self.__ftbeam = csu.enforce_array(data.copy(), data.size, dtype=np.complex64)

    _ftbeam = property(lambda x: x.__ftbeam, set_ftbeam)

    def set_hrmap(self, data):
        """ TODO : docstring
        """
        self.__hrmap = csu.enforce_arrayMultiDim(data.copy(), data.shape, dtype=np.int32)

    _hrmap = property(lambda x: x.__hrmap, set_hrmap)

    def set_binmap(self, data):
        """ TODO : docstring
        """
        self.__binmap = csu.enforce_arrayMultiDim(data.copy(), data.shape,
                                                  dtype=np.int32)

    _binmap = property(lambda x: x.__binmap, set_binmap)

    def set_phasemap(self, data):
        """ TODO : docstring
        """
        self.__phasemap = csu.enforce_arrayMultiDim(data.copy(), data.shape,
                                                    dtype=np.int32)

    _phasemap = property(lambda x: x.__phasemap, set_phasemap)

    def set_istart(self, data):
        """ TODO : docstring
        """
        self.__istart = csu.enforce_array(data.copy(), data.size, dtype=np.int32)

    _istart = property(lambda x: x.__istart, set_istart)

    def set_jstart(self, data):
        """ TODO : docstring
        """
        self.__jstart = csu.enforce_array(data.copy(), data.size, dtype=np.int32)

    _jstart = property(lambda x: x.__jstart, set_jstart)

    def set_isvalid(self, data):
        """ TODO : docstring
        """
        self.__isvalid = csu.enforce_arrayMultiDim(data.copy(), data.shape,
                                                   dtype=np.int32)

    _isvalid = property(lambda x: x.__isvalid, set_isvalid)

    def set_pyr_pos(self, data):
        """ TODO : docstring
        """
        self.__pyr_pos = csu.enforce_array(data.copy(), data.size, dtype=np.complex64)

    pyr_pos = property(lambda x: x.__pyr_pos, set_pyr_pos)
