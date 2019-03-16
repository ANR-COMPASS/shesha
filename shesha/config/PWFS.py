'''
Param_wfs class definition
Parameters for WFS
'''

import numpy as np
from . import config_setter_utils as csu
import shesha.constants as scons

#################################################
# P-Class (parametres) Param_wfs
#################################################


class Param_wfs:

    def __init__(self, roket=False):
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
        self.__roket = roket
        """ If True, enable error budget analysis for the simulation"""
        self.__is_low_order = False
        """If True, WFS is considered as a low order one and so will not profit from array mutualisation"""
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

        # Fakecam mode (uint16)
        self.__fakecam = False
        """ uint16 computation flag for WFS image """
        self.__maxFluxPerPix = 0
        """ Maximum number of photons allowed before pixel computation (only used if fakecam is True) """
        self.__maxPixValue = 0
        """ Maximum number of ADU photons allowed in the uint16 image (only used if fakecam is True) """
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

        self.__validpuppixx = None
        """ (int*) x start indexes for cutting phase screens"""
        self.__validpuppixy = None
        """ (int*) y start indexes for cutting phase screens"""
        # cdef np.ndarray _validsubs    # (i,j) indices of valid subaps
        self.__validsubsx = None
        """ (int*) X-indices of bottom left pixel of each valid subaps [pixels]"""
        self.__validsubsy = None
        """ (int*) Y-indices of bottom left pixel of each valid subaps [pixels]"""
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
        """ Number of Pyramid facets """
        self.__nPupils = 0
        """ Pyramid pupil separation. (default: long(wfs.nxsub))"""
        self.__pyr_misalignments = None
        """ Pyramid quadrant misalignments: by how much pupil subimages
        are off from their expected positions (in rebinned pixels)"""

        # pyramid internal kwrds
        self._pyr_offsets = None  # (float*)
        self.__pyr_cx = None  # (float*)
        self.__pyr_cy = None  # (float*)

    def get_type(self):
        """ Get the type of wfs

        :return: (str) : type of wfs ("sh" or "pyr")
        """
        return self.__type

    def set_type(self, typewfs):
        """ Set the type of wfs

        :param t: (str) : type of wfs ("sh" or "pyr")
        """
        self.__type = typewfs  #scons.check_enum(scons.WFSType, type)0

    type = property(get_type, set_type)

    def get_nxsub(self):
        """ Get the linear number of subaps

        :return: (long) : linear number of subaps
        """
        return self.__nxsub

    def set_nxsub(self, n):
        """ Set the linear number of subaps

        :param n: (long) : linear number of subaps
        """
        self.__nxsub = csu.enforce_int(n)

    nxsub = property(get_nxsub, set_nxsub)

    def get_nPupils(self):
        """ Get the number of pupil images

        :return: (long) : number of pupil images
        """
        return self.__nPupils

    def set_nPupils(self, n):
        """ Set the number of pupil images

        :param n: (long) : number of pupil images
        """
        self.__nPupils = csu.enforce_int(n)

    nPupils = property(get_nPupils, set_nPupils)

    def get_npix(self):
        """ Get the number of pixels per subap

        :return: (long) : number of pixels per subap
        """
        return self.__npix

    def set_npix(self, n):
        """ Set the number of pixels per subap

        :param n: (long) : number of pixels per subap
        """
        self.__npix = csu.enforce_int(n)

    npix = property(get_npix, set_npix)

    def get_pixsize(self):
        """ Get the pixel size

        :return: (float) : pixel size (in arcsec) for a subap
        """
        return self.__pixsize

    def set_pixsize(self, p):
        """ Set the pixel size

        :param p: (float) : pixel size (in arcsec) for a subap
        """
        self.__pixsize = csu.enforce_float(p)

    pixsize = property(get_pixsize, set_pixsize)

    def get_Lambda(self):
        """ Get the observation wavelength

        :return: (float) : observation wavelength (in um) for a subap
        """
        return self.__Lambda

    def set_Lambda(self, L):
        """ Set the observation wavelength

        :param L: (float) : observation wavelength (in um) for a subap
        """
        self.__Lambda = L

    Lambda = property(get_Lambda, set_Lambda)

    def get_optthroughput(self):
        """ Get the wfs global throughput

        :return: (float) : wfs global throughput
        """
        return self.__optthroughput

    def set_optthroughput(self, o):
        """ Set the wfs global throughput

        :param o: (float) : wfs global throughput
        """
        self.__optthroughput = csu.enforce_float(o)

    optthroughput = property(get_optthroughput, set_optthroughput)

    def get_fracsub(self):
        """ Get the minimal illumination fraction for valid subaps

        :return: (float) : minimal illumination fraction for valid subaps
        """
        return self.__fracsub

    def set_fracsub(self, f):
        """ Set the minimal illumination fraction for valid subaps

        :param f: (float) : minimal illumination fraction for valid subaps
        """
        self.__fracsub = csu.enforce_float(f)

    fracsub = property(get_fracsub, set_fracsub)

    def get_openloop(self):
        """ Get the loop state (open or closed)

        :return: (long) : 1 if in "open-loop" mode (i.e. does not see dm)
        """
        return self.__openloop

    def set_openloop(self, o):
        """ Set the loop state (open or closed)

        :param o: (long) : 1 if in "open-loop" mode (i.e. does not see dm)
        """
        self.__openloop = csu.enforce_or_cast_bool(o)

    openloop = property(get_openloop, set_openloop)

    def get_fssize(self):
        """ Get the size of field stop

        :return: (float) : size of field stop in arcsec
        """
        return self.__fssize

    def set_fssize(self, f):
        """ Set the size of field stop

        :param f: (float) : size of field stop in arcsec
        """
        self.__fssize = csu.enforce_float(f)

    fssize = property(get_fssize, set_fssize)

    def get_fstop(self):
        """ Get the size of field stop

        :return: (str) : size of field stop in arcsec
        """
        return self.__fstop

    def set_fstop(self, f):
        """ Set the size of field stop

        :param f: (str) : size of field stop in arcsec
        """
        self.__fstop = scons.check_enum(scons.FieldStopType, f)

    fstop = property(get_fstop, set_fstop)

    def get_atmos_seen(self):
        """ Gells if the wfs sees the atmosphere layers

        :return: (bool) :True if the WFS sees the atmosphere layers
        """
        return self.__atmos_seen

    def set_atmos_seen(self, i):
        """ Tells if the wfs sees the atmosphere layers

        :param i: (bool) :True if the WFS sees the atmosphere layers
        """
        self.__atmos_seen = csu.enforce_or_cast_bool(i)

    atmos_seen = property(get_atmos_seen, set_atmos_seen)

    def get_xpos(self):
        """ Get the guide star x position on sky

        :return: (float) : guide star x position on sky (in arcsec)
        """
        return self.__xpos

    def set_xpos(self, x):
        """ Set the guide star x position on sky

        :param x: (float) : guide star x position on sky (in arcsec)
        """
        self.__xpos = csu.enforce_float(x)

    xpos = property(get_xpos, set_xpos)

    def get_ypos(self):
        """ Get the guide star y position on sky

        :return: (float) : guide star y position on sky (in arcsec)
        """
        return self.__ypos

    def set_ypos(self, y):
        """ Set the guide star y position on sky

        :param y: (float) : guide star y position on sky (in arcsec)
        """
        self.__ypos = csu.enforce_float(y)

    ypos = property(get_ypos, set_ypos)

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

    def get_thetaML(self):
        """ Get the rotation angle in the pupil

        :return: (float) : rotation angle (rad)
        """
        return self.__thetaML

    def set_thetaML(self, thetaML):
        """ Set the rotation angle in the pupil

        :param thetaML: (float) : rotation angle (rad)
        """
        self.__thetaML = csu.enforce_float(thetaML)

    thetaML = property(get_thetaML, set_thetaML)

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

    def get_fakecam(self):
        """ Get the fakecam flag

        :return: (bool) : fakecam flag
        """
        return self.__fakecam

    def set_fakecam(self, fakecam):
        """ Set the fakecam flag

        :return: (bool) : fakecam flag
        """
        self.__fakecam = csu.enforce_or_cast_bool(fakecam)

    fakecam = property(get_fakecam, set_fakecam)

    def get_maxFluxPerPix(self):
        """ Get the maxFluxPerPix

        :return: (int) : maxFluxPerPix
        """
        return self.__maxFluxPerPix

    def set_maxFluxPerPix(self, maxFluxPerPix):
        """ Set the maxFluxPerPix

        :return: (int) : maxFluxPerPix
        """
        self.__maxFluxPerPix = csu.enforce_int(maxFluxPerPix)

    maxFluxPerPix = property(get_maxFluxPerPix, set_maxFluxPerPix)

    def get_maxPixValue(self):
        """ Get the maxPixValue

        :return: (int) : maxPixValue
        """
        return self.__maxPixValue

    def set_maxPixValue(self, maxPixValue):
        """ Set the maxPixValue

        :return: (int) : maxPixValue
        """
        self.__maxPixValue = csu.enforce_int(maxPixValue)

    maxPixValue = property(get_maxPixValue, set_maxPixValue)

    def get_gsalt(self):
        """ Get the altitude of guide star

        :return: (float) : altitude of guide star (in m) 0 if ngs
        """
        return self.__gsalt

    def set_gsalt(self, g):
        """ Set the altitude of guide star

        :param g: (float) : altitude of guide star (in m) 0 if ngs
        """
        self.__gsalt = csu.enforce_float(g)

    gsalt = property(get_gsalt, set_gsalt)

    def get_gsmag(self):
        """ Get the magnitude of guide star

        :return: (float) : magnitude of guide star
        """
        return self.__gsmag

    def set_gsmag(self, g):
        """ Set the magnitude of guide star

        :param g: (float) : magnitude of guide star
        """
        self.__gsmag = csu.enforce_float(g)

    gsmag = property(get_gsmag, set_gsmag)

    def get_zerop(self):
        """ Get the detector zero point

        :return: (float) : detector zero point
        """
        return self.__zerop

    def set_zerop(self, z):
        """ Set the detector zero point

        :param z: (float) : detector zero point
        """
        self.__zerop = csu.enforce_float(z)

    zerop = property(get_zerop, set_zerop)

    def get_noise(self):
        """ Get the desired noise

        :return: (float) : desired noise : < 0 = no noise / 0 = photon only / > 0 photon + ron
        """
        return self.__noise

    def set_noise(self, n):
        """ Set the desired noise

        :param n: (float) : desired noise : < 0 = no noise / 0 = photon only / > 0 photon + ron
        """
        self.__noise = csu.enforce_float(n)

    noise = property(get_noise, set_noise)

    def get_nphotons4imat(self):
        """ Get the desired numner of photons used for doing imat

        :return: (float) : desired number of photons
        """
        return self.__nphotons4imat

    def set_nphotons4imat(self, nphot):
        """ Set the desired numner of photons used for doing imat

        :param nphot: (float) : desired number of photons
        """
        self.__nphotons4imat = csu.enforce_float(nphot)

    nphotons4imat = property(get_nphotons4imat, set_nphotons4imat)

    def get_kernel(self):
        """ Get the attribute kernel

        :return: (float) :
        """
        return self.__kernel

    def set_kernel(self, k):
        """ Set the attribute kernel

        :param k: (float) :
        """
        self.__kernel = csu.enforce_float(k)

    kernel = property(get_kernel, set_kernel)

    def get_laserpower(self):
        """ Get the laser power

        :return: (float) : laser power in W
        """
        return self.__laserpower

    def set_laserpower(self, l):
        """ Set the laser power

        :param l: (float) : laser power in W
        """
        self.__laserpower = csu.enforce_float(l)

    laserpower = property(get_laserpower, set_laserpower)

    def get_lltx(self):
        """ Get the x position of llt

        :return: (float) : x position (in meters) of llt
        """
        return self.__lltx

    def set_lltx(self, l):
        """ Set the x position of llt

        :param l: (float) : x position (in meters) of llt
        """
        self.__lltx = csu.enforce_float(l)

    lltx = property(get_lltx, set_lltx)

    def get_llty(self):
        """ Get the y position of llt

        :return: (float) : y position (in meters) of llt
        """
        return self.__llty

    def set_llty(self, l):
        """ Set the y position of llt

        :param l: (float) : y position (in meters) of llt
        """
        self.__llty = csu.enforce_float(l)

    llty = property(get_llty, set_llty)

    def get_proftype(self):
        """ Get the type of sodium profile

        :return: (str) : type of sodium profile "gauss", "exp", etc ...
        """
        return self.__proftype

    def set_proftype(self, p):
        """ Set the type of sodium profile

        :param p: (str) : type of sodium profile "gauss", "exp", etc ...
        """
        self.__proftype = scons.check_enum(scons.ProfType, p)

    proftype = property(get_proftype, set_proftype)

    def get_beamsize(self):
        """ Get the laser beam fwhm on-sky

        :return: (float) : laser beam fwhm on-sky (in arcsec)
        """
        return self.__beamsize

    def set_beamsize(self, b):
        """ Set the laser beam fwhm on-sky

        :param b: (float) : laser beam fwhm on-sky (in arcsec)
        """
        self.__beamsize = csu.enforce_float(b)

    beamsize = property(get_beamsize, set_beamsize)

    def get_pyr_ampl(self):
        """ Get the pyramid wfs modulation amplitude radius

        :return: (float) : pyramid wfs modulation amplitude radius (in arsec)
        """
        return self.__pyr_ampl

    def set_pyr_ampl(self, p):
        """ Set the pyramid wfs modulation amplitude radius

        :param p: (float) : pyramid wfs modulation amplitude radius (in arsec)
        """
        self.__pyr_ampl = csu.enforce_float(p)

    pyr_ampl = property(get_pyr_ampl, set_pyr_ampl)

    def get_pyr_npts(self):
        """ Get the total number of point along modulation circle

        :return: (long) : total number of point along modulation circle
        """
        return self.__pyr_npts

    def set_pyr_npts(self, p):
        """ Set the total number of point along modulation circle

        :param p: (long) : total number of point along modulation circle
        """
        self.__pyr_npts = csu.enforce_int(p)

    pyr_npts = property(get_pyr_npts, set_pyr_npts)

    def get_pyr_loc(self):
        """ Get the location of modulation

        :return: (str) : location of modulation, before/after the field stop.
                          valid value are "before" or "after" (default "after")
        """
        return self.__pyr_loc

    def set_pyr_loc(self, p):
        """ Set the location of modulation

        :param p: (str) : location of modulation, before/after the field stop.
                          valid value are "before" or "after" (default "after")
        """
        self.__pyr_loc = bytes(p.encode('UTF-8'))

    pyr_loc = property(get_pyr_loc, set_pyr_loc)

    def get_pyrtype(self):
        """ Get the type of pyramid,

        :return: (str) : type of pyramid, either 0 for "Pyramid" or 1 for "RoofPrism"
        """
        return self.__pyrtype

    def set_pyrtype(self, p):
        """ Set the type of pyramid,

        :param p: (str) : type of pyramid, either 0 for "Pyramid" or 1 for "RoofPrism"
        """
        self.__pyrtype = bytes(p.encode('UTF-8'))

    pyrtype = property(get_pyrtype, set_pyrtype)

    def get_pyr_cx(self):
        """ Get the x position of modulation points for pyramid sensor

        :return: (np.ndarray[ndim=1,dtype=np.floatt32_t) : x positions
        """
        return self.__pyr_cx

    def set_pyr_cx(self, cx):
        """ Set the x position of modulation points for pyramid sensor

        :param cx: (np.ndarray[ndim=1,dtype=np.floatt32_t) : x positions
        """
        self.__pyr_cx = csu.enforce_array(cx.copy(), self.__pyr_npts, dtype=np.float32)

    _pyr_cx = property(get_pyr_cx, set_pyr_cx)

    def get_pyr_cy(self):
        """ Get the y position of modulation points for pyramid sensor

        :return: (np.ndarray[ndim=1,dtype=np.floatt32_t) : y positions
        """
        return self.__pyr_cy

    def set_pyr_cy(self, cy):
        """ Set the y position of modulation points for pyramid sensor

        :param cy: (np.ndarray[ndim=1,dtype=np.floatt32_t) : y positions
        """
        self.__pyr_cy = csu.enforce_array(cy.copy(), self.__pyr_npts, dtype=np.float32)

    _pyr_cy = property(get_pyr_cy, set_pyr_cy)

    def get_dms_seen(self):
        """ Get the index of dms seen by the WFS

        :return: (np.ndarray[ndim=1,dtype=np.int32_t) : index of dms seen by the WFS
        """
        return self.__dms_seen

    def set_dms_seen(self, dms_seen):
        """ Set the index of dms seen by the WFS

        :param dms_seen: (np.ndarray[ndim=1,dtype=np.int32_t) : index of dms seen by the WFS
        """
        self.__dms_seen = csu.enforce_array(dms_seen.copy(), dms_seen.size,
                                            dtype=np.int32)

    dms_seen = property(get_dms_seen, set_dms_seen)

    def get_lgsreturnperwatt(self):
        """ Get the return per watt factor

        :return: (float) : return per watt factor (high season : 10 ph/cm2/s/W)
        """
        return self.__lgsreturnperwatt

    def set_lgsreturnperwatt(self, lpw):
        """ Set the return per watt factor

        :param lpw: (float) : return per watt factor (high season : 10 ph/cm2/s/W)
        """
        self.__lgsreturnperwatt = csu.enforce_float(lpw)

    lgsreturnperwatt = property(get_lgsreturnperwatt, set_lgsreturnperwatt)

    def get_altna(self):
        """ Get the corresponding altitude

        :return: (np.ndarray[ndim=1,dtype=np.float32]) : corresponding altitude
        """
        return self.__altna

    def set_altna(self, a):
        """ Set the corresponding altitude

        :param a: (np.ndarray[ndim=1,dtype=np.float32]) : corresponding altitude
        """
        self.__altna = csu.enforce_array(a.copy(), a.size, dtype=np.float32)

    _altna = property(get_altna, set_altna)

    def get_profna(self):
        """ Get the sodium profile

        :return: (np.ndarray[ndim=1,dtype=np.float32]) : sodium profile
        """
        return self.__profna

    def set_profna(self, p):
        """ Set the sodium profile

        :param p: (np.ndarray[ndim=1,dtype=np.float32]) : sodium profile
        """
        self.__profna = csu.enforce_array(p.copy(), p.size, dtype=np.float32)

    _profna = property(get_profna, set_profna)

    def get_roket(self):
        """ Get the error budget flag : if True, enable error budget analysis
        for this simulation

        :return: (bool) : error budget flag
        """
        return self.__roket

    def set_roket(self, roket):
        """ Set the error budget flag : if True, enable error budget analysis
        for this simulation

        :param roket: (bool) : error budget flag
        """
        self.__roket = csu.enforce_or_cast_bool(roket)

    roket = property(get_roket, set_roket)

    def get_is_low_order(self):
        """ Get the low order flag : if True, WFS arrays will not be mutualised

        :return: (bool) : low order flag
        """
        return self.__is_low_order

    def set_is_low_order(self, is_low_order):
        """ Set the low order flag : if True, WFS arrays will not be mutualised

        :param is_low_order: (bool) : low order flag
        """
        self.__is_low_order = csu.enforce_or_cast_bool(is_low_order)

    is_low_order = property(get_is_low_order, set_is_low_order)

    def get_pyr_pup_sep(self):
        """ Get the pyramid pupil separation. (default: long(wfs.nxsub))

        :return: (long) : pyramid pupil separation wanted
        """
        return self.__pyr_pup_sep

    def set_pyr_pup_sep(self, pyr_pup_sep):
        """ Set the pyramid pupil separation. (default: long(wfs.nxsub))

        :param pyr_pup_sep: (long) : pyramid pupil separation wanted
        """
        self.__pyr_pup_sep = csu.enforce_int(pyr_pup_sep)

    pyr_pup_sep = property(get_pyr_pup_sep, set_pyr_pup_sep)

    def get_pyr_misalignments(self) -> None:

        return self.__pyr_misalignments

    def set_pyr_misalignments(self, misalignments: np.ndarray) -> None:

        self.__pyr_misalignments = csu.enforce_arrayMultiDim(misalignments.copy(),
                                                             (self.nPupils, 2),
                                                             dtype=np.float32)

    pyr_misalignments = property(get_pyr_misalignments, set_pyr_misalignments)

    def get_nvalid(self):
        """ Get the number of valid subapertures

        :return: (long) : number of valid subapertures
        """
        return self.__nvalid

    def set_nvalid(self, n):
        """ Set the number of valid subapertures

        :param n: (long) : number of valid subapertures
        """
        self.__nvalid = csu.enforce_int(n)

    _nvalid = property(get_nvalid, set_nvalid)

    def get_validsubsx(self):
        """ Get the valid subapertures along X-axis

        :return: (np.array(dim=1, dtype=np.int32)) : validsubsx
        """
        return self.__validsubsx

    def set_validsubsx(self, vx):
        """ Set the valid subapertures along X-axis

        :param vx: (np.array(dim=1, dtype=np.int32)) : validsubsx
        """
        if self.__type == scons.WFSType.PYRHR or self.__type == scons.WFSType.PYRLR:
            self.__validsubsx = csu.enforce_array(vx, vx.size, dtype=np.int32)
        else:
            self.__validsubsx = csu.enforce_array(vx, self.__nvalid, dtype=np.int32)

    _validsubsx = property(get_validsubsx, set_validsubsx)

    def get_validsubsy(self):
        """ Get the valid subapertures along Y-axis

        :return: (np.array(dim=1, dtype=np.int32)) : validsubsy
        """
        return self.__validsubsy

    def set_validsubsy(self, vy):
        """ Set the valid subapertures along Y-axis

        :param vy: (np.array(dim=1, dtype=np.int32)) : validsubsy
        """
        if self.__type == scons.WFSType.PYRHR or self.__type == scons.WFSType.PYRLR:
            self.__validsubsy = csu.enforce_array(vy, vy.size, dtype=np.int32)
        else:
            self.__validsubsy = csu.enforce_array(vy, self.__nvalid, dtype=np.int32)

    _validsubsy = property(get_validsubsy, set_validsubsy)

    def get_validsub(self):
        """ Return both validsubsx and validsubsy

        :return: (tuple) : (self._validsubsx, self._validsubsy)
        """
        return np.stack([self._validsubsx, self._validsubsy])

    def get_Nfft(self):
        """ Get the size of FFT support for a subap

        :return: (long) : size of FFT support
        """
        return self.__Nfft

    def set_Nfft(self, n):
        """ Set the size of FFT support for a subap

        :param n: (long) : size of FFT support
        """
        self.__Nfft = csu.enforce_int(n)

    _Nfft = property(get_Nfft, set_Nfft)

    def get_Ntot(self):
        """ Get the size of hr image for a subap

        :return: (long) : size of hr image for a subap
        """
        return self.__Ntot

    def set_Ntot(self, n):
        """ Set the size of hr image for a subap

        :param n: (long) : size of hr image for a subap
        """
        self.__Ntot = csu.enforce_int(n)

    _Ntot = property(get_Ntot, set_Ntot)

    def get_nrebin(self):
        """ Get the rebin factor from hr to binned image for a subap

        :return: (long) : rebin factor
        """
        return self.__nrebin

    def set_nrebin(self, n):
        """ Set the rebin factor from hr to binned image for a subap

        :param n: (long) : rebin factor
        """
        self.__nrebin = csu.enforce_int(n)

    _nrebin = property(get_nrebin, set_nrebin)

    def get_pdiam(self):
        """ Get the subap diameter in pixels

        :return: (long) : subap diam in pixels
        """
        return self.__pdiam

    def set_pdiam(self, n):
        """ Set the subap diameter in pixels

        :param n: (long) : subap diam in pixels
        """
        self.__pdiam = csu.enforce_int(n)

    _pdiam = property(get_pdiam, set_pdiam)

    def get_nphotons(self):
        """ Get number of photons per subap

        :return: (float) : number of photons per subap
        """
        return self.__nphotons

    def set_nphotons(self, n):
        """ Set number of photons per subap

        :param n: (float) : number of photons per subap
        """
        self.__nphotons = csu.enforce_float(n)

    _nphotons = property(get_nphotons, set_nphotons)

    def get_qpixsize(self):
        """ Get the quantum pixel size for the simulation

        :return: (float) : quantum pixel size
        """
        return self.__qpixsize

    def set_qpixsize(self, n):
        """ Set the quantum pixel size for the simulation

        :param n: (float) : quantum pixel size
        """
        self.__qpixsize = csu.enforce_float(n)

    _qpixsize = property(get_qpixsize, set_qpixsize)

    def get_subapd(self):
        """ Get the subap diameter (m)

        :return: (float) : subap diameter (m)
        """
        return self.__subapd

    def set_subapd(self, n):
        """ Set the subap diameter (m)

        :param n: (float) : subap diameter (m)
        """
        self.__subapd = csu.enforce_float(n)

    _subapd = property(get_subapd, set_subapd)

    def get_fluxPerSub(self):
        """ Get the subap diameter (m)

        :return: (np.array(ndim=2, dtype=np.float32)) : subap diameter (m)
        """
        return self.__fluxPerSub

    def set_fluxPerSub(self, data):
        """ Set the subap diameter (m)

        :param data: (np.array(ndim=2, dtype=np.float32)) : subap diameter (m)
        """
        if self.__type == scons.WFSType.PYRHR or self.__type == scons.WFSType.PYRLR:
            self.__fluxPerSub = csu.enforce_arrayMultiDim(
                    data.copy(), (self.__nxsub + 2, self.__nxsub + 2), dtype=np.float32)
        else:
            self.__fluxPerSub = csu.enforce_arrayMultiDim(data.copy(),
                                                          (self.__nxsub, self.__nxsub),
                                                          dtype=np.float32)

    _fluxPerSub = property(get_fluxPerSub, set_fluxPerSub)

    def get_ftkernel(self):
        """ TODO : docstring
        """
        return self.__ftkernel

    def set_ftkernel(self, data):
        """ TODO : docstring
        """
        self.__ftkernel = csu.enforce_arrayMultiDim(data.copy(), data.shape,
                                                    dtype=np.complex64)

    _ftkernel = property(get_ftkernel, set_ftkernel)

    def get_sincar(self):
        """ TODO : docstring
        """
        return self.__sincar

    def set_sincar(self, data):
        """ TODO : docstring
        """
        self.__sincar = csu.enforce_arrayMultiDim(data.copy(), data.shape,
                                                  dtype=np.float32)

    _sincar = property(get_sincar, set_sincar)

    def get_halfxy(self):
        """ TODO : docstring
        """
        return self.__halfxy

    def set_halfxy(self, data):
        """ TODO : docstring
        """
        self.__halfxy = csu.enforce_arrayMultiDim(data.copy(), data.shape,
                                                  dtype=np.float32)

    _halfxy = property(get_halfxy, set_halfxy)

    def get_submask(self):
        """ TODO : docstring
        """
        return self.__submask

    def set_submask(self, data):
        """ TODO : docstring
        """
        self.__submask = csu.enforce_arrayMultiDim(data.copy(), data.shape,
                                                   dtype=np.float32)

    _submask = property(get_submask, set_submask)

    def get_lgskern(self):
        """ TODO : docstring
        """
        return self.__lgskern

    def set_lgskern(self, data):
        """ TODO : docstring
        """
        self.__lgskern = csu.enforce_arrayMultiDim(data.copy(), data.shape,
                                                   dtype=np.float32)

    _lgskern = property(get_lgskern, set_lgskern)

    def get_azimuth(self):
        """ TODO : docstring
        """
        return self.__azimuth

    def set_azimuth(self, data):
        """ TODO : docstring
        """
        self.__azimuth = csu.enforce_arrayMultiDim(data.copy(), data.shape,
                                                   dtype=np.float32)

    _azimuth = property(get_azimuth, set_azimuth)

    def get_prof1d(self):
        """ TODO : docstring
        """
        return self.__prof1d

    def set_prof1d(self, data):
        """ TODO : docstring
        """
        self.__prof1d = csu.enforce_arrayMultiDim(data.copy(), data.shape,
                                                  dtype=np.float32)

    _prof1d = property(get_prof1d, set_prof1d)

    def get_profcum(self):
        """ TODO : docstring
        """
        return self.__profcum

    def set_profcum(self, data):
        """ TODO : docstring
        """
        self.__profcum = csu.enforce_array(data.copy(), data.size, dtype=np.float32)

    _profcum = property(get_profcum, set_profcum)

    def get_beam(self):
        """ TODO : docstring
        """
        return self.__beam

    def set_beam(self, data):
        """ TODO : docstring
        """
        self.__beam = csu.enforce_array(data.copy(), data.size, dtype=np.float32)

    _beam = property(get_beam, set_beam)

    def get_ftbeam(self):
        """ TODO : docstring
        """
        return self.__ftbeam

    def set_ftbeam(self, data):
        """ TODO : docstring
        """
        self.__ftbeam = csu.enforce_array(data.copy(), data.size, dtype=np.complex64)

    _ftbeam = property(get_ftbeam, set_ftbeam)

    def get_hrmap(self):
        """ TODO : docstring
        """
        return self.__hrmap

    def set_hrmap(self, data):
        """ TODO : docstring
        """
        self.__hrmap = csu.enforce_arrayMultiDim(data.copy(), data.shape, dtype=np.int32)

    _hrmap = property(get_hrmap, set_hrmap)

    def get_binmap(self):
        """ TODO : docstring
        """
        return self.__binmap

    def set_binmap(self, data):
        """ TODO : docstring
        """
        self.__binmap = csu.enforce_arrayMultiDim(data.copy(), data.shape,
                                                  dtype=np.int32)

    _binmap = property(get_binmap, set_binmap)

    def get_phasemap(self):
        """ TODO : docstring
        """
        return self.__phasemap

    def set_phasemap(self, data):
        """ TODO : docstring
        """
        self.__phasemap = csu.enforce_arrayMultiDim(data.copy(), data.shape,
                                                    dtype=np.int32)

    _phasemap = property(get_phasemap, set_phasemap)

    def get_validpuppixx(self):
        """ TODO : docstring
        """
        return self.__validpuppixx

    def set_validpuppixx(self, data):
        """ TODO : docstring
        """
        self.__validpuppixx = csu.enforce_array(data.copy(), data.size, dtype=np.int32)

    _validpuppixx = property(get_validpuppixx, set_validpuppixx)

    def get_validpuppixy(self):
        """ TODO : docstring
        """
        return self.__validpuppixy

    def set_validpuppixy(self, data):
        """ TODO : docstring
        """
        self.__validpuppixy = csu.enforce_array(data.copy(), data.size, dtype=np.int32)

    _validpuppixy = property(get_validpuppixy, set_validpuppixy)

    def get_isvalid(self):
        """ Get the valid subapertures array

        :return: (int*) array of 0/1 for valid subaps
        """
        return self.__isvalid

    def set_isvalid(self, data):
        """ Set the valid subapertures array

        :param data: (int*) array of 0/1 for valid subaps
        """
        self.__isvalid = csu.enforce_arrayMultiDim(data.copy(), data.shape,
                                                   dtype=np.int32)

    _isvalid = property(get_isvalid, set_isvalid)

    def get_pyr_pos(self):
        """ TODO : docstring
        """
        return self.__pyr_pos

    def set_pyr_pos(self, data):
        """ TODO : docstring
        """
        self.__pyr_pos = csu.enforce_array(data.copy(), data.size, dtype=np.complex64)

    pyr_pos = property(get_pyr_pos, set_pyr_pos)
