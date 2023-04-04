## @package   shesha.util.psfMap
## @brief     class PSF_map
## @author    COMPASS Team <https://github.com/ANR-COMPASS>
## @version   5.4.2
## @date      2022/01/24
## @copyright GNU Lesser General Public License
#
#  This file is part of COMPASS <https://anr-compass.github.io/compass/>
#
#  Copyright (C) 2011-2023 COMPASS Team <https://github.com/ANR-COMPASS>
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
import astropy.io.fits as fits
import matplotlib.pyplot as plt


class PSF_map:

    def __init__(self, SR=np.zeros((0, 0)), radius=0, wl=0, NGS=None, LGS=None, TS=None,
                 AOtype="AO", sup=None, filename=None):
        """ SR_map class

        SR      : np.ndarray        : array of strehl (organised by position)
        radius  : float             : radius of the map (in arcsec)
        wl      : float             : wavelength of observation (in nm)
        NGS     : np.ndarray        : position (arcsec) of the NGS (1st line:x axis, 2nd:y axis)
        LGS     : np.ndarray        : position (arcsec) of the LGS (1st line:x axis, 2nd:y axis)
        TS      : np.ndarray        : position (arcsec) of the TS (1st line:x axis, 2nd:y axis)
        sup     : shesha.supervisor : shesha supervisor to retrieve all data from
        filename: str               : filename to read the psf map from

        warning:
        using 'sup' will take only the supervisor into consideration
        using 'filename' will overide previous values of the psf map (unless called with sup)

        """
        self.map = SR

        self._Rtar = radius
        self._Rngs = 0
        self._Rlgs = 0
        self._Rlgs = 0
        if (NGS is not None):
            self.NGSx = NGS[0]
            self.NGSy = NGS[1]
            self._Rngs = max((self.NGSx.max(), self.NGSy.max()))
        else:
            self.NGSx = np.zeros((0))
            self.NGSy = np.zeros((0))

        if (LGS is not None):
            self.LGSx = LGS[0]
            self.LGSy = LGS[1]
            self._Rlgs = max(self.LGSx.max(), self.LGSy.max())
        else:
            self.LGSx = np.zeros((0))
            self.LGSy = np.zeros((0))

        if (TS is not None):
            self.TSx = TS[0]
            self.TSy = TS[1]
            self._Rts = max(self.TSx.max(), self.TSy.max())
        else:
            self.TSx = np.zeros((0))
            self.TSy = np.zeros((0))

        self.wavelength = wl
        self.type = AOtype

        if (sup is not None):
            self.NGSx = np.array([t.xpos for t in sup.config.p_wfs_ngs[:-1]])
            self.NGSy = np.array([t.ypos for t in sup.config.p_wfs_ngs[:-1]])
            self._Rngs = max((self.NGSx.max(), self.NGSy.max()))
            self.LGSx = np.array([t.xpos for t in sup.config.p_wfs_lgs])
            self.LGSy = np.array([t.ypos for t in sup.config.p_wfs_lgs])
            self._Rlgs = max(self.LGSx.max(), self.LGSy.max())
            self.TSx = np.array([t.xpos for t in sup.config.p_wfs_ts])
            self.TSy = np.array([t.ypos for t in sup.config.p_wfs_ts])
            self._Rts = max(self.TSx.max(), self.TSy.max())
            self.wavelength = sup.config.p_targets[0].Lambda
            NTAR = len(sup.config.p_targets)
            NTAR_side = int(np.sqrt(NTAR))
            if (NTAR != NTAR_side**2):
                raise ValueError("not a square nb of targets")
            self.map = np.zeros((NTAR_side, NTAR_side))
            for i in range(NTAR):
                #self.map.itemset(i,sup._sim.get_strehl(i)[1])
                self.map.itemset(i, sup._sim.get_strehl(i)[0])
                tar = sup._sim.tar.d_targets[i]
                self._Rtar = max(self._Rtar, tar.posx, tar.posy)

        elif (filename is not None):
            self.read(filename)

    def setNGS(self, NGS, NGSy=None):
        if (NGSy is None):
            self.NGSx = NGS[0]
            self.NGSy = NGS[1]
        else:
            self.NGS = NGS
            self.NGSy = NGS
        self._Rngs = max((self.NGSx.max(), self.NGSy.max()))

    def setLGS(self, LGS, LGSy=None):
        if (LGSy is None):
            self.LGSx = LGS[0]
            self.LGSy = LGS[1]
        else:
            self.LGS = LGS
            self.LGSy = LGS
        self._Rlgs = max(self.LGSx.max(), self.LGSy.max())

    def setTS(self, TS, TSy=None):
        if (TSy is None):
            self.TSx = TS[0]
            self.TSy = TS[1]
        else:
            self.TSx = TS
            self.TSy = TSy
        self._Rts = max(self.TSx.max(), self.TSy.max())

    def setWaveLength(self, wl):
        self.wavelength = wl

    def plot(self, title=False, GS=False, WFS=False, LGS=False, NGS=False, TS=False):
        if (self.map.shape[0] > 0 and self.map.shape[1] > 0):
            plt.matshow(self.map,
                        extent=[-self._Rtar, self._Rtar, -self._Rtar, self._Rtar])
            plt.colorbar()
            if (GS or WFS or LGS):
                plt.scatter(self.LGSy, self.LGSx, color="red")
            if (GS or WFS or NGS):
                plt.scatter(self.NGSy, self.NGSx, color="blue")
            if (GS or TS):
                plt.scatter(self.TSy, self.TSx, color="yellow", s=1)

            t = self.type + " Strehl"
            if (self.wavelength > 0):
                t += " @ {:.3f} nm".format(self.wavelength)

            if (self._Rts > 0):
                t += ", {:.2f}".format(self._Rts) + " arcsec ring optim"
            if (title):
                plt.title(t)

    def save(self, name=""):
        hdu_map = fits.PrimaryHDU(self.map)
        hdu_map.header["TYPE"] = self.type
        hdu_map.header["LAMBDA"] = self.wavelength
        hdu_map.header["RTAR"] = self._Rtar
        hdu_map.header["RLGS"] = self._Rlgs
        hdu_map.header["RNGS"] = self._Rngs
        hdu_map.header["RTS"] = self._Rts
        hdu_LGSX = fits.ImageHDU(self.LGSx, name="LGSX")
        hdu_LGSY = fits.ImageHDU(self.LGSy, name="LGSY")
        hdu_NGSX = fits.ImageHDU(self.NGSx, name="NGSX")
        hdu_NGSY = fits.ImageHDU(self.NGSy, name="NGSY")
        hdu_TSX = fits.ImageHDU(self.TSx, name="TSX")
        hdu_TSY = fits.ImageHDU(self.TSy, name="TSY")

        hdul = fits.HDUList([
                hdu_map, hdu_LGSX, hdu_LGSY, hdu_NGSX, hdu_NGSY, hdu_TSX, hdu_TSY
        ])
        t = self.type + "_StrehlMap"
        if (self.wavelength > 0):
            t += "_{:.3f}nm".format(self.wavelength)
        if (self._Rts > 0):
            t += "_{:.2f}arcsec".format(self._Rts)
        t += ".fits"
        if (name == ""):
            name = t
        hdul.writeto(name, overwrite=1)

    def read(self, name):
        hdu_map = fits.open(name)
        self.type = hdu_map[0].header["TYPE"]
        self.wavelength = hdu_map[0].header["LAMBDA"]
        self._Rtar = hdu_map[0].header["RTAR"]
        self._Rlgs = hdu_map[0].header["RLGS"]
        self._Rngs = hdu_map[0].header["RNGS"]
        self._Rts = hdu_map[0].header["RTS"]
        self.map = hdu_map[0].data
        self.LGSx = hdu_map["LGSx"].data
        self.LGSy = hdu_map["LGSy"].data
        self.NGSx = hdu_map["NGSx"].data
        self.NGSy = hdu_map["NGSy"].data
        self.TSx = hdu_map["TSx"].data
        self.TSy = hdu_map["TSy"].data
