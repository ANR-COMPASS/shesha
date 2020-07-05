## @package   shesha.supervisor
## @brief     User layer for initialization and execution of a COMPASS simulation
## @author    COMPASS Team <https://github.com/ANR-COMPASS>
## @version   5.0.0
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
from shesha.init.wfs_init import wfs_init
import shesha.util.utilities as util
import shesha.ao.wfs as wfs_util
from shesha.supervisor.components.sourceCompass import SourceCompass
import numpy as np
from typing import List

class WfsCompass(SourceCompass):
    """ WFS handler for compass simulation

    Attributes:
        sources : (List) : List of SutraSource instances used for raytracing

        _wfs : (sutraWrap.Wfs) : SutraSensors instance

        _context : (carmaContext) : CarmaContext instance

        _config : (config module) : Parameters configuration structure module
    """
    def __init__(self, context, config, tel):
        """ Initialize a wfsCompass component for wfs related supervision

        Parameters:
            context : (carmaContext) : CarmaContext instance

            config : (config module) : Parameters configuration structure module

            tel : (TelescopeCompass) : A TelescopeCompass instance
        """
        self._context = context
        self._config = config # Parameters configuration coming from supervisor init
        print("->wfs init")
        self._wfs = wfs_init(self._context, tel._tel, self._config.p_wfss,
                                self._config.p_tel, self._config.p_geom, self._config.p_dms,
                                self._config.p_atmos)
        self.sources = [wfs.d_gs for wfs in self._wfs.d_wfs]
        
    def get_wfs_image(self, wfs_index : int) -> np.ndarray:
        """ Get an image from the WFS (wfs[0] by default), or from the centroider handling the WFS
        to get the calibrated image

        Parameters:
            wfs_index : (int) : index of the WFS (or the centroider) to request an image

        Return:
            image : (np.ndarray) : WFS image
        """
        if self._config.p_wfss[wfs_index].fakecam:
            return np.array(self._wfs.d_wfs[wfs_index].d_camimg)
        else:
            return np.array(self._wfs.d_wfs[wfs_index].d_binimg)

    def set_pyr_modulation_points(self, wfs_index : int, cx: np.ndarray, cy: np.ndarray,
                                  *, weights: np.ndarray = None) -> None:
        """ Set pyramid modulation positions

        Parameters:
            wfs_index : (int) : WFS index

            cx : (np.ndarray) : X positions of the modulation points [arcsec]

            cy : (np.ndarray) : Y positions of the modulation points [arcsec]

            weights : (np.ndarray, optional) : Weights to apply on each modulation point contribution
        """
        pyr_npts = len(cx)
        pwfs = self._config.p_wfss[wfs_index]
        pwfs.set_pyr_npts(pyr_npts)
        pwfs.set_pyr_cx(cx)
        pwfs.set_pyr_cy(cy)
        if weights is not None:
            self._wfs.d_wfs[wfs_index].set_pyr_modulation_points(cx, cy, pyr_npts)
        else:
            self._wfs.d_wfs[wfs_index].set_pyr_modulation_points(
                    cx, cy, weights, pyr_npts)

    def set_pyr_modulation_ampli(self, wfs_index: int, pyr_mod: float) -> float:
        """ Set pyramid circular modulation amplitude value - in lambda/D units.

        Compute new modulation points corresponding to the new amplitude value
        and upload them. 
        /!\ WARNING : if you are using slopes-based centroider with the PWFS,
        also update the centroider scale (rtc.set_scale) with the returned
        value

        Parameters:
            wfs_index : (int) : WFS index

            pyr_mod : (float) : new pyramid modulation amplitude value
        
        Return:
            scale : (float) : scale factor
        """
        p_wfs = self._config.p_wfss[wfs_index]

        cx, cy, scale, pyr_npts = wfs_util.comp_new_pyr_ampl(wfs_index, pyr_mod,
                                                    self._config.p_wfss,
                                                    self._config.p_tel)
        p_wfs.set_pyr_ampl(pyr_mod)
        self.set_pyr_modulation_points(wfs_index, cx, cy)

        if (len(p_wfs._halfxy.shape) == 2):
            print("PYR modulation set to: %f L/D using %d points" % (pyr_mod, pyr_npts))
        elif (len(p_wfs._halfxy.shape) == 3):
            newhalfxy = np.tile(p_wfs._halfxy[0, :, :], (pyr_npts, 1, 1))
            print("Loading new modulation arrays")
            self._wfs.d_wfs[wfs_index].set_phalfxy(
                    np.exp(1j * newhalfxy).astype(np.complex64).T)
            print("Done. PYR modulation set to: %f L/D using %d points" % (pyr_mod,
                                                                           pyr_npts))
        else:
            raise ValueError("Error unknown p_wfs._halfxy shape")

        return scale

    def set_pyr_multiple_stars_source(self, wfs_index: int, coords: List,
                                      *, weights: List = None, pyr_mod: float = 3.,
                                      niters: int = None) -> None:
        """ Sets the Pyramid modulation points with a multiple star system

        Parameters:
            wfs_index : (int) : WFS index

            coords : (list) : list of couples of length n, coordinates of the n stars in lambda/D

            weights : (list, optional) : list of weights to apply on each modulation points. Default is None

            pyr_mod : (float, optional): modulation amplitude of the pyramid in lambda/D. Default is 3

            niters : (int, optional) : number of iteration. Default is None
        """
        if niters is None:
            perim = pyr_mod * 2 * np.pi
            niters = int((perim // 4 + 1) * 4)
            print(niters)
        scale_circ = self._config.p_wfss[wfs_index]._pyr_scale_pos * pyr_mod
        temp_cx = []
        temp_cy = []
        for k in coords:
            temp_cx.append(scale_circ * \
                np.sin((np.arange(niters)) * 2. * np.pi / niters) + \
                k[0] * self._config.p_wfss[wfs_index]._pyr_scale_pos)
            temp_cy.append(scale_circ * \
                np.cos((np.arange(niters)) * 2. * np.pi / niters) + \
                k[1] * self._config.p_wfss[wfs_index]._pyr_scale_pos)
        cx = np.concatenate(np.array(temp_cx))
        cy = np.concatenate(np.array(temp_cy))
        #Gives the arguments to the simulation
        if weights is not None:
            w = []
            for k in weights:
                w += niters * [k]
            weights = np.array(w)
        self.set_pyr_modulation_points(wfs_index, cx, cy, weights=weights)

    def set_pyr_disk_source_hexa(self, wfs_index: int, radius: float) -> None:
        """ Create disk object by packing PSF in a given radius, using hexagonal packing
        and set it as modulation pattern

        /!\ There is no modulation

        Parameters:
            wfs_index  : (int) : WFS index

            radius : (float) : radius of the disk object in lambda/D
        """
        #Vectors used to generate the hexagonal paving
        gen_xp, gen_yp = np.array([1,
                                   0.]), np.array([np.cos(np.pi / 3),
                                                   np.sin(np.pi / 3)])
        n = 1 + int(1.2 * radius)
        mat_circ = []
        for k in range(-n, n):
            for l in range(-n, n):
                coord = k * gen_xp + l * gen_yp
                if np.sqrt(coord[0]**2 + coord[1]**2) <= radius:
                    mat_circ.append(coord)
        mat_circ = np.array(mat_circ)
        cx, cy = mat_circ[:, 0], mat_circ[:, 1]
        self.set_pyr_modulation_points(wfs_index, cx, cy)


    def set_pyr_disk_source(self, wfs_index: int, radius: float, *, density: float = 1.) -> None:
        """ Create disk object by packing PSF in a given radius, using square packing
        and set it as modulation pattern

        /!\ There is no modulation

        Parameters:
            wfs_index  : (int) : WFS index

            radius : (float) : radius of the disk object in lambda/D

            density : (float, optional) : Spacing between the packed PSF in the disk object, in lambda/D.
                                          Default is 1
        """
        cx, cy = util.generate_circle(radius, density)
        cx = cx.flatten() * self._config.p_wfss[wfs_index]._pyr_scale_pos
        cy = cy.flatten() * self._config.p_wfss[wfs_index]._pyr_scale_pos
        self.set_pyr_modulation_points(wfs_index, cx, cy)

    def set_pyr_square_source(self, wfs_index: int, radius: float, *, density: float = 1.) -> None:
        """ Create a square object by packing PSF in a given radius, using square packing
        and set it as modulation pattern

        /!\ There is no modulation

        Parameters:
            wfs_index  : (int) : WFS index

            radius : (float) : radius of the disk object in lambda/D

            density : (float, optional) : Spacing between the packed PSF in the disk object, in lambda/D.
                                          Default is 1
        """
        cx, cy = util.generate_square(radius, density)
        cx = cx.flatten() * self._config.p_wfss[wfs_index]._pyr_scale_pos
        cy = cy.flatten() * self._config.p_wfss[wfs_index]._pyr_scale_pos
        self.set_pyr_modulation_points(wfs_index, cx, cy)


    def set_pyr_pseudo_source(self, wfs_index: int, radius: float, *,
                              additional_psf: int = 0, density: float = 1.) -> None:
        """ TODO : DESCRIPTION

        Parameters:
            wfs_index : (int) : WFS index

            radius : (float) : TODO : DESCRIPTION

            additional_psf : (int, optional) : TODO : DESCRIPTION

            density : (float, optional) :TODO : DESCRIPTION
        """
        cx, cy, weights, _, _ = util.generate_pseudo_source(radius, additional_psf,
                                                            density)
        cx = cx.flatten() * self._config.p_wfss[wfs_index]._pyr_scale_pos
        cy = cy.flatten() * self._config.p_wfss[wfs_index]._pyr_scale_pos
        self.set_pyr_modulation_points(wfs_index, cx, cy, weights)

    def set_fourier_mask(self, wfs_index : int, new_mask: np.ndarray) -> None:
        """ Set a mask in the Fourier Plane of the given WFS

        Parameters:
            wfs_index : (int, optional) : WFS index

            new_mask : (ndarray) : mask to set
        """
        if new_mask.shape != self._config.p_wfss[wfs_index].get_halfxy().shape:
            print('Error : mask shape should be {}'.format(
                    self._config.p_wfss[wfs_index].get_halfxy().shape))
        else:
            self._wfs.d_wfs[wfs_index].set_phalfxy(
                    np.exp(1j * np.fft.fftshift(new_mask)).astype(np.complex64).T)

    def set_noise(self, wfs_index : int, noise: float, *, seed: int = 1234) -> None:
        """ Set noise value of WFS wfs_index

        Parameters:
            wfs_index : (int, optional) : WFS index

            noise : (float) : readout noise value in e-

            seed : (int, optional) : RNG seed. The seed used will be computed as seed + wfs_index
                                     Default is 1234
        """
        self._wfs.d_wfs[wfs_index].set_noise(noise, int(seed + wfs_index))
        print("Noise set to: %f on WFS %d" % (noise, wfs_index))

    def set_gs_mag(self, wfs_index : int, mag : float) -> None:
        """ Change the guide star magnitude for the given WFS

        Parameters:
            wfs_index : (int, optional) : WFS index

            mag : (float) : New magnitude of the guide star
        """
        wfs = self._wfs.d_wfs[wfs_index]
        if (self._config.p_wfs0.type == "pyrhr"):
            r = wfs.comp_nphot(self._config.p_loop.ittime,
                               self._config.p_wfss[wfs_index].optthroughput,
                               self._config.p_tel.diam, self._config.p_tel.cobs,
                               self._config.p_wfss[wfs_index].zerop, mag)
        else:
            r = wfs.comp_nphot(self._config.p_loop.ittime,
                               self._config.p_wfss[wfs_index].optthroughput,
                               self._config.p_tel.diam, self._config.p_wfss[wfs_index].nxsub,
                               self._config.p_wfss[wfs_index].zerop, mag)
        if (r == 0):
            print("GS magnitude is now %f on WFS %d" % (mag, wfs_index))

    def compute_wfs_image(self, wfs_index : int, *, noise: bool = True) -> None:
        """ Computes the image produced by the WFS from its phase screen

        Parameters :
            wfs_index : (int): WFS index

            noise : (bool, optional) : Flag to enable noise for image computation. Default is True
        """
        self._wfs.d_wfs[wfs_index].comp_image(noise)

    def reset_noise(self) -> None:
        """ Reset all the WFS RNG to their original state
        """
        for wfs_index, p_wfs in enumerate(self._config.p_wfss):
            self._wfs.d_wfs[wfs_index].set_noise(p_wfs.noise, 1234 + wfs_index)

    def get_ncpa_wfs(self, wfs_index : int) -> np.ndarray:
        """ Return the current NCPA phase screen of the WFS path

        Parameters:
            wfs_index : (int) : Index of the WFS

        Return:
            ncpa : (np.ndarray) : NCPA phase screen
        """
        return np.array(self._wfs.d_wfs[wfs_index].d_gs.d_ncpa_phase)

    def get_wfs_phase(self, wfs_index : int) -> np.ndarray:
        """ Return the WFS phase screen of WFS number wfs_index

        Parameters:
            wfs_index : (int) : Index of the WFS

        Return:
            phase : (np.ndarray) : WFS phase screen
        """
        return np.array(self._wfs.d_wfs[wfs_index].d_gs.d_phase)

    def get_pyrhr_image(self, wfs_index : int) -> np.ndarray:
        """ Get an high resolution image from the PWFS

        Parameters:
            wfs_index : (int) : Index of the WFS

        Return:
            image : (np.ndarray) : PWFS high resolution image

        """
        return np.array(self._wfs.d_wfs[wfs_index].d_hrimg)

    def set_ncpa_wfs(self, wfs_index : int, ncpa: np.ndarray) -> None:
        """ Set the additional fixed NCPA phase in the WFS path.
        ncpa must be of the same size of the mpupil support

        Parameters:
            wfs_index : (int) : WFS index

            ncpa : (ndarray) : NCPA phase screen to set [µm]
        """
        self._wfs.d_wfs[wfs_index].d_gs.set_ncpa(ncpa)

    def set_wfs_phase(self, wfs_index : int, phase : np.ndarray) -> None:
        """ Set the phase screen seen by the WFS

        Parameters:
            wfs_index : (int) : WFS index

            phase : (np.ndarray) : phase screen to set
        """
        self._wfs.d_wfs[wfs_index].d_gs.set_phase(phase)

    def set_wfs_pupil(self, wfs_index : int, pupil : np.ndarray) -> None:
        """ Set the pupil seen by the WFS
        Other pupils remain unchanged, i.e. DM and target can see an other
        pupil than the WFS after this call.
        <pupil> must have the same shape than p_geom._mpupil support

        Parameters:
            wfs_index : (int) : WFS index

            pupil : (np.ndarray) : new pupil to set
        """
        old_mpup = self._config.p_geom._mpupil
        dimx = old_mpup.shape[0]
        dimy = old_mpup.shape[1]
        if ((pupil.shape[0] != dimx) or (pupil.shape[1] != dimy)):
            print("Error pupil shape on wfs %d must be: (%d,%d)" % (wfs_index, dimx,
                                                                     dimy))
        else:
            self._wfs.d_wfs[wfs_index].set_pupil(pupil.copy())

    def get_pyr_focal_plane(self, wfs_index : int) -> np.ndarray:
        """ Returns the psf on the top of the pyramid.
        pyrhr WFS only

        Parameters:
            wfs_index : (int) : WFS index

        Return:
            focal_plane : (np.ndarray) : psf on the top of the pyramid
        """
        return np.fft.fftshift(np.array(self._wfs.d_wfs[wfs_index].d_pyrfocalplane))
