## @package   shesha.components.coronagraph.coronagraphCompass
## @brief     User layer for compass coronagraph object
## @author    COMPASS Team <https://github.com/ANR-COMPASS>
## @version   5.4.4
## @date      2023/03/02
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
import shesha.config as conf
import shesha.constants as scons
from carmaWrap import context
from shesha.supervisor.components.targetCompass import TargetCompass
from abc import ABC, abstractmethod
from shesha.util.coronagraph_utils import compute_contrast

class CoronagraphCompass():
    """ Class for Compass Coronagraph modules

    Attributes:
        _spupil: (np.ndarray[ndim=2, dtype=np.float32]): Telescope pupil mask

        _pupdiam : (int): Number of pixels along the pupil diameter

        _dim_image :(int): Coronagraphic image dimension

        _p_corono: (Param_corono): Coronagraph parameters

        _target: (TargetCompass): Compass Target used as input for the coronagraph

        _norm_img : (float): Normalization factor for coronagraphic image

        _norm_psf : (float): Normalization factor for PSF

        _coronos: (Coronagraph): Coronagraph instances
    """
    def __init__(self):
        """ Initialize a coronagraph instance with generic attributes

        Args:
            p_corono: (Param_corono): Compass coronagraph parameters

            p_geom: (Param_geom): Compass geometry parameters 

            targetCompass: (TargetCompass): Compass Target used as input for the coronagraph

            coroType: (CoronoType): type of coronagraph
        """

        self._coronos = []

    def add_corono(self, context:context, p_corono: conf.Param_corono, p_geom: conf.Param_geom, 
                   targetCompass: TargetCompass):
        """ Add a coronagraph

        Args:
            p_corono: (Param_corono): Compass coronagraph parameters

            p_geom: (Param_geom): Compass geometry parameters 

            targetCompass: (TargetCompass): Compass Target used as input for the coronagraph  
        """
        if(p_corono._type == scons.CoronoType.CUSTOM) or (p_corono._type == scons.CoronoType.SPHERE_APLC):
            from shesha.supervisor.components.coronagraph.stellarCoronagraph import StellarCoronagraphCompass
            self._coronos.append(StellarCoronagraphCompass(context, targetCompass, p_corono, p_geom))

        elif(p_corono._type == scons.CoronoType.PERFECT):
            from shesha.supervisor.components.coronagraph.perfectCoronagraph import PerfectCoronagraphCompass
            self._coronos.append(PerfectCoronagraphCompass(context, targetCompass, p_corono, p_geom))

    def compute_image(self, coro_index: int, *, comp_psf: bool = True, accumulate: bool = True):
        """ Compute the SE coronagraphic image, and accumulate it in the LE image

        Args:
            coro_index: (int): Index of the coronagraph

            comp_psf: (bool, optionnal): If True (default), also compute the PSF SE & LE

            accumulate: (bool, optional): If True (default), the computed SE image is accumulated in 
                                            long exposure
        """
        self._coronos[coro_index].compute_image(comp_psf=comp_psf, accumulate=accumulate)

    def compute_psf(self, coro_index: int, *, accumulate: bool = True):
        """ Compute the SE psf, and accumulate it in the LE image

        Args:
            coro_index: (int): Index of the coronagraph

            accumulate: (bool, optional): If True (default), the computed SE psf is accumulated in 
                                            long exposure
        """
        self._coronos[coro_index].compute_psf(accumulate=accumulate)
        
    def get_image(self, coro_index: int, *, expo_type:str=scons.ExposureType.LE):
        """ Return the coronagraphic image

        Args:
            coro_index: (int): Index of the coronagraph

            expo_type: (str, optional): If "le" (default), returns the long exposure image.
                                        If "se", returns short exposure one.
        
        Return:
            img: (np.ndarra[ndim=2,dtype=np.float32]): coronagraphic image
        """
        return self._coronos[coro_index].get_image(expo_type=expo_type)

    def get_psf(self, coro_index: int, *, expo_type:str=scons.ExposureType.LE):
        """ Return the psf

        Args:
            coro_index: (int): Index of the coronagraph

            expo_type: (str, optional): If "le" (default), returns the long exposure psf.
                                        If "se", returns short exposure one.
        
        Return:
            img: (np.ndarra[ndim=2,dtype=np.float32]): psf
        """
        return self._coronos[coro_index].get_psf(expo_type=expo_type)

    def reset(self, *, coro_index: int=None):
        """ Reset long exposure image and PSF

        Args:
            coro_index: (int): Index of the coronagraph to reset. If not provided, reset all coronagraphs
        """
        if coro_index is None:
            for corono in self._coronos:
                corono.reset()
        else:
            self._coronos[coro_index].reset()

    def get_contrast(self, coro_index, *, expo_type=scons.ExposureType.LE, d_min=None, d_max=None, width=None, normalized_by_psf=True):
        """ Computes average, standard deviation, minimum and maximum of coronagraphic
        image intensity, over rings at several angular distances from the optical axis.

        A ring includes the pixels between the following angular distances :
        d_min + k * width - width / 2 and d_min + k * width + width / 2 (in lambda/D units)
        with k = 0, 1, 2... until d_min + k * width > d_max (excluded).

        Args:
            coro_index: (int): Index of the coronagraph

            expo_type: (str, optional): If "le" (default), computes contrast on the long exposure image.
                                        If "se", it uses the short exposure one.

            d_min: (float, optional): Angular radius of the first ring in lambda/D unit.
                                      Default = width

            d_max: (float, optional): Maximum angular distances in lambda/D unit.
                                      Default includes the whole image.

            width: (float, optional): Width of one ring in lambda/D unit.
                                      Default = 1 [lambda/D]

            normalized_by_psf: (bool, optional): If True (default), the coronagraphic image
                                                 is normalized by the maximum of the PSF

        Returns:
            distances: (1D array): angular distances to the optical axis in lambda/D unit

            mean: (1D array): corresponding average intensities

            std: (1D array): corresponding standard deviations

            mini: (1D array): corresponding minimums

            maxi: (1D array): corresponding maximums
        """
        return self._coronos[coro_index].get_contrast(expo_type=expo_type, d_min=d_min, d_max=d_max,
                                                       width=width, 
                                                       normalized_by_psf=normalized_by_psf)

    def set_electric_field_amplitude(self, coro_index, amplitude:np.ndarray):
        """ Set the amplitude of the electric field

        Args:
            coro_index: (int): Index of the coronagraph

            amplitude: (np.ndarray[ndim=3, dtype=np.float32]): amplitude for each wavelength
        """
        self._coronos[coro_index].set_electric_field_amplitude(amplitude)
