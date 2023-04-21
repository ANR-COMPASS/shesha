## @package   shesha.components.coronagraph.genericCoronagraph
## @brief     Abstracted layer for coronagraph object
## @author    COMPASS Team <https://github.com/ANR-COMPASS>
## @version   5.4.3
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
from shesha.supervisor.components.targetCompass import TargetCompass
from abc import ABC, abstractmethod
from shesha.util.coronagraph_utils import compute_contrast

class GenericCoronagraph(ABC):
    """ Generic class for Compass Coronagraph modules

    Attributes:
        _spupil: (np.ndarray[ndim=2, dtype=np.float32]): Telescope pupil mask

        _pupdiam : (int): Number of pixels along the pupil diameter

        _dim_image :(int): Coronagraphic image dimension

        _p_corono: (Param_corono): Coronagraph parameters

        _target: (TargetCompass): Compass Target used as input for the coronagraph

        _norm_img : (float): Normalization factor for coronagraphic image

        _norm_psf : (float): Normalization factor for PSF

        _coronagraph: (SutraCoronagraph): Sutra coronagraph instance
    """
    def __init__(self, p_corono: conf.Param_corono, p_geom: conf.Param_geom, targetCompass: TargetCompass):
        """ Initialize a coronagraph instance with generic attributes

        Args:
            p_corono: (Param_corono): Compass coronagraph parameters

            p_geom: (Param_geom): Compass geometry parameters 

            targetCompass: (TargetCompass): Compass Target used as input for the coronagraph
        """
        self._spupil = p_geom.get_spupil()
        self._pupdiam = self._spupil.shape[0]
        self._dim_image = p_corono._dim_image
        self._p_corono = p_corono
        self._target = targetCompass
        self._norm_img = 1
        self._norm_psf = 1

        self._coronagraph = None

    def compute_image(self, *, comp_psf: bool = True, accumulate: bool = True):
        """ Compute the SE coronagraphic image, and accumulate it in the LE image

        Args:
            comp_psf: (bool, optionnal): If True (default), also compute the PSF SE & LE
            accumulate: (bool, optional): If True (default), the computed SE image is accumulated in 
                                            long exposure
        """
        self._coronagraph.compute_image(accumulate=accumulate)
        if comp_psf:
            self.compute_psf(accumulate=accumulate)

    def compute_psf(self, *, accumulate: bool = True):
        """ Compute the SE psf, and accumulate it in the LE image

        Args:
            accumulate: (bool, optional): If True (default), the computed SE psf is accumulated in 
                                            long exposure
        """
        self._coronagraph.compute_psf(accumulate=accumulate)
        
    def get_image(self, *, expo_type:str=scons.ExposureType.LE):
        """ Return the coronagraphic image

        Args:
            expo_type: (str, optional): If "le" (default), returns the long exposure image.
                                        If "se", returns short exposure one.
        
        Return:
            img: (np.ndarra[ndim=2,dtype=np.float32]): coronagraphic image
        """
        if expo_type == scons.ExposureType.LE:
            img = np.array(self._coronagraph.d_image_le)
            if(self._coronagraph.cntImg):
                img /= self._coronagraph.cntImg
        if expo_type == scons.ExposureType.SE:
            img = np.array(self._coronagraph.d_image_se)
        return img / self._norm_img

    def get_psf(self, *, expo_type:str=scons.ExposureType.LE):
        """ Return the psf

        Args:
            expo_type: (str, optional): If "le" (default), returns the long exposure psf.
                                        If "se", returns short exposure one.
        
        Return:
            img: (np.ndarra[ndim=2,dtype=np.float32]): psf
        """
        if expo_type == scons.ExposureType.LE:
            img = np.array(self._coronagraph.d_psf_le) 
            if(self._coronagraph.cntPsf):
                img /= self._coronagraph.cntPsf
        if expo_type == scons.ExposureType.SE:
            img = np.array(self._coronagraph.d_psf_se)
        return img / self._norm_psf

    def reset(self):
        """ Reset long exposure image and PSF
        """
        self._coronagraph.reset()

    def get_contrast(self, *, expo_type=scons.ExposureType.LE, d_min=None, d_max=None, width=None, normalized_by_psf=True):
        """ Computes average, standard deviation, minimum and maximum of coronagraphic
        image intensity, over rings at several angular distances from the optical axis.

        A ring includes the pixels between the following angular distances :
        d_min + k * width - width / 2 and d_min + k * width + width / 2 (in lambda/D units)
        with k = 0, 1, 2... until d_min + k * width > d_max (excluded).

        Args:
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
        image_sampling = self._p_corono._image_sampling
        if width == None:
            width = image_sampling
        else:
            width = width * image_sampling
        if d_min == None:
            d_min = width
        else:
            d_min = d_min * image_sampling
        if d_max == None:
            d_max = self._dim_image / 2 - width / 2
        else:
            d_max = d_max * image_sampling

        center = self._dim_image / 2 - (1 / 2)
        image = self.get_image(expo_type=expo_type)
        if normalized_by_psf and np.max(self.get_psf(expo_type=expo_type)):
            image = image / np.max(self.get_psf(expo_type=expo_type))

        distances, mean, std, mini, maxi = compute_contrast(image, center, d_min, d_max, width)
        angular_distances = distances / image_sampling
        return angular_distances, mean, std, mini, maxi

    def set_electric_field_amplitude(self, amplitude:np.ndarray):
        """ Set the amplitude of the electric field

        Args:
            amplitude: (np.ndarray[ndim=3, dtype=np.float32]): amplitude for each wavelength
        """
        self._coronagraph.set_amplitude(amplitude)
