## @package   shesha.components.coronagraph.stellarCoronagraph
## @brief     Stellar Coronagraph Class definition
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
from shesha.supervisor.components.coronagraph.genericCoronagraph import GenericCoronagraph
from shesha.init.coronagraph_init import init_coronagraph, init_mft, mft_multiplication
from shesha.supervisor.components.targetCompass import TargetCompass
from sutraWrap import StellarCoronagraph
from carmaWrap import context
class StellarCoronagraphCompass(GenericCoronagraph):
    """ Class supervising stellar coronagraph component

    Attributes:
        _spupil: (np.ndarray[ndim=2, dtype=np.float32]): Telescope pupil mask

        _pupdiam : (int): Number of pixels along the pupil diameter

        _dim_image :(int): Coronagraphic image dimension

        _p_corono: (Param_corono): Coronagraph parameters

        _target: (TargetCompass): Compass Target used as input for the coronagraph

        _norm_img : (float): Normalization factor for coronagraphic image

        _norm_psf : (float): Normalization factor for PSF

        _coronagraph: (SutraCoronagraph): Sutra coronagraph instance

        _wav_vec: (np.ndarray[ndim=1, dtype=np.float32]): Vector of wavelength

        _AA_apod_to_fpm: (np.ndarray[ndim=3, dtype=np.complex64]): MFT matrix for focal plane

        _BB_apod_to_fpm: (np.ndarray[ndim=2, dtype=np.complex64]): MFT matrix for focal plane

        _norm0_apod_to_fpm: (np.ndarray[ndim=2, dtype=np.complex64]): MFT matrix for focal plane

        _AA_fpm_to_lyot: (np.ndarray[ndim=3, dtype=np.complex64]): MFT matrix for lyot plane

        _BB_fpm_to_lyot: (np.ndarray[ndim=2, dtype=np.complex64]): MFT matrix for lyot plane

        _norm0_fpm_to_lyot: (np.ndarray[ndim=2, dtype=np.complex64]): MFT matrix for lyot plane

        _AA_lyot_to_image_c: (np.ndarray[ndim=2, dtype=np.complex64]): MFT matrix for image computation (centered on pixel)

        _BB_lyot_to_image_c: (np.ndarray[ndim=2, dtype=np.complex64]): MFT matrix for psf computation (centered on pixel)
        
        _norm0_lyot_to_image_c: (np.ndarray[ndim=2, dtype=np.complex64]): MFT matrix for psf computation (centered on pixel)

        _AA_lyot_to_image: (np.ndarray[ndim=2, dtype=np.complex64]): MFT matrix for image computation

        _BB_lyot_to_image: (np.ndarray[ndim=2, dtype=np.complex64]): MFT matrix for psf computation
        
        _norm0_lyot_to_image: (np.ndarray[ndim=2, dtype=np.complex64]): MFT matrix for psf computation

        _indices_pup: (tuple): Tuple of ndarray containing X and Y indices of illuminated 
                                pixels in the pupil
    """
    def __init__(self, context: context, targetCompass: TargetCompass, p_corono: conf.Param_corono, 
                 p_geom: conf.Param_geom):
        """ Initialize a stellar coronagraph instance

        Args:
            context: (CarmaWrap.context): GPU context

            targetCompass: (TargetCompass): Compass Target used as input for the coronagraph

            p_corono: (Param_corono): Coronagraph parameters

            p_geom: (Param_geom): Compass geometry parameters
        """

        init_coronagraph(p_corono, p_geom.pupdiam)
        GenericCoronagraph.__init__(self, p_corono, p_geom, targetCompass)
        self._wav_vec = p_corono._wav_vec

        self._AA_apod_to_fpm, self._BB_apod_to_fpm, self._norm0_apod_to_fpm = init_mft(self._p_corono,
                                                                                       self._pupdiam,
                                                                                       planes='apod_to_fpm')
        self._AA_fpm_to_lyot, self._BB_fpm_to_lyot, self._norm0_fpm_to_lyot = init_mft(self._p_corono,
                                                                                       self._pupdiam,
                                                                                       planes='fpm_to_lyot')
        self._AA_lyot_to_image, self._BB_lyot_to_image, self._norm0_lyot_to_image = init_mft(self._p_corono,
                                                                                             self._pupdiam,
                                                                                             planes='lyot_to_image')
        self._AA_lyot_to_image_c, self._BB_lyot_to_image_c, self._norm0_lyot_to_image_c = init_mft(self._p_corono,
                                                                                                   self._pupdiam,
                                                                                                   planes='lyot_to_image',
                                                                                                   center_on_pixel=True)
        self._coronagraph = StellarCoronagraph(context, self._target.sources[0], 
                                               self._dim_image, self._dim_image, 
                                               self._p_corono._dim_fpm, self._p_corono._dim_fpm, 
                                               self._wav_vec, self._wav_vec.size, 
                                               self._p_corono._babinet_trick, 0)
        
        self._coronagraph.set_mft(self._AA_lyot_to_image, 
                                  self._BB_lyot_to_image, 
                                  self._norm0_lyot_to_image, scons.MftType.IMG)
        self._coronagraph.set_mft(self._AA_lyot_to_image_c, 
                                  self._BB_lyot_to_image_c, 
                                  self._norm0_lyot_to_image_c, scons.MftType.PSF)
        self._coronagraph.set_mft(self._AA_apod_to_fpm, 
                                  self._BB_apod_to_fpm, 
                                  self._norm0_apod_to_fpm, scons.MftType.FPM)
        self._coronagraph.set_mft(self._AA_fpm_to_lyot, 
                                  self._BB_fpm_to_lyot, 
                                  self._norm0_fpm_to_lyot, scons.MftType.LYOT)

        self._coronagraph.set_apodizer(self._p_corono._apodizer)
        self._coronagraph.set_lyot_stop(self._p_corono._lyot_stop)
        fpm = np.rollaxis(np.array(self._p_corono._focal_plane_mask), 0, 3)
        if self._p_corono._babinet_trick:
            fpm = 1. - fpm
        self._coronagraph.set_focal_plane_mask(fpm)
        self._compute_normalization()

    def _compute_normalization(self):
        """ Compute the normalization factor of coronagraphic images
        """
        self._target.reset_tar_phase(0)
        self._coronagraph.compute_image_normalization()
        self._norm_img = np.max(self.get_image(expo_type=scons.ExposureType.SE))
        self.compute_psf(accumulate=False)
        self._norm_psf = np.max(self.get_psf(expo_type=scons.ExposureType.SE))
