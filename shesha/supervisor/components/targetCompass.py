## @package   shesha.supervisor
## @brief     User layer for initialization and execution of a COMPASS simulation
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
from shesha.init.target_init import target_init
from shesha.supervisor.components.sourceCompass import SourceCompass
import numpy as np

class TargetCompass(SourceCompass):
    """ Target handler for compass simulation

    Attributes:
        sources : (List) : List of SutraSource instances used for raytracing

        _target : (sutraWrap.Target) : Sutra target instance

        _context : (carmaContext) : CarmaContext instance

        _config : (config module) : Parameters configuration structure module
    """
    def __init__(self, context, config, tel):
        """ Initialize a TargetCompass component for target related supervision

        Args:
            context : (carmaContext) : CarmaContext instance

            config : (config module) : Parameters configuration structure module

            tel : (TelescopeCompass) : A TelescopeCompass instance
        """
        self._context = context
        self._config = config # Parameters configuration coming from supervisor init
        print("->target init")
        self._target = target_init(self._context, tel._tel, self._config.p_targets,
                                   self._config.p_atmos, self._config.p_tel,
                                   self._config.p_geom, self._config.p_dms, brahma=False)
        self.sources = self._target.d_targets

    def get_tar_image(self, tar_index : int, *, expo_type: str = "se") -> np.ndarray:
        """ Get the PSF in the direction of the given target

        Args:
            tar_index : (int) : Index of target

        Kwargs:
            expo_type : (str) : "se" for short exposure (default)
                                          "le" for long exposure

        Returns:
            psf : (np.ndarray) : PSF
        """
        if (expo_type == "se"):
            return np.fft.fftshift(
                    np.array(self._target.d_targets[tar_index].d_image_se))
        elif (expo_type == "le"):
            nb = self._target.d_targets[tar_index].strehl_counter
            if nb == 0: nb = 1
            return np.fft.fftshift(np.array(self._target.d_targets[tar_index].d_image_le)) / nb
        else:
            raise ValueError("Unknown exposure type")

    def set_tar_phase(self, tar_index : int, phase : np.ndarray) -> None:
        """ Set the phase screen seen by the tar

        Args:
            tar_index : (int) : target index

            phase : (np.ndarray) : phase screen to set
        """
        self._target.d_targets[tar_index].set_phase(phase)

    def get_tar_phase(self, tar_index: int, *, pupil: bool = False) -> np.ndarray:
        """ Returns the target phase screen of target number tar_index

        Args:
            tar_index : (int) : Target index

        Kwargs:
            pupil : (bool) : If True, applies the pupil on top of the phase screen
                                       Default is False

        Returns:
            tar_phase : (np.ndarray) : Target phase screen
        """
        tar_phase = np.array(self._target.d_targets[tar_index].d_phase)
        if pupil:
            pup = self._config.p_geom._spupil
            tar_phase *= pup
        return tar_phase

    def reset_strehl(self, tar_index: int) -> None:
        """ Reset the Strehl Ratio of the target tar_index

        Args:
            tar_index : (int) : Target index
        """
        self._target.d_targets[tar_index].reset_strehlmeter()

    def reset_tar_phase(self, tar_index: int) -> None:
        """ Reset the phase screen of the target tar_index

        Args:
            tar_index : (int) : Target index
        """
        self._target.d_targets[tar_index].d_phase.reset()

    def get_strehl(self, tar_index: int, *, do_fit: bool = True) -> np.ndarray:
        """ Return the Strehl Ratio of target number tar_index.
        This fuction will return an array of 4 values as
        [SR SE, SR LE, phase variance SE [µm²], phase variance LE [µm²]]

        Args:
            tar_index : (int) : Target index

        Kwargs:
            do_fit : (bool) : If True (default), fit the PSF
                                        with a sinc before computing SR

        Returns:
            strehl : (np.ndarray) : Strehl ratios and phase variances
        """
        src = self._target.d_targets[tar_index]
        src.comp_strehl(do_fit)
        avg_var = 0
        if (src.phase_var_count > 0):
            avg_var = src.phase_var_avg / src.phase_var_count
        return [src.strehl_se, src.strehl_le, src.phase_var, avg_var]

    def get_ncpa_tar(self, tar_index : int) -> np.ndarray:
        """ Return the current NCPA phase screen of the target path

        Args:
            tar_index : (int) : Index of the target

        Returns:
            ncpa : (np.ndarray) : NCPA phase screen
        """
        return np.array(self._target.d_targets[tar_index].d_ncpa_phase)

    def set_ncpa_tar(self, tar_index: int, ncpa: np.ndarray) -> None:
        """ Set the additional fixed NCPA phase in the target path.
        ncpa must be of the same size of the spupil support

        Args:
            tar_index : (int) : WFS index

            ncpa : (ndarray) : NCPA phase screen to set [µm]
        """
        self._target.d_targets[tar_index].set_ncpa(ncpa)

    def comp_tar_image(self, tarNum: int, *, puponly: int = 0, compLE: bool = True) -> None:
        """ Computes the PSF

        Args:
            tarNum: (int): target index

        Kwargs:
            puponly: (int) : if set to 1, computes Airy (default=0)

            compLE: (bool) : if True, the computed image is taken into account in long exposure image (default=True)
        """
        self._target.d_targets[tarNum].comp_image(puponly, compLE)

    def comp_strehl(self, tarNum: int, *, do_fit: bool = True) -> None:
        """ Computes the Strehl ratio

        Args:
            tarNum: (int): target index

        Kwargs:
            do_fit: (bool): Flag for enabling fitting by sinc
        """
        self._target.d_targets[tarNum].comp_strehl(do_fit)