## @package   shesha.supervisor
## @brief     User layer for initialization and execution of a COMPASS simulation
## @author    COMPASS Team <https://github.com/ANR-COMPASS>
## @version   5.4.1
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
from shesha.init.geom_init import tel_init
import numpy as np


class TelescopeCompass(object):
    """ Telescope handler for compass simulation

    Attributes:
        _tel : (sutraWrap.Tel) : Sutra telescope instance

        _context : (carmaContext) : CarmaContext instance

        _config : (config module) : Parameters configuration structure module
    """

    def __init__(self, context, config):
        """ Initialize an AtmosCompass component for atmosphere related supervision

        Args:
            context : (carmaContext) : CarmaContext instance

            config : (config module) : Parameters configuration structure module
        """
        self._context = context
        self._config = config  # Parameters configuration coming from supervisor init
        if self._config.p_atmos is not None:
            r0 = self._config.p_atmos.r0
        else:
            raise ValueError('A r0 value through a Param_atmos is required.')

        if self._config.p_loop is not None:
            ittime = self._config.p_loop.ittime
        else:
            raise ValueError(
                    'An ittime (iteration time in seconds) value through a Param_loop is required.'
            )
        print("->telescope init")
        self._tel = tel_init(self._context, self._config.p_geom, self._config.p_tel, r0,
                             ittime, self._config.p_wfss)

    def set_input_phase(self, phase : np.ndarray):
        """ Set a circular buffer of phase screens to be raytraced as a 
        Telescope layer. Buffer size shall be (mpup size, mpup size, N).

        Args:
            phase : (np.ndarray[ndim=3, dtype=float]) : circular buffer of phase screen
        """
        if (phase.ndim != 3 or phase.shape[0] != self._config.p_geom._mpupil.shape[0] or phase.shape[1] != self._config.p_geom._mpupil.shape[1]):
            print("Input shall be a np.ndarray of dimensions (mpup size, mpup size, N)")
            return
        self._tel.set_input_phase(phase)
    
    def update_input_phase(self):
        """ Update the index of the current phase screen in the circular buffer, so it passes to the next one
        """
        self._tel.update_input_phase()

    def reset_input_phase(self):
        """ Reset circular buffer d_input_phase
        """
        self._tel.reset_input_phase()
        
    def get_input_phase(self):
        """ Return the circular buffer of telescope phase screens

        Returns:
            phase : (np.ndarray[ndim=3, dtype=float]) : circular buffer of phase screen
        """
        if (self._tel.d_input_phase is None):
            return None
        return np.array(self._tel.d_input_phase)
    
    def get_input_phase_counter(self):
        """ Return the index of the current phase screen to be raytraced inside the telescope circular buffer

        Returns:
            counter : (int) : Phase screen index in the circular buffer
        """
        return self._tel.input_phase_counter
