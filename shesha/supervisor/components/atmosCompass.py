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
from shesha.init.atmos_init import atmos_init
from shesha.constants import CONST
import numpy as np

class AtmosCompass(object):
    """ Atmosphere handler for compass simulation

    Attributes:
        is_enable : (bool) : Flag to enable/disable atmophere

        _atmos : (sutraWrap.Atmos) : Sutra atmos instance

        _context : (carmaContext) : CarmaContext instance

        _config : (config module) : Parameters configuration structure module
    """
    def __init__(self, context, config):
        """ Initialize an AtmosCompass component for atmosphere related supervision

        Args:
            context : (carmaContext) : CarmaContext instance

            config : (config module) : Parameters configuration structure module
        """
        self.is_enable = True # Flag to enable/disable atmophere
        self._context = context
        self._config = config # Parameters configuration coming from supervisor init
        print("->atmosphere init")
        self._atmos = atmos_init(self._context, self._config.p_atmos, self._config.p_tel,
                        self._config.p_geom, self._config.p_loop.ittime, p_wfss=self._config.p_wfss,
                        p_targets=self._config.p_targets)


    def enable_atmos(self, enable : bool) -> None:
        """ Set or unset whether atmos is enabled when running loop

        Args:
            enable : (bool) : True to enable, False to disable
        """
        self.is_enable = enable

    def set_r0(self, r0 : float, *, reset_seed : int=-1) -> None:
        """ Change the current r0 for all layers

        Args:
            r0 : (float) : r0 @ 0.5 µm

        Kwargs:
            reset_seed : (int): if -1 (default), keep same seed and same screen
                                if 0 random seed is applied and refresh screens
                                if (value) set the given seed and refresh screens
        """
        self._atmos.set_r0(r0)
        if reset_seed != -1:
            if reset_seed == 0:
                ilayer = np.random.randint(1e4)
            else:
                ilayer = reset_seed
            for k in range(self._atmos.nscreens):
                self._atmos.set_seed(k, self._config.p_atmos.seeds[ilayer])
                self._atmos.refresh_screen(k)
                ilayer += 1
        self._config.p_atmos.set_r0(r0)

    def set_wind(self, screen_index : int, *, windspeed : float = None, winddir : float = None) -> None:
        """ Set new wind information for the given screen

        Args:
            screen_index : (int) : Atmos screen to change

        Kwargs:
            windspeed : (float) [m/s] : new wind speed of the screen. If None, the wind speed is unchanged

            winddir : (float) [deg]: new wind direction of the screen. If None, the wind direction is unchanged
        """
        if windspeed is not None:
            self._config.p_atmos.windspeed[screen_index] = windspeed
        if winddir is not None:
            self._config.p_atmos.winddir[screen_index] = winddir

        lin_delta = self._config.p_geom.pupdiam / self._config.p_tel.diam * self._config.p_atmos.windspeed[screen_index] * \
                    np.cos(CONST.DEG2RAD * self._config.p_geom.zenithangle) * self._config.p_loop.ittime
        oldx = self._config.p_atmos._deltax[screen_index]
        oldy = self._config.p_atmos._deltay[screen_index]
        self._config.p_atmos._deltax[screen_index] = lin_delta * np.sin(CONST.DEG2RAD * self._config.p_atmos.winddir[screen_index] + np.pi)
        self._config.p_atmos._deltay[screen_index] = lin_delta * np.cos(CONST.DEG2RAD * self._config.p_atmos.winddir[screen_index] + np.pi)
        self._atmos.d_screens[screen_index].set_deltax(self._config.p_atmos._deltax[screen_index])
        self._atmos.d_screens[screen_index].set_deltay(self._config.p_atmos._deltay[screen_index])
        if(oldx * self._config.p_atmos._deltax[screen_index] < 0): #Sign has changed, must change the stencil
            stencilx = np.array(self._atmos.d_screens[screen_index].d_istencilx)
            n = self._config.p_atmos.dim_screens[screen_index]
            stencilx = (n * n - 1) - stencilx
            self._atmos.d_screens[screen_index].set_istencilx(stencilx)
        if(oldy * self._config.p_atmos._deltay[screen_index] < 0): #Sign has changed, must change the stencil
            stencily = np.array(self._atmos.d_screens[screen_index].d_istencily)
            n = self._config.p_atmos.dim_screens[screen_index]
            stencily = (n * n - 1) - stencily
            self._atmos.d_screens[screen_index].set_istencily(stencily)

    def reset_turbu(self) -> None:
        """ Reset the turbulence layers to their original state
        """
        ilayer = 0
        for k in range(self._atmos.nscreens):
            self._atmos.set_seed(k, self._config.p_atmos.seeds[ilayer])
            self._atmos.refresh_screen(k)
            ilayer += 1

    def get_atmos_layer(self, indx: int) -> np.ndarray:
        """ Return the selected atmos screen

        Args:
            indx : (int) : Index of the turbulent layer to return

        Returns:
            layer : (np.ndarray) : turbulent layer phase screen
        """
        return np.array(self._atmos.d_screens[indx].d_screen)

    def move_atmos(self) -> None:
        """ Move the turbulent layers according to wind speed and direction for a single iteration
        """
        self._atmos.move_atmos()
