## @package   shesha.supervisor
## @brief     User layer for initialization and execution of a COMPASS simulation
## @author    COMPASS Team <https://github.com/ANR-COMPASS>
## @version   5.4.3
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
from typing import List

class SourceCompass(object):
    """ Source handler for compass simulation

    Attributes:
        sources : (List) : List of SutraSource instances
    """
    def __init__(self, sources : List):
        """ Initialize a SourceCompass component for target and wfs source related supervision

        Args:
            sources : (List) : List of SutraSource instances
        """
        self.sources = sources

    def raytrace(self, index, *, tel=None, atm=None, dms=None, ncpa : bool=True, reset : bool = True, comp_avg_var : bool = True) -> None:
        """ Performs the raytracing operation through provided object phase screens
        to obtain the phase screen of the SutraSource

        Args:
            index : (int) : Index of the source  to raytrace in self.sources list

        Kwargs:
            tel : (TelescopeCompass) : TelescopeCompass instance.
                                                 If provided, raytrace through the telescope aberration phase in the pupil

            atm : (AtmosCompass) : AtmosCompass instance.
                                            If provided, raytrace through the layers phase screens

            dms : (dmsCompass) : DmCompass instance.
                                            If provided, raytrace through the DM shapes

            ncpa : (bool) : If True (default), raytrace through NCPA phase screen of the source (default is array of 0, i.e. no impact)

            reset: (bool): reset the phase screen before raytracing. Default is True
        """
        if (reset):
            self.sources[index].d_phase.reset()

        if atm is not None:
            self.sources[index].raytrace(atm._atmos) # Must be done first because of automatic reset of the phase screen when call
        if tel is not None:
            self.sources[index].raytrace(tel._tel)
        if ncpa:
            self.sources[index].raytrace()
        if dms is not None:
            self.sources[index].raytrace(dms._dms, do_phase_var=comp_avg_var)
