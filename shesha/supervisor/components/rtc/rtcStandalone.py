## @package   shesha.supervisor
## @brief     User layer for initialization and execution of a COMPASS simulation
## @author    COMPASS Team <https://github.com/ANR-COMPASS>
## @version   5.4.4
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
from shesha.init.rtc_init import rtc_standalone
from shesha.supervisor.components.sourceCompass import SourceCompass
import shesha.constants as scons
import numpy as np
from typing import Union

from shesha.supervisor.components.rtc.rtcAbstract import RtcAbstract, carmaWrap_context


class RtcStandalone(RtcAbstract):
    """ RTC handler for compass standalone
    """

    def __init__(self, context: carmaWrap_context, config, nwfs: int, nvalid: list,
                 nactu: int, centroider_type: list, delay: list, offset: list,
                 scale: list, *, brahma: bool = False, fp16: bool = False,
                 cacao: bool = False):
        """ Initialize a RtcStandalone component for rtc related supervision

        Args:
            context : (carmaContext) : CarmaContext instance

            config : (config module) : Parameters configuration structure module

            nwfs: (int): number of wavefront sensors

            nvalid: (int): number of valid measures as input

            nactu: (int): number of actuators as output

            centroider_type: (list): type of centroiders

            delay: (list): delay of each controller

            offset: (list): offset added in the cog computation of each WFS

            scale: (list): scale factor used in the cog computation of each WFS

        Kwargs:
            brahma : (bool, optional) : If True, enables BRAHMA features in RTC (Default is False)
                                      Requires BRAHMA to be installed

            fp16 : (bool, optional) : If True, enables FP16 features in RTC (Default is False)
                                      Requires CUDA_SM>60 to be installed

            cacao : (bool) : If True, enables CACAO features in RTC (Default is False)
                                      Requires OCTOPUS to be installed
        """
        RtcAbstract.__init__(self, context, config, brahma=brahma, fp16=fp16,
                             cacao=cacao)

        self.rtc_init(nwfs, nvalid, nactu, centroider_type, delay, offset, scale)

    def rtc_init(self, nwfs: int, nvalid: list, nactu: int, centroider_type: list,
                 delay: list, offset: list, scale: list):
        """ Initialize a RtcStandalone component for rtc related supervision

        Args:
            nwfs: (int): number of wavefront sensors

            nvalid: (int): number of valid measures as input

            nactu: (int): number of actuators as output

            centroider_type: (list): type of centroiders

            delay: (list): delay of each controller

            offset: (list): offset added in the cog computation of each WFS

            scale: (list): scale factor used in the cog computation of each WFS
        """
        self._rtc = rtc_standalone(self._context, nwfs, nvalid, nactu, centroider_type,
                                   delay, offset, scale, brahma=self.brahma,
                                   fp16=self.fp16, cacao=self.cacao)
