## @package   shesha.supervisor
## @brief     User layer for initialization and execution of a COMPASS simulation
## @author    COMPASS Team <https://github.com/ANR-COMPASS>
## @version   5.5.0
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

from shesha.init.rtc_init import rtc_init

from shesha.supervisor.components.rtc.rtcAbstract import RtcAbstract, carmaWrap_context


class RtcCompass(RtcAbstract):
    """ RTC handler for compass simulation
    """

    def __init__(self, context: carmaWrap_context, config, tel, wfs, dms, atm, *,
                 brahma: bool = False, fp16: bool = False, cacao: bool = False):
        """ Initialize a RtcCompass component for rtc related supervision

        Args:
            context : (carmaContext) : CarmaContext instance

            config : (config module) : Parameters configuration structure module

            tel: (Telescope) : Telescope object

            wfs: (Sensors) : Sensors object

            dms: (Dms) : Dms object

            atm: (Atmos) : Atmos object

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
        self.rtc_init(tel, wfs, dms, atm)

    def rtc_init(self, tel, wfs, dms, atm):
        """ Initialize a RtcCompass component for rtc related supervision

        Args:
            tel: (Telescope) : Telescope object

            wfs: (Sensors) : Sensors object

            dms: (Dms) : Dms object

            atm: (Atmos) : Atmos object
        """
        self._rtc = rtc_init(self._context, tel._tel, wfs._wfs, dms._dms, atm._atmos,
                             self._config.p_wfss, self._config.p_tel,
                             self._config.p_geom, self._config.p_atmos,
                             self._config.p_loop.ittime, self._config.p_centroiders,
                             self._config.p_controllers, self._config.p_dms,
                             cacao=self.cacao)
