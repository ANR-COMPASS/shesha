## @package   shesha.sutra_wrap
## @brief     Sutra import handler 
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

import importlib


def smart_import(mod, cls, verbose=False, silent=False):
    try:
        if verbose:
            print("trying from " + mod + " import " + cls)
        # my_module = __import__(mod, globals(), locals(), [cls], 0)
        my_module = importlib.import_module(mod)
        return getattr(my_module, cls)

    except ImportError as err:
        if not silent:
            import warnings
            warnings.warn(
                    "Error importing %s, it will be simulated due to: %s" %
                    (cls, err.msg), Warning)

        class tmp_cls:

            def __init__(self, *args, **kwargs):
                raise RuntimeError("Can not initilize the simulation with fake objects")

        return tmp_cls

    except AttributeError as err:
        if not silent:
            import warnings
            warnings.warn(
                    "Error importing %s, it will be simulated due to: %s" %
                    (cls, err.args), Warning)

        class tmp_cls:

            def __init__(self, *args, **kwargs):
                raise RuntimeError("Can not initialize the simulation with fake objects")

        return tmp_cls


# The import of carmaWrap MUST be done first
# since MAGMA >= 2.5.0
# Otherwise, it causes huge memory leak
# plus not working code
# Why ? We don't know... TB check with further version of MAGMA
carmaWrap_context = smart_import("carmaWrap", "context")

Dms = smart_import("sutraWrap", "Dms")
Rtc_FFF = smart_import("sutraWrap", "Rtc_FFF")
Rtc_FHF = smart_import("sutraWrap", "Rtc_FHF", silent=True)
Rtc_UFF = smart_import("sutraWrap", "Rtc_UFF", silent=True)
Rtc_UHF = smart_import("sutraWrap", "Rtc_UHF", silent=True)
Rtc_FFU = smart_import("sutraWrap", "Rtc_FFU", silent=True)
Rtc_FHU = smart_import("sutraWrap", "Rtc_FHU", silent=True)
Rtc_UFU = smart_import("sutraWrap", "Rtc_UFU", silent=True)
Rtc_UHU = smart_import("sutraWrap", "Rtc_UHU", silent=True)
Rtc_brahma = smart_import("sutraWrap", "Rtc_brahma", silent=True)
Rtc_cacao_FFF = smart_import("sutraWrap", "Rtc_cacao_FFF", silent=True)
Rtc_cacao_UFF = smart_import("sutraWrap", "Rtc_cacao_UFF", silent=True)
Rtc_cacao_FFU = smart_import("sutraWrap", "Rtc_cacao_FFU", silent=True)
Rtc_cacao_UFU = smart_import("sutraWrap", "Rtc_cacao_UFU", silent=True)
Rtc_cacao_FHF = smart_import("sutraWrap", "Rtc_cacao_FHF", silent=True)
Rtc_cacao_UHF = smart_import("sutraWrap", "Rtc_cacao_UHF", silent=True)
Rtc_cacao_FHU = smart_import("sutraWrap", "Rtc_cacao_FHU", silent=True)
Rtc_cacao_UHU = smart_import("sutraWrap", "Rtc_cacao_UHU", silent=True)
Sensors = smart_import("sutraWrap", "Sensors")
Atmos = smart_import("sutraWrap", "Atmos")
Telescope = smart_import("sutraWrap", "Telescope")
Target = smart_import("sutraWrap", "Target")
Target_brahma = smart_import("sutraWrap", "Target_brahma", silent=True)
Gamora = smart_import("sutraWrap", "Gamora")
Groot = smart_import("sutraWrap", "Groot")
