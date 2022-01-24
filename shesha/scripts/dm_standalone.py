## @package   shesha.scripts.dm_standalone
## @brief     Python dm standalone script
## @author    COMPASS Team <https://github.com/ANR-COMPASS>
## @version   5.2.1
## @date      2022/01/24
## @copyright GNU Lesser General Public License
#
#  This file is part of COMPASS <https://anr-compass.github.io/compass/>
#
#  Copyright (C) 2011-2022 COMPASS Team <https://github.com/ANR-COMPASS>
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

# import cProfile
# import pstats as ps

import sys
import os
# import numpy as np
import carmaWrap as ch
import shesha.config as conf
import time
import numpy as np
import matplotlib.pyplot as plt
plt.ion()

from shesha.init.geom_init import geom_init_generic
from shesha.init.dm_init import dm_init_standalone

#geom
p_geom = conf.Param_geom()
geom_init_generic(p_geom, 500)

#dm
p_dm0 = conf.Param_dm()
p_dms = [p_dm0]
p_dm0.set_type("pzt")
# p_dm0.set_pattern("hexa")
p_dm0.set_nact(80)
p_dm0.set_alt(0.)
p_dm0.set_thresh(0.3)
p_dm0.set_coupling(0.2)
p_dm0.set_unitpervolt(0.01)
p_dm0.set_pzt_extent(0)

#   context
c = ch.context.get_instance_1gpu(0)

#   dm
print("->dm")
dms = dm_init_standalone(c, p_dms, p_geom)

print("====================")
print("init done")
print("====================")
print("objects initialzed on GPU:")
print("--------------------------------------------------------")
print(dms)

cmd = np.zeros(5268)
cmd[1111] = 1
dms.set_full_com(cmd)
plt.matshow(dms.d_dms[0].d_shape)
