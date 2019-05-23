''' @package shesha.script.dm_standalone

Python dm standalone script

'''
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
