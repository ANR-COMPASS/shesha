
# import cProfile
# import pstats as ps

import sys
import os
# import numpy as np
import naga as ch
import shesha as ao
import time
import hdf5_utils as h5u
import numpy as np

print("TEST SHESHA\n closed loop: call loop(int niter)")


if(len(sys.argv) != 2):
    error = 'command line should be:"python -i test.py parameters_filename"\n with "parameters_filename" the path to the parameters file'
    raise Exception(error)

# get parameters from file
param_file = sys.argv[1]
if(param_file.split('.')[-1] == "py"):
    filename = param_file.split('/')[-1]
    param_path = param_file.split(filename)[0]
    sys.path.insert(0, param_path)
    exec("import %s as config" % filename.split(".py")[0])
    sys.path.remove(param_path)
elif(param_file.split('.')[-1] == "h5"):
    sys.path.insert(0, os.environ["SHESHA_ROOT"] + "/data/par/par4bench/")
    import scao_sh_16x16_8pix as config
    sys.path.remove(os.environ["SHESHA_ROOT"] + "/data/par/par4bench/")
    h5u.configFromH5(param_file, config)
else:
    raise ValueError("Parameter file extension must be .py or .h5")

print("param_file is", param_file)


if(hasattr(config, "simul_name")):
    if(config.simul_name is None):
        simul_name = ""
    else:
        simul_name = config.simul_name
        print("simul name is", simul_name)
else:
    simul_name = ""

clean = 1
matricesToLoad = {}

config.p_geom.set_pupdiam(500)

# initialisation:

#   context
# c = ch.naga_context(0)
# c = ch.naga_context(devices=np.array([0,1], dtype=np.int32))
# c.set_activeDevice(0) #useful only if you use ch.naga_context()
c = ch.naga_context(devices=config.p_loop.devices)

#    wfs
print("->wfs")
wfs, tel = ao.wfs_init(config.p_wfss, config.p_atmos, config.p_tel,
                       config.p_geom, None, config.p_loop, config.p_dms)

#   atmos
print("->atmos")
atm = ao.atmos_init(c, config.p_atmos, config.p_tel, config.p_geom,
                    config.p_loop, config.p_wfss, wfs, None,
                    clean=clean, load=matricesToLoad)

#   dm
print("->dm")
dms = ao.dm_init(config.p_dms, config.p_wfss, wfs, config.p_geom, config.p_tel)
ao.correct_dm(config.p_dms, dms, config.p_controller0, config.p_geom,
              np.ones((config.p_wfs0._nvalid, config.p_dm0._ntotact), dtype = np.float32),
              b'', {}, 1)


if not clean:
    h5u.validDataBase(os.environ["SHESHA_ROOT"] + "/data/", matricesToLoad)

print("====================")
print("init done")
print("====================")
print("objects initialzed on GPU:")
print("--------------------------------------------------------")
print(atm)
print(wfs)
print(dms)

mimg = 0.  # initializing average image


def loop(n):
    print("----------------------------------------------------")
    print("iter# | S.E. SR | L.E. SR | Est. Rem. | framerate")
    print("----------------------------------------------------")
    t0 = time.time()
    for i in range(n):
        atm.move_atmos()

        for w in range(len(config.p_wfss)):
            wfs.sensors_trace(w, b"all", tel, atm, dms)

    t1 = time.time()
    print(" loop execution time:", t1 - t0, "  (", n, "iterations), ", (t1 - t0) / n, "(mean)  ", n / (t1 - t0), "Hz")


# loop(config.p_loop.niter)
