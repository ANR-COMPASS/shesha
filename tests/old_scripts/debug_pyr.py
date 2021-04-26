import cProfile
import pstats as ps

import sys
import os
import numpy as np
import carmaWrap as ch
import shesha as ao
import time
import matplotlib.pyplot as pl
import hdf5_util as h5u

print("TEST SHESHA\n closed loop: call loop(int niter)")

if (len(sys.argv) != 2):
    error = 'command line should be:"python -i test.py parameters_filename"\n with "parameters_filename" the path to the parameters file'
    raise Exception(error)

# get parameters from file
param_file = sys.argv[1]
if (param_file.split('.')[-1] == b"py"):
    filename = param_file.split('/')[-1]
    param_path = param_file.split(filename)[0]
    sys.path.insert(0, param_path)
    exec("import %s as config" % filename.split(".py")[0])
    sys.path.remove(param_path)
elif (param_file.split('.')[-1] == b"h5"):
    sys.path.insert(0, os.environ["SHESHA_ROOT"] + "/data/par/par4bench/")
    import scao_sh_16x16_8pix as config
    sys.path.remove(os.environ["SHESHA_ROOT"] + "/data/par/par4bench/")
    h5u.configFromH5(param_file, config)
else:
    raise ValueError("Parameter file extension must be .py or .h5")

print("param_file is", param_file)

if (hasattr(config, "simul_name")):
    if (config.simul_name is None):
        simul_name = ""
    else:
        simul_name = config.simul_name
        print("simul name is", simul_name)
else:
    simul_name = ""

matricesToLoad = {}
if (simul_name == b""):
    clean = 1
else:
    clean = 0
    param_dict = h5u.params_dictionary(config)
    matricesToLoad = h5u.checkMatricesDataBase(os.environ["SHESHA_ROOT"] + "/data/",
                                               config, param_dict)
# initialisation:
#   context
c = ch.carmaWrap_context(0)
# c.set_active_device(0) #useful only if you use ch.carmaWrap_context()

# wfs
config.p_wfs0.set_atmos_seen(0)
config.p_wfs0.set_pyr_ampl(3)

# dm
config.p_dm1 = ao.Param_dm()
config.p_dms = [config.p_dm1]
config.p_dm1.set_type("tt")
config.p_dm1.set_alt(0.)
config.p_dm1.set_unitpervolt(1.)
lambda_d = 1.  #config.p_wfs0.Lambda / config.p_tel.diam * 180 / np.pi * 3600
config.p_dm1.set_push4imat(2. * lambda_d)

# controllers
config.p_controller0.set_ndm([0])

import matplotlib.pyplot as plt
plt.ion()
npts = 16
index = 1
while npts <= 512:
    config.p_wfs0.set_pyr_npts(npts)
    #    wfs
    print("->wfs")
    wfs, tel = ao.wfs_init(config.p_wfss, config.p_atmos, config.p_tel, config.p_geom,
                           config.p_target, config.p_loop, config.p_dms)

    #   dm
    print("->dm")
    dms = ao.dm_init(config.p_dms, config.p_wfss, wfs, config.p_geom, config.p_tel)

    print("->rtc")
    #   rtc
    rtc = ao.rtc_init(tel, wfs, config.p_wfss, dms, config.p_dms, config.p_geom,
                      config.p_rtc, None, None, config.p_tel, config.p_loop, clean=clean,
                      simul_name=simul_name, load=matricesToLoad)

    if not clean:
        h5u.validDataBase(os.environ["SHESHA_ROOT"] + "/data/", matricesToLoad)

    print("====================")
    print("init done")
    print("====================")
    print("objects initialzed on GPU:")
    print("--------------------------------------------------------")
    print(wfs)
    print(dms)
    print(rtc)

    imat = rtc.get_imat(0)

    plt.subplot(2, 3, index)
    plt.plot(imat[:, -1], label="Tip")
    plt.plot(imat[:, -2], label="Tilt")
    plt.legend()
    plt.title("%s_npts%d_ampl%.2f" % (param_file, config.p_wfs0.pyr_npts,
                                      config.p_wfs0.pyr_ampl))
    npts <<= 1
    index += 1
