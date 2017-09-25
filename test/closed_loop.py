
# import cProfile
# import pstats as ps

import sys
import os
# import numpy as np
import naga as ch
import shesha as ao
import time
import matplotlib.pyplot as plt
import hdf5_utils as h5u
plt.ion()

print "TEST SHESHA\n closed loop: call loop(int niter)"


if(len(sys.argv) != 2):
    error = 'command line should be:"python -i test.py parameters_filename"\n with "parameters_filename" the path to the parameters file'
    raise StandardError(error)

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

print "param_file is", param_file


if(hasattr(config, "simul_name")):
    if(config.simul_name is None):
        simul_name = ""
    else:
        simul_name = config.simul_name
        print "simul name is", simul_name
else:
    simul_name = ""

clean = 1
matricesToLoad = {}
'''
if(simul_name != ""):
    clean = 0
    param_dict = h5u.params_dictionary(config)
    matricesToLoad = h5u.checkMatricesDataBase(
        os.environ["SHESHA_ROOT"] + "/data/", config, param_dict)
'''
# initialisation:

#   context
# c = ch.naga_context(0)
# c = ch.naga_context(devices=np.array([0,1], dtype=np.int32))
# c.set_activeDevice(0) #useful only if you use ch.naga_context()
c = ch.naga_context(devices=config.p_loop.devices)

#    wfs
print "->wfs"
wfs, tel = ao.wfs_init(config.p_wfss, config.p_atmos, config.p_tel,
                       config.p_geom, config.p_target, config.p_loop, config.p_dms)

#   atmos
print "->atmos"
atm = ao.atmos_init(c, config.p_atmos, config.p_tel, config.p_geom, config.p_loop, config.p_wfss, wfs, config.p_target,
                    clean=clean, load=matricesToLoad)

#   dm
print "->dm"
dms = ao.dm_init(config.p_dms, config.p_wfss, wfs, config.p_geom, config.p_tel)

#   target
print "->target"
tar = ao.target_init(c, tel, config.p_target, config.p_atmos, config.p_geom, config.p_tel, config.p_dms, config.p_wfss)

print "->rtc"
#   rtc
rtc = ao.rtc_init(tel, wfs, config.p_wfss, dms, config.p_dms, config.p_geom, config.p_rtc, config.p_atmos, atm,
                  config.p_tel, config.p_loop, clean=clean, simul_name=simul_name,
                  load=matricesToLoad)

if not clean:
    h5u.validDataBase(os.environ["SHESHA_ROOT"] + "/data/", matricesToLoad)

print "===================="
print "init done"
print "===================="
print "objects initialzed on GPU:"
print "--------------------------------------------------------"
print atm
print wfs
print dms
print tar
print rtc

mimg = 0.  # initializing average image


def loop(n):
    print "----------------------------------------------------"
    print "iter# | S.E. SR | L.E. SR | Est. Rem. | framerate"
    print "----------------------------------------------------"
    t0 = time.time()
    for i in range(n):
        atm.move_atmos()

        if(config.p_controllers[0].type_control == "geo"):
            for t in range(config.p_target.ntargets):
                tar.atmos_trace(t, atm, tel)
                rtc.docontrol_geo(0, dms, tar, 0)
                rtc.applycontrol(0, dms)
                tar.dmtrace(0, dms)
        else:
            for t in range(config.p_target.ntargets):
                tar.atmos_trace(t, atm, tel)
                tar.dmtrace(t, dms)
            for w in range(len(config.p_wfss)):
                wfs.sensors_trace(w, "all", tel, atm, dms)
                wfs.sensors_compimg(w)

            rtc.docentroids(0)
            rtc.docontrol(0)

            rtc.applycontrol(0, dms)

        if((i + 1) % 100 == 0):
            strehltmp = tar.get_strehl(0)
            print i + 1, "\t", strehltmp[0], "\t", strehltmp[1]
    t1 = time.time()
    print " loop execution time:", t1 - t0, "  (", n, "iterations), ", (t1 - t0) / n, "(mean)  ", n / (t1 - t0), "Hz"


loop(config.p_loop.niter)
