# -*- coding: utf-8 -*-
"""
Created on Wed Nov  9 14:03:29 2016

@author: sdurand
"""

# import min
import sys
import os
sys.path.insert(0, os.environ["SHESHA_ROOT"] + "/widgets/")
import carmaWrap as ch
import shesha as ao

print("TEST SHESHA_dm\n")

# read param file :
param_file = sys.argv[1]
if (param_file.split('.')[-1] == b"py"):
    filename = param_file.split('/')[-1]
    param_path = param_file.split(filename)[0]
    sys.path.insert(0, param_path)
    exec("import %s as config" % filename.split(".py")[0])
else:
    raise ValueError("Parameter file extension must be .py or .h5")

print("param_file is", param_file)

#initialisation:
#   context : gpu 0
gpudevice = 0
c = ch.carmaWrap_context(gpudevice)


# fonction for init dm and geometry
def initSimuDM(config):

    # init geom
    print("->geom")
    ao.Param_geom.geom_init(config.p_geom, config.p_tel, config.p_geom.pupdiam,
                            config.p_geom.apod)  #apod = apodizer
    # init dm
    print("->dm")
    dms = ao.dm_init_standalone(config.p_dms, config.p_geom, config.p_tel.diam,
                                config.p_tel.cobs)

    # Print DM information
    print("====================")
    print("init done")
    print("====================")
    print("objects initialzed on GPU:")
    print("--------------------------------------------------------")

    print(dms)

    return dms


# use init function
dms = initSimuDM(config)  # Init Simu
