import os

import cProfile
import pstats as ps

import sys
import numpy as np
import carmaWrap as ch
import shesha as ao
import time

rank = int(os.environ['OMPI_COMM_WORLD_RANK'])
c = ch.carmaWrap_context()
c.set_active_device(rank % c.get_ndevice())

# Delay import because of cuda_aware
# mpi_init called during the import
import mpi4py
from mpi4py import MPI
import hdf5_util as h5u

comm = MPI.COMM_WORLD
comm_size = comm.Get_size()
rank = comm.Get_rank()

print("TEST SHESHA\n closed loop with MPI")

if (len(sys.argv) != 2):
    error = 'command line should be:"python test.py parameters_filename"\n with "parameters_filename" the path to the parameters file'
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
else:
    simul_name = ""
print("simul name is", simul_name)

matricesToLoad = {}
if (simul_name == b""):
    clean = 1
else:
    clean = 0
    param_dict = h5u.params_dictionary(config)
    matricesToLoad = h5u.checkMatricesDataBase(os.environ["SHESHA_ROOT"] + "/data/",
                                               config, param_dict)

# initialisation:
#    wfs
print("->wfs")
wfs, tel = ao.wfs_init(config.p_wfss, config.p_atmos, config.p_tel, config.p_geom,
                       config.p_target, config.p_loop, comm_size, rank, config.p_dms)

#   atmos
print("->atmos")
atm = ao.atmos_init(c, config.p_atmos, config.p_tel, config.p_geom, config.p_loop,
                    rank=rank, load=matricesToLoad)

#   dm
print("->dm")
dms = ao.dm_init(config.p_dms, config.p_wfss, wfs, config.p_geom, config.p_tel)

#   target
print("->target")
tar = ao.target_init(c, tel, config.p_target, config.p_atmos, config.p_geom,
                     config.p_tel, config.p_dms)

#   rtc
print("->rtc")
rtc = ao.rtc_init(tel, wfs, config.p_wfss, dms, config.p_dms, config.p_geom,
                  config.p_rtc, config.p_atmos, atm, config.p_tel, config.p_loop,
                  clean=clean, simul_name=simul_name, load=matricesToLoad)

if not clean and rank == 0:
    h5u.validDataBase(os.environ["SHESHA_ROOT"] + "/data/", matricesToLoad)

comm.Barrier()
if (rank == 0):
    print("====================")
    print("init done")
    print("====================")
    print("objects initialzed on GPU:")
    print("--------------------------------------------------------")
    print(atm)
    print(wfs)
    print(dms)
    print(tar)
    print(rtc)

    print("----------------------------------------------------")
    print("iter# | S.E. SR | L.E. SR | Est. Rem. | framerate")
    print("----------------------------------------------------")
comm.Barrier()

mimg = 0.  # initializing average image

#import matplotlib.pyplot as pl


def loop(n):
    # if(rank==0):
    #fig,((turbu,image),(shak,defMir))=pl.subplots(2,2, figsize=(15,15))
    # pl.ion()
    # pl.show()

    t0 = time.time()
    for i in range(n):
        if (rank == 0):
            atm.move_atmos()
            for t in range(config.p_target.ntargets):
                tar.atmos_trace(t, atm, tel)
                tar.dmtrace(t, dms)
            for w in range(len(config.p_wfss)):
                wfs.sensors_trace(w, "all", tel, atm, dms)
        wfs.Bcast_dscreen()
        for w in range(len(config.p_wfss)):
            wfs.sensors_compimg(w)
            wfs.gather_bincube(w)
        if (rank == 0):
            rtc.docentroids(0)
            rtc.docontrol(0)
            rtc.applycontrol(0, dms)

        if ((i + 1) % 50 == 0):
            # s=rtc.get_centroids(0)
            if (rank == 0):
                """ FOR DEBUG PURPOSE
                turbu.clear()
                image.clear()
                shak.clear()
                defMir.clear()

                screen=atm.get_screen(0.)

                im=tar.get_image(0,"se")
                im=np.roll(im,im.shape[0]/2,axis=0)
                im=np.roll(im,im.shape[1]/2,axis=1)

                #sh=wfs.get_binimg(0)

                dm=dms.get_dm("pzt",0.)

                f1=turbu.matshow(screen,cmap='Blues_r')
                f2=image.matshow(im,cmap='Blues_r')
                #f3=shak.matshow(sh,cmap='Blues_r')
                f4=defMir.matshow(dm)
                pl.draw()


                c=rtc.get_command(0)
                v=rtc.get_voltages(0)

                sh_file="dbg/shak_"+str(i)+"_np_"+str(comm.Get_size())+".npy"
                im_file="dbg/imag_"+str(i)+"_np_"+str(comm.Get_size())+".npy"
                dm_file="dbg/DM_"+str(i)+"_np_"+str(comm.Get_size())+".npy"
                s_file="dbg/cent_"+str(i)+"_np_"+str(comm.Get_size())+".npy"
                c_file="dbg/comm_"+str(i)+"_np_"+str(comm.Get_size())+".npy"
                v_file="dbg/volt_"+str(i)+"_np_"+str(comm.Get_size())+".npy"

                np.save(sh_file,sh)
                np.save(im_file,im)
                np.save(dm_file,dm)
                np.save(s_file,s)
                np.save(c_file,c)
                np.save(v_file,v)
                """

                strehltmp = tar.get_strehl(0)
                print("%5d" % (i + 1), "  %1.5f" % strehltmp[0],
                      "  %1.5f" % strehltmp[1])

    t1 = time.time()
    print(rank, "| loop execution time:", t1 - t0, "  (", n, "iterations), ",
          (t1 - t0) / n, "(mean)  ", n / (t1 - t0), "Hz")


loop(config.p_loop.niter)
