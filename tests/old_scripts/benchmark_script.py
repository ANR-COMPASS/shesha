import datetime
#import hdf5_util as h5u
import os
import platform
import re
import sys
import time
from subprocess import check_output

import numpy as np
from pandas import DataFrame, HDFStore

import shesha.constants as scons
import shesha.init as init
from carmaWrap.context import context as carmaWrap_context
from carmaWrap.timer import timer as carmaWrap_timer
from carmaWrap.timer import threadSync


def get_processor_name():
    command = "cat /proc/cpuinfo"
    all_info = check_output(command, shell=True).strip().decode("utf-8")
    nb_cpu = 0
    cpu = []
    for line in all_info.split("\n"):
        if "model name" in line:
            cpu.append(re.sub(".*model name.*:", "", line, 1))
            nb_cpu += 1
    return nb_cpu, cpu


def script4bench(param_file, centroider, controller, devices, fwrite=True):
    """

    Args:
        param_file: (str) : parameters filename

        centroider: (str) : centroider type

        controller: (str) : controller type
    """

    c = carmaWrap_context(devices=np.array(devices, dtype=np.int32))
    #     c.set_active_device(device)

    timer = carmaWrap_timer()

    # times measured
    synctime = 0.
    move_atmos_time = 0.
    t_raytrace_atmos_time = 0.
    t_raytrace_dm_time = 0.
    s_raytrace_atmos_time = 0.
    s_raytrace_dm_time = 0.
    comp_img_time = 0.
    docentroids_time = 0.
    docontrol_time = 0.
    applycontrol_time = 0.

    # reading parfile
    filename = param_file.split('/')[-1]
    param_path = param_file.split(filename)[0]
    sys.path.insert(0, param_path)
    #exec("import %s as config" % filename.split(".py")[0])
    config = __import__(filename.split(".py")[0])
    sys.path.remove(param_path)

    config.p_centroiders[0].set_type(centroider)

    if (centroider == "tcog"):
        config.p_centroiders[0].set_thresh(0.)
    elif (centroider == "bpcog"):
        config.p_centroiders[0].set_nmax(16)
    elif (centroider == "geom"):
        config.p_centroiders[0].set_type("cog")
    elif (centroider == "wcog"):
        config.p_centroiders[0].set_type_fct("gauss")
        config.p_centroiders[0].set_width(2.0)
    elif (centroider == "corr"):
        config.p_centroiders[0].set_type_fct("gauss")
        config.p_centroiders[0].set_width(2.0)

    if (controller == "modopti"):
        config.p_controllers[0].set_type("ls")
        config.p_controllers[0].set_modopti(1)
    else:
        config.p_controllers[0].set_type(controller)

    config.p_loop.set_niter(2000)

    threadSync()
    timer.start()
    threadSync()
    synctime = timer.stop()
    timer.reset()

    # init system
    timer.start()
    tel = init.tel_init(c, config.p_geom, config.p_tel, config.p_atmos.r0,
                        config.p_loop.ittime, config.p_wfss)
    threadSync()
    tel_init_time = timer.stop() - synctime
    timer.reset()

    timer.start()
    atm = init.atmos_init(c, config.p_atmos, config.p_tel, config.p_geom,
                          config.p_loop.ittime)
    threadSync()
    atmos_init_time = timer.stop() - synctime
    timer.reset()

    timer.start()
    dms = init.dm_init(c, config.p_dms, config.p_tel, config.p_geom, config.p_wfss)
    threadSync()
    dm_init_time = timer.stop() - synctime
    timer.reset()

    timer.start()
    target = init.target_init(c, tel, config.p_target, config.p_atmos, config.p_tel,
                              config.p_geom, config.p_dms)
    threadSync()
    target_init_time = timer.stop() - synctime
    timer.reset()

    timer.start()
    wfs = init.wfs_init(c, tel, config.p_wfss, config.p_tel, config.p_geom, config.p_dms,
                        config.p_atmos)
    threadSync()
    wfs_init_time = timer.stop() - synctime
    timer.reset()

    timer.start()
    rtc = init.rtc_init(c, tel, wfs, dms, atm, config.p_wfss, config.p_tel,
                        config.p_geom, config.p_atmos, config.p_loop.ittime,
                        config.p_centroiders, config.p_controllers, config.p_dms)
    threadSync()
    rtc_init_time = timer.stop() - synctime
    timer.reset()

    print("... Done with inits !")
    # h5u.validDataBase(os.environ["SHESHA_ROOT"]+"/data/",matricesToLoad)

    strehllp = []
    strehlsp = []
    ############################################################
    #                  _         _
    #                 (_)       | |
    #  _ __ ___   __ _ _ _ __   | | ___   ___  _ __
    # | '_ ` _ \ / _` | | '_ \  | |/ _ \ / _ \| '_ \
    # | | | | | | (_| | | | | | | | (_) | (_) | |_) |
    # |_| |_| |_|\__,_|_|_| |_| |_|\___/ \___/| .__/
    #                                         | |
    #                                         |_|
    ###########################################################
    if (controller == "modopti"):
        for zz in range(2048):
            atm.move_atmos()

    for cc in range(config.p_loop.niter):
        threadSync()
        timer.start()
        atm.move_atmos()
        threadSync()
        move_atmos_time += timer.stop() - synctime
        timer.reset()

        if (config.p_controllers[0].type != b"geo"):
            if ((config.p_target is not None) and (rtc is not None)):
                for i in range(config.p_target.ntargets):
                    timer.start()
                    target.raytrace(i, b"atmos", tel, atm)
                    threadSync()
                    t_raytrace_atmos_time += timer.stop() - synctime
                    timer.reset()

                    if (dms is not None):
                        timer.start()
                        target.raytrace(i, b"dm", tel, dms=dms)
                        threadSync()
                        t_raytrace_dm_time += timer.stop() - synctime
                        timer.reset()

            if (config.p_wfss is not None and wfs is not None):
                for i in range(len(config.p_wfss)):
                    timer.start()
                    wfs.raytrace(i, b"atmos", tel, atm)
                    threadSync()
                    s_raytrace_atmos_time += timer.stop() - synctime
                    timer.reset()

                    if (not config.p_wfss[i].open_loop and dms is not None):
                        timer.start()
                        wfs.raytrace(i, b"dm", tel, atm, dms)
                        threadSync()
                        s_raytrace_dm_time += timer.stop() - synctime
                        timer.reset()

                    timer.start()
                    wfs.comp_img(i)
                    threadSync()
                    comp_img_time += timer.stop() - synctime
                    timer.reset()

            if (rtc is not None and config.p_wfss is not None and wfs is not None):
                if (centroider == "geom"):
                    timer.start()
                    rtc.do_centroids_geom(0)
                    threadSync()
                    docentroids_time += timer.stop() - synctime
                    timer.reset()
                else:
                    timer.start()
                    rtc.do_centroids(0)
                    threadSync()
                    docentroids_time += timer.stop() - synctime
                    timer.reset()

            if (dms is not None):
                timer.start()
                rtc.do_control(0)
                threadSync()
                docontrol_time += timer.stop() - synctime
                timer.reset()

                timer.start()
                rtc.apply_control(0)
                threadSync()
                applycontrol_time += timer.stop() - synctime
                timer.reset()

        else:
            if (config.p_target is not None and target is not None):
                for i in range(config.p_target.ntargets):
                    timer.start()
                    target.raytrace(i, b"atmos", tel, atm)
                    threadSync()
                    t_raytrace_atmos_time += timer.stop() - synctime
                    timer.reset()

                    if (dms is not None):
                        timer.start()
                        rtc.do_control_geo(0, dms, target, i)
                        threadSync()
                        docontrol_time += timer.stop() - synctime
                        timer.reset()

                        timer.start()
                        rtc.apply_control(0)
                        threadSync()
                        applycontrol_time += timer.stop() - synctime
                        timer.reset()

                        timer.start()
                        target.raytrace(i, b"dm", tel, atm, dms)
                        threadSync()
                        t_raytrace_dm_time += timer.stop() - synctime
                        timer.reset()
        target.comp_image(0)
        strehltmp = target.get_strehl(0)
        strehlsp.append(strehltmp[0])
        if (cc > 50):
            strehllp.append(strehltmp[1])

    print("\n done with simulation \n")
    print("\n Final strehl : \n", strehllp[len(strehllp) - 1])
    ###################################################################
    #  _   _
    # | | (_)
    # | |_ _ _ __ ___   ___ _ __ ___
    # | __| | '_ ` _ \ / _ \ '__/ __|
    # | |_| | | | | | |  __/ |  \__ \
    #  \__|_|_| |_| |_|\___|_|  |___/
    ###################################################################

    move_atmos_time /= config.p_loop.niter / 1000.
    t_raytrace_atmos_time /= config.p_loop.niter / 1000.
    t_raytrace_dm_time /= config.p_loop.niter / 1000.
    s_raytrace_atmos_time /= config.p_loop.niter / 1000.
    s_raytrace_dm_time /= config.p_loop.niter / 1000.
    comp_img_time /= config.p_loop.niter / 1000.
    docentroids_time /= config.p_loop.niter / 1000.
    docontrol_time /= config.p_loop.niter / 1000.
    applycontrol_time /= config.p_loop.niter / 1000.

    time_per_iter = move_atmos_time + t_raytrace_atmos_time +\
        t_raytrace_dm_time + s_raytrace_atmos_time +\
        s_raytrace_dm_time + comp_img_time +\
        docentroids_time + docontrol_time +\
        applycontrol_time

    ###########################################################################
    #  _         _  __ _____
    # | |       | |/ _| ____|
    # | |__   __| | |_| |__    ___  __ ___   _____
    # | '_ \ / _` |  _|___ \  / __|/ _` \ \ / / _ \
    # | | | | (_| | |  ___) | \__ \ (_| |\ V /  __/
    # |_| |_|\__,_|_| |____/  |___/\__,_| \_/ \___|
    ###############################################################################

    if (config.p_wfss[0].gsalt > 0):
        stype = "lgs "
    else:
        stype = "ngs "

    if (config.p_wfss[0].gsmag > 3):
        stype += "noisy "

    stype += str(config.p_wfss[0].type)

    if (controller == "modopti"):
        G = np.mean(rtc.get_modal_gains(0))
    else:
        G = 0.

    date = datetime.datetime.now()
    date = [date.year, date.month, date.day]

    version = check_output(["git", "rev-parse", "--short", "HEAD"]).decode('utf8')

    # version=str(check_output(["svnversion",os.getenv("COMPASS_ROOT")]).replace("\n",""))
    hostname = check_output("hostname").replace(b"\n", b"").decode('UTF-8')
    nb_cpu, cpu = get_processor_name()
    keys_dict = {
            "date": date,
            "simulname": config.simul_name,
            "hostname": hostname,
            "ndevices": c.get_ndevice(),
            "device": c.get_device_names()[0],
            "cuda_version": c.get_cuda_runtime_get_version(),
            "magma_version": c.get_magma_info(),
            "platform": platform.platform(),
            "ncpu": nb_cpu,
            "processor": cpu[0],
            "tel.diam": config.p_tel.diam,
            "sensor_type": config.p_wfss[0].type.decode('UTF-8'),
            "LGS": config.p_wfss[0].gsalt > 0,
            "noisy": config.p_wfss[0].gsmag > 3,
            "nxsub": config.p_wfss[0].nxsub,
            "npix": config.p_wfss[0].npix,
            "nphotons": config.p_wfss[0]._nphotons,
            "controller": controller,
            "centroider": centroider,
            "finalSRLE": strehllp[len(strehllp) - 1],
            "rmsSRLE": np.std(strehllp),
            "wfs_init": wfs_init_time,
            "atmos_init": atmos_init_time,
            "dm_init": dm_init_time,
            "target_init": target_init_time,
            "rtc_init": rtc_init_time,
            "move_atmos": move_atmos_time,
            "target_trace_atmos": t_raytrace_atmos_time,
            "target_trace_dm": t_raytrace_dm_time,
            "sensor_trace_atmos": s_raytrace_atmos_time,
            "sensor_trace_dm": s_raytrace_dm_time,
            "comp_img": comp_img_time,
            "docentroids": docentroids_time,
            "docontrol": docontrol_time,
            "applycontrol": applycontrol_time,
            "iter_time": time_per_iter,
            "Avg.gain": G,
            "residualPhase": target.get_phase(0)
    }

    store = HDFStore(BENCH_SAVEPATH + "/benchmarks.h5")
    try:
        df = store.get(version)
    except KeyError:
        df = DataFrame(columns=list(keys_dict.keys()), dtype=object)

    ix = len(df.index)

    if (fwrite):
        print("writing files")
        for i in list(keys_dict.keys()):
            df.loc[ix, i] = keys_dict[i]
        store.put(version, df)
        store.close()


#############################################################
#                 _
#                | |
#   ___ _ __   __| |
#  / _ \ '_ \ / _` |
# |  __/ | | | (_| |
#  \___|_| |_|\__,_|
#############################################################
if __name__ == '__main__':

    if (len(sys.argv) < 4 or len(sys.argv) > 6):
        error = "wrong number of argument. Got %d (expect 4)\ncommande line should be: 'python benchmark_script.py <filename> <centroider> <controller>" % len(
                sys.argv)
        raise Exception(error)

    SHESHA = os.environ.get('SHESHA_ROOT')
    if (SHESHA is None):
        raise EnvironmentError("Environment variable 'SHESHA_ROOT' must be define")

    SHESHA_SAVEPATH = SHESHA + "/data"
    PARPATH = SHESHA_SAVEPATH + "/par/par4bench"
    BENCH_SAVEPATH = SHESHA_SAVEPATH + "/bench-results"

    store = HDFStore(BENCH_SAVEPATH + "/benchmarks.h5")

    filename = PARPATH + "/" + sys.argv[1]
    centroider = sys.argv[2]
    controller = sys.argv[3]
    device = 5
    fwrite = True
    if (len(sys.argv) > 4):
        devices = []
        if (len(sys.argv[4]) > 1):
            for k in range(len(sys.argv[4])):
                devices.append(int(sys.argv[4][k]))
        else:
            devices.append(int(sys.argv[4]))
    if (len(sys.argv) == 6):
        fwrite = int(sys.argv[5])

    script4bench(filename, centroider, controller, devices, fwrite)
