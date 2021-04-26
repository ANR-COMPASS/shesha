"""
Created on Tue Jul 12 09:28:23 2016

@author: fferreira
"""

import cProfile
import pstats as ps

import sys, os
import numpy as np
import carmaWrap as ch
import shesha as ao
import time
import matplotlib.pyplot as plt
plt.ion()
import hdf5_util as h5u
import pandas
from scipy.sparse import csr_matrix

if (len(sys.argv) < 2):
    error = 'command line should be at least:"python -i test.py parameters_filename"\n with "parameters_filename" the path to the parameters file'
    raise Exception(error)

#get parameters from file
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

#if(len(sys.argv) > 2):
#    device=int(sys.argv[2])
#else:
#    device = 0
if (len(sys.argv) > 2):
    savename = sys.argv[2]
else:
    savename = "roket_default.h5"

print("save file is ", savename)
############################################################################
#  _       _ _
# (_)_ __ (_) |_ ___
# | | '_ \| | __/ __|
# | | | | | | |_\__ \
# |_|_| |_|_|\__|___/
############################################################################

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
#initialisation:
#   context
#c=ch.carmaWrap_context(device)
c = ch.carmaWrap_context(devices=np.array([0, 1], dtype=np.int32))
#c.set_active_device(device)

#    wfs
print("->wfs")
wfs, tel = ao.wfs_init(config.p_wfss, config.p_atmos, config.p_tel, config.p_geom,
                       config.p_target, config.p_loop, config.p_dms)

#   atmos
print("->atmos")
atm = ao.atmos_init(c, config.p_atmos, config.p_tel, config.p_geom, config.p_loop,
                    config.p_wfss, wfs, config.p_target, rank=0, clean=clean,
                    load=matricesToLoad)

#   dm
print("->dm")
dms = ao.dm_init(config.p_dms, config.p_wfss, wfs, config.p_geom, config.p_tel)

#   target
print("->target")
tar = ao.target_init(c, tel, config.p_target, config.p_atmos, config.p_geom,
                     config.p_tel, config.p_dms)

print("->rtc")
#   rtc
rtc = ao.rtc_init(tel, wfs, config.p_wfss, dms, config.p_dms, config.p_geom,
                  config.p_rtc, config.p_atmos, atm, config.p_tel, config.p_loop,
                  clean=clean, simul_name=simul_name, load=matricesToLoad)

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
print(tar)
print(rtc)

print("----------------------------------------------------")
print("iter# | SE SR image | LE SR image | Fitting | LE SR phase var")
print("----------------------------------------------------")

error_flag = True in [w.roket for w in config.p_wfss]


##############################################################################
#    _   ___    _
#   /_\ / _ \  | |___  ___ _ __
#  / _ \ (_) | | / _ \/ _ \ '_ \
# /_/ \_\___/  |_\___/\___/ .__/
#                         |_|
##############################################################################
def loop(n):
    """
    Performs the main AO loop for n interations. First, initialize buffers
    for error breakdown computations. Then, at the end of each iteration, just
    before applying the new DM shape, calls the error_breakdown function.

    :param n: (int) : number of iterations

    :return:
        com : (np.array((n,nactus))) : full command buffer

        noise_com : (np.array((n,nactus))) : noise contribution for error breakdown

        alias_wfs_com : (np.array((n,nactus))) : aliasing estimation in the WFS direction

        tomo_com : (np.array((n,nactus))) : tomography error estimation

        H_com : (np.array((n,nactus))) : Filtered modes contribution for error breakdown

        trunc_com : (np.array((n,nactus))) : Truncature and sampling error of WFS

        bp_com : (np.array((n,nactus))) : Bandwidth error estimation on target

        mod_com : (np.array((n,nactus))) : Commanded modes expressed on the actuators

        fit : (float) : fitting (mean variance of the residual target phase after projection)

        SR : (float) : final strehl ratio returned by the simulation
    """
    if (error_flag):
        # Initialize buffers for error breakdown
        nactu = rtc.get_command(0).size
        nslopes = rtc.get_centroids(0).size
        com = np.zeros((n, nactu), dtype=np.float32)
        noise_com = np.zeros((n, nactu), dtype=np.float32)
        alias_wfs_com = np.copy(noise_com)
        wf_com = np.copy(noise_com)
        tomo_com = np.copy(noise_com)
        trunc_com = np.copy(noise_com)
        H_com = np.copy(noise_com)
        mod_com = np.copy(noise_com)
        bp_com = np.copy(noise_com)
        fit = np.zeros(n)
    #    covm = np.zeros((nslopes,nslopes))
    #    covv = np.zeros((nactu,nactu))

    t0 = time.time()
    for i in range(-10, n):
        atm.move_atmos()

        if (config.p_controllers[0].type == b"geo"):
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
            #m = np.reshape(rtc.get_centroids(0),(nslopes,1))
            #v = np.reshape(rtc.get_command(0),(nactu,1))
            if (error_flag and i > -1):
                #compute the error breakdown for this iteration
                #covm += m.dot(m.T)
                #covv += v.dot(v.T)
                roket.computeBreakdown()
            rtc.applycontrol(0, dms)

        if ((i + 1) % 100 == 0 and i > -1):
            strehltmp = tar.get_strehl(0)
            print(i + 1, "\t", strehltmp[0], "\t", strehltmp[1], "\t",
                  np.exp(-strehltmp[2]), "\t", np.exp(-strehltmp[3]))
    t1 = time.time()
    print(" loop execution time:", t1 - t0, "  (", n, "iterations), ", (t1 - t0) / n,
          "(mean)  ", n / (t1 - t0), "Hz")
    if (error_flag):
        #Returns the error breakdown
        SR2 = np.exp(-tar.get_strehl(0, comp_strehl=False)[3])
        SR = tar.get_strehl(0, comp_strehl=False)[1]
        #bp_com[-1,:] = bp_com[-2,:]
        #SR = tar.get_strehl(0,comp_strehl=False)[1]
    return SR, SR2


def preloop(n):
    """
    Performs the main AO loop for n interations. First, initialize buffers
    for error breakdown computations. Then, at the end of each iteration, just
    before applying the new DM shape, calls the error_breakdown function.

    :param n: (int) : number of iterations

    :return:
        com : (np.array((n,nactus))) : full command buffer

        noise_com : (np.array((n,nactus))) : noise contribution for error breakdown

        alias_wfs_com : (np.array((n,nactus))) : aliasing estimation in the WFS direction

        tomo_com : (np.array((n,nactus))) : tomography error estimation

        H_com : (np.array((n,nactus))) : Filtered modes contribution for error breakdown

        trunc_com : (np.array((n,nactus))) : Truncature and sampling error of WFS

        bp_com : (np.array((n,nactus))) : Bandwidth error estimation on target

        mod_com : (np.array((n,nactus))) : Commanded modes expressed on the actuators

        fit : (float) : fitting (mean variance of the residual target phase after projection)

        SR : (float) : final strehl ratio returned by the simulation
    """
    for i in range(0, n):
        atm.move_atmos()

        if (config.p_controllers[0].type == b"geo"):
            for t in range(config.p_target.ntargets):
                tar.atmos_trace(t, atm, tel)
                rtc.docontrol_geo(0, dms, tar, 0)
                rtc.applycontrol(0, dms)
        else:
            for t in range(config.p_target.ntargets):
                tar.atmos_trace(t, atm, tel)
            for w in range(len(config.p_wfss)):
                wfs.sensors_trace(w, "all", tel, atm, dms)
                wfs.sensors_compimg(w)
            rtc.docentroids(0)
            rtc.docontrol(0)

            rtc.applycontrol(0, dms)


################################################################################
#  ___          _
# | _ ) __ _ __(_)___
# | _ \/ _` (_-< (_-<
# |___/\__,_/__/_/__/
################################################################################
def compute_btt():
    IF = rtc.get_IFsparse(1).T
    N = IF.shape[0]
    n = IF.shape[1]
    #T = IF[:,-2:].copy()
    T = rtc.get_IFtt(1)
    #IF = IF[:,:n-2]
    n = IF.shape[1]

    delta = IF.T.dot(IF).toarray() / N

    # Tip-tilt + piston
    Tp = np.ones((T.shape[0], T.shape[1] + 1))
    Tp[:, :2] = T.copy()  #.toarray()
    deltaT = IF.T.dot(Tp) / N
    # Tip tilt projection on the pzt dm
    tau = np.linalg.inv(delta).dot(deltaT)

    # Famille generatrice sans tip tilt
    G = np.identity(n)
    tdt = tau.T.dot(delta).dot(tau)
    subTT = tau.dot(np.linalg.inv(tdt)).dot(tau.T).dot(delta)
    G -= subTT

    # Base orthonormee sans TT
    gdg = G.T.dot(delta).dot(G)
    U, s, V = np.linalg.svd(gdg)
    U = U[:, :U.shape[1] - 3]
    s = s[:s.size - 3]
    L = np.identity(s.size) / np.sqrt(s)
    B = G.dot(U).dot(L)

    # Rajout du TT
    TT = T.T.dot(T) / N  #.toarray()/N
    Btt = np.zeros((n + 2, n - 1))
    Btt[:B.shape[0], :B.shape[1]] = B
    mini = 1. / np.sqrt(TT)
    mini[0, 1] = 0
    mini[1, 0] = 0
    Btt[n:, n - 3:] = mini

    # Calcul du projecteur actus-->modes
    delta = np.zeros((n + T.shape[1], n + T.shape[1]))
    #IF = rtc.get_IFsparse(1).T
    delta[:-2, :-2] = IF.T.dot(IF).toarray() / N
    delta[-2:, -2:] = T.T.dot(T) / N
    P = Btt.T.dot(delta)

    return Btt.astype(np.float32), P.astype(np.float32)


def compute_cmatWithBtt(Btt, nfilt):
    D = rtc.get_imat(0)
    #D = ao.imat_geom(wfs,config.p_wfss,config.p_controllers[0],dms,config.p_dms,meth=0)
    # Filtering on Btt modes
    Btt_filt = np.zeros((Btt.shape[0], Btt.shape[1] - nfilt))
    Btt_filt[:, :Btt_filt.shape[1] - 2] = Btt[:, :Btt.shape[1] - (nfilt + 2)]
    Btt_filt[:, Btt_filt.shape[1] - 2:] = Btt[:, Btt.shape[1] - 2:]

    # Modal interaction basis
    Dm = D.dot(Btt_filt)
    # Direct inversion
    Dmp = np.linalg.inv(Dm.T.dot(Dm)).dot(Dm.T)
    # Command matrix
    cmat = Btt_filt.dot(Dmp)

    return Dm.astype(np.float32), cmat.astype(np.float32)


def compute_cmatWithBtt2(Btt, nfilt):
    D = rtc.get_imat(0)

    # Modal interaction basis
    Dm = D.dot(Btt)
    # Filtering on modal imat
    DmtDm = Dm.T.dot(Dm)
    U, s, V = np.linalg.svd(DmtDm)
    s = 1. / s
    s[s.shape[0] - nfilt - 2:s.shape[0] - 2] = 0.
    DmtDm1 = U.dot(np.diag(s)).dot(U.T)
    Dmp = DmtDm1.dot(Dm.T)
    # Command matrix
    cmat = Btt.dot(Dmp)

    return Dm.astype(np.float32), cmat.astype(np.float32)


###########################################################################################
#     ___                  _                    __                          _      _   _
#    / __|_____ ____ _ _ _(_)__ _ _ _  __ ___  / _|___   __ ___ _ _ _ _ ___| |__ _| |_(_)___ _ _
#   | (__/ _ \ V / _` | '_| / _` | ' \/ _/ -_) > _|_ _| / _/ _ \ '_| '_/ -_) / _` |  _| / _ \ ' \
#    \___\___/\_/\__,_|_| |_\__,_|_||_\__\___| \_____|  \__\___/_| |_| \___|_\__,_|\__|_\___/_||_|
#
###########################################################################################


def cov_cor(P, noise, trunc, alias, H, bp, tomo):
    cov = np.zeros((6, 6))
    bufdict = {
            "0": noise.T,
            "1": trunc.T,
            "2": alias.T,
            "3": H.T,
            "4": bp.T,
            "5": tomo.T
    }
    for i in range(cov.shape[0]):
        for j in range(cov.shape[1]):
            if (j >= i):
                tmpi = P.dot(bufdict[str(i)])
                tmpj = P.dot(bufdict[str(j)])
                cov[i, j] = np.sum(
                        np.mean(tmpi * tmpj, axis=1) -
                        np.mean(tmpi, axis=1) * np.mean(tmpj, axis=1))
            else:
                cov[i, j] = cov[j, i]

    s = np.reshape(np.diag(cov), (cov.shape[0], 1))
    sst = np.dot(s, s.T)
    cor = cov / np.sqrt(sst)

    return cov, cor


###########################################################################################
#  ___
# / __| __ ___ _____
# \__ \/ _` \ V / -_)
# |___/\__,_|\_/\___|
###########################################################################################


def save_it(filename):
    IF = rtc.get_IFsparse(1)
    TT = rtc.get_IFtt(1)
    noise_com = roket.getContributor("noise")
    trunc_com = roket.getContributor("nonlinear")
    alias_wfs_com = roket.getContributor("aliasing")
    H_com = roket.getContributor("filtered")
    bp_com = roket.getContributor("bandwidth")
    tomo_com = roket.getContributor("tomo")
    fit = roket.getContributor("fitting")

    tmp = (config.p_geom._ipupil.shape[0] -
           (config.p_dms[0]._n2 - config.p_dms[0]._n1 + 1)) / 2
    tmp_e0 = config.p_geom._ipupil.shape[0] - tmp
    tmp_e1 = config.p_geom._ipupil.shape[1] - tmp
    pup = config.p_geom._ipupil[tmp:tmp_e0, tmp:tmp_e1]
    indx_pup = np.where(pup.flatten() > 0)[0].astype(np.int32)
    dm_dim = config.p_dms[0]._n2 - config.p_dms[0]._n1 + 1
    cov, cor = cov_cor(P, noise_com, trunc_com, alias_wfs_com, H_com, bp_com, tomo_com)
    psf = tar.get_image(0, "le", fluxNorm=False)
    psfortho = roket.get_tar_imageortho()
    covv = roket.get_covv()
    covm = roket.get_covm()

    fname = "/home/fferreira/Data/" + filename
    pdict = {
            "noise": noise_com.T,
            "aliasing": alias_wfs_com.T,
            "tomography": tomo_com.T,
            "filtered modes": H_com.T,
            "non linearity": trunc_com.T,
            "bandwidth": bp_com.T,
            "P": P,
            "Btt": Btt,
            "IF.data": IF.data,
            "IF.indices": IF.indices,
            "IF.indptr": IF.indptr,
            "TT": TT,
            "dm_dim": dm_dim,
            "indx_pup": indx_pup,
            "fitting": fit,
            "SR": SR,
            "SR2": SR2,
            "cov": cov,
            "cor": cor,
            "psfortho": psfortho,
            "covm": covm,
            "covv": covv
    }
    h5u.save_h5(fname, "psf", config, psf)
    #h5u.writeHdf5SingleDataset(fname,com.T,datasetName="com")
    for k in list(pdict.keys()):
        h5u.save_hdf5(fname, k, pdict[k])


###############################################################################################
#  _            _
# | |_ ___  ___| |_ ___
# | __/ _ \/ __| __/ __|
# | ||  __/\__ \ |_\__ \
#  \__\___||___/\__|___/
###############################################################################################
nfiltered = config.p_controllers[0].maxcond
niters = config.p_loop.niter
#config.p_loop.set_niter(niters)
Btt, P = compute_btt()
rtc.load_Btt(1, Btt.dot(Btt.T))
Dm, cmat = compute_cmatWithBtt(Btt, nfiltered)
rtc.set_cmat(0, cmat)
R = rtc.get_cmat(0)
imat = rtc.get_imat(0)
RD = np.dot(R, imat).astype(np.float32)

gRD = (np.identity(RD.shape[0]) - config.p_controllers[0].gain * RD).astype(np.float32)
roket = ao.roket_init(rtc, wfs, tar, dms, tel, atm, 0, 1, Btt.shape[0], Btt.shape[1],
                      nfiltered, niters, Btt, P, gRD, RD)
#diagRD = np.diag(gRD)
#gRD = np.diag(diagRD)
#gRD=np.diag(gRD)

#imat_geom = ao.imat_geom(wfs,config.p_wfss,config.p_controllers[0],dms,config.p_dms,meth=0)
#RDgeom = np.dot(R,imat_geom)
preloop(1000)
SR, SR2 = loop(niters)

save_it(savename)
