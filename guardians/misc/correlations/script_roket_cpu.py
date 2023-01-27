"""
Created on Wed Apr 27 09:28:23 2016

@author: fferreira
"""

import cProfile
import pstats as ps

import sys, os
import numpy as np
import carmaWrap as ch
import shesha as ao
import time
import matplotlib.pyplot as pl
pl.ion()
import hdf5_util as h5u
import pandas
from scipy.sparse import csr_matrix

c = ch.carmaWrap_context(devices=np.array([6], dtype=np.int32))

############################################################################
#  _       _ _
# (_)_ __ (_) |_ ___
# | | '_ \| | __/ __|
# | | | | | | |_\__ \
# |_|_| |_|_|\__|___/
############################################################################


def init_config(config):
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

    #c=ch.carmaWrap_context(devices=np.array([4,5,6,7], dtype=np.int32))
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

    return atm, wfs, tel, dms, tar, rtc


##############################################################################
#    _   ___    _
#   /_\ / _ \  | |___  ___ _ __
#  / _ \ (_) | | / _ \/ _ \ '_ \
# /_/ \_\___/  |_\___/\___/ .__/
#                         |_|
##############################################################################
def loop(config, n):
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
        psf_ortho = tar.get_image(0, 'se') * 0.

    t0 = time.time()
    for i in range(n):
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

            if (error_flag):
                #compute the error breakdown for this iteration
                error_breakdown(com, noise_com, alias_wfs_com, tomo_com, H_com,
                                trunc_com, bp_com, wf_com, mod_com, fit, psf_ortho, i)

            rtc.applycontrol(0, dms)

        if ((i + 1) % 10 == 0 and i > -1):
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
        return com, noise_com, alias_wfs_com, tomo_com, H_com, trunc_com, bp_com, mod_com, np.mean(
                fit[N_preloop:]), SR, SR2, psf_ortho


def preloop(config, n):
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


###################################################################################
#  ___                   ___              _      _
# | __|_ _ _ _ ___ _ _  | _ )_ _ ___ __ _| |____| |_____ __ ___ _
# | _|| '_| '_/ _ \ '_| | _ \ '_/ -_) _` | / / _` / _ \ V  V / ' \
# |___|_| |_| \___/_|   |___/_| \___\__,_|_\_\__,_\___/\_/\_/|_||_|
###################################################################################
def error_breakdown(config, com, noise_com, alias_wfs_com, tomo_com, H_com, trunc_com, bp_com,
                    wf_com, mod_com, fit, psf_ortho, i):
    """
    Compute the error breakdown of the AO simulation. Returns the error commands of
    each contributors. Suppose no delay (for now) and only 2 controllers : the main one, controller #0, (specified on the parameter file)
    and the geometric one, controller #1 (automatically added if roket is asked in the parameter file)
    Commands are computed by applying the loop filter on various kind of commands : (see schema_simulation_budget_erreur_v2)

        - Ageom : Aliasing contribution on WFS direction
            Obtained by computing commands from DM orthogonal phase (projection + slopes_geom)

        - B : Projection on the target direction
            Obtained as the commmands output of the geometric controller

        - C : Wavefront
            Obtained by computing commands from DM parallel phase (RD*B)

        - E : Wavefront + aliasing + ech/trunc + tomo
            Obtained by performing the AO loop iteration without noise on the WFS

        - F : Wavefront + aliasing + tomo
            Obtained by performing the AO loop iteration without noise on the WFS and using phase deriving slopes

        - G : tomo

    Note : rtc.get_err returns to -CMAT.slopes

    Args:
        noise_com : np.array((niter,nactu)) : Noise contribution
            Computed with com-E

        alias_wfs_com : np.array((niter,nactu)) : Aliasing on WFS direction contribution
            Computed with Ageom

        tomo_com : np.array((niter,nactu)) : Tomographic error contribution
            Computed with C-B

        H_com : np.array((niter,nactu)) : Filtered modes error
            Computed with B

        trunc_com : np.array((niter,nactu)) : sampling/truncature error contribution
            Computed with E-F

        bp_com : np.array((niter,nactu)) : Bandwidth error

        wf_com : np.array((niter,nactu)) : Reconstructed wavefront

        mod_com : np.array((niter,nactu)) : commanded modes

        fit : np.array((niter)) : fitting value

        i : (int) : current iteration number

    """
    g = config.p_controllers[0].gain
    Dcom = rtc.get_command(0)
    Derr = rtc.get_err(0)
    com[i, :] = Dcom
    tarphase = tar.get_phase(0)
    ###########################################################################
    ## Noise contribution
    ###########################################################################
    if (config.p_wfss[0].type == b"sh"):
        ideal_bincube = wfs.get_bincubeNotNoisy(0)
        bincube = wfs.get_bincube(0)
        if (config.p_centroiders[0].type == b"tcog"
            ):  # Select the same pixels with or without noise
            invalidpix = np.where(bincube <= config.p_centroiders[0].thresh)
            ideal_bincube[invalidpix] = 0
            rtc.setthresh(0, -1e16)
        wfs.set_bincube(0, ideal_bincube)
    elif (config.p_wfss[0].type == b"pyrhr"):
        ideal_pyrimg = wfs.get_binimg_notnoisy(0)
        wfs.set_pyrimg(0, ideal_pyrimg)

    rtc.docentroids(0)
    if (config.p_centroiders[0].type == b"tcog"):
        rtc.setthresh(0, config.p_centroiders[0].thresh)

    rtc.docontrol(0)
    E = rtc.get_err(0)
    # Apply loop filter to get contribution of noise on commands
    if (i + 1 < config.p_loop.niter):
        noise_com[i + 1, :] = gRD.dot(noise_com[i, :]) + g * (Derr - E)

    ###########################################################################
    ## Sampling/truncature contribution
    ###########################################################################
    rtc.docentroids_geom(0)
    rtc.docontrol(0)
    F = rtc.get_err(0)
    # Apply loop filter to get contribution of sampling/truncature on commands
    if (i + 1 < config.p_loop.niter):
        trunc_com[i + 1, :] = gRD.dot(trunc_com[i, :]) + g * (E - F)

    ###########################################################################
    ## Aliasing contribution on WFS direction
    ###########################################################################
    rtc.docontrol_geo_onwfs(1, dms, wfs, 0)
    rtc.applycontrol(1, dms)
    for w in range(len(config.p_wfss)):
        wfs.sensors_trace(w, "dm", tel, atm, dms)
    """
        wfs.sensors_compimg(0)
    if(config.p_wfss[0].type == b"sh"):
        ideal_bincube = wfs.get_bincubeNotNoisy(0)
        bincube = wfs.get_bincube(0)
        if(config.p_centroiders[0].type == b"tcog"): # Select the same pixels with or without noise
            invalidpix = np.where(bincube <= config.p_centroiders[0].thresh)
            ideal_bincube[invalidpix] = 0
            rtc.setthresh(0,-1e16)
        wfs.set_bincube(0,ideal_bincube)
    elif(config.p_wfss[0].type == b"pyrhr"):
        ideal_pyrimg = wfs.get_binimg_notnoisy(0)
        wfs.set_pyrimg(0,ideal_pyrimg)
    """
    rtc.docentroids_geom(0)
    rtc.docontrol(0)
    Ageom = rtc.get_err(0)
    if (i + 1 < config.p_loop.niter):
        alias_wfs_com[i + 1, :] = gRD.dot(alias_wfs_com[i, :]) + g * (Ageom)

    ###########################################################################
    ## Wavefront + filtered modes reconstruction
    ###########################################################################
    tar.atmos_trace(0, atm, tel)
    rtc.docontrol_geo(1, dms, tar, 0)
    B = rtc.get_command(1)

    ###########################################################################
    ## Fitting
    ###########################################################################
    rtc.applycontrol(1, dms)
    tar.dmtrace(0, dms, do_phase_var=0)
    fit[i] = tar.get_strehl(0, comp_strehl=False)[2]
    if (i >= N_preloop):
        psf_ortho += tar.get_image(0, 'se') / niters

    ###########################################################################
    ## Filtered modes error & Commanded modes
    ###########################################################################
    modes = P.dot(B)
    modes_filt = modes.copy() * 0.
    modes_filt[-nfiltered - 2:-2] = modes[-nfiltered - 2:-2]
    H_com[i, :] = Btt.dot(modes_filt)
    modes[-nfiltered - 2:-2] = 0
    mod_com[i, :] = Btt.dot(modes)

    ###########################################################################
    ## Bandwidth error
    ###########################################################################
    C = mod_com[i, :] - mod_com[i - 1, :]

    bp_com[i, :] = gRD.dot(bp_com[i - 1, :]) - C

    ###########################################################################
    ## Tomographic error
    ###########################################################################
    #G = F - (mod_com[i,:] + Ageom - np.dot(RDgeom,com[i-1,:]))
    for w in range(len(config.p_wfss)):
        wfs.sensors_trace(w, "atmos", tel, atm, dms)
    rtc.docontrol_geo_onwfs(1, dms, wfs, 0)
    G = rtc.get_command(1)
    modes = P.dot(G)
    modes[-nfiltered - 2:-2] = 0
    wf_com[i, :] = Btt.dot(modes)

    G = mod_com[i, :] - wf_com[i, :]
    if (i + 1 < config.p_loop.niter):
        tomo_com[i + 1, :] = gRD.dot(tomo_com[i, :]) - g * RD.dot(G)

    # Without anyone noticing...
    tar.set_phase(0, tarphase)
    rtc.setCom(0, Dcom)


################################################################################
#  ___          _
# | _ ) __ _ __(_)___
# | _ \/ _` (_-< (_-<
# |___/\__,_/__/_/__/
################################################################################
def compute_btt2():
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


def compute_btt():
    IF = rtc.get_IFsparse(1).T
    N = IF.shape[0]
    n = IF.shape[1]
    T = IF[:, -2:].copy()
    IF = IF[:, :n - 2]
    n = IF.shape[1]

    delta = IF.T.dot(IF).toarray() / N

    # Tip-tilt + piston
    Tp = np.ones((T.shape[0], T.shape[1] + 1))
    Tp[:, :2] = T.toarray()
    deltaT = IF.T.dot(Tp) / N
    # Tip tilt projection on the pzt dm
    tau = np.linalg.inv(delta).dot(deltaT)

    # Famille génératrice sans tip tilt
    G = np.identity(n)
    tdt = tau.T.dot(delta).dot(tau)
    subTT = tau.dot(np.linalg.inv(tdt)).dot(tau.T).dot(delta)
    G -= subTT

    # Base orthonormée sans TT
    gdg = G.T.dot(delta).dot(G)
    U, s, V = np.linalg.svd(gdg)
    U = U[:, :U.shape[1] - 3]
    s = s[:s.size - 3]
    L = np.identity(s.size) / np.sqrt(s)
    B = G.dot(U).dot(L)

    # Rajout du TT
    TT = T.T.dot(T).toarray() / N
    Btt = np.zeros((n + 2, n - 1))
    Btt[:B.shape[0], :B.shape[1]] = B
    mini = 1. / np.sqrt(TT)
    mini[0, 1] = 0
    mini[1, 0] = 0
    Btt[n:, n - 3:] = mini

    # Calcul du projecteur actus-->modes
    IF = rtc.get_IFsparse(1).T
    delta = IF.T.dot(IF).toarray() / N
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


def save_it(config, filename):
    IF = rtc.get_IFsparse(1)
    TT = rtc.get_IFtt(1)

    tmp = (config.p_geom._ipupil.shape[0] -
           (config.p_dms[0]._n2 - config.p_dms[0]._n1 + 1)) / 2
    tmp_e0 = config.p_geom._ipupil.shape[0] - tmp
    tmp_e1 = config.p_geom._ipupil.shape[1] - tmp
    pup = config.p_geom._ipupil[tmp:tmp_e0, tmp:tmp_e1]
    indx_pup = np.where(pup.flatten() > 0)[0].astype(np.int32)
    dm_dim = config.p_dms[0]._n2 - config.p_dms[0]._n1 + 1
    cov, cor = cov_cor(P, noise_com, trunc_com, alias_wfs_com, H_com, bp_com, tomo_com)
    psf = tar.get_image(0, "le", fluxNorm=False)

    fname = "/home/fferreira/Data/" + filename
    pdict = {
            "noise": noise_com.T,
            "aliasing": alias_wfs_com.T,
            "tomography": tomo_com.T,
            "filtered modes": H_com.T,
            "non linearity": trunc_com.T,
            "bandwidth": bp_com.T,
            "wf_com": wf_com.T,
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
            "psfortho": np.fft.fftshift(psf_ortho),
            "dm.xpos": config.p_dms[0]._xpos,
            "dm.ypos": config.p_dms[0]._ypos
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
param_file = "/home/fferreira/compass/trunk/shesha/data/par/par4roket/correlation_study/roket_8m_1layer.py"
error_flag = True
config = None
if (param_file.split('.')[-1] == b"py"):
    filename = param_file.split('/')[-1]
    param_path = param_file.split(filename)[0]
    sys.path.insert(0, param_path)
    exec("import %s as config" % filename.split(".py")[0])
    #sys.path.remove(param_path)
nfiltered = 20
N_preloop = 1000
niters = config.p_loop.niter
config.p_loop.set_niter(niters + N_preloop)
winddirs = [0, 45, 90, 135, 180]
windspeeds = [5., 10., 15., 20.]

d = float(sys.argv[1])
s = float(sys.argv[2])
g = float(sys.argv[3])

savename = "roket_8m_1layer_dir%d_speed%d_g%d_cpu.h5" % (d, s, g * 10)
config.p_atmos.set_winddir([d])
config.p_atmos.set_windspeed([s])
config.p_controllers[0].set_gain(g)

atm, wfs, tel, dms, tar, rtc = init_config(config)
#config.p_loop.set_niter(niters)
Btt, P = compute_btt2()
rtc.load_Btt(1, Btt.dot(Btt.T))
Dm, cmat = compute_cmatWithBtt(Btt, nfiltered)
rtc.set_cmat(0, cmat)
R = rtc.get_cmat(0)
imat = rtc.get_imat(0)
RD = np.dot(R, imat).astype(np.float32)
gRD = (np.identity(RD.shape[0]) - config.p_controllers[0].gain * RD).astype(np.float32)

com, noise_com, alias_wfs_com, tomo_com, H_com, trunc_com, bp_com, wf_com, fit, SR, SR2, psf_ortho = loop( config, 
        niters + N_preloop)
noise_com = noise_com[N_preloop:, :]
trunc_com = trunc_com[N_preloop:, :]
alias_wfs_com = alias_wfs_com[N_preloop:, :]
H_com = H_com[N_preloop:, :]
bp_com = bp_com[N_preloop:, :]
tomo_com = tomo_com[N_preloop:, :]
save_it(config, savename)
