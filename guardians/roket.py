"""
ROKET (erROr breaKdown Estimation Tool)

Computes the error breakdown during a COMPASS simulation
and saves it in a HDF5 file
Error contributors are bandwidth, tomography, noise, aliasing,
WFS non linearity, filtered modes and fitting
Saved file contained temporal buffers of those contributors
"""

import cProfile
import pstats as ps

import sys, os
import numpy as np
from shesha.supervisor.compassSupervisor import CompassSupervisor
from shesha.util.rtc_util import centroid_gain
from shesha.ao.tomo import create_nact_geom
from shesha.ao.basis import compute_btt, compute_cmat_with_Btt
import shesha.constants as scons
import time
import matplotlib.pyplot as pl
pl.ion()
import shesha.util.hdf5_util as h5u
import pandas
from scipy.sparse import csr_matrix


class Roket(CompassSupervisor):
    """
    ROKET class
    Inherits from CompassSupervisor
    """

    def __init__(self, str=None, N_preloop=1000, gamma=1.):
        """
        Initializes an instance of Roket class

        Args:
            str: (str): (optional) path to a parameter file
            N_preloop: (int): (optional) number of iterations before starting error breakdown estimation
            gamma: (float): (optional) centroid gain
        """
        super().__init__(str)
        self.N_preloop = N_preloop
        self.gamma = gamma

    def init_config(self):
        """
        Initializes the COMPASS simulation and the ROKET buffers
        """
        #super().init_config()
        self.iter_number = 0
        self.n = self.config.p_loop.niter + self.N_preloop
        #self.nfiltered = int(self.config.p_controllers[0].maxcond)
        self.nfiltered = 20
        #self.nactus = self.get_command(0).size
        self.nactus = self.rtc.get_command(0).size
        self.nslopes = self.rtc.get_slopes(0).size
        self.com = np.zeros((self.n, self.nactus), dtype=np.float32)
        self.noise_com = np.zeros((self.n, self.nactus), dtype=np.float32)
        self.noise_buf = np.zeros((self.n, self.nactus), dtype=np.float32)
        self.alias_wfs_com = np.copy(self.noise_com)
        self.ageom = np.copy(self.noise_com)
        self.alias_meas = np.zeros((self.n, self.nslopes), dtype=np.float32)
        self.wf_com = np.copy(self.noise_com)
        self.tomo_com = np.copy(self.noise_com)
        self.tomo_buf = np.copy(self.noise_com)
        self.trunc_com = np.copy(self.noise_com)
        self.trunc_buf = np.copy(self.noise_com)
        self.trunc_meas = np.copy(self.alias_meas)
        self.H_com = np.copy(self.noise_com)
        self.mod_com = np.copy(self.noise_com)
        self.bp_com = np.copy(self.noise_com)
        self.fit = np.zeros(self.n)
        self.psf_ortho = self.target.get_tar_image(0) * 0
        self.centroid_gain = 0
        self.centroid_gain2 = 0
        self.slopes = np.zeros((self.n, self.nslopes), dtype=np.float32)
        #gamma = 1.0
        self.config.p_loop.set_niter(self.n)
        #self.IFpzt = self.get_influ_basis_sparse(1)
        self.IFpzt = self.rtc._rtc.d_control[1].d_IFsparse.get_csr().astype(np.float32)
        #self.IFpzt.P = scipy.sparse.csr_matrix.transpose(self.IFpzt.get_csr())
        #self.TT = self.get_tt_influ_basis(1)
        self.TT = np.array(self.rtc._rtc.d_control[1].d_TT)

        self.Btt, self.P = compute_btt(self.IFpzt.T, self.TT)
        tmp = self.Btt.dot(self.Btt.T)
        self.rtc._rtc.d_control[1].load_Btt(tmp[:-2, :-2], tmp[-2:, -2:])
        compute_cmat_with_Btt(self.rtc._rtc, self.Btt, self.nfiltered)
        self.cmat = self.rtc.get_command_matrix(0)
        self.D = self.rtc.get_interaction_matrix(0)
        self.RD = np.dot(self.cmat, self.D)
        # self.gRD = np.identity(
        #         self.RD.
        #         shape[0]) - self.config.p_controllers[0].gain * self.gamma * self.RD
        self.gRD = self.config.p_controllers[0].gain * self.gamma * self.RD

        self.Nact = create_nact_geom(self.config.p_dms[0])
        self.delay = int(self.config.p_controllers[0].delay) + 1

    def loop_next(self, **kwargs):
        """
            function next
            Iterates the AO loop, with optional parameters

        :param move_atmos (bool): move the atmosphere for this iteration, default: True
        :param nControl (int): Controller number to use, default 0 (single control configurations)
        :param tar_trace (None or list[int]): list of targets to trace. None equivalent to all.
        :param wfs_trace (None or list[int]): list of WFS to trace. None equivalent to all.
        :param apply_control (bool): (optional) if True (default), apply control on DMs
        """
        self.next(apply_control=False)
        self.error_breakdown()
        self.rtc.apply_control(0)
        self.iter_number += 1

    def loop(self, monitoring_freq=100, **kwargs):
        """
        Performs the AO loop for n iterations

        Args:
            monitoring_freq: (int): (optional) Loop monitoring frequency [frames] in the terminal
        """
        print("-----------------------------------------------------------------")
        print("iter# | SE SR | LE SR | FIT SR | PH SR | ETR (s) | Framerate (Hz)")
        print("-----------------------------------------------------------------")
        t0 = time.time()
        for i in range(self.n):
            self.loop_next(**kwargs)
            if ((i + 1) % monitoring_freq == 0):
                framerate = (i + 1) / (time.time() - t0)
                # self.target.comp_tar_image(0)
                # self.target.comp_strehl(0)
                strehltmp = self.target.get_strehl(0)
                etr = (self.n - i) / framerate
                print("%d \t %.2f \t  %.2f\t %.2f \t %.2f \t    %.1f \t %.1f" %
                      (i + 1, strehltmp[0], strehltmp[1], np.exp(-strehltmp[2]),
                       np.exp(-strehltmp[3]), etr, framerate))
        t1 = time.time()

        print(" loop execution time:", t1 - t0, "  (", self.n, "iterations), ",
              (t1 - t0) / self.n, "(mean)  ", self.n / (t1 - t0), "Hz")

        #self.tar.comp_image(0)
        SRs = self.target.get_strehl(0)
        self.SR2 = np.exp(-SRs[3])
        self.SR = SRs[1]

    def error_breakdown(self):
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
        """
        g = self.config.p_controllers[0].gain
        Dcom = self.rtc.get_command(0)
        Derr = self.rtc.get_err(0)
        self.com[self.iter_number, :] = Dcom
        tarphase = self.target.get_tar_phase(0)
        self.slopes[self.iter_number, :] = self.rtc.get_slopes(0)
        
        ###########################################################################
        ## Noise contribution
        ###########################################################################
        if (self.config.p_wfss[0].type == scons.WFSType.SH):
            ideal_img = np.array(self.wfs._wfs.d_wfs[0].d_binimg_notnoisy)
            binimg = np.array(self.wfs._wfs.d_wfs[0].d_binimg)
            if (self.config.p_centroiders[0].type == scons.CentroiderType.TCOG
                ):  # Select the same pixels with or without noise
                invalidpix = np.where(binimg <= self.config.p_centroiders[0].thresh)
                ideal_img[invalidpix] = 0
                self.rtc.set_centroider_threshold(0, -1e16)
            self.wfs._wfs.d_wfs[0].set_binimg(ideal_img, ideal_img.size)
        elif (self.config.p_wfss[0].type == scons.centroiderType.PYRHR):
            ideal_pyrimg = np.array(self.wfs._wfs.d_wfs[0].d_binimg_notnoisy)
            self.wfs._wfs.d_wfs[0].set_binimg(ideal_pyrimg, ideal_pyrimg.size)

        self.rtc.do_centroids(0)
        if (self.config.p_centroiders[0].type == scons.CentroiderType.TCOG):
            self.rtc.set_centroider_threshold(0, config.p_centroiders[0].thresh)

        self.rtc.do_control(0)
        E = self.rtc.get_err(0)
        E_meas = self.rtc.get_slopes(0)
        self.noise_buf[self.iter_number, :] = (Derr - E)
        # Apply loop filter to get contribution of noise on commands
        # if (self.iter_number + 1 < self.config.p_loop.niter):
        self.noise_com[self.iter_number, :] = self.noise_com[self.iter_number - 1, :] - self.gRD.dot(
                    self.noise_com[self.iter_number - self.delay, :]) + g * self.noise_buf[self.iter_number - self.delay, :]
        ###########################################################################
        ## Sampling/truncature contribution
        ###########################################################################
        self.rtc.do_centroids_geom(0)
        self.rtc.do_control(0)
        F = self.rtc.get_err(0)
        F_meas = self.rtc.get_slopes(0)
        self.trunc_meas[self.iter_number, :] = E_meas - F_meas
        self.trunc_buf[self.iter_number, :] = (E - self.gamma * F)
        # Apply loop filter to get contribution of sampling/truncature on commands
        #if (self.iter_number + 1 < self.config.p_loop.niter):
        self.trunc_com[self.iter_number, :] = self.trunc_com[self.iter_number - 1, :] - self.gRD.dot(
                    self.trunc_com[self.iter_number - self.delay, :]) + g * self.trunc_buf[self.iter_number - self.delay, :]
        self.centroid_gain += centroid_gain(E, F)
        #Derr = np.ones(Derr)
        #print(Derr)
        #print(F)
        self.centroid_gain2 += centroid_gain(Derr, F)
        ###########################################################################
        ## Aliasing contribution on WFS direction
        ###########################################################################
        #self.rtc.do_control(1, 0, wfs_direction=True)
        self.rtc.do_control(1, sources=self.wfs.sources, is_wfs_phase=True)
        #self.rtc.do_control(0)
        #self.rtc.do_control(1)
        self.rtc.apply_control(1)
        for w in range(len(self.config.p_wfss)):
            self.wfs.raytrace(w, dms=self.dms, reset=False)
        """
            wfs.sensors_compimg(0)
        if(config.p_wfss[0].type == scons.WFSType.SH):
            ideal_img = wfs.get_binimgNotNoisy(0)
            binimg = wfs.get_binimg(0)
            if(config.p_centroiders[0].type == scons.CentroiderType.TCOG): # Select the same pixels with or without noise
                invalidpix = np.where(binimg <= config.p_centroiders[0].thresh)
                ideal_img[self.iter_numbernvalidpix] = 0
                rtc.setthresh(0,-1e16)
            wfs.set_binimg(0,ideal_img)
        elif(config.p_wfss[0].type == scons.centroiderType.PYRHR):
            ideal_pyrimg = wfs.get_binimg_notnoisy(0)
            wfs.set_pyrimg(0,ideal_pyrimg)
        """
        self.rtc.do_centroids_geom(0)
        self.rtc.do_control(0)
        self.ageom[self.iter_number, :] = self.rtc.get_err(0)
        self.alias_meas[self.iter_number, :] = self.rtc.get_slopes(0)
        # if (self.iter_number + 1 < self.config.p_loop.niter):
        self.alias_wfs_com[self.iter_number, :] = self.alias_wfs_com[self.iter_number - 1, :] - self.gRD.dot(
                    self.alias_wfs_com[self.iter_number - self.delay, :]) + self.gamma * g * self.ageom[self.iter_number - self.delay, :]# - (E-F))

        ###########################################################################
        ## Wavefront + filtered modes reconstruction
        ###########################################################################
        self.target.raytrace(0, atm=self.atmos, ncpa=False)
        #self.rtc.do_control(1, 0, wfs_direction=False)
        self.rtc.do_control(1, sources=self.target.sources, is_wfs_phase=False)
        #self.rtc.do_control(0)
        #self.rtc.do_control(1)
        B = self.rtc.get_command(1)

        ###########################################################################
        ## Fitting
        ###########################################################################
        self.rtc.apply_control(1)
        self.target.raytrace(0, dms=self.dms, ncpa=False, reset=False, comp_avg_var=False)
        self.target.comp_tar_image(0, compLE=False)
        self.target.comp_strehl(0)
        self.fit[self.iter_number] = self.target.get_strehl(0)[2]
        if (self.iter_number >= self.N_preloop):
            self.psf_ortho += self.target.get_tar_image(0, expo_type='se')

        ###########################################################################
        ## Filtered modes error & Commanded modes
        ###########################################################################
        modes = self.P.dot(B)
        modes_filt = modes.copy() * 0.
        modes_filt[-self.nfiltered - 2:-2] = modes[-self.nfiltered - 2:-2]
        self.H_com[self.iter_number, :] = self.Btt.dot(modes_filt)
        modes[-self.nfiltered - 2:-2] = 0
        self.mod_com[self.iter_number, :] = self.Btt.dot(modes)

        ###########################################################################
        ## Bandwidth error
        ###########################################################################
        C = self.mod_com[self.iter_number, :] - self.mod_com[self.iter_number - 1, :]

        self.bp_com[self.iter_number, :] = self.bp_com[self.iter_number - 1, :] - self.gRD.dot(
                self.bp_com[self.iter_number - self.delay, :]) - C

        ###########################################################################
        ## Tomographic error
        ###########################################################################
        #G = F - (mod_com[self.iter_number,:] + Ageom - np.dot(RDgeom,com[self.iter_number-1,:]))
        for w in range(len(self.config.p_wfss)):
            self.wfs.raytrace(w, atm=self.atmos)

        #self.rtc.do_control(1, 0, wfs_direction=True)
        self.rtc.do_control(1, sources=self.wfs.sources, is_wfs_phase=True)
        #self.rtc.do_control(0)
        #self.rtc.do_control(1)
        G = self.rtc.get_command(1)
        modes = self.P.dot(G)
        modes[-self.nfiltered - 2:-2] = 0
        self.wf_com[self.iter_number, :] = self.Btt.dot(modes)

        self.tomo_buf[self.iter_number, :] = self.mod_com[self.iter_number, :] - self.wf_com[self.iter_number, :]
        # if (self.iter_number + 1 < self.config.p_loop.niter):
        self.tomo_com[self.iter_number, :] = self.tomo_com[self.iter_number - 1, :] - self.gRD.dot(
                    self.tomo_com[self.iter_number - self.delay, :]) - g * self.gamma * self.RD.dot(self.tomo_buf[self.iter_number - self.delay, :])

        # Without anyone noticing...
        #self._sim.tar.d_targets[0].set_phase(tarphase)
        self.target.set_tar_phase(0, tarphase)
        #self._sim.rtc.d_control[0].set_com(Dcom, Dcom.size)
        self.rtc.set_command(0, Dcom)

    def save_in_hdf5(self, savename):
        """
        Saves all the ROKET buffers + simuation parameters in a HDF5 file

        Args:
            savename: (str): name of the output file
        """
        tmp = (self.config.p_geom._ipupil.shape[0] -
               (self.config.p_dms[0]._n2 - self.config.p_dms[0]._n1 + 1)) // 2
        tmp_e0 = self.config.p_geom._ipupil.shape[0] - tmp
        tmp_e1 = self.config.p_geom._ipupil.shape[1] - tmp
        pup = self.config.p_geom._ipupil[tmp:tmp_e0, tmp:tmp_e1]
        indx_pup = np.where(pup.flatten() > 0)[0].astype(np.int32)
        dm_dim = self.config.p_dms[0]._n2 - self.config.p_dms[0]._n1 + 1
        self.cov_cor()
        psf = self.target.get_tar_image(0, expo_type='le')
        if(os.getenv("DATA_GUARDIAN") is not None):
                fname = os.getenv("DATA_GUARDIAN") + "/" + savename
        else:
                fname = savename
        #fname = "test"
        pdict = {
                "noise":
                        self.noise_com[self.N_preloop:, :].T,
                "aliasing":
                        self.alias_wfs_com[self.N_preloop:, :].T,
                "tomography":
                        self.tomo_com[self.N_preloop:, :].T,
                "filtered modes":
                        self.H_com[self.N_preloop:, :].T,
                "non linearity":
                        self.trunc_com[self.N_preloop:, :].T,
                "bandwidth":
                        self.bp_com[self.N_preloop:, :].T,
                "wf_com":
                        self.wf_com[self.N_preloop:, :].T,
                "P":
                        self.P,
                "Btt":
                        self.Btt,
                "IF.data":
                        self.IFpzt.data,
                "IF.indices":
                        self.IFpzt.indices,
                "IF.indptr":
                        self.IFpzt.indptr,
                "TT":
                        self.TT,
                "dm_dim":
                        dm_dim,
                "indx_pup":
                        indx_pup,
                "fitting":
                        np.mean(self.fit[self.N_preloop:]),
                "SR":
                        self.SR,
                "SR2":
                        self.SR2,
                "cov":
                        self.cov,
                "cor":
                        self.cor,
                "psfortho":
                        np.fft.fftshift(self.psf_ortho) /
                        (self.config.p_loop.niter - self.N_preloop),
                "centroid_gain":
                        self.centroid_gain / (self.config.p_loop.niter - self.N_preloop),
                "centroid_gain2":
                        self.centroid_gain2 /
                        (self.config.p_loop.niter - self.N_preloop),
                "dm.xpos":
                        self.config.p_dms[0]._xpos,
                "dm.ypos":
                        self.config.p_dms[0]._ypos,
                "R":
                        self.rtc.get_command_matrix(0),
                "D":
                        self.rtc.get_interaction_matrix(0),
                "Nact":
                        self.Nact,
                "com":
                        self.com[self.N_preloop:, :].T,
                "slopes":
                        self.slopes[self.N_preloop:, :].T,
                "alias_meas":
                        self.alias_meas[self.N_preloop:, :].T,
                "trunc_meas":
                        self.trunc_meas[self.N_preloop:, :].T
        }
        h5u.save_h5(fname, "psf", self.config, psf)
        for k in list(pdict.keys()):
            h5u.save_hdf5(fname, k, pdict[k])

    def cov_cor(self):
        """
        Computes covariance matrix and correlation matrix between all the contributors
        """
        self.cov = np.zeros((6, 6))
        self.cor = np.zeros((6, 6))
        bufdict = {
                "0": self.noise_com.T,
                "1": self.trunc_com.T,
                "2": self.alias_wfs_com.T,
                "3": self.H_com.T,
                "4": self.bp_com.T,
                "5": self.tomo_com.T
        }
        for i in range(self.cov.shape[0]):
            for j in range(self.cov.shape[1]):
                if (j >= i):
                    tmpi = self.P.dot(bufdict[str(i)])
                    tmpj = self.P.dot(bufdict[str(j)])
                    self.cov[i, j] = np.sum(
                            np.mean(tmpi * tmpj, axis=1) -
                            np.mean(tmpi, axis=1) * np.mean(tmpj, axis=1))
                else:
                    self.cov[i, j] = self.cov[j, i]

        s = np.reshape(np.diag(self.cov), (self.cov.shape[0], 1))
        sst = np.dot(s, s.T)
        ok = np.where(sst)
        self.cor[ok] = self.cov[ok] / np.sqrt(sst[ok])


###############################################################################
#
#                                 MAIN
#
###############################################################################
if __name__ == "__main__":
    from shesha.config import ParamConfig

    if (len(sys.argv) < 2):
        error = 'command line should be at least:"python -i test.py parameters_filename"\n with "parameters_filename" the path to the parameters file'
        raise Exception(error)

    #get parameters from file
    param_file = sys.argv[1]

    if (len(sys.argv) > 2):
        savename = sys.argv[2]
    else:
        savename = "roket_default.h5"
    config = ParamConfig(param_file)
    roket = Roket(config)
    roket.init_config()
    roket.loop()
    roket.save_in_hdf5(savename)
