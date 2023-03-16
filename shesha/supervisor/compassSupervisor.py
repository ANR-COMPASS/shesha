## @package   shesha.supervisor.compassSupervisor
## @brief     Initialization and execution of a COMPASS supervisor
## @author    COMPASS Team <https://github.com/ANR-COMPASS>
## @version   5.4.1
## @date      2022/01/24
## @copyright GNU Lesser General Public License
#
#  This file is part of COMPASS <https://anr-compass.github.io/compass/>
#
#  Copyright (C) 2011-2023 COMPASS Team <https://github.com/ANR-COMPASS>
#  All rights reserved.
#  Distributed under GNU - LGPL
#
#  COMPASS is free software: you can redistribute it and/or modify it under the terms of the GNU Lesser
#  General Public License as published by the Free Software Foundation, either version 3 of the License,
#  or any later version.
#
#  COMPASS: End-to-end AO simulation tool using GPU acceleration
#  The COMPASS platform was designed to meet the need of high-performance for the simulation of AO systems.
#
#  The final product includes a software package for simulating all the critical subcomponents of AO,
#  particularly in the context of the ELT and a real-time core based on several control approaches,
#  with performances consistent with its integration into an instrument. Taking advantage of the specific
#  hardware architecture of the GPU, the COMPASS tool allows to achieve adequate execution speeds to
#  conduct large simulation campaigns called to the ELT.
#
#  The COMPASS platform can be used to carry a wide variety of simulations to both testspecific components
#  of AO of the E-ELT (such as wavefront analysis device with a pyramid or elongated Laser star), and
#  various systems configurations such as multi-conjugate AO.
#
#  COMPASS is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the
#  implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
#  See the GNU Lesser General Public License for more details.
#
#  You should have received a copy of the GNU Lesser General Public License along with COMPASS.
#  If not, see <https://www.gnu.org/licenses/lgpl-3.0.txt>.

from shesha.supervisor.genericSupervisor import GenericSupervisor
from shesha.supervisor.components import AtmosCompass, DmCompass, RtcCompass, TargetCompass, TelescopeCompass, WfsCompass, CoronagraphCompass
from shesha.supervisor.optimizers import ModalBasis, Calibration, ModalGains
import numpy as np
import time

import shesha.constants as scons

from typing import Iterable


class CompassSupervisor(GenericSupervisor):
    """ This class implements generic supervisor to handle compass simulation

    Attributes inherited from GenericSupervisor:
        context : (CarmaContext) : a CarmaContext instance

        config : (config) : Parameters structure

        is_init : (bool) : Flag equals to True if the supervisor has already been initialized

        iter : (int) : Frame counter

    Attributes:
        telescope : (TelescopeComponent) : a TelescopeComponent instance

        atmos : (AtmosComponent) : An AtmosComponent instance

        target : (targetComponent) : A TargetComponent instance

        wfs : (WfsComponent) : A WfsComponent instance

        dms : (DmComponent) : A DmComponent instance

        rtc : (RtcComponent) : A Rtc component instance

        cacao : (bool) : CACAO features enabled in the RTC

        silence_tqdm : (bool) : Silence tqdm's output

        basis : (ModalBasis) : a ModalBasis instance (optimizer)

        calibration : (Calibration) : a Calibration instance (optimizer)

        modalgains : (ModalGains) : a ModalGain instance (optimizer) using CLOSE algorithm

        close_modal_gains : (list of floats) : list of the previous values of the modal gains
    """

    def __init__(self, config, *, cacao: bool = False, silence_tqdm: bool = False):
        """ Instantiates a CompassSupervisor object

        Args:
            config: (config module) : Configuration module

        Kwargs:
            cacao : (bool) : If True, enables CACAO features in RTC (Default is False)
                                      Requires OCTOPUS to be installed
        """
        self.cacao = cacao
        self.tel = None
        self.atmos = None
        self.target = None
        self.wfs = None
        self.dms = None
        self.rtc = None
        self.corono = None
        
        GenericSupervisor.__init__(self, config, silence_tqdm=silence_tqdm)
        self.basis = ModalBasis(self.config, self.dms, self.target)
        self.calibration = Calibration(self.config, self.tel, self.atmos, self.dms,
                                       self.target, self.rtc, self.wfs)
        if config.p_controllers is not None:
            self.modalgains = ModalGains(self.config, self.rtc)
        self.close_modal_gains = []

#     ___                  _      __  __     _   _            _
#    / __|___ _ _  ___ _ _(_)__  |  \/  |___| |_| |_  ___  __| |___
#   | (_ / -_) ' \/ -_) '_| / _| | |\/| / -_)  _| ' \/ _ \/ _` (_-<
#    \___\___|_||_\___|_| |_\__| |_|  |_\___|\__|_||_\___/\__,_/__/

    def _init_tel(self):
        """Initialize the telescope component of the supervisor as a TelescopeCompass
        """
        self.tel = TelescopeCompass(self.context, self.config)

    def _init_atmos(self):
        """Initialize the atmosphere component of the supervisor as a AtmosCompass
        """
        self.atmos = AtmosCompass(self.context, self.config, silence_tqdm=self.silence_tqdm)

    def _init_dms(self):
        """Initialize the DM component of the supervisor as a DmCompass
        """
        self.dms = DmCompass(self.context, self.config, silence_tqdm=self.silence_tqdm)

    def _init_target(self):
        """Initialize the target component of the supervisor as a TargetCompass
        """
        if self.tel is not None:
            self.target = TargetCompass(self.context, self.config, self.tel)
        else:
            raise ValueError("Configuration not loaded or Telescope not initilaized")

    def _init_wfs(self):
        """Initialize the wfs component of the supervisor as a WfsCompass
        """
        if self.tel is not None:
            self.wfs = WfsCompass(self.context, self.config, self.tel)
        else:
            raise ValueError("Configuration not loaded or Telescope not initilaized")

    def _init_rtc(self):
        """Initialize the rtc component of the supervisor as a RtcCompass
        """
        if self.wfs is not None:
            self.rtc = RtcCompass(self.context, self.config, self.tel, self.wfs,
                                  self.dms, self.atmos, cacao=self.cacao, silence_tqdm=self.silence_tqdm)
        else:
            raise ValueError("Configuration not loaded or Telescope not initilaized")

    def _init_components(self) -> None:
        """ Initialize all the components
        """

        if self.config.p_tel is None or self.config.p_geom is None:
            raise ValueError("Telescope geometry must be defined (p_geom and p_tel)")
        self._init_tel()

        if self.config.p_atmos is not None:
            self._init_atmos()
        if self.config.p_dms is not None:
            self._init_dms()
        if self.config.p_targets is not None:
            self._init_target()
        if self.config.p_wfss is not None:
            self._init_wfs()
        if self.config.p_controllers is not None or self.config.p_centroiders is not None:
            self._init_rtc()
        if self.config.p_coronos is not None:
            self._init_coronagraph()

        GenericSupervisor._init_components(self)

    def _init_coronagraph(self):
        """ Initialize the coronagraph
        """
        self.corono = CoronagraphCompass()
        for p_corono in self.config.p_coronos:
            self.corono.add_corono(self.context, p_corono, self.config.p_geom, self.target)

    def next(self, *, move_atmos: bool = True, nControl: int = 0,
             tar_trace: Iterable[int] = None, wfs_trace: Iterable[int] = None,
             do_control: bool = True, apply_control: bool = True,
             compute_tar_psf: bool = True, compute_corono: bool=True) -> None:
        """Iterates the AO loop, with optional parameters.

        Overload the GenericSupervisor next() method to handle the GEO controller
        specific raytrace order operations

        Kwargs:
            move_atmos: (bool): move the atmosphere for this iteration. Default is True

            nControl: (int): Controller number to use. Default is 0 (single control configuration)

            tar_trace: (List): list of targets to trace. None is equivalent to all (default)

            wfs_trace: (List): list of WFS to trace. None is equivalent to all (default)

            do_control : (bool) : Performs RTC operations if True (Default)

            apply_control: (bool): if True (default), apply control on DMs

            compute_tar_psf : (bool) : If True (default), computes the PSF at the end of the iteration

            compute_corono: (bool): If True (default), computes the coronagraphic image
        """
        try:
            iter(nControl)
        except TypeError:
            # nControl is not an iterable creating a list
            nControl = [nControl]

        #get the index of the first GEO controller (-1 if there is no GEO controller)
        geo_index = next(( i for i,c in enumerate(self.config.p_controllers)
            if c.type== scons.ControllerType.GEO ), -1)

        if tar_trace is None and self.target is not None:
            tar_trace = range(len(self.config.p_targets))
        if wfs_trace is None and self.wfs is not None:
            wfs_trace = range(len(self.config.p_wfss))

        if move_atmos and self.atmos is not None:
            self.atmos.move_atmos()
        # in case there is at least 1 controller GEO in the controller list : use this one only
        self.tel.update_input_phase()
        if ( geo_index > -1):
            nControl = geo_index
            if tar_trace is not None:
                for t in tar_trace:
                    if self.atmos.is_enable:
                        self.target.raytrace(t, tel=self.tel, atm=self.atmos, ncpa=False)
                    else:
                        self.target.raytrace(t, tel=self.tel, ncpa=False)

                    if do_control and self.rtc is not None:
                        self.rtc.do_control(nControl, sources=self.target.sources)
                        self.target.raytrace(t, dms=self.dms, ncpa=True, reset=False)
                        if apply_control:
                            self.rtc.apply_control(nControl)
                        if self.cacao:
                            self.rtc.publish()
        else:
            if tar_trace is not None: # already checked at line 213?
                for t in tar_trace:
                    if self.atmos.is_enable:
                        self.target.raytrace(t, tel=self.tel, atm=self.atmos,
                                             dms=self.dms)
                    else:
                        self.target.raytrace(t, tel=self.tel, dms=self.dms)

            if wfs_trace is not None: # already checked at line 215?
                for w in wfs_trace:
                    if self.atmos.is_enable:
                        self.wfs.raytrace(w, tel=self.tel, atm=self.atmos)
                    else:
                        self.wfs.raytrace(w, tel=self.tel)

                    if not self.config.p_wfss[w].open_loop and self.dms is not None:
                        self.wfs.raytrace(w, dms=self.dms, ncpa=False, reset=False)
                    self.wfs.compute_wfs_image(w)
            if do_control and self.rtc is not None:
                for ncontrol in nControl : # range(len(self.config.p_controllers)):
                    self.rtc.do_centroids(ncontrol)
                    self.rtc.do_control(ncontrol)
                    self.rtc.do_clipping(ncontrol)

            if apply_control:
                for ncontrol in nControl :
                    self.rtc.apply_control(ncontrol)

            if self.cacao:
                self.rtc.publish()

        if compute_tar_psf:
            for tar_index in tar_trace:
                self.target.comp_tar_image(tar_index)
                self.target.comp_strehl(tar_index)

        if self.corono is not None and compute_corono:
            for coro_index in range(len(self.config.p_coronos)):
                self.corono.compute_image(coro_index)

        if self.config.p_controllers[0].close_opti and (not self.rtc._rtc.d_control[0].open_loop):
            self.modalgains.update_mgains()
            self.close_modal_gains.append(self.modalgains.get_modal_gains())

        self.iter += 1

    def _print_strehl(self, monitoring_freq: int, iters_time: float, total_iters: int, *,
                      tar_index: int = 0):
        """ Print the Strehl ratio SE and LE from a target on the terminal, the estimated remaining time and framerate

        Args:
            monitoring_freq : (int) : Number of frames between two prints

            iters_time : (float) : time elapsed between two prints

            total_iters : (int) : Total number of iterations

        Kwargs:
            tar_index : (int) : Index of the target. Default is 0
        """
        framerate = monitoring_freq / iters_time
        strehl = self.target.get_strehl(tar_index)
        etr = (total_iters - self.iter) / framerate
        print("%d \t %.3f \t  %.3f\t     %.1f \t %.1f" % (self.iter + 1, strehl[0],
                                                          strehl[1], etr, framerate))

    def loop(self, number_of_iter: int, *, monitoring_freq: int = 100,
             compute_tar_psf: bool = True, **kwargs):
        """ Perform the AO loop for <number_of_iter> iterations

        Args:
            number_of_iter: (int) : Number of iteration that will be done

        Kwargs:
            monitoring_freq: (int) : Monitoring frequency [frames]. Default is 100

            compute_tar_psf : (bool) : If True (default), computes the PSF at each iteration
                                                 Else, only computes it each <monitoring_freq> frames
        """
        if not compute_tar_psf:
            print("WARNING: Target PSF will be computed (& accumulated) only during monitoring"
                  )

        print("----------------------------------------------------")
        print("iter# | S.E. SR | L.E. SR | ETR (s) | Framerate (Hz)")
        print("----------------------------------------------------")
        # self.next(**kwargs)
        t0 = time.time()
        t1 = time.time()
        if number_of_iter == -1:  # Infinite loop
            while (True):
                self.next(compute_tar_psf=compute_tar_psf, **kwargs)
                if ((self.iter + 1) % monitoring_freq == 0):
                    if not compute_tar_psf:
                        self.target.comp_tar_image(0)
                        self.target.comp_strehl(0)
                    self._print_strehl(monitoring_freq, time.time() - t1, number_of_iter)
                    t1 = time.time()

        for _ in range(number_of_iter):
            self.next(compute_tar_psf=compute_tar_psf, **kwargs)
            if ((self.iter + 1) % monitoring_freq == 0):
                if not compute_tar_psf:
                    self.target.comp_tar_image(0)
                    self.target.comp_strehl(0)
                self._print_strehl(monitoring_freq, time.time() - t1, number_of_iter)
                t1 = time.time()
        t1 = time.time()
        print(" loop execution time:", t1 - t0, "  (", number_of_iter, "iterations), ",
              (t1 - t0) / number_of_iter, "(mean)  ", number_of_iter / (t1 - t0), "Hz")

    def reset(self):
        """ Reset the simulation to return to its original state
        """
        self.atmos.reset_turbu()
        self.wfs.reset_noise()
        for tar_index in range(len(self.config.p_targets)):
            self.target.reset_strehl(tar_index)
        self.dms.reset_dm()
        self.rtc.open_loop()
        self.rtc.close_loop()


#    ___              _  __ _      __  __     _   _            _
#   / __|_ __  ___ __(_)/ _(_)__  |  \/  |___| |_| |_  ___  __| |___
#   \__ \ '_ \/ -_) _| |  _| / _| | |\/| / -_)  _| ' \/ _ \/ _` (_-<
#   |___/ .__/\___\__|_|_| |_\__| |_|  |_\___|\__|_||_\___/\__,_/__/
#       |_|

    def record_ao_circular_buffer(
            self, cb_count: int, sub_sample: int = 1, controller_index: int = 0,
            tar_index: int = 0, see_atmos: bool = True, cube_data_type: str = None,
            cube_data_file_path: str = "", ncpa: int = 0, ncpa_wfs: np.ndarray = None,
            ref_slopes: np.ndarray = None, ditch_strehl: bool = True,
            projection_matrix: np.ndarray = None):
        """ Used to record a synchronized circular buffer AO loop data.

        Args:
            cb_count: (int) : the number of iterations to record.

            sub_sample:  (int) : sub sampling of the data (default=1, I.e no subsampling)

            controller_index:  (int) :

            tar_index:  (int) : target number

            see_atmos:  (int) : used for the next function to enable or not the Atmos

            cube_data_type:   (int) : if  specified ("tarPhase" or "psfse") returns the target phase or short exposure PSF data cube in the output variable

            cube_data_file_path:  (int) : if specified it will also save the target phase cube data (full path on the server)

            ncpa:  (int) : !!experimental!!!: Used only in the context of PYRWFS + NCPA compensation on the fly (with optical gain)
            defines how many iters the NCPA refslopes are updates with the proper optical gain. Ex: if NCPA=10 refslopes will be updates every 10 iters.

            ncpa_wfs:  (int) : the ncpa phase as seen from the wfs array with dims = size of Mpupil

            ref_slopes:  (int) : the reference slopes to use.

            ditch_strehl:  (int) : resets the long exposure SR computation at the beginning of the Circular buffer (default= True)

            projection_matrix : (np.ndarray) : projection matrix on modal basis to compute residual coefficients

        Returns:
            slopes:  (int) : the slopes CB

            volts:  (int) : the volts applied to the DM(s) CB

            ai:  (int) : the modal coefficient of the residual phase projected on the currently used modal Basis

            psf_le:  (int) : Long exposure PSF over the <cb_count> iterations (I.e SR is reset at the begining of the CB if ditch_strehl=True)

            strehl_se_list:  (int) : The SR short exposure evolution during CB recording

            strehl_le_list:  (int) : The SR long exposure evolution during CB recording

            g_ncpa_list:  (int) : the gain applied to the NCPA (PYRWFS CASE) if NCPA is set to True

            cube_data:  (int) : the tarPhase or psfse cube data (see cube_data_type)
        """
        slopes_data = None
        volts_data = None
        cube_data = None
        ai_data = None
        k = 0
        sthrel_se_list = []
        sthrel_le_list = []
        g_ncpa_list = []

        # Resets the target so that the PSF LE is synchro with the data
        # Doesn't reset it if Ditch_strehl == False (used for real time gain computation)
        if ditch_strehl:
            for i in range(len(self.config.p_targets)):
                self.target.reset_strehl(i)

        # Starting CB loop...
        for j in range(cb_count):
            print(j, end="\r")
            if (ncpa):
                if (j % ncpa == 0):
                    ncpa_diff = ref_slopes[None, :]
                    ncpa_turbu = self.calibration.do_imat_phase(
                            controller_index, -ncpa_wfs[None, :, :], noise=False,
                            with_turbu=True)
                    g_ncpa = float(
                            np.sqrt(
                                    np.dot(ncpa_diff, ncpa_diff.T) / np.dot(
                                            ncpa_turbu, ncpa_turbu.T)))
                    if (g_ncpa > 1e18):
                        g_ncpa = 0
                        print('Warning NCPA ref slopes gain too high!')
                        g_ncpa_list.append(g_ncpa)
                        self.rtc.set_ref_slopes(-ref_slopes * g_ncpa)
                    else:
                        g_ncpa_list.append(g_ncpa)
                        print('NCPA ref slopes gain: %4.3f' % g_ncpa)
                        self.rtc.set_ref_slopes(-ref_slopes / g_ncpa)

            self.atmos.enable_atmos(see_atmos)
            self.next()
            for t in range(len(self.config.p_targets)):
                self.target.comp_tar_image(t)

            srse, srle, _, _ = self.target.get_strehl(tar_index)
            sthrel_se_list.append(srse)
            sthrel_le_list.append(srle)
            if (j % sub_sample == 0):
                if (projection_matrix is not None):
                    ai_vector = self.calibration.compute_modal_residuals(
                            projection_matrix, selected_actus=self.basis.selected_actus)
                    if (ai_data is None):
                        ai_data = np.zeros((len(ai_vector), int(cb_count / sub_sample)))
                    ai_data[:, k] = ai_vector

                slopes_vector = self.rtc.get_slopes(controller_index)
                if (slopes_data is None):
                    slopes_data = np.zeros((len(slopes_vector),
                                            int(cb_count / sub_sample)))
                slopes_data[:, k] = slopes_vector

                volts_vector = self.rtc.get_command(
                        controller_index)  # get_command or get_voltages ?
                if (volts_data is None):
                    volts_data = np.zeros((len(volts_vector),
                                           int(cb_count / sub_sample)))
                volts_data[:, k] = volts_vector

                if (cube_data_type):
                    if (cube_data_type == "tarPhase"):
                        dataArray = self.target.get_tar_phase(tar_index, pupil=True)
                    elif (cube_data_type == "psfse"):
                        dataArray = self.target.get_tar_image(tar_index, expo_type="se")
                    else:
                        raise ValueError("unknown dataData" % cube_data_type)
                    if (cube_data is None):
                        cube_data = np.zeros((*dataArray.shape,
                                              int(cb_count / sub_sample)))
                    cube_data[:, :, k] = dataArray
                k += 1
        if (cube_data_file_path != ""):
            print("Saving tarPhase cube at: ", cube_data_file_path)
            from astropy.io import fits as pf
            pf.writeto(cube_data_file_path, cube_data, overwrite=True)

        psf_le = self.target.get_tar_image(tar_index, expo_type="le")
        return slopes_data, volts_data, ai_data, psf_le, sthrel_se_list, sthrel_le_list, g_ncpa_list, cube_data

    def export_config(self):
        """
        Extract and convert compass supervisor configuration parameters
        into 2 dictionnaries containing relevant AO parameters

        Args:
            root: (object), COMPASS supervisor object to be parsed

        Returns :
            2 dictionaries... See F. Vidal :)
        """
        return self.config.export_config()

    def get_s_pupil(self):
        """
        Returns the so called S Pupil of COMPASS

        Return:
            s_pupil: (np.array) : S Pupil of COMPASS
        """
        return self.config.p_geom.get_spupil()

    def get_i_pupil(self):
        """
        Returns the so called I Pupil of COMPASS

        Return:
            i_pupil: (np.array) : I Pupil of COMPASS
        """
        return self.config.p_geom.get_ipupil()

    def get_m_pupil(self):
        """
        Returns the so called M Pupil of COMPASS

        Return:
            m_pupil: (np.array) : M Pupil of COMPASS
        """
        return self.config.p_geom.get_mpupil()
