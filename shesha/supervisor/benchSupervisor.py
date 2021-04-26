## @package   shesha.supervisor.benchSupervisor
## @brief     Initialization and execution of a Bench supervisor
## @author    COMPASS Team <https://github.com/ANR-COMPASS>
## @version   5.1.0
## @date      2020/05/18
## @copyright GNU Lesser General Public License
#
#  This file is part of COMPASS <https://anr-compass.github.io/compass/>
#
#  Copyright (C) 2011-2019 COMPASS Team <https://github.com/ANR-COMPASS>
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

import numpy as np

from shesha.constants import CentroiderType, WFSType
from shesha.init.dm_init import dm_init_standalone
from shesha.init.rtc_init import rtc_standalone
from shesha.sutra_wrap import carmaWrap_context

from shesha.supervisor.components import RtcStandalone

from shesha.supervisor.genericSupervisor import GenericSupervisor

from shesha.util.utilities import load_config_from_file

from typing import Callable


class BenchSupervisor(GenericSupervisor):
    """ This class implements generic supervisor to handle compass simulation

    Attributes inherited from GenericSupervisor:
        context : (CarmaContext) : a CarmaContext instance

        config : (config) : Parameters structure

        is_init : (bool) : Flag equals to True if the supervisor has already been initialized

        iter : (int) : Frame counter

    Attributes:
        rtc : (RtcComponent) : A Rtc component instance

        cacao : (bool) : CACAO features enabled in the RTC

        basis : (ModalBasis) : a ModalBasis instance (optimizer)

        calibration : (Calibration) : a Calibration instance (optimizer)
    """

    def __init__(self, config_file: str = None, cacao: bool = False):
        """ Init the COMPASS wih the config_file

        Kwargs:
            config_file : (str) : path to the configuration file. Default is None

            cacao : (bool) : Flag to use cacao rtc. Default is False
        """
        self.pause_loop = None
        self.rtc = None
        self.frame = None
        self.cacao = cacao
        self.iter = 0
        self.slopes_index = None
        config = load_config_from_file(config_file)

        GenericSupervisor.__init__(self, config)

    def _init_rtc(self):
        """Initialize the rtc component of the supervisor as a RtcCompass
        """
        print("->RTC")
        self.number_of_wfs = len(self.config.p_wfss)
        print("Configuration of", self.number_of_wfs, "wfs ...")

        if (hasattr(self.config, 'p_loop') and self.config.p_loop.devices.size > 1):
            self.context = carmaWrap_context.get_instance_ngpu(
                    self.config.p_loop.devices.size, self.config.p_loop.devices)
        else:
            self.context = carmaWrap_context.get_instance_1gpu(
                    self.config.p_loop.devices[0])
        nact = self.config.p_controllers[0].nactu

        nvalid = []
        centroider_type = []
        delay = []
        offset = []
        scale = []
        gain = []
        cmat_size = []
        npix = []

        # Get parameters
        for wfs in range(self.number_of_wfs):

            if self.config.p_wfss[wfs].type == WFSType.SH:
                npix.append(self.config.p_wfss[wfs].npix)
                if self.config.p_wfss[wfs]._validsubsx is None or \
                        self.config.p_wfss[wfs]._validsubsy is None:

                    from hraa.tools.doit import makessp
                    roiTab = makessp(self.config.p_wfss[wfs].nxsub, obs=0., rmax=0.98)
                    self.config.p_wfss[wfs]._nvalid = roiTab[0].size
                    self.config.p_wfss[
                            wfs]._validsubsx = roiTab[0] * self.config.p_wfss[wfs].npix
                    self.config.p_wfss[
                            wfs]._validsubsy = roiTab[1] * self.config.p_wfss[wfs].npix
                else:
                    self.config.p_wfss[wfs]._nvalid = self.config.p_wfss[
                            wfs]._validsubsx.size

                nvalid.append(
                        np.array([self.config.p_wfss[wfs]._nvalid], dtype=np.int32))
                # print("nvalid : %d" % nvalid[wfs])
                centroider_type.append(self.config.p_centroiders[wfs].type)
                delay.append(self.config.p_controllers[0].delay)  # ???
                offset.append((self.config.p_wfss[wfs].npix - 1) / 2)
                scale.append(1)
                gain.append(1)
                cmat_size.append(2 * nvalid[wfs][0])

            elif self.config.p_wfss[wfs].type == WFSType.PYRHR or self.config.p_wfss[
                    wfs].type == WFSType.PYRLR:
                nvalid.append(
                        np.array([self.config.p_wfss[wfs]._nvalid],
                                 dtype=np.int32))  # Number of valid SUBAPERTURES
                centroider_type.append(self.config.p_centroiders[wfs].type)
                delay.append(self.config.p_controllers[0].delay)  # ???
                offset.append(0)
                scale.append(1)
                gain.append(1)
                cmat_size.append(self.config.p_wfss[wfs].nPupils * nvalid[wfs][0])
                npix.append(0)
            else:
                raise ValueError('WFS type not supported')

        # Create RTC
        self.rtc = RtcStandalone(self.context, self.config, self.number_of_wfs, nvalid,
                                 nact, centroider_type, delay, offset, scale,
                                 cacao=self.cacao)

        self.slopes_index = np.cumsum([0] +
                                      [wfs.nslopes for wfs in self.rtc._rtc.d_centro])

        # Create centroiders
        for wfs in range(self.number_of_wfs):
            self.rtc._rtc.d_centro[wfs].load_validpos(
                    self.config.p_wfss[wfs]._validsubsx,
                    self.config.p_wfss[wfs]._validsubsy,
                    self.config.p_wfss[wfs]._validsubsx.size)
            if self.config.p_centroiders[wfs].type is CentroiderType.BPCOG:
                self.rtc._rtc.d_centro[wfs].set_nmax(self.config.p_centroiders[wfs].nmax)
            self.rtc._rtc.d_centro[wfs].set_npix(npix[wfs])
            # finally
            self.config.p_centroiders[wfs]._nslope = self.rtc._rtc.d_centro[wfs].nslopes
            print("wfs ", wfs, " set as ", centroider_type[wfs])
        size = sum(cmat_size)
        cMat = np.zeros((nact, size), dtype=np.float32)
        print("Size of cMat:", cMat.shape)

        # Initiate RTC
        self.rtc._rtc.d_control[0].set_cmat(cMat)
        self.rtc._rtc.d_control[0].set_decayFactor(
                np.ones(nact, dtype=np.float32) * (gain[0] - 1))
        self.rtc._rtc.d_control[0].set_matE(np.identity(nact, dtype=np.float32))
        self.rtc._rtc.d_control[0].set_modal_gains(
                np.ones(nact, dtype=np.float32) * -gain[0])

        print("RTC initialized")
        self.is_init = True

    def _init_components(self) -> None:
        """ Initialize all the components
        """
        if self.config.p_controllers is not None or self.config.p_centroiders is not None:
            self._init_rtc()

        GenericSupervisor._init_components(self)

    #     _    _         _                  _
    #    / \  | |__  ___| |_ _ __ __ _  ___| |_
    #   / _ \ | '_ \/ __| __| '__/ _` |/ __| __|
    #  / ___ \| |_) \__ \ |_| | | (_| | (__| |_
    # /_/   \_\_.__/|___/\__|_|  \__,_|\___|\__|
    #
    #  __  __      _   _               _
    # |  \/  | ___| |_| |__   ___   __| |___
    # | |\/| |/ _ \ __| '_ \ / _ \ / _` / __|
    # | |  | |  __/ |_| | | | (_) | (_| \__ \
    # |_|  |_|\___|\__|_| |_|\___/ \__,_|___/

    def next(self) -> None:
        """ Performs a single loop iteration
        """
        self.load_new_wfs_frame()

        if (self.pause_loop is not True):
            self.compute_wfs_frame()
            self.set_command(0, np.array(self.rtc._rtc.d_control[0].d_voltage))
        if self.cacao:
            self.rtc.publish()
        self.iter += 1

    def get_tar_image(self, tar_index, expo_type: str = "se") -> np.ndarray:
        """ NOT IMPLEMENTED
        """
        raise NotImplementedError("Not implemented")

    def set_command(self, nctrl: int, command: np.ndarray) -> None:
        """ Immediately sets provided command to DMs - does not affect integrator

        Args:
            nctrl : (int) : Controller index (unused)

            command : (np.ndarray) : Command vector to send
        """
        # Do stuff
        self.dm_set_callback(command)
        # Btw, update the RTC state with the information
        # self.rtc._rtc.d_control[nctrl].set_com(command, command.size)

    def get_command(self) -> np.ndarray:
        """ Get command from DM

        Returns:
            command : (np.ndarray) : Command vector
        """
        # Do something
        command = self.dm_get_callback()
        # Btw, update the RTC state with the information
        # self.rtc._rtc.d_control[nControl].set_com(command, command.size)

        return command

    #  ____                  _ _   _        __  __      _   _               _
    # / ___| _ __   ___  ___(_) |_(_) ___  |  \/  | ___| |_| |__   ___   __| |___
    # \___ \| '_ \ / _ \/ __| | __| |/ __| | |\/| |/ _ \ __| '_ \ / _ \ / _` / __|
    #  ___) | |_) |  __/ (__| | |_| | (__  | |  | |  __/ |_| | | | (_) | (_| \__ \
    # |____/| .__/ \___|\___|_|\__|_|\___| |_|  |_|\___|\__|_| |_|\___/ \__,_|___/
    #       |_|

    def __repr__(self):

        s = '--- BenchSupervisor ---\nRTC: ' + repr(self.rtc)
        if hasattr(self, '_cam'):
            s += '\nCAM: ' + repr(cam)
        if hasattr(self, '_dm'):
            s += '\nDM: ' + repr(dm)
        return s

    def load_new_wfs_frame(self, centro_index: int = 0) -> None:
        """ Acquire a new WFS frame and load it

        Args:
            centro_index : (int) : Index of the centroider where to load the frame
        """
        self.frame = self.cam_callback()
        if (type(self.frame) is tuple):
            centro_index = len(self.frame)
            for i in range(centro_index):
                self.rtc._rtc.d_centro[i].load_img(self.frame[i], self.frame[i].shape[0],
                                                   self.frame[i].shape[1], -1)
        else:
            self.rtc._rtc.d_centro[centro_index].load_img(self.frame,
                                                          self.frame.shape[0],
                                                          self.frame.shape[1], -1)

    def compute_wfs_frame(self):
        """ Compute the WFS frame: calibrate, centroid, commands.
        """
        # for index, centro in enumerate(self.rtc._rtc.d_centro):
        for centro in self.rtc._rtc.d_centro:
            centro.calibrate_img()
        self.rtc.do_centroids(0)
        self.rtc.do_control(0)
        self.rtc.do_clipping(0)
        self.rtc._rtc.comp_voltage(0)

    def set_one_actu(self, nctrl: int, nactu: int, *, ampli: float = 1,
                     reset: bool = True) -> None:
        """ Push the selected actuator

        Args:
            nctrl : (int) : controller index

            nactu : (int) : actuator index to push

        Kwargs:
            ampli : (float) : amplitude to apply. Default is 1 volt

            reset : (bool) : reset the previous command vector. Default is True
        """
        command = self.get_command()
        if reset:
            command *= 0
        command[nactu] = ampli
        self.set_command(nctrl, command)

    def force_context(self) -> None:
        """ Active all the GPU devices specified in the parameters file
        Required for using with widgets, due to multithreaded init
        and in case GPU 0 is not used by the simu
        """
        if self.is_init and self.context is not None:
            current_device = self.context.active_device
            for device in range(len(self.config.p_loop.devices)):
                self.context.set_active_device_force(device)
            self.context.set_active_device(current_device)

    def reset_dm(self) -> None:
        """ Reset the DM
        """
        if hasattr(self, '_dm'):
            dm.reset_dm()

    def reset_command(self, nctrl: int = -1) -> None:
        """ Reset the nctrl Controller command buffer, reset all controllers if nctrl  == -1

        Kwargs:
            nctrl : (int) : Controller index. If -1 (default), all controllers commands are reset
        """
        if (nctrl == -1):  # All Dms reset
            for control in self.rtc._rtc.d_control:
                control.d_com.reset()
        else:
            self.rtc._rtc.d_control[nctrl].d_com.reset()

    def load_config(self, config_file: str = None) -> None:
        """ Init the COMPASS with the config_file

        Args:
            config_file : (str) : path to the configuration file
        """
        from shesha.config import ParamConfig
        self.config = ParamConfig(config_file)

    def set_cam_callback(self, cam_callback: Callable):
        """ Set the externally defined function that allows to grab frames

        Args:
            cam_callback : (Callable) : function that allows to grab frames
        """
        self.cam_callback = cam_callback

    def set_dm_callback(self, dm_get_callback: Callable, dm_set_callback: Callable):
        """ Set the externally defined function that allows to communicate with the DM

        Args:
            dm_get_callback : (Callable) : function that allows to retrieve commands
            dm_set_callback : (Callable) : function that allows to set commands
        """
        self.dm_get_callback = dm_get_callback
        self.dm_set_callback = dm_set_callback

    def adaptive_windows(self, init_config=False, centro_index: int = 0):
        """ Re-centre the centroiding boxes around the spots, and loads
        the new box coordinates in the slopes computation supervisor
        pipeline.

        Args:
            init_config : (bool): Flag to reset to the default positions of boxes. Default is False

            centro_index : (int) : centroider index
        """
        if init_config:
            # reset de la configuration initiale
            ij_subap = self.config.p_wfss[centro_index].get_validsub()
            nsubap = ij_subap.shape[1]
            self.rtc._rtc.d_centro[centro_index].load_validpos(
                    ij_subap[0], ij_subap[1], nsubap)
        else:
            # acquire slopes first
            nslopes = 10
            s = 0.
            for i in range(nslopes):
                self.load_new_wfs_frame()  # sinon toutes les slopes sont les memes
                self.compute_wfs_frame()
                s = s + self.get_slopes()[self.slopes_index[centro_index]:self.
                                          slopes_index[centro_index + 1]]
            s /= nslopes
            # get coordinates of valid sub-apertures
            #ij_subap = self.config.p_wfss[centro_index].get_validsub()
            i_subap = np.array(self.rtc._rtc.d_centro[centro_index].d_validx)
            j_subap = np.array(self.rtc._rtc.d_centro[centro_index].d_validy)
            # get number of subaps
            nsubap = i_subap.shape[0]
            # reshape the array <s> to be conformable with <ij_subap>
            s = np.resize(s, (2, nsubap))
            # re-centre the boxes around the spots
            new_i_subap = (i_subap + s[0, :].round()).astype(int)
            new_j_subap = (j_subap + s[1, :].round()).astype(int)
            # load the new positions of boxes
            self.rtc._rtc.d_centro[centro_index].load_validpos(
                    new_i_subap, new_j_subap, nsubap)

    def get_current_windows_pos(self, centro_index: int = 0):
        """ Returns the currently used subapertures positions

        Args:
            centro_index : (int) : Index of the centroider

        Returns:
            current_pos : (tuple) : (i_subap, j_subap)
        """
        i_subap = np.array(self.rtc._rtc.d_centro[centro_index].d_validx)
        j_subap = np.array(self.rtc._rtc.d_centro[centro_index].d_validy)
        return i_subap, j_subap

    def get_slopes_index(self):
        """ Return the index of the first position of each WFS slopes vector
        inside the global RTC slopes vector

        Returns:
            slopes_index : (np.ndarray) : Slopes index
        """
        return self.slopes_index

    def export_config(self):
        """
        Extract and convert compass supervisor configuration parameters
        into 2 dictionnaries containing relevant AO parameters

        Args:
            root: (object), COMPASS supervisor object to be parsed

        Returns : 2 dictionnaries
        """
        return self.config.export_config()
