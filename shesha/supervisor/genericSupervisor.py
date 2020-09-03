## @package   shesha.supervisor.aoSupervisor
## @brief     Abstract layer for initialization and execution of a AO supervisor
## @author    COMPASS Team <https://github.com/ANR-COMPASS>
## @version   5.0.0
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

from abc import abstractmethod
import numpy as np
import time
from shesha.sutra_wrap import carmaWrap_context
from typing import Iterable


class GenericSupervisor(object):
    """ This class defines generic methods and behavior of a supervisor
    It is not intended to be instantiated as it is : prefer to build
    a supervisor class inheriting from it. This approach allows to build multiple
    supervisors with various components and less effort

    Attributes:
        context : (CarmaContext) : a CarmaContext instance

        config : (config) : Parameters structure

        telescope : (TelescopeComponent) : a TelescopeComponent instance

        atmos : (AtmosComponent) : An AtmosComponent instance

        target : (targetComponent) : A TargetComponent instance

        wfs : (WfsComponent) : A WfsComponent instance

        dms : (DmComponent) : A DmComponent instance

        rtc : (RtcComponent) : A Rtc component instance

        is_init : (bool) : Flag equals to True if the supervisor has already been initialized

        iter : (int) : Frame counter
    """

    def __init__(self, config):
        """ Init the a supervisor

        Args:
            config : (config module) : Configuration module
        """
        self.context = None
        self.config = config
        self.telescope = None
        self.atmos = None
        self.target = None
        self.wfs = None
        self.dms = None
        self.rtc = None
        self.is_init = False
        self.iter = 0
        self._init_components()

    def get_config(self):
        """ Returns the configuration in use, in a supervisor specific format ?

        Return:
            config : (config module) : Current supervisor configuration
        """
        return self.config

    def get_frame_counter(self) -> int:
        """Return the current iteration number of the loop

        Return:
            framecounter : (int) : Number of iteration already performed
        """
        return self.iter

    def force_context(self) -> None:
        """ Active all the GPU devices specified in the parameters file
        """
        if self.context is not None:
            current_device = self.context.active_device
            for device in range(len(self.config.p_loop.devices)):
                self.context.set_active_device_force(device)
            self.context.set_active_device(current_device)

    def _init_components(self) -> None:
        """ Initialize all the components
        """
        if (self.config.p_loop.devices.size > 1):
            self.context = carmaWrap_context.get_instance_ngpu(
                    self.config.p_loop.devices.size, self.config.p_loop.devices)
        else:
            self.context = carmaWrap_context.get_instance_1gpu(
                    self.config.p_loop.devices[0])
        self.force_context()

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

        self.is_init = True

    @abstractmethod
    def _init_tel(self):
        """ Initialize the telescope component of the supervisor
        """
        pass

    @abstractmethod
    def _init_atmos(self):
        """ Initialize the atmos component of the supervisor
        """
        pass

    @abstractmethod
    def _init_dms(self):
        """ Initialize the dms component of the supervisor
        """
        pass

    @abstractmethod
    def _init_target(self):
        """ Initialize the target component of the supervisor
        """
        pass

    @abstractmethod
    def _init_wfs(self):
        """ Initialize the wfs component of the supervisor
        """
        pass

    @abstractmethod
    def _init_rtc(self):
        """ Initialize the rtc component of the supervisor
        """
        pass

    def next(self, *, move_atmos: bool = True, nControl: int = 0,
             tar_trace: Iterable[int] = None, wfs_trace: Iterable[int] = None,
             do_control: bool = True, apply_control: bool = True,
             compute_tar_psf: bool = True) -> None:
        """Iterates the AO loop, with optional parameters

        Kwargs:
            move_atmos: (bool): move the atmosphere for this iteration. Default is True

            nControl: (int): Controller number to use. Default is 0 (single control configuration)

            tar_trace: (List): list of targets to trace. None is equivalent to all (default)

            wfs_trace: (List): list of WFS to trace. None is equivalent to all (default)

            do_control : (bool) : Performs RTC operations if True (Default)

            apply_control: (bool): if True (default), apply control on DMs

            compute_tar_psf : (bool) : If True (default), computes the PSF at the end of the iteration
        """
        if tar_trace is None and self.target is not None:
            tar_trace = range(len(self.config.p_targets))
        if wfs_trace is None and self.wfs is not None:
            wfs_trace = range(len(self.config.p_wfss))

        if move_atmos and self.atmos is not None:
            self.atmos.move_atmos()

        if tar_trace is not None:
            for t in tar_trace:
                if self.atmos.is_enable:
                    self.target.raytrace(t, tel=self.tel, atm=self.atmos, dms=self.dms)
                else:
                    self.target.raytrace(t, tel=self.tel, dms=self.dms)

        if wfs_trace is not None:
            for w in wfs_trace:
                if self.atmos.is_enable:
                    self.wfs.raytrace(w, tel=self.tel, atm=self.atmos)
                else:
                    self.wfs.raytrace(w, tel=self.tel)

                if not self.config.p_wfss[w].open_loop and self.dms is not None:
                    self.wfs.raytrace(w, dms=self.dms, ncpa=False, reset=False)
                self.wfs.compute_wfs_image(w)
        if do_control and self.rtc is not None:
            for ncontrol in range(len(self.config.p_controllers)):
                self.rtc.do_centroids(ncontrol)
                self.rtc.do_control(ncontrol)
                self.rtc.do_clipping(ncontrol)

        if apply_control:
            self.rtc.apply_control(ncontrol)

        if self.cacao:
            self.rtc.publish()

        if compute_tar_psf:
            for tar_index in tar_trace:
                self.target.comp_tar_image(tar_index)
                self.target.comp_strehl(tar_index)

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
                        self.comp_tar_image(0)
                        self.comp_strehl(0)
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
