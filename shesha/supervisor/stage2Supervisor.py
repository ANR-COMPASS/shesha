## @package   shesha.supervisor.stage2Supervisor
## @brief     Initialization and execution of a second stage supervisor
## @author    SAXO+ Team (Clementine Bechet)
## @version   5.3.0
## @date      2023/01/31
## @copyright GNU Lesser General Public License
#
#  This file is part of COMPASS <https://anr-compass.github.io/compass/>
#
#  Copyright (C) 2011-2022 COMPASS Team <https://github.com/ANR-COMPASS>
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

from shesha.supervisor.compassSupervisor import CompassSupervisor
from shesha.supervisor.components import AtmosCompass, DmCompass, RtcCompass, TargetCompass, TelescopeCompass, WfsCompass
from shesha.supervisor.optimizers import ModalBasis, Calibration, ModalGains
import numpy as np
import time

import shesha.constants as scons

from typing import Iterable


class Stage2Supervisor(CompassSupervisor):
    """ This class implements a second stage supervisor to handle compass cascaded simulations.

    Attributes inherited from CompassSupervisor:
        context : (CarmaContext) : a CarmaContext instance

        config : (config) : Parameters structure

        is_init : (bool) : Flag equals to True if the supervisor has already been initialized

        iter : (int) : Frame counter

        telescope : (TelescopeComponent) : a TelescopeComponent instance

        atmos : (AtmosComponent) : An AtmosComponent instance

        target : (targetComponent) : A TargetComponent instance

        wfs : (WfsComponent) : A WfsComponent instance

        dms : (DmComponent) : A DmComponent instance

        rtc : (RtcComponent) : A Rtc component instance

        cacao : (bool) : CACAO features enabled in the RTC

        basis : (ModalBasis) : a ModalBasis instance (optimizer)

        calibration : (Calibration) : a Calibration instance (optimizer)

        modalgains : (ModalGains) : a ModalGain instance (optimizer) using CLOSE algorithm
        close_modal_gains : (list of floats) : list of the previous values of the modal gains
    """

    def next(self, *, move_atmos: bool = True, nControl: int = 0,
             tar_trace: Iterable[int] = None, wfs_trace: Iterable[int] = None,
             do_control: bool = True, apply_control: bool = True,
             compute_tar_psf: bool = True) -> None:
        """Iterates the AO loop, with optional parameters, considering it is a second 
        stage. 

        Overload the CompassSupervisor next() method to fix the order of the AO tasks.

        Kwargs:
            move_atmos: (bool): move the atmosphere for this iteration. Default is True

            nControl: (int): Controller number to use. Default is 0 (single control configuration)

            tar_trace: (List): list of targets to trace. None is equivalent to all (default)

            wfs_trace: (List): list of WFS to trace. None is equivalent to all (default)

            do_control : (bool) : Performs RTC operations if True (Default)

            apply_control: (bool): if True (default), apply control on DMs

            compute_tar_psf : (bool) : If True (default), computes the PSF at the end of the iteration
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

                    if apply_control:
                        self.rtc.apply_control(nControl)

                    if self.atmos.is_enable:
                        self.target.raytrace(t, tel=self.tel, atm=self.atmos, ncpa=False)
                    else:
                        self.target.raytrace(t, tel=self.tel, ncpa=False)

                    if do_control and self.rtc is not None:
                        self.rtc.do_control(nControl, sources=self.target.sources)
                        self.target.raytrace(t, dms=self.dms, ncpa=True, reset=False)
                        
        else:
            # start updating the DM shape
            if apply_control:
                for ncontrol in nControl :
                    self.rtc.apply_control(ncontrol)

            # start with the propagations
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
                    
            if self.rtc is not None:
                for ncontrol in nControl : # range(len(self.config.p_controllers)):       
                   self.rtc.do_centroids(ncontrol)

            # modified to allow do_centroids even if do_control is False.
            # Required for calibration. (CBE 2023.01.19)
            if do_control:
                for ncontrol in nControl:
                    self.rtc.do_control(ncontrol)

        if self.cacao:
            self.rtc.publish()
           
        if compute_tar_psf:
            for tar_index in tar_trace:
                self.target.comp_tar_image(tar_index)
                self.target.comp_strehl(tar_index)

        if self.config.p_controllers[0].close_opti and (not self.rtc._rtc.d_control[0].open_loop):
            self.modalgains.update_mgains()
            self.close_modal_gains.append(self.modalgains.get_modal_gains())

        self.iter += 1

