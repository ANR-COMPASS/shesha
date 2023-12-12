## @package   shesha.supervisor.stageSupervisor
## @brief     Initialization and execution of a single stage supervisor for cascaded AO systems
## @author    SAXO+ Team <https://github.com/ANR-COMPASS> (Clementine Bechet)
## @version   5.5.0
## @date      2023/01/31
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

from shesha.supervisor.compassSupervisor import CompassSupervisor

import shesha.constants as scons

from typing import Iterable


class StageSupervisor(CompassSupervisor):
    """ This class implements a single stage (e.g. first stage, second stage) supervisor
    to handle compass simulations of cascaded AO. The main supervision will be handled by another
    supervisor (manager).

    Attributes:
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
             compute_tar_psf: bool = True, stack_wfs_image: bool = False,
             do_centroids: bool = True, compute_corono: bool=True) -> None:
        """Iterates the AO loop, with optional parameters, considering it is a single
        stage and may be called in the middle of WFS frames.
        Overload the CompassSupervisor next() method to arrange tasks orders and allow cascaded
        simulation.

        Kwargs:
            move_atmos: (bool): move the atmosphere for this iteration. Default is True

            nControl: (int): Controller number to use. Default is 0 (single control configuration)

            tar_trace: (List): list of targets to trace. None is equivalent to all (default)

            wfs_trace: (List): list of WFS to trace. None is equivalent to all (default)

            do_control : (bool) : Performs RTC operations if True (Default)

            apply_control: (bool): if True (default), apply control on DMs

            compute_tar_psf : (bool) : If True (default), computes the PSF at the end of the
                                        iteration

            stack_wfs_image : (bool) : If False (default), the Wfs image is computed as
                                        usual. Otherwise, a newly computed WFS image is accumulated
                                        to the previous one.

            do_centroids : (bool) : If True (default), the last WFS image is stacked and
                                    centroids computation is done. WFS image must be reset before
                                    next loop (in the manager).

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

                    if apply_control:
                        self.rtc.apply_control(nControl)

                    if self.atmos.is_enable:
                        self.target.raytrace(t, tel=self.tel, atm=self.atmos, ncpa=False)
                    else:
                        self.target.raytrace(t, tel=self.tel, ncpa=False)

                    if do_control and self.rtc is not None:
                        self.rtc.do_control(nControl, sources=self.target.sources)
                        self.target.raytrace(t, dms=self.dms, ncpa=True, reset=False)

                    if self.cacao:
                        self.rtc.publish()

        else:
            # start updating the DM shape
            if apply_control:
                for ncontrol in nControl :
                    # command buffer is updated and commands voltages update is applied
                    self.rtc.apply_control(ncontrol)
                    # Note: clipping is always made by apply_control (CBE. 2023.01.27)

            # start the propagations
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

                    if stack_wfs_image:
                        # accumulate image during sub-integration frames
                        wfs_image = self.wfs.get_wfs_image(w)
                        self.wfs.compute_wfs_image(w)
                        self.wfs.set_wfs_image(w, self.wfs.get_wfs_image(w) + wfs_image)
                    else:
                        self.wfs.compute_wfs_image(w)

            if self.rtc is not None:
                for ncontrol in nControl : # range(len(self.config.p_controllers)):
                    # modified to allow do_centroids when the WFS exposure is over.
                    # Also useful for calibration. (CBE 2023.01.30)
                    if do_centroids:
                        self.rtc.do_centroids(ncontrol)

                    if do_control:
                        self.rtc.do_control(ncontrol)

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


    def reset(self):
        """
        Reset the simulation to return to its original state.
        Overwrites the compassSupervisor reset function, reseting explicitely the WFS image,
        to force a new integration of the frame.
        """
        self.atmos.reset_turbu()
        self.wfs.reset_noise()
        self.wfs.reset_image()
        for tar_index in range(len(self.config.p_targets)):
            self.target.reset_strehl(tar_index)
        self.dms.reset_dm()
        self.rtc.open_loop()
        self.rtc.close_loop()