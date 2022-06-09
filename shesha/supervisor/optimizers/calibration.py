## @package   shesha.supervisor.optimizers
## @brief     User layer for optimizing AO supervisor loop
## @author    COMPASS Team <https://github.com/ANR-COMPASS>
## @version   5.3.0
## @date      2022/01/24
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
import numpy as np
from tqdm import trange

class Calibration(object):
    """ This optimizer class handles all the modal basis and DM Influence functions
    related operations.

    Attributes:
        _config : (config) : Configuration parameters module

        _tel : (TelescopeCompass) : TelescopeCompass instance

        _atmos : (AtmosScompass) : AtmosCompass instance

        _dms : (DmCompass) : DmCompass instance

        _target : (TargetCompass) : TargetCompass instance

        _rtc : (RtcCompass) : RtcCompass instance

        _wfs : (WfsCompass) : WfsCompass instance
    """
    def __init__(self, config, tel, atmos, dms, target, rtc, wfs):
        """ Instantiate a ModalBasis object

        Args:
            config : (config) : Configuration parameters module

            tel : (TelescopeCompass) : TelescopeCompass instance

            atmos : (AtmosScompass) : AtmosCompass instance

            dms : (DmCompass) : DmCompass instance

            target : (TargetCompass) : TargetCompass instance

            rtc : (RtcCompass) : RtcCompass instance

            wfs : (WfsCompass) : WfsCompass instance
        """
        self._config = config
        self._tel = tel
        self._atmos = atmos
        self._dms = dms
        self._target = target
        self._rtc = rtc
        self._wfs = wfs

    def apply_volts_and_get_slopes(self, controller_index: int, *, noise: bool = False,
                                   turbu: bool = False, reset: bool = True):
        """ Apply voltages, raytrace, compute WFS image, compute slopes and returns it

        Args:
            controller_index : (int) : Controller index

        Kwargs:
            noise : (bool) : Flag to enable noise for WFS image computation. Default is False

            turbu : (bool) : Flag to enable atmosphere for WFS phase screen raytracing.
                                       Default is False

            reset : (bool) : Flag to reset previous phase screen before raytracing.
                                       Default is True
        """
        self._rtc.apply_control(controller_index)
        for w in range(len(self._config.p_wfss)):
            if (turbu):
                self._wfs.raytrace(w, tel=self._tel, atm=self._atmos, dms=self._dms)
            else:
                self._wfs.raytrace(w, dms=self._dms, reset=reset)
            self._wfs.compute_wfs_image(w, noise=noise)
        return self._rtc.compute_slopes(controller_index)

    def do_imat_modal(self, controller_index : int, ampli : np.ndarray, modal_basis : np.ndarray,
                      *, noise : bool=False, nmodes_max : int=0, with_turbu : bool=False, push_pull : bool=False) -> np.ndarray:
        """ Computes an interaction matrix from provided modal basis

        Args:
            controller_index : (int) : Controller index

            ampli : (np.ndarray) : amplitude to apply on each mode

            modal_basis : (np.ndarray) : modal basis matrix

        Kwargs:
            noise : (bool) : Flag to enable noise for WFS image compuation. Default is False

            nmodes_max : (int) : Default is 0. TODO : description

            with_turbu : (bool) : Flag to enable atmosphere for WFS phase screen raytracing.
                                            Default is False

            push_pull : (bool) : If True, imat is computed as an average of push and pull ampli
                                            on each mode

        Returns:
            modal_imat : (np.ndarray) : Modal interaction matrix
        """
        modal_imat = np.zeros((self._config.p_controllers[controller_index].nslope, modal_basis.shape[1]))
        print("Starting Modal imat...")
        if (nmodes_max == 0):
            nmodes_max = modal_basis.shape[1]
        v_old = self._rtc.get_command(controller_index)
        self._rtc.open_loop(controller_index, reset=False)
        for m in trange(nmodes_max):
            v = ampli[m] * modal_basis[:, m]
            if ((push_pull is True) or
                (with_turbu is True)):  # with turbulence/aberrations => push/pull
                self._rtc.set_perturbation_voltage(
                        controller_index, "imat_modal",
                        v_old + v)  # Adding Perturbation voltage on current iteration
                devpos = self.apply_volts_and_get_slopes(controller_index,
                                                         turbu=with_turbu, noise=noise)
                self._rtc.set_perturbation_voltage(controller_index, "imat_modal", v_old - v)
                devmin = self.apply_volts_and_get_slopes(controller_index,
                                                         turbu=with_turbu, noise=noise)
                modal_imat[:, m] = (devpos - devmin) / (2. * ampli[m])
            else:  # No turbulence => push only
                self._rtc.open_loop(controller_index)  # open_loop
                self._rtc.set_perturbation_voltage(controller_index, "imat_modal", v)
                modal_imat[:, m] = self.apply_volts_and_get_slopes(controller_index, noise=noise) / ampli[m]
        self._rtc.remove_perturbation_voltage(controller_index, "imat_modal")
        if ((push_pull is True) or (with_turbu is True)):
            self._rtc.close_loop(controller_index)  # We are supposed to be in close loop now
        return modal_imat

    def do_imat_phase(self, controller_index: int, cube_phase: np.ndarray, *, noise : bool=False,
                      nmodes_max : int=0, with_turbu : bool=False, push_pull : bool=False, wfs_index : int=0) -> np.ndarray:
        """ Computes an interaction matrix with the provided cube phase

        Args:
            controller_index : (int) : Controller index

            cube_phase : (np.ndarray) : Cube of phase to insert as NCPA

        Kwargs:
            noise : (bool) : Flag to enable noise for WFS image compuation. Default is False

            nmodes_max : (int) : Default is 0. TODO : description

            with_turbu : (bool) : Flag to enable atmosphere for WFS phase screen raytracing.
                                            Default is False

            push_pull : (bool) : If True, imat is computed as an average of push and pull ampli
                                            on each mode

            wfs_index : (int) : WFS index. Default is 0

        Returns:
            phase_imat : (np.ndarray) : Phase interaction matrix
        """
        imat_phase = np.zeros((cube_phase.shape[0], self._config.p_controllers[controller_index].nslope))
        for nphase in range(cube_phase.shape[0]):
            if ((push_pull is True) or (with_turbu is True)
                ):  # with turbulence/aberrations => push/pullADOPT/projects/cosmic/
                self._wfs.set_ncpa_wfs(wfs_index, cube_phase[nphase, :, :])
                devpos = self.apply_volts_and_get_slopes(controller_index,
                                                         turbu=with_turbu, noise=noise)
                self._wfs.set_ncpa_wfs(wfs_index, -cube_phase[nphase, :, :])
                devmin = self.apply_volts_and_get_slopes(controller_index,
                                                         turbu=with_turbu, noise=noise)
                imat_phase[nphase, :] = (devpos - devmin) / 2
            else:  # No turbulence => push only
                self._rtc.open_loop(controller_index)  # open_loop
                self._wfs.set_ncpa_wfs(wfs_index, cube_phase[nphase, :, :])
                imat_phase[nphase, :] = self.apply_volts_and_get_slopes(
                        controller_index, noise=noise)
        self._wfs.set_ncpa_wfs(wfs_index,
                          cube_phase[nphase, :, :] * 0.)  # Remove the Phase on WFS
        _ = self.apply_volts_and_get_slopes(controller_index, turbu=with_turbu,
                                            noise=noise)

        return imat_phase

    def compute_modal_residuals(self, projection_matrix : np.ndarray,
                                *, selected_actus : np.ndarray=None) -> np.ndarray:
        """ Computes the modal residual coefficients of the residual phase.

        It supposed that roket is enabled, and the associated GEO controller is index 1.

        Uses the projection matrix computed from compute_modes_to_volts_basis (modalBasis module)

        Args:
            projection_matrix : (np.ndarray) : Modal projection matrix

        Kwargs:
            selected_actus : (np.ndarray) : TODO : description

        Returns:
            ai : (np.ndarray) : Modal coefficients
        """
        try:
            self._rtc.do_control(1, sources=self._target.sources)
        except:
            return [0]
        v = self._rtc.get_command(1)  # We compute here the residual phase on the DM modes. Gives the Equivalent volts to apply/
        if (selected_actus is None):
            ai = projection_matrix.dot(v) * 1000.  # np rms units
        else:  # Slaving actus case
            v2 = v[:-2][list(
                    selected_actus)]  # If actus are slaved then we select them.
            v3 = v[-2:]
            ai = projection_matrix.dot(np.concatenate((v2, v3))) * 1000.
        return ai
