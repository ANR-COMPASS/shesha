## @package   shesha.supervisor.aoSupervisor
## @brief     Abstract layer for initialization and execution of a AO supervisor
## @author    COMPASS Team <https://github.com/ANR-COMPASS>
## @version   4.3.1
## @date      2011/01/28
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

from .abstractSupervisor import AbstractSupervisor
from shesha.constants import CentroiderType
import numpy as np
from tqdm import trange


class AoSupervisor(AbstractSupervisor):

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

    def getConfig(self):
        ''' Returns the configuration in use, in a supervisor specific format ? '''
        return self.config

    def loadFlat(self, flat: np.ndarray, nctrl: int = 0):
        """
        Load flat field for the given controller

        """
        self.rtc.d_centro[nctrl].set_flat(flat, flat.shape[0])

    def loadBackground(self, background: np.ndarray, nctrl: int = 0):
        """
        Load background for the given controller

        """
        self.rtc.d_centro[nctrl].set_dark(background, background.shape[0])

    def computeSlopes(self, nControl: int = 0):
        self.rtc.do_centroids(nControl)
        return self.getCentroids(nControl)

    def getWfsImage(self, numWFS: int = 0, calPix=False) -> np.ndarray:
        '''
        Get an image from the WFS
        '''
        if (calPix):
            if self.rtc.d_centro[numWFS].d_img is None:
                return np.array(self._sim.wfs.d_wfs[numWFS].d_binimg)
            if self.rtc.d_centro[numWFS].type == CentroiderType.MASKEDPIX:
                self.rtc.d_centro[numWFS].fill_selected_pix(
                        self.rtc.d_control[0].d_centroids)
                return np.array(self.rtc.d_centro[numWFS].d_selected_pix)
            return np.array(self.rtc.d_centro[numWFS].d_img)
        else:
            if self.rtc.d_centro[numWFS].d_img is None:
                return np.array(self._sim.wfs.d_wfs[numWFS].d_binimg)
            return np.array(self.rtc.d_centro[numWFS].d_img_raw)

    def getAllDataLoop(self, nIter: int, slope: bool, command: bool, target: bool,
                       intensity: bool, targetPhase: bool) -> np.ndarray:
        '''
        Returns a sequence of data at continuous loop steps.
        Requires loop to be asynchronously running
        '''
        raise NotImplementedError("Not implemented")
        # return np.empty(1)

    def getIntensities(self) -> np.ndarray:
        '''
        Return sum of intensities in subaps. Size nSubaps, same order as slopes
        '''
        raise NotImplementedError("Not implemented")
        # return np.empty(1)

    def computeIMatModal(self, M2V: np.ndarray, pushVector: np.ndarray,
                         refOffset: np.ndarray, noise: bool,
                         useAtmos: bool) -> np.ndarray:
        '''
        TODO
        Computes a modal interaction matrix for the given modal matrix
        with given push values (length = nModes)
        around an (optional) offset value
        optionally with noise
        with/without atmos shown to WFS
        '''
        raise NotImplementedError("Not implemented")
        # return np.empty(1)

    def setPerturbationVoltage(self, nControl: int, name: str,
                               command: np.ndarray) -> None:
        ''' Add this offset value to integrator (will be applied at the end of next iteration)'''
        if len(command.shape) == 1:
            self.rtc.d_control[nControl].set_perturb_voltage(name, command, 1)
        elif len(command.shape) == 2:
            self.rtc.d_control[nControl].set_perturb_voltage(name, command,
                                                             command.shape[0])
        else:
            raise AttributeError("command should be a 1D or 2D array")

    def resetPerturbationVoltage(self, nControl: int) -> None:
        '''
        Reset the perturbation voltage of the nControl controller
        (i.e. will remove ALL perturbation voltages.)
        If you want to reset just one, see the function removePerturbationVoltage().
        '''
        self.rtc.d_control[nControl].reset_perturb_voltage()

    def removePerturbationVoltage(self, nControl: int, name: str) -> None:
        '''
        Remove the perturbation voltage called <name>, from the controller number <nControl>.
        If you want to remove all of them, see function resetPerturbationVoltage().
        '''
        self.rtc.d_control[nControl].remove_perturb_voltage(name)

    def getCentroids(self, nControl: int = 0):
        '''
        Return the centroids of the nControl controller
        '''
        return np.array(self.rtc.d_control[nControl].d_centroids)

    def getErr(self, nControl: int = 0) -> np.ndarray:
        '''
        Get command increment from nControl controller
        '''
        return np.array(self.rtc.d_control[nControl].d_err)

    def getVoltage(self, nControl: int = 0) -> np.ndarray:
        '''
        Get voltages from nControl controller
        '''
        return np.array(self.rtc.d_control[nControl].d_voltage)

    def setIntegratorLaw(self, nControl: int = 0):
        """
        Set the control law to Integrator

        Parameters
        ------------
        nControl: (int): controller index
        """
        self.rtc.d_control[nControl].set_commandlaw("integrator")

    def setDecayFactor(self, decay, nControl: int = 0):
        """
        Set the decay factor

        Parameters
        ------------
        nControl: (int): controller index
        """
        self.rtc.d_control[nControl].set_decayFactor(decay)

    def setEMatrix(self, eMat, nControl: int = 0):
        self.rtc.d_control[nControl].set_matE(eMat)

    def doRefslopes(self, nControl: int = 0):
        print("Doing refslopes...")
        self.rtc.do_centroids_ref(nControl)
        print("refslopes done")

    def resetRefslopes(self, nControl: int = 0):
        for centro in self.rtc.d_centro:
            centro.d_centroids_ref.reset()

    def closeLoop(self) -> None:
        '''
        DM receives controller output + pertuVoltage
        '''
        self.rtc.d_control[0].set_openloop(0)  # closeLoop

    def openLoop(self, rst=True) -> None:
        '''
        Integrator computation goes to /dev/null but pertuVoltage still applied
        '''
        self.rtc.d_control[0].set_openloop(1, rst)  # openLoop

    def setRefSlopes(self, refSlopes: np.ndarray) -> None:
        '''
        Set given ref slopes in controller
        '''
        self.rtc.set_centroids_ref(refSlopes)

    def getRefSlopes(self) -> np.ndarray:
        '''
        Get the currently used reference slopes
        '''
        refSlopes = np.empty(0)
        for centro in self.rtc.d_centro:
            refSlopes = np.append(refSlopes, np.array(centro.d_centroids_ref))
        return refSlopes

    def getImat(self, nControl: int = 0):
        """
        Return the interaction matrix of the controller

        Parameters
        ------------
        nControl: (int): controller index
        """
        return np.array(self.rtc.d_control[nControl].d_imat)

    def getCmat(self, nControl: int = 0):
        """
        Return the command matrix of the controller

        Parameters
        ------------
        nControl: (int): controller index
        """
        return np.array(self.rtc.d_control[nControl].d_cmat)

    # Warning: SH specific
    def setCentroThresh(self, nCentro: int = 0, thresh: float = 0.):
        """
        Set the threshold value of a thresholded COG

        Parameters
        ------------
        nCentro: (int): centroider index
        thresh: (float): new threshold value
        """
        self.rtc.d_centro[nCentro].set_threshold(thresh)

    # Warning: PWFS specific
    def getPyrMethod(self, nCentro):
        '''
        Get pyramid compute method
        '''
        return self.rtc.d_centro[nCentro].pyr_method

    # Warning: PWFS specific
    def setPyrMethod(self, pyrMethod, nCentro: int = 0):
        '''
        Set pyramid compute method
        '''
        self.rtc.d_centro[nCentro].set_pyr_method(pyrMethod)  # Sets the pyr method
        self.rtc.do_centroids(0)  # To be ready for the next getSlopes
        print("PYR method set to " + self.rtc.d_centro[nCentro].pyr_method)

    def getSlope(self) -> np.ndarray:
        '''
        Immediately gets one slope vector for all WFS at the current state of the system
        '''
        return self.computeSlopes()

    def next(self, nbiters, see_atmos=True):
        ''' Move atmos -> getSlope -> applyControl ; One integrator step '''
        for i in range(nbiters):
            self.singleNext(showAtmos=see_atmos)

    def setGain(self, gain) -> None:
        '''
        Set the scalar gain or mgain of feedback controller loop
        '''
        if self.rtc.d_control[0].command_law == 'integrator':  # Integrator law
            if np.isscalar(gain):
                self.rtc.d_control[0].set_gain(gain)
            else:
                raise ValueError("Cannot set array gain w/ generic + integrator law")
        else:  # E matrix mode
            if np.isscalar(gain):  # Automatic scalar expansion
                gain = np.ones(np.sum(self.rtc.d_control[0].nactu),
                               dtype=np.float32) * gain
            # Set array
            self.rtc.d_control[0].set_mgain(gain)

    def setCommandMatrix(self, cMat: np.ndarray) -> None:
        '''
        Set the cmat for the controller to use
        '''
        self.rtc.d_control[0].set_cmat(cMat)

    def getFrameCounter(self) -> int:
        '''
        return the current frame counter of the loop
        '''
        if not self.is_init:
            print('Warning - requesting frame counter of uninitialized BenchSupervisor.')
        return self.iter
