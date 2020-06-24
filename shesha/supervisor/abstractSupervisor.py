## @package   shesha.supervisor.abstractSupervisor
## @brief     Abstract layer for initialization and execution of a COMPASS supervisor
## @author    COMPASS Team <https://github.com/ANR-COMPASS>
## @version   4.4.2
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

from abc import ABC, abstractmethod
import numpy as np
from tqdm import trange


class AbstractSupervisor(ABC):

    @abstractmethod
    def getConfig(self):
        ...

    ''' Returns the configuration in use, in a supervisor specific format ? '''

    @abstractmethod
    def loadConfig(self):
        ...

    ''' Load the configuration for the supervisor'''

    @abstractmethod
    def initConfig(self):
        ...

    ''' Init the configuration for the supervisor'''

    @abstractmethod
    def setCommand(self, command: np.ndarray) -> None:
        ...

    ''' Immediately sets provided command to DMs - does not affect integrator '''

    @abstractmethod
    def setPerturbationVoltage(self, nControl: int, command: np.ndarray) -> None:
        ...

    ''' Add this offset value to integrator (will be applied at the end of next iteration)'''

    @abstractmethod
    def getSlope(self) -> np.ndarray:
        ...

    ''' Immediately gets one slope vector for all WFS at the current state of the system '''

    @abstractmethod
    def singleNext(self, moveAtmos: bool = True, showAtmos: bool = True,
                   getPSF: bool = False, getResidual: bool = False) -> None:
        ...

    def next(self, nbiters, see_atmos=True):
        for i in range(nbiters):
            print(i, end="\r")
            self.singleNext(showAtmos=see_atmos)

    ''' Move atmos -> getSlope -> applyControl ; One integrator step '''

    @abstractmethod
    def closeLoop(self) -> None:
        ...

    ''' DM receives controller output + pertuVoltage '''

    @abstractmethod
    def openLoop(self) -> None:
        ...

    ''' Integrator computation goes to /dev/null but pertuVoltage still applied '''

    @abstractmethod
    def setRefSlopes(self, refSlopes: np.ndarray) -> None:
        ...

    ''' Set given ref slopes in controller '''

    @abstractmethod
    def getRefSlopes(self) -> np.ndarray:
        ...

    ''' Get the currently used reference slopes '''

    @abstractmethod
    def setGain(self, gain: float):
        ...

    ''' Set the scalar gain of feedback controller loop '''

    @abstractmethod
    def setCommandMatrix(self, cMat: np.ndarray):
        ...

    ''' Set the cmat for the controller to use '''

    @abstractmethod
    def getTarImage(self, tarID):
        ...

    ''' Get an image from a target '''

    @abstractmethod
    def getWfsImage(self, numWFS: int = 0, calPix: bool = False) -> np.ndarray:
        ...

    ''' Get an image from the WFS. If calpix = True returns the calibrated image (background + flat + pixels selection )'''

    @abstractmethod
    def getIntensities(self):
        ...

    ''' Return sum of intensities in subaps. Size nSubaps, same order as slopes '''

    @abstractmethod
    def getAllDataLoop(self, nIter: int, slope: bool, command: bool, target: bool,
                       intensity: bool, targetPhase: bool) -> np.ndarray:
        ...

    '''
    Returns a sequence of data at continuous loop steps.
    Requires loop to be asynchronously running
    '''

    @abstractmethod
    def getFrameCounter(self):
        ...
