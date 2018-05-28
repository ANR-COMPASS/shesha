from abc import ABC, abstractmethod
import numpy as np


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
    def setPerturbationVoltage(self, command: np.ndarray) -> None:
        ...

    ''' Add this offset value to integrator (will be applied at the end of next iteration)'''

    @abstractmethod
    def getSlope(self) -> np.ndarray:
        ...

    ''' Immediately gets one slope vector for all WFS at the current state of the system '''

    @abstractmethod
    def singleNext(self, moveAtmos: bool=True, showAtmos: bool=True, getPSF: bool=False,
                   getResidual: bool=False) -> None:
        ...

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
    def getRawWFSImage(self, numWFS: int=0) -> np.ndarray:
        ...

    ''' Get an image from the WFS '''

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
