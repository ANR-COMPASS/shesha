from .abstractSupervisor import AbstractSupervisor
import numpy as np

import shesha.constants as scons
from shesha.constants import CONST

from tqdm import trange


class CompassSupervisor(AbstractSupervisor):

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
        '''
        Returns the configuration in use, in a supervisor specific format ?
        '''
        return self._sim.config

    def enableAtmos(self, enable=True) -> None:
        ''' TODO
        Set or unset whether atmos is enabled when running loop (see singleNext)
        '''
        self._seeAtmos = enable

    def setCommand(self, command: np.ndarray) -> None:
        '''
        Immediately sets provided command to DMs - does not affect integrator
        '''
        self._sim.dms.set_full_comm((command).astype(np.float32).copy())

    def setPerturbationVoltage(self, command: np.ndarray) -> None:
        '''
        Add this offset value to integrator (will be applied at the end of next iteration)
        '''
        self._sim.rtc.set_perturbcom(0, command.astype(np.float32).copy())

    def getSlope(self) -> np.ndarray:
        '''
        Immediately gets one slope vector for all WFS at the current state of the system
        '''
        return self.computeSlopes()

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
        return np.empty(1)

    def next(self, nbiters, see_atmos=True):
        for _ in trange(nbiters):
            self._sim.next(see_atmos=see_atmos)

    def singleNext(self, moveAtmos: bool=True, showAtmos: bool=True, getPSF: bool=False,
                   getResidual: bool=False) -> None:
        '''
        Move atmos -> getSlope -> applyControl ; One integrator step
        '''
        self._sim.next(see_atmos=showAtmos)  # why not self._seeAtmos?

    def closeLoop(self) -> None:
        '''
        DM receives controller output + pertuVoltage
        '''
        self._sim.rtc.set_openloop(0, 0)  # closeLoop

    def openLoop(self) -> None:
        '''
        Integrator computation goes to /dev/null but pertuVoltage still applied
        '''
        self._sim.rtc.set_openloop(0, 1)  # openLoop

    def setRefSlopes(self, refSlopes: np.ndarray) -> None:
        '''
        Set given ref slopes in controller
        '''
        self._sim.rtc.set_centroids_ref(0, refSlopes)

    def getRefSlopes(self) -> np.ndarray:
        '''
        Get the currently used reference slopes
        '''
        self._sim.rtc.get_centroids_ref(0)

    def setGain(self, gainMat) -> None:
        '''
        Set the scalar gain of feedback controller loop
        '''
        if ((type(gainMat) is float) or (type(gainMat) is int)):
            gainMat = np.ones(
                    np.sum(self._sim.config.p_controller0.nactu),
                    dtype=np.float32) * gainMat
        self._sim.rtc.set_mgain(0, gainMat)

    def setCommandMatrix(self, cMat: np.ndarray) -> None:
        '''
        Set the cmat for the controller to use
        '''
        self._sim.rtc.set_cmat(0, cMat)

    def setNoise(self, noise, numwfs=0):
        '''
        Set noise value of WFS numwfs
        '''
        self._sim.wfs.set_noise(numwfs, noise)
        print("Noise set to: %d" % noise)

    def setPyrModulation(self, pyrMod: float) -> None:
        '''
        Set pyramid modulation value - in l/D units
        '''
        self._sim.rtc.set_pyr_ampl(0, pyrMod, self._sim.config.p_wfss,
                                   self._sim.config.p_tel)
        print("PYR modulation set to: %f L/D" % pyrMod)

    def setPyrMethod(self, pyrMethod):
        '''
        Set pyramid compute method
        '''
        self._sim.rtc.set_pyr_method(
                0, pyrMethod, self._sim.config.p_centroiders)  # Sets the pyr method
        print("PYR method set to: %d" % self._sim.rtc.get_pyr_method(0))

    def getRawWFSImage(self, numWFS: int=0) -> np.ndarray:
        '''
        Get an image from the WFS
        '''
        if self._sim.config.p_wfss[numWFS].type == scons.WFSType.PYRHR:
            return self._sim.wfs.get_pyrimg(numWFS)
        elif self._sim.config.p_wfss[numWFS].type == scons.WFSType.SH:
            return self._sim.wfs.get_binimg(numWFS)
        else:
            raise "WFSType not handled"

    def getTarImage(self, tarID, expoType: str="se") -> np.ndarray:
        '''
        Get an image from a target
        '''
        return self._sim.tar.get_image(tarID, bytes(expoType, "utf-8"))

    def getIntensities(self) -> np.ndarray:
        '''
        Return sum of intensities in subaps. Size nSubaps, same order as slopes
        '''
        raise NotImplementedError("Not implemented")
        return np.empty(1)

    def getAllDataLoop(self, nIter: int, slope: bool, command: bool, target: bool,
                       intensity: bool, targetPhase: bool) -> np.ndarray:
        '''
        Returns a sequence of data at continuous loop steps.
        Requires loop to be asynchronously running
        '''
        raise NotImplementedError("Not implemented")
        return np.empty(1)

    #  ____                  _ _   _        __  __      _   _               _
    # / ___| _ __   ___  ___(_) |_(_) ___  |  \/  | ___| |_| |__   ___   __| |___
    # \___ \| '_ \ / _ \/ __| | __| |/ __| | |\/| |/ _ \ __| '_ \ / _ \ / _` / __|
    #  ___) | |_) |  __/ (__| | |_| | (__  | |  | |  __/ |_| | | | (_) | (_| \__ \
    # |____/| .__/ \___|\___|_|\__|_|\___| |_|  |_|\___|\__|_| |_|\___/ \__,_|___/
    #       |_|

    def __init__(self, configFile: str=None, BRAHMA: bool=False):
        '''
        Init the COMPASS wih the configFile
        '''
        self._sim = None
        self._seeAtmos = False

        if configFile is not None:
            self.loadConfig(configFile, BRAHMA)

    def __repr__(self):
        return str(self._sim)

    def computeSlopes(self):
        for w in range(len(self._sim.config.p_wfss)):
            self._sim.wfs.comp_img(w)
        self._sim.rtc.do_centroids(0)
        return self._sim.rtc.get_centroids(0)

    def resetDM(self, numdm: int=-1) -> None:
        '''
        Reset the DM number nDM or all DMs if  == -1
        '''
        if (numdm == -1):  # All Dms reset
            for numdm in range(len(self._sim.config.p_dms)):
                self._sim.dms.resetdm(self._sim.config.p_dms[numdm].type,
                                      self._sim.config.p_dms[numdm].alt)
        else:
            self._sim.dms.resetdm(self._sim.config.p_dms[numdm].type,
                                  self._sim.config.p_dms[numdm].alt)

    def resetSimu(self, noiseList):
        self.resetTurbu()
        self.resetNoise(noiseList)

    def resetTurbu(self):
        ilayer = 0
        for layerAlt in self._sim.atm.list_alt().tolist():
            self._sim.atm.set_seed(layerAlt, 1234 + ilayer)
            self._sim.atm.refresh_screen(layerAlt)
            ilayer += 1

    def resetNoise(self, noiseList):
        for nwfs in range(len(self._sim.config.p_wfss)):
            self._sim.wfs.set_noise(nwfs, noiseList[nwfs], 1234 + nwfs)

    def resetStrehl(self, nTar: int) -> None:
        '''
        Reset the Strehl Ratio of the target nTar
        '''
        self._sim.tar.reset_strehl(nTar)

    def loadConfig(self, configFile: str, BRAMA: bool=False) -> None:
        '''
        Init the COMPASS wih the configFile
        '''
        if self._sim is None:
            if BRAMA:
                from shesha.sim.simulatorBrahma import SimulatorBrahma
                self._sim = SimulatorBrahma(configFile)
            else:
                from shesha.sim.simulator import Simulator
                self._sim = Simulator(configFile)
        else:
            self._sim.clear_init()
            self._sim.load_from_file(configFile)

    def isInit(self) -> bool:
        '''
        return the status on COMPASS init
        '''
        return self._sim.is_init

    def clearInitSim(self) -> None:
        '''
        Clear the initialization of the simulation
        '''
        self._sim.clear_init()

    def forceContext(self) -> None:
        '''
        Clear the initialization of the simulation
        '''
        self._sim.force_context()

    def initConfig(self) -> None:
        '''
        Initialize the simulation
        '''
        self._sim.init_sim()

    def getAtmScreen(self, alt: int) -> np.ndarray:
        '''
        return the atmos screen at the altitude alt
        '''
        return self._sim.atm.get_screen(alt)

    def getWfsPhase(self, numWFS: int) -> np.ndarray:
        '''
        return the WFS screen of WFS number numWFS
        '''
        return self._sim.wfs.get_phase(numWFS)

    def getDmPhase(self, dm_type: str, alt: int) -> np.ndarray:
        '''
        return the DM screen of type dm_type conjugatide at the altitude alt
        '''
        return self._sim.dms.get_dm(dm_type, alt)

    def getTarPhase(self, numTar: int) -> np.ndarray:
        '''
        return the target screen of target number numTar
        '''
        return self._sim.tar.get_phase(numTar)

    def getPyrHRImage(self, numWFS: int=0) -> np.ndarray:
        '''
        Get an HR image from the WFS
        '''
        if self._sim.config.p_wfss[numWFS].type == scons.WFSType.PYRHR:
            return self._sim.wfs.get_pyrimghr(numWFS)
        else:
            raise "WFSType not handled"

    def getSlopeGeom(self, numWFS: int) -> np.ndarray:
        '''
        return the slopes geom of WFS number numWFS
        '''
        self._sim.rtc.do_centroids_geom(0)
        slopesGeom = self._sim.rtc.get_centroids(0)
        self._sim.rtc.do_centroids(0)
        return slopesGeom

    def getStrehl(self, numTar: int) -> np.ndarray:
        '''
        return the Strehl Ratio of target number numTar
        '''
        self._sim.tar.comp_image(numTar)
        return self._sim.tar.get_strehl(numTar)

    def getFrameCounter(self) -> int:
        '''
        return the current frame counter of the loop
        '''
        return self._sim.iter
