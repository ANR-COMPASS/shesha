import numpy as np

from shesha.constants import CentroiderType, WFSType
from shesha.init.dm_init import dm_init_standalone
from shesha.init.rtc_init import rtc_standalone
from shesha.sim.simulator import Simulator
from shesha.sim.simulatorBrahma import SimulatorBrahma
from shesha.sutra_bind.wrap import naga_context

from .abstractSupervisor import AbstractSupervisor


class BenchSupervisor(AbstractSupervisor):

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
        raise NotImplementedError("Not implemented")
        # self._seeAtmos = enable

    def setCommand(self, command: np.ndarray) -> None:
        '''
        Immediately sets provided command to DMs - does not affect integrator
        '''
        self._sim.dms.set_full_comm((command).astype(np.float32).copy())

    def setPerturbationVoltage(self, command: np.ndarray) -> None:
        '''
        Add this offset value to integrator (will be applied at the end of next iteration)
        '''
        self.rtc.set_perturbcom(0, command.astype(np.float32).copy())

    def getSlope(self) -> np.ndarray:
        '''
        Immediately gets one slope vector for all WFS at the current state of the system
        '''
        return self.rtc.get_centroids(0)

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

    def singleNext(self, moveAtmos: bool=True, showAtmos: bool=True, getPSF: bool=False,
                   getResidual: bool=False) -> None:
        '''
        Move atmos -> getSlope -> applyControl ; One integrator step
        '''
        self.frame = self.cam.getFrame()
        self.rtc.load_rtc_img(0, self.frame.astype(np.float32))
        if self._sim.config.p_wfss[0].type == WFSType.SH:
            #for SH
            self.rtc.fill_rtc_bincube(0, self.npix)
        self.rtc.do_centroids(0)
        self.rtc.do_control(0)
        self.rtc.save_com(0)

    def closeLoop(self) -> None:
        '''
        DM receives controller output + pertuVoltage
        '''
        self.rtc.set_openloop(0, 0)  # closeLoop

    def openLoop(self) -> None:
        '''
        Integrator computation goes to /dev/null but pertuVoltage still applied
        '''
        self.rtc.set_openloop(0, 1)  # openLoop

    def setRefSlopes(self, refSlopes: np.ndarray) -> None:
        '''
        Set given ref slopes in controller
        '''
        self.rtc.set_centroids_ref(0, refSlopes)

    def getRefSlopes(self) -> np.ndarray:
        '''
        Get the currently used reference slopes
        '''
        self.rtc.get_centroids_ref(0)

    def setGain(self, gain: float) -> None:
        '''
        Set the scalar gain of feedback controller loop
        '''
        self.rtc.set_gain(gain)

    def setCommandMatrix(self, cMat: np.ndarray) -> None:
        '''
        Set the cmat for the controller to use
        '''
        self.rtc.set_cmat(cMat)

    def setPyrModulation(self, pyrMod: float) -> None:
        '''
        Set pyramid modulation value - in l/D units
        '''
        raise NotImplementedError("Not implemented")

    def getRawWFSImage(self, numWFS: int=0) -> np.ndarray:
        '''
        Get an image from the WFS
        '''
        return self.frame

    def getTarImage(self, tarID, expoType: str="se") -> np.ndarray:
        '''
        Get an image from a target
        '''
        raise NotImplementedError("Not implemented")

    def getIntensities(self) -> np.ndarray:
        '''
        Return sum of intensities in subaps. Size nSubaps, same order as slopes
        '''
        raise NotImplementedError("Not implemented")

    def getAllDataLoop(self, nIter: int, slope: bool, command: bool, target: bool,
                       intensity: bool, targetPhase: bool) -> np.ndarray:
        '''
        Returns a sequence of data at continuous loop steps.
        Requires loop to be asynchronously running
        '''
        raise NotImplementedError("Not implemented")

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
        self.cam = None
        self.rtc = None
        self.npix = 0
        self.frame = None

        if configFile is not None:
            self.loadConfig(configFile, BRAHMA)

    def __repr__(self):
        return str(self._sim)

    def resetDM(self, nDM: int) -> None:
        '''
        Reset the DM number nDM
        '''
        self._sim.dms.resetdm(nDM)

    def loadConfig(self, configFile: str, BRAMA: bool=False) -> None:
        '''
        Init the COMPASS wih the configFile
        '''

        if self._sim is None:
            if BRAMA:
                self._sim = SimulatorBrahma(configFile)
            else:
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
        import hraa.devices.camera as m_cam

        Camera = m_cam.getCamClass(self._sim.config.p_cams[0].type)
        print("->cam")
        self.cam = Camera(
                self._sim.config.p_cams[0].camAddr, self._sim.config.p_cams[0].width,
                self._sim.config.p_cams[0].height, self._sim.config.p_cams[0].offset_w,
                self._sim.config.p_cams[0].offset_h,
                self._sim.config.p_cams[0].expo_usec,
                self._sim.config.p_cams[0].framerate)

        print("->RTC")
        wfsNb = len(self._sim.config.p_wfss)
        p_wfs = self._sim.config.p_wfss[0]
        if wfsNb > 1:
            raise RuntimeError("multi WFS not supported")

        if p_wfs.type == WFSType.SH:
            self.npix = p_wfs.npix

            if p_wfs._validsubsx is None or \
                    p_wfs._validsubsy is None:
                import rtcData.DataInit as di
                dataS = di.makeSH(wfsNb=wfsNb, frameSize=self.cam.getWidth(),
                                  roiSize=p_wfs.nxsub, subSize=self.npix)
                p_wfs._nvalid = dataS.data["roiTab"].data.shape[1]
                p_wfs._validsubsx = dataS.data["roiTab"].data[0, :]
                p_wfs._validsubsy = dataS.data["roiTab"].data[1, :]
            else:
                p_wfs._nvalid = p_wfs._validsubsx.size

            self.context = naga_context(devices=np.array([0], dtype=np.int32))
            nvalid = np.array([p_wfs._nvalid], dtype=np.int32)
            print("nvalid : %d" % nvalid)
            offset = (p_wfs.npix - 1) / 2
            scale = 1
            gain = 1
            nact = self._sim.config.p_dms[0].nact

            self.rtc = rtc_standalone(self.context, wfsNb, nvalid, nact,
                                      self._sim.config.p_centroiders[0].type, 1,
                                      offset * 0, scale)
            # put pixels in the SH grid coordonates
            self.rtc.load_rtc_validpos(0, p_wfs._validsubsx // self.npix,
                                       p_wfs._validsubsy // self.npix)

            cMat = np.zeros((nact, 2 * nvalid[0]), dtype=np.float32)
            self.rtc.set_cmat(0, cMat)
            self.rtc.set_decayFactor(0, np.ones(nact, dtype=np.float32) * (gain - 1))
            self.rtc.set_matE(0, np.identity(nact, dtype=np.float32))
            self.rtc.set_mgain(0, np.ones(nact, dtype=np.float32) * -gain)
        elif p_wfs.type == WFSType.PYRHR:
            raise RuntimeError("PYRHR not usable")
        self._sim.is_init = True

    def getWfsPhase(self, numWFS: int) -> np.ndarray:
        '''
        return the WFS screen of WFS number numWFS
        '''
        raise NotImplementedError("Not implemented")
        # return self._sim.atm.get_screen(numWFS)

    def getDmPhase(self, dm_type: str, alt: int) -> np.ndarray:
        '''
        return the DM screen of type dm_type conjugatide at the altitude alt
        '''
        return self._sim.dms.get_dm(dm_type, alt)

    def getTarPhase(self, numTar: int) -> np.ndarray:
        '''
        return the target screen of target number numTar
        '''
        raise NotImplementedError("Not implemented")
        # return self._sim.tar.get_phase(numTar)

    def getFrameCounter(self) -> int:
        '''
        return the current frame counter of the loop
        '''
        return self._sim.iter
