import numpy as np

from shesha.constants import CentroiderType, WFSType
from shesha.init.dm_init import dm_init_standalone
from shesha.init.rtc_init import rtc_standalone
from shesha.sutra_wrap import carmaWrap_context

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
        return self.config

    def setCommand(self, nctrl: int, command: np.ndarray) -> None:
        ''' TODO
        Immediately sets provided command to DMs - does not affect integrator
        '''
        # self.dm.set_command(command=command)
        self.rtc.d_control[nctrl].set_com(command, command.size)

    def setOneActu(self, nctrl: int, ndm: int, nactu: int, ampli: float = 1,
                   reset: bool = True) -> None:
        '''
        Push the selected actuator
        '''
        command = self.dm.get_command()
        if reset:
            command *= 0
        command[nactu] = ampli
        self.dm.set_command(command)

    def setPerturbationVoltage(self, nControl: int, name: str,
                               command: np.ndarray) -> None:
        '''
        Add this offset value to integrator (will be applied at the end of next iteration)
        '''
        if len(command.shape) == 1:
            self.rtc.d_control[nControl].add_perturb_voltage(name, command, 1)
        elif len(command.shape) == 2:
            self.rtc.d_control[nControl].add_perturb_voltage(name, command,
                                                             command.shape[0])
        else:
            raise AttributeError("command should be a 1D or 2D array")

    def getWfsImage(self, numWFS: int = 0) -> np.ndarray:
        '''
        Get an image from the WFS
        '''
        return self.cam.getFrame(1)

    def getCentroids(self, nControl: int = 0):
        '''
        Return the centroids of the nControl controller
        '''
        return np.array(self.rtc.d_control[nControl].d_centroids)

    def getSlope(self) -> np.ndarray:
        '''
        Immediately gets one slope vector for all WFS at the current state of the system
        '''
        return self.computeSlopes()

    def getCom(self, nControl: int = 0):
        '''
        Get command from nControl controller
        '''
        return np.array(self.rtc.d_control[nControl].d_com)

    def getErr(self, nControl: int = 0):
        '''
        Get command increment from nControl controller
        '''
        return np.array(self.rtc.d_control[nControl].d_err)

    def getVoltage(self, nControl: int = 0):
        '''
        Get voltages from nControl controller
        '''
        return np.array(self.rtc.d_control[nControl].d_voltage)

    def setIntegratorLaw(self, nControl: int = 0):
        self.rtc.d_control[nControl].set_commandlaw("integrator")

    def setDecayFactor(self, decay, nControl: int = 0):
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

    def computeSlopes(self, do_centroids=False, nControl: int = 0):
        if do_centroids:
            self.rtc.do_centroids(nControl)
        return self.getCentroids(nControl)

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

    def singleNext(self, moveAtmos: bool = True, showAtmos: bool = True,
                   getPSF: bool = False, getResidual: bool = False) -> None:
        '''
        Move atmos -> getSlope -> applyControl ; One integrator step
        '''
        self.frame = self.getWfsImage()
        if self.config.p_wfss[0].type == WFSType.SH:
            #for SH
            self.rtc.d_centro[0].load_img(self.frame, self.frame.shape[0])
            self.rtc.d_centro[0].fill_bincube(self.npix)
        elif self.config.p_wfss[0].type == WFSType.PYRHR:
            self.rtc.d_centro[0].load_pyr_img(self.frame, self.frame.shape[0])
        self.rtc.do_centroids(0)
        self.rtc.do_control(0)
        self.rtc.do_clipping(0, -1e5, 1e5)
        self.rtc.comp_voltage(0)
        self.dm.set_command(np.array(self.rtc.d_control[0].d_voltage))
        self.rtc.publish()

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

    def setGain(self, gainMat) -> None:
        '''
        Set the scalar gain of feedback controller loop
        '''
        if type(gainMat) in [int, float]:
            gainMat = np.ones(np.sum(self.rtc.d_control[0].nactu),
                              dtype=np.float32) * gainMat
        self.rtc.d_control[0].set_mgain(gainMat)

    def setCommandMatrix(self, cMat: np.ndarray) -> None:
        '''
        Set the cmat for the controller to use
        '''
        self.rtc.d_control[0].set_cmat(cMat)

    def getRawWFSImage(self, numWFS: int = 0) -> np.ndarray:
        '''
        Get an image from the WFS
        '''
        return self.frame

    def getTarImage(self, tarID, expoType: str = "se") -> np.ndarray:
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

    def __init__(self, configFile: str = None, BRAHMA: bool = False):
        '''
        Init the COMPASS wih the configFile
        '''
        self.cam = None
        self.rtc = None
        self.npix = 0
        self.frame = None
        self.BRAHMA = BRAHMA

        if configFile is not None:
            self.loadConfig(configFile=configFile)

    def __repr__(self):
        return str(self.rtc)

    def resetDM(self, nDM: int) -> None:
        '''
        Reset the DM number nDM
        '''
        raise NotImplementedError("Not implemented")

    def resetCommand(self, nctrl: int = -1) -> None:
        '''
        Reset the nctrl Controller command buffer, reset all controllers if nctrl  == -1
        '''
        if (nctrl == -1):  # All Dms reset
            for control in self.rtc.d_control:
                control.d_com.reset()
        else:
            self.rtc.d_control[nctrl].d_com.reset()

    def resetPerturbationVoltage(self, nControl: int = 0) -> None:
        '''
        Reset the perturbation voltage of the nControl controller
        '''
        if self.rtc.d_control[nControl].d_perturb is not None:
            self.rtc.d_control[nControl].d_perturb.reset()

    def loadConfig(self, configFile: str = None, sim=None) -> None:
        '''
        Init the COMPASS wih the configFile
        '''
        from shesha.util.utilities import load_config_from_file
        load_config_from_file(self, configFile)

    def isInit(self) -> bool:
        '''
        return the status on COMPASS init
        '''
        return self.is_init

    def clearInitSim(self) -> None:
        '''
        Clear the initialization of the simulation
        '''
        self.clear_init()

    def initConfig(self) -> None:
        '''
        Initialize the simulation
        '''
        import hraa.devices.camera as m_cam
        from hraa.lib.camSDK import cam_attributes
        Camera = m_cam.getCamClass(self.config.p_cams[0].type)
        print("->cam")
        self.cam = Camera(
                self.config.p_cams[0].camAddr,
                cam_attributes(self.config.p_cams[0].width, self.config.p_cams[0].height,
                               self.config.p_cams[0].offset_w,
                               self.config.p_cams[0].offset_h,
                               self.config.p_cams[0].expo_usec,
                               self.config.p_cams[0].framerate))

        print("->RTC")
        wfsNb = len(self.config.p_wfss)
        p_wfs = self.config.p_wfss[0]
        if wfsNb > 1:
            raise RuntimeError("multi WFS not supported")

        if (hasattr(self.config, 'p_loop') and self.config.p_loop.devices.size > 1):
            self.c = carmaWrap_context.get_instance_ngpu(self.config.p_loop.devices.size,
                                                         self.config.p_loop.devices)
        else:
            self.c = carmaWrap_context.get_instance_1gpu(self.config.p_loop.devices[0])

        if p_wfs.type == WFSType.SH:
            self.npix = p_wfs.npix
            if p_wfs._validsubsx is None or \
                    p_wfs._validsubsy is None:
                # import rtcData.DataInit as di
                # dataS = di.makeSH(wfsNb=wfsNb, frameSize=self.cam.getWidth(),
                #                   roiSize=p_wfs.nxsub, subSize=self.npix)
                # p_wfs._nvalid = dataS.data["roiTab"].data.shape[1]
                # p_wfs._validsubsx = dataS.data["roiTab"].data[0, :]
                # p_wfs._validsubsy = dataS.data["roiTab"].data[1, :]

                from hraa.tools.doit import makessp
                roiTab = makessp(p_wfs.nxsub, obs=0., rmax=0.98)
                # for pos in self.roiTab: pos *= self.pitch
                p_wfs._nvalid = roiTab[0].size
                p_wfs._validsubsx = roiTab[0]
                p_wfs._validsubsy = roiTab[1]
            else:
                p_wfs._nvalid = p_wfs._validsubsx.size

            nvalid = np.array([p_wfs._nvalid], dtype=np.int32)
            print("nvalid : %d" % nvalid)
            offset = (p_wfs.npix - 1) / 2
            scale = 1
            gain = 1
            nact = self.config.p_dms[0].nact

            self.rtc = rtc_standalone(self.context, wfsNb, nvalid, nact,
                                      self.config.p_centroiders[0].type, 1, offset * 0,
                                      scale, brahma=True)
            # put pixels in the SH grid coordonates
            self.rtc.d_centro[0].load_validpos(p_wfs._validsubsx, p_wfs._validsubsy,
                                               nvalid)

            cMat = np.zeros((nact, 2 * nvalid[0]), dtype=np.float32)
            self.rtc.d_control[0].set_cmat(cMat)
            self.rtc.d_control[0].set_decayFactor(
                    np.ones(nact, dtype=np.float32) * (gain - 1))
            self.rtc.d_control[0].set_matE(np.identity(nact, dtype=np.float32))
            self.rtc.d_control[0].set_mgain(np.ones(nact, dtype=np.float32) * -gain)

        elif p_wfs.type == WFSType.PYRHR or p_wfs.type == WFSType.PYRLR:
            raise RuntimeError("PYRHR not usable")

        self.is_init = True

    def getFrameCounter(self) -> int:
        '''
        return the current frame counter of the loop
        '''
        return self.iter

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

    def setCentroThresh(self, nCentro: int = 0, thresh: float = 0.):
        """
        Set the threshold value of a thresholded COG

        Parameters
        ------------
        nCentro: (int): centroider index
        thresh: (float): new threshold value
        """
        self.rtc.d_centro[nCentro].set_threshold(thresh)

    def setPyrModulation(self, pyrMod: float) -> None:
        '''
        Set pyramid modulation value - in l/D units
        '''
        raise NotImplementedError("Not implemented")

    def setPyrMethod(self, pyrMethod, nCentro: int = 0):
        '''
        Set pyramid compute method
        '''
        self.rtc.d_centro[nCentro].set_pyr_method(pyrMethod)  # Sets the pyr method
        self.rtc.do_centroids(0)  # To be ready for the next getSlopes
        print("PYR method set to " + self.rtc.d_centro[nCentro].pyr_method)

    def getPyrMethod(self, nCentro):
        return self.rtc.d_centro[nCentro].pyr_method
