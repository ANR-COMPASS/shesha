""" @package shesha.supervisor.benchSupervisor

COMPASS simulation package

Initialization and execution of a Bench supervisor

"""

import numpy as np

from shesha.constants import CentroiderType, WFSType
from shesha.init.dm_init import dm_init_standalone
from shesha.init.rtc_init import rtc_standalone
from shesha.sutra_wrap import carmaWrap_context

from .abstractSupervisor import AbstractSupervisor

from typing import Callable


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

    def setOneActu(self, nctrl: int, ndm: int, nactu: int, ampli: float = 1,
                   reset: bool = True) -> None:
        '''
        Push the selected actuator
        '''
        command = self.getCommand()
        if reset:
            command *= 0
        command[nactu] = ampli
        self.setCommand(nctrl, command)

    def setPerturbationVoltage(self, nControl: int, name: str,
                               command: np.ndarray) -> None:
        '''
        Add this offset value to integrator (will be applied at the end of next iteration)
        '''
        if len(command.shape) == 1:
            self.rtc.d_control[nControl].set_perturb_voltage(name, command, 1)
        elif len(command.shape) == 2:
            self.rtc.d_control[nControl].set_perturb_voltage(name, command,
                                                             command.shape[0])
        else:
            raise AttributeError("command should be a 1D or 2D array")

    def getWfsImage(self, numWFS: int = 0, calPix=False) -> np.ndarray:
        '''
        Get an image from the WFS
        '''
        if (calPix):
            return np.array(self.rtc.d_centro[0].d_img)
        else:
            return np.array(self.rtc.d_centro[0].d_img_raw)

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

    def setCommand(self, nctrl: int, command: np.ndarray) -> None:
        ''' TODO
        Immediately sets provided command to DMs - does not affect integrator
        '''
        # Do stuff
        self.dmSetCallback(command)
        # Btw, update the RTC state with the information
        # self.rtc.d_control[nctrl].set_com(command, command.size)

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

    def getCom(self, nControl: int = 0) -> np.ndarray:
        '''
        Get command from DM, and set it back to nCtrl controller.
        These should be equivalent, unless an external source controls the DM as well
        '''
        # Do something
        command = self.dmGetCallback()
        # Btw, update the RTC state with the information
        # self.rtc.d_control[nControl].set_com(command, command.size)

        return command

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

    def loadNewWfsFrame(self, numWFS: int = 0) -> None:
        '''
            Acquire a new WFS frame, load, calibrate, centroid.
        '''

        self.frame = self.camCallback()
        self.rtc.d_centro[0].load_img(self.frame, self.frame.shape[0])
        self.rtc.d_centro[0].calibrate_img()
        self.rtc.do_centroids(0)

    def singleNext(self) -> None:
        '''
        Move atmos -> getSlope -> applyControl ; One integrator step
        '''
        self.loadNewWfsFrame()
        #self.rtc.do_control(0)
        self.rtc.do_clipping(0)
        self.rtc.comp_voltage(0)
        self.setCommand(0, np.array(self.rtc.d_control[0].d_voltage))
        if self.BRAHMA or self.CACAO:
            self.rtc.publish()

    def getAllDataLoop(self, nb):
        ...

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

    def forceContext(self) -> None:
        """
        Active all the GPU devices specified in the parameters file
        Required for using with widgets, due to multithreaded init
        and in case GPU 0 is not used by the simu
        """
        if self.isInit() and self.c is not None:
            current_Id = self.c.activeDevice
            for devIdx in range(len(self.config.p_loop.devices)):
                self.c.set_activeDeviceForce(devIdx)
            self.c.set_activeDevice(current_Id)

    #  ____                  _ _   _        __  __      _   _               _
    # / ___| _ __   ___  ___(_) |_(_) ___  |  \/  | ___| |_| |__   ___   __| |___
    # \___ \| '_ \ / _ \/ __| | __| |/ __| | |\/| |/ _ \ __| '_ \ / _ \ / _` / __|
    #  ___) | |_) |  __/ (__| | |_| | (__  | |  | |  __/ |_| | | | (_) | (_| \__ \
    # |____/| .__/ \___|\___|_|\__|_|\___| |_|  |_|\___|\__|_| |_|\___/ \__,_|___/
    #       |_|

    def __init__(self, configFile: str = None, BRAHMA: bool = False,
                 CACAO: bool = False):
        '''
        Init the COMPASS wih the configFile
        '''

        self.rtc = None
        self.frame = None
        self.BRAHMA = BRAHMA
        self.CACAO = CACAO
        self.dm = None

        if configFile is not None:
            self.loadConfig(configFile=configFile)

    def __repr__(self):

        s = '--- BenchSupervisor ---\nRTC: ' + repr(self.rtc)
        if hasattr(self, '_cam'):
            s += '\nCAM: ' + repr(self._cam)
        if hasattr(self, '_dm'):
            s += '\nDM: ' + repr(self._dm)
        return s

    def resetDM(self, nDM: int) -> None:
        '''
        Reset the DM number nDM
        '''
        if self.dm is not None:
            self.dm.reset_dm()

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
        Removes all the perturbation voltage buffers currently existing in this controller
        '''
        self.rtc.d_control[nControl].reset_perturb_voltage()

    def loadConfig(self, configFile: str = None, sim=None) -> None:
        '''
        Init the COMPASS wih the configFile
        '''
        from shesha.util.utilities import load_config_from_file
        load_config_from_file(self, configFile)

    def setCamCallback(self, camCallback: Callable):
        '''
        Set the externally defined function that allows to grab frames
        '''
        self.camCallback = camCallback

    def setDmCallback(self, dmGetCallback: Callable, dmSetCallback: Callable):
        '''
        Set the externally defined function that allows to grab frames
        '''
        self.dmGetCallback = dmGetCallback
        self.dmSetCallback = dmSetCallback

    def isInit(self) -> bool:
        '''
        return the status on COMPASS init
        '''
        return self.is_init

    def initConfig(self) -> None:
        '''
        Initialize the bench
        '''
        print("->CAM")
        if not hasattr(self, 'camCallback') or self.camCallback is None:
            print('No user provided camera getFrame handle. Creating from config file.')
            from hraa.devices.camera.cam_attributes import cam_attributes
            import hraa.devices.camera as m_cam
            Camera = m_cam.getCamClass(self.config.p_cams[0].type)
            self._cam = Camera(
                    self.config.p_cams[0].camAddr,
                    cam_attributes(self.config.p_cams[0].width,
                                   self.config.p_cams[0].height,
                                   self.config.p_cams[0].offset_w,
                                   self.config.p_cams[0].offset_h,
                                   self.config.p_cams[0].expo_usec,
                                   self.config.p_cams[0].framerate,
                                   self.config.p_cams[0].symcode))
            self._cam.acquisitionStart()
            self.camCallback = lambda: self._cam.getFrame(1)
        print("->DM")
        if not hasattr(self, 'dmSetCallback') or self.dmSetCallback is None:
            print('No user provided get setCommand handle. Creating from config file.')
            from hraa.devices.dm.kacou import Kacou
            self.dm = Kacou(
                    reset=True,
                    calibFits="/home/micado/codes/hraa/devices/dm/kacouCalib.fits")
            self.dmSetCallback = lambda cmd_dm: self.dm.set_command(
                    cmd_dm, useChecksum=True)
            self.dmGetCallback = lambda: self.dm.get_command()

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
                p_wfs._validsubsx = roiTab[0] * p_wfs.npix
                p_wfs._validsubsy = roiTab[1] * p_wfs.npix
            else:
                p_wfs._nvalid = p_wfs._validsubsx.size

            nvalid = np.array([p_wfs._nvalid], dtype=np.int32)
            print("nvalid : %d" % nvalid)
            offset = (p_wfs.npix - 1) / 2
            scale = 1
            gain = 1
            nact = self.config.p_dms[0].nact

            self.rtc = rtc_standalone(self.c, wfsNb, nvalid, nact,
                                      self.config.p_centroiders[0].type,
                                      self.config.p_controllers[0].delay, offset, scale,
                                      brahma=self.BRAHMA, cacao=self.CACAO)
            # put pixels in the SH grid coordonates
            self.rtc.d_centro[0].load_validpos(p_wfs._validsubsx, p_wfs._validsubsy,
                                               nvalid)
            if self.config.p_centroiders[0].type is CentroiderType.BPCOG:
                self.rtc.d_centro[0].set_nmax(self.config.p_centroiders[0].nmax)

            self.rtc.d_centro[0].set_npix(self.npix)

            cMat = np.zeros((nact, 2 * nvalid[0]), dtype=np.float32)

        elif p_wfs.type == WFSType.PYRHR or p_wfs.type == WFSType.PYRLR:
            nvalid = np.array([p_wfs._nvalid],
                              dtype=np.int32)  # Number of valid SUBAPERTURES
            nact = sum([p_dm.get_ntotact()
                        for p_dm in self.config.p_dms])  # Number of actu over all DMs
            gain = 1.
            self.rtc = rtc_standalone(self.c, wfsNb, nvalid, nact,
                                      self.config.p_centroiders[0].type,
                                      self.config.p_controllers[0].delay, 0, 1,
                                      brahma=self.BRAHMA, cacao=self.CACAO)

            self.rtc.d_centro[0].load_validpos(p_wfs._validsubsx, p_wfs._validsubsy,
                                               len(p_wfs._validsubsx))

            cMat = np.zeros((nact, p_wfs.nPupils * nvalid[0]), dtype=np.float32)
        else:
            raise ValueError('WFS type not supported')

        self.rtc.d_control[0].set_cmat(cMat)
        self.rtc.d_control[0].set_decayFactor(
                np.ones(nact, dtype=np.float32) * (gain - 1))
        self.rtc.d_control[0].set_matE(np.identity(nact, dtype=np.float32))
        self.rtc.d_control[0].set_mgain(np.ones(nact, dtype=np.float32) * -gain)

        self.is_init = True

    def getFrameCounter(self) -> int:
        '''
        return the current frame counter of the loop
        '''
        if not self.is_init:
            print('Warning - requesting frame counter of uninitialized BenchSupervisor.')
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
    def setPyrModulation(self, pyrMod: float) -> None:
        '''
        Set pyramid modulation value - in l/D units
        '''
        raise NotImplementedError("Not implemented")

    # Warning: PWFS specific
    def setPyrMethod(self, pyrMethod, nCentro: int = 0):
        '''
        Set pyramid compute method
        '''
        self.rtc.d_centro[nCentro].set_pyr_method(pyrMethod)  # Sets the pyr method
        self.rtc.do_centroids(0)  # To be ready for the next getSlopes
        print("PYR method set to " + self.rtc.d_centro[nCentro].pyr_method)

    # Warning: PWFS specific
    def getPyrMethod(self, nCentro):
        '''
        Get pyramid compute method
        '''
        return self.rtc.d_centro[nCentro].pyr_method
