import numpy as np

import Octopus
from shesha.constants import CentroiderType, WFSType
from shesha.sim.simulator import load_config_from_file

from .benchSupervisor import BenchSupervisor, naga_context, rtc_standalone


class RTCSupervisor(BenchSupervisor):

    def __init__(self, configFile: str=None, BRAHMA: bool=False):
        '''
        Init the COMPASS wih the configFile
        '''
        self._sim = lambda: None
        self.cam = None
        self.rtc = None
        self.npix = 0
        self.BRAHMA = BRAHMA
        self._sim = lambda x: None

        self._sim.is_init = False  # type: bool
        self._sim.loaded = False  # type: bool
        self._sim.config = None  # type: Any # types.ModuleType ?

        if configFile is not None:
            self.loadConfig(configFile)

    def clearInitSim(self) -> None:
        """
        Delete objects initialized in a previous simulation
        """
        if self._sim.loaded and self._sim.is_init:
            self._sim.iter = 0

            del self._sim.rtc
            self._sim.rtc = None

            # del self._sim.c  # What is this supposed to do ... ?
            # self._sim.c = None

        self._sim.is_init = False

    def forceContext(self) -> None:
        '''
        Clear the initialization of the simulation
        '''
        ...

    def singleNext(self, moveAtmos: bool=True, showAtmos: bool=True, getPSF: bool=False,
                   getResidual: bool=False) -> None:
        '''
        Move atmos -> getSlope -> applyControl ; One integrator step
        '''
        # print("Wait a frame...")
        self.fakewfs.recv(self.frame, 0)
        p_wfs = self._sim.config.p_wfss[0]
        if p_wfs.type == WFSType.SH:
            self.rtc.load_rtc_img(0, self.frame)
            #for SH
            self.rtc.fill_rtc_bincube(0, self.npix)
        elif p_wfs.type == WFSType.PYRHR:
            self.rtc.load_rtc_pyrimg(0, self.frame)
        else:
            raise RuntimeError("WFS Type not usable")
        self.rtc.do_centroids(0)
        self.rtc.do_control(0)
        self.rtc.save_com(0)
        # print("Send a command")
        comms = self.rtc.get_com(0)
        self.fakedms.send(comms)

    def loadConfig(self, configFile: str) -> None:
        '''
        Init the COMPASS wih the configFile
        '''
        load_config_from_file(self._sim, configFile)

    def initConfig(self) -> None:
        '''
        Initialize the simulation
        '''
        wfsNb = len(self._sim.config.p_wfss)
        if wfsNb > 1:
            raise RuntimeError("multi WFS not supported")
        p_wfs = self._sim.config.p_wfss[0]

        self.context = naga_context(devices=np.array([0], dtype=np.int32))

        print("->cam")
        self.frame = np.zeros((p_wfs._framesizex, p_wfs._framesizey), dtype=np.float32)
        self.fakewfs = Octopus.getInterface(**p_wfs._frameInterface)

        print("->dm")
        self.fakedms = Octopus.getInterface(**self._sim.config.p_dms[0]._actuInterface)

        print("->RTC")
        nact = self._sim.config.p_dms[0].nact
        nvalid = p_wfs._nvalid
        self.valid = Octopus.getInterface(**p_wfs._validsubsInterface)
        tmp_valid = np.zeros((2, self.valid.size // 2), dtype=np.float32)
        self.valid.recv(tmp_valid, 0)
        self._sim.config.p_nvalid = tmp_valid

        self.cmat = Octopus.getInterface(
                **self._sim.config.p_controllers[0]._cmatInterface)
        cMat_data = np.zeros((nact, nvalid * 2), dtype=np.float32)
        self.cmat.recv(cMat_data, 0)

        if p_wfs.type == WFSType.SH:
            self.npix = p_wfs.npix

            # if "p_nvalid" not in self._sim.config.__dict__.keys(
            # ) or self._sim.config.p_nvalid is None:
            #     import rtcData.DataInit as di
            #     dataS = di.makeSH(wfsNb=wfsNb, frameSize=self.fakewfs.data.md.size[0],
            #                       roiSize=p_wfs.nxsub,
            #                       subSize=self.npix)
            #     xvalid = dataS.data["roiTab"].data[0, :] / self.npix
            #     yvalid = dataS.data["roiTab"].data[1, :] / self.npix
            # else:
            xvalid = self._sim.config.p_nvalid[0, :] / self.npix
            yvalid = self._sim.config.p_nvalid[1, :] / self.npix
            offset = (self.npix - 1) / 2
            scale = 0.29005988378497927
        elif p_wfs.type == WFSType.PYRHR:
            xvalid = self._sim.config.p_nvalid[1, :]
            yvalid = self._sim.config.p_nvalid[0, :]
            offset = 0
            scale = self._sim.config.p_centroiders[0].pyrscale
        else:
            raise RuntimeError("WFS Type not usable")
        print("nvalid : %d" % nvalid)
        print("nact : %d" % nact)
        p_wfs._nvalid = nvalid

        gain = self._sim.config.p_controllers[0].gain
        print("gain : %f" % gain)
        self.rtc = rtc_standalone(self.context, wfsNb, [nvalid], nact,
                                  self._sim.config.p_centroiders[0].type, 1, offset,
                                  scale, brahma=self.BRAHMA)
        self.rtc.set_decayFactor(0, np.ones(nact, dtype=np.float32))
        self.rtc.set_matE(0, np.identity(nact, dtype=np.float32))
        self.rtc.set_mgain(0, np.ones(nact, dtype=np.float32) * gain)
        self.rtc.set_cmat(0, cMat_data)
        self.rtc.load_rtc_validpos(0, xvalid.astype(np.int32), yvalid.astype(np.int32))

        self._sim.is_init = True

    def getRawWFSImage(self, numWFS: int=0) -> np.ndarray:
        '''
        Get an image from the WFS
        '''
        return np.array(self.fakewfs.data)
