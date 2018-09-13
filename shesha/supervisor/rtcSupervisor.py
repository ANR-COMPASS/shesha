import numpy as np

import Octopus
from shesha.constants import CentroiderType, WFSType
from shesha.util.utilities import load_config_from_file

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

    def loop(self, n: int=1, monitoring_freq: int=100, **kwargs):
        """
        Perform the AO loop for n iterations

        :parameters:
            n: (int): (optional) Number of iteration that will be done
            monitoring_freq: (int): (optional) Monitoring frequency [frames]
        """
        self._sim.loop(n, monitoring_freq=monitoring_freq)

    def singleNext(self, moveAtmos: bool=True, showAtmos: bool=True, getPSF: bool=False,
                   getResidual: bool=False) -> None:
        '''
        Move atmos -> getSlope -> applyControl ; One integrator step
        '''
        p_wfs = self._sim.config.p_wfss[0]

        try:
            # from GPUIPCInterfaceWrap import GPUIPCInterfaceFloat
            # print("Wait a frame...")
            if type(self.fakewfs) is not GPUIPCInterfaceFloat:
                raise RuntimeError("Fallback to basic OCtopus API")
            if p_wfs.type == WFSType.SH:
                self.fakewfs.wait()
                self.rtc.load_rtc_img_gpu(0, np.array(self.fakewfs.buffer, copy=False))
                self.rtc.fill_rtc_bincube(0, self.npix)
                # print("Received a frame using GPUIPCInterfaceFloat...")
            else:
                raise RuntimeError("WFS Type not usable")
        except:
            self.fakewfs.recv(self.frame, 0)
            p_wfs = self._sim.config.p_wfss[0]
            if p_wfs.type == WFSType.SH:
                self.rtc.d_centro[0].load_img(self.frame, self.frame.shape[0])
                #for SH
                self.rtc.d_centro[0].fill_bincube(self.npix)
            elif p_wfs.type == WFSType.PYRHR or p_wfs.type == WFSType.PYRLR:
                self.rtc.d_centro[0].load_pyrimg(self.frame, self.frame.shape[0])
            else:
                raise RuntimeError("WFS Type not usable")
        # print("frame")
        # print(self.frame)
        self.rtc.do_centroids(0)
        # print("slopes")
        # print(self.rtc.get_centroids(0))
        self.rtc.do_control(0)
        self.rtc.d_control[0].command_delay()
        # print("Send a command")
        comms = np.array(self.rtc.d_control[0].d_com)
        # print("comms")
        # print(comms)
        self.fakedms.send(comms)

    def loadConfig(self, configFile: str=None) -> None:
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

        # self.context = naga_context.get_instance_1gpu(0)
        self.context = naga_context.get_instance_ngpu(1, np.array([0], dtype=np.int32))

        print("->cam")
        self.frame = np.zeros((p_wfs._framesizex, p_wfs._framesizey), dtype=np.float32,
                              order="F")
        self.fakewfs = Octopus.getInterface(**p_wfs._frameInterface)

        print("->dm")
        self.fakedms = Octopus.getInterface(**self._sim.config.p_dms[0]._actuInterface)

        print("->RTC")
        nact = self._sim.config.p_dms[0].nact
        nvalid = p_wfs._nvalid
        self.cmat = Octopus.getInterface(
                **self._sim.config.p_controllers[0]._cmatInterface)
        cMat_data = np.zeros((nact, nvalid * 2), dtype=np.float32, order="F")
        self.cmat.recv(cMat_data, 0)

        self.valid = Octopus.getInterface(**p_wfs._validsubsInterface)

        if p_wfs.type == WFSType.SH:
            tmp_valid = np.zeros((2, nvalid), dtype=np.int32)
            self.valid.recv(tmp_valid, 0)
            self._sim.config.p_nvalid = tmp_valid

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

            xvalid = tmp_valid[0, :] // self.npix
            yvalid = tmp_valid[1, :] // self.npix
            offset = (self.npix + 1) / 2
            scale = p_wfs.pixsize
        elif p_wfs.type == WFSType.PYRHR or p_wfs.type == WFSType.PYRLR:
            tmp_valid = np.zeros((2, nvalid * 4), dtype=np.int32)
            self.valid.recv(tmp_valid, 0)
            self._sim.config.p_nvalid = tmp_valid
            xvalid = tmp_valid[0, :]
            yvalid = tmp_valid[1, :]
            offset = 0
            scale = self._sim.config.p_centroiders[0].pyrscale
        else:
            raise RuntimeError("WFS Type not usable")
        print("nvalid : %d" % nvalid)
        print("nact : %d" % nact)
        p_wfs._nvalid = nvalid
        p_wfs.set_validsubsx(xvalid)
        p_wfs.set_validsubsy(yvalid)

        gain = self._sim.config.p_controllers[0].gain
        print("gain : %f" % gain)
        self.rtc = rtc_standalone(self.context, wfsNb, [nvalid], nact,
                                  self._sim.config.p_centroiders[0].type, 1, offset,
                                  scale, brahma=self.BRAHMA)
        self.rtc.d_control[0].set_decayFactor(np.ones(nact, dtype=np.float32))
        self.rtc.d_control[0].set_matE(np.identity(nact, dtype=np.float32))
        self.rtc.d_control[0].set_mgain(np.ones(nact, dtype=np.float32) * gain)
        self.rtc.d_control[0].set_cmat(cMat_data)
        self.rtc.d_centro[0].load_validpos(xvalid, yvalid, nvalid)

        self._sim.is_init = True

    def getRawWFSImage(self, numWFS: int=0) -> np.ndarray:
        '''
        Get an image from the WFS
        '''
        return np.array(self.fakewfs.data)
