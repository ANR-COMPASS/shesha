"""
Class SimulatorRTC: COMPASS simulation linked to real RTC with Octopus
"""
import os
import sys

import numpy as np

import Octopus
from shesha.constants import CentroiderType, WFSType
from shesha.util.utilities import load_config_from_file

from .simulator import Iterable, Simulator


class SimulatorRTC(Simulator):
    """
        Class SimulatorRTC: COMPASS simulation linked to real RTC with Octopus
    """

    def __init__(self, filepath: str=None, fastMode: bool=False,
                 location: str="CPUSHM") -> None:
        """
        Initializes a Simulator instance

        :parameters:
            filepath: (str): (optional) path to the parameters file
            rtcfilepath: (str): (optional) path to the rtc parameters file
            use_DB: (bool): (optional) flag to use dataBase system
        """
        Simulator.__init__(self)
        self.rtcconf = lambda x: None
        self.frame = None
        self.fastMode = fastMode
        self.location = location

        if filepath is not None:
            self.load_from_file(filepath)

    def load_from_file(self, filepath: str) -> None:
        """
        Load the parameters from the parameters file

        :parameters:
            filepath: (str): path to the parameters file

        """
        if "KRAKEN_ROOT" in os.environ:
            kraken_pathfile = "{}/KrakenConf/data".format(os.environ["KRAKEN_ROOT"])
            if (kraken_pathfile not in sys.path):
                sys.path.insert(0, kraken_pathfile)

        load_config_from_file(self.rtcconf, filepath)
        Simulator.load_from_file(self, self.rtcconf.config.p_sim)
        if self.fastMode:
            self.config.p_controllers[0].set_type("generic")

    def init_sim(self) -> None:
        Simulator.init_sim(self)

        wfsNb = len(self.rtcconf.config.p_wfss)
        if wfsNb > 1:
            raise RuntimeError("multi WFS not supported")
        p_wfs = self.rtcconf.config.p_wfss[0]

        nact = self.rtcconf.config.p_dms[0].nact
        framesizex = p_wfs._framesizex
        framesizey = p_wfs._framesizey
        nvalid = p_wfs._nvalid
        self.frame = np.array(self.wfs.d_wfs[0].d_binimg)
        nslopes = self.rtc.d_centro[0].nslopes

        if self.frame.shape != (framesizex, framesizey):
            raise RuntimeError("framesize not match with the simulation")

        if self.rtc.d_control[0].d_voltage.nbElem != nact:
            raise RuntimeError("nact not match with the simulation")

        if np.any(self.rtc.d_control[0].d_cmat.shape != [nact, nslopes]):
            raise RuntimeError("cmat not match with the simulation")

        self.fakewfs = Octopus.getInterface(**p_wfs._frameInterface)

        self.frame = np.ones((framesizex, framesizey), dtype=np.float32, order="F")
        self.comp = np.zeros(nact, dtype=np.float32)
        self.fakedms = Octopus.getInterface(
                **self.rtcconf.config.p_dms[0]._actuInterface)

        tmp_cmat = np.array(self.rtc.d_control[0].d_cmat)
        self.cmat = Octopus.getInterface(
                **self.rtcconf.config.p_controllers[0]._cmatInterface)
        self.cmat.send(tmp_cmat)

        tmp_valid = self.config.p_wfss[0].get_validsub()
        self.valid = Octopus.getInterface(
                **self.rtcconf.config.p_wfss[0]._validsubsInterface)
        if self.rtcconf.config.p_wfss[0].type == WFSType.SH:
            self.valid.send(tmp_valid * self.rtcconf.config.p_wfss[0].npix)
        elif self.rtcconf.config.p_wfss[0].type == WFSType.PYRHR or self.rtcconf.config.p_wfss[0].type == WFSType.PYRLR:
            self.valid.send(tmp_valid)
        else:
            raise RuntimeError("WFS Type not usable")

        if self.fastMode:
            p_wfs = self.rtcconf.config.p_wfss[0]
            self.frame = np.random.random(
                    (p_wfs._framesizex,
                     p_wfs._framesizey)).astype(np.float32, order="F")

    def next(self, *, move_atmos: bool=True, see_atmos: bool=True, nControl: int=0,
             tar_trace: Iterable[int]=None, wfs_trace: Iterable[int]=None,
             do_control: bool=True, apply_control: bool=True) -> None:
        """
        Overload of the Simulator.next() function
        """
        # print("Send a frame")
        p_wfs = self.rtcconf.config.p_wfss[0]

        # try:
        #     # from GPUIPCInterfaceWrap import GPUIPCInterfaceFloat
        #     if type(self.fakewfs) is not GPUIPCInterfaceFloat:
        #         raise RuntimeError("Fallback to basic OCtopus API")
        #     if not self.fastMode:
        #         if p_wfs.type == WFSType.SH:
        #             Simulator.next(self, move_atmos=move_atmos, see_atmos=see_atmos,
        #                            nControl=nControl, tar_trace=[0], wfs_trace=[0],
        #                            do_control=False)
        #         else:
        #             raise RuntimeError("WFS Type not usable")
        #     self.wfs.get_binimg_gpu(0, np.array(self.fakewfs.buffer, copy=False))
        #     self.fakewfs.notify()
        #     # print("Send a frame using GPUIPCInterfaceFloat...")
        # except:
        if not self.fastMode:
            Simulator.next(self, move_atmos=move_atmos, see_atmos=see_atmos,
                           nControl=nControl, tar_trace=[0], wfs_trace=[0],
                           do_control=False)
            if (self.location == "CPUSHM"):
                self.frame = np.array(self.wfs.d_wfs[0].d_binimg)
        if (self.location == "CPUSHM"):
            self.fakewfs.send(self.frame)
        elif (self.location == "GPUSHM"):
            self.fakewfs.copyFrom(self.wfs.d_wfs[0].d_binimg)
            self.fakewfs.notify()
        else:
            raise ValueError("location not known")
        if apply_control:
            self.fakedms.recv(self.comp, 0)
            self.dms.set_full_com(self.comp)
