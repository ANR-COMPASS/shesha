import sys
import os

from .simulator import Simulator, init

class SimulatorBrama(Simulator):
    """
        Class SimulatorBrama: Brama overloaded simulator
        _tar_init and _rtc_init to instantiate Brama classes instead of regular classes
        next() to call rtc/tar.publish()
    """

    def _tar_init(self) -> None:
        '''
            TODO
        '''
        if self.config.p_target is not None:
            print("->target")
            self.tar = init.target_init(self.c, self.tel, self.config.p_target,
                                        self.config.p_atmos, self.config.p_tel,
                                        self.config.p_geom, self.config.p_dms,
                                        brama=True)
        else:
            self.tar = None

    def _rtc_init(self, ittime) -> None:
        '''
            TODO
        '''
        if self.config.p_controllers is not None or self.config.p_centroiders is not None:
            print("->rtc")
            #   rtc
            self.rtc = init.rtc_init(
                    self.c, self.tel, self.wfs, self.dms, self.atm, self.config.p_wfss,
                    self.config.p_tel, self.config.p_geom, self.config.p_atmos, ittime,
                    self.config.p_centroiders, self.config.p_controllers,
                    self.config.p_dms, brama=True)
        else:
            self.rtc = None

    def next(self, **kwargs) -> None:
        Simulator.next(self, **kwargs)
        if self.rtc is not None:
            self.rtc.publish()
        if self.tar is not None:
            self.tar.publish()

