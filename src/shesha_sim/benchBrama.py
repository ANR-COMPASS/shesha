import time

from typing import Callable, TypeVar

from .simulatorBrama import SimulatorBrama

class BenchBrama(SimulatorBrama):
    '''
        Class BenchBrama
    '''

    def next(self, *, see_atmos: bool=False, nControl: int=0) -> None:
        '''
            function next
            Iterates on centroiding and control, with optional parameters

        :param nControl (int): Controller number to use, default 0 (single control configurations)
        '''
        self.rtc.do_centroids(nControl)
        self.rtc.do_control(nControl)
        self.rtc.do_clipping(0, -1e5, 1e5)

        if self.rtc is not None:
            self.rtc.publish()
        if self.tar is not None:
            self.tar.publish()

