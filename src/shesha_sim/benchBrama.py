"""
Benchmark class for COMPASS with BRAMA simulation timing
(Not used, incomplete)
"""
from time import time, sleep

from .simulatorBrama import SimulatorBrama


class BenchBrama(SimulatorBrama):
    '''
        Class BenchBrama
    '''

    def next(self, *, nControl: int=0) -> None:
        '''
            function next
            Iterates on centroiding and control, with optional parameters

        :param nControl (int): Controller number to use, default 0 (single control configurations)
        '''

        # self.rtc.do_centroids(nControl)
        # self.rtc.do_control(nControl)
        # self.rtc.do_clipping(0, -1e5, 1e5)

        if self.rtc is not None:
            self.rtc.publish()
        # if self.tar is not None:
        #     self.tar.publish()

    def loop(self, freq=0., monitoring_freq=100, **kwargs):
        """
        TODO: docstring
        """
        print("----------------------------------------------------")
        print("iter# | S.E. SR | L.E. SR | ETR (s) | Framerate (Hz)")
        print("----------------------------------------------------")
        niter = 0
        t0 = time()
        while True:
            start = time()
            self.next(**kwargs)
            if ((niter + 1) % monitoring_freq == 0):
                framerate = (monitoring_freq) / (time() - t0)
                t0 = time()
                print("%d \t %.1f" % (niter + 1, framerate))
            tmp = time() - start
            if freq > 0:
                sleep(1 / freq - tmp)
            niter += 1
