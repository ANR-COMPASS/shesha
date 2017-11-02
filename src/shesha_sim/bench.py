"""
Benchmark class for COMPASS simulation timing
(Not used, incomplete)
"""
import time

from typing import Callable, TypeVar

from .simulator import Simulator

_O = TypeVar('_O')


def timeit(function: Callable[..., _O]) -> Callable[..., _O]:
    '''
        Function timing decorator
    '''

    def new_func(*args, **kwargs) -> _O:
        print('** Timed call to function {}'.format(function.__name__))
        t1 = time.time()
        ret = function(*args, **kwargs)
        t2 = time.time()
        print('** Execution time of {}: {} seconds'.format(function.__name__, t2 - t1))
        return ret

    return new_func


class Bench(Simulator):
    '''
        Class Bench

        Timed version of the simulator class using decorated overloads
    '''

    @timeit
    def __init__(self, filepath: str=None) -> None:
        Simulator.__init__(self, filepath)

    @timeit
    def load_from_file(self, filepath: str) -> None:
        Simulator.load_from_file(self, filepath)

    @timeit
    def init_sim(self) -> None:
        Simulator.init_sim(self)

    @timeit
    def timed_next(self) -> None:
        Simulator.next(self)

    @timeit
    def loop(self, n: int=1, monitoring_freq: int=100, **kwargs) -> None:
        Simulator.loop(self, n=n, monitoring_freq=monitoring_freq, **kwargs)
