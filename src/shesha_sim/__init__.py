"""
COMPASS simulation package
Abstraction layer for initialization and execution of a COMPASS simulation
"""
__all__ = ['simulator', 'simulatorBrahma', 'bench', 'benchBrahma']

from .simulator import Simulator
from .bench import Bench
