"""
COMPASS simulation package
Abstraction layer for initialization and execution of a COMPASS simulation
"""
__all__ = ['simulator', 'simulatorBrama', 'bench', 'benchBrama']

from .simulator import Simulator
from .simulatorBrama import SimulatorBrama
from .bench import Bench
from .benchBrama import BenchBrama
