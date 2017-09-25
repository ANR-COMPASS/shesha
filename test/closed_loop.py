"""script test to simulate a closed loop

Usage:
  closed_loop.py <parameters_filename> [--brama] [--bench]

with 'parameters_filename' the path to the parameters file

Options:
  -h --help          Show this help message and exit
  --brama            Distribute data with BRAMA
  --bench            For a timed call
"""

from docopt import docopt

import sys
import os
import shesha_sim

arguments = docopt(__doc__)
param_file = arguments["<parameters_filename>"]

# Get parameters from file
if arguments["--bench"]:
    sim = shesha_sim.Bench(param_file)
elif arguments["--brama"]:
    sim = shesha_sim.SimulatorBrama(param_file)
else:
    sim = shesha_sim.Simulator(param_file)

sim.init_sim()
sim.loop(sim.config.p_loop.niter)
