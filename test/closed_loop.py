"""script test to simulate a closed loop

Usage:
  closed_loop.py <parameters_filename> [options]

with 'parameters_filename' the path to the parameters file

Options:
  -h --help          Show this help message and exit
  --brama            Distribute data with BRAMA
  --bench            For a timed call
  -d, --devices devices      Specify the devices
"""

from docopt import docopt

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

if arguments["--devices"]:
    sim.config.p_loop.set_devices([
            int(device) for device in arguments["--devices"].split(",")
    ])

sim.init_sim()
sim.loop(sim.config.p_loop.niter)
