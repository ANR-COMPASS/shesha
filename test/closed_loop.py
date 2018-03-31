"""script test to simulate a closed loop

Usage:
  closed_loop.py <parameters_filename> [options]

with 'parameters_filename' the path to the parameters file

Options:
  -h --help          Show this help message and exit
  --brahma            Distribute data with BRAHMA
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
elif arguments["--brahma"]:
    sim = shesha_sim.SimulatorBrahma(param_file)
else:
    sim = shesha_sim.Simulator(param_file)

if arguments["--devices"]:
    sim.config.p_loop.set_devices([
            int(device) for device in arguments["--devices"].split(",")
    ])

sim.init_sim()
sim.loop(sim.config.p_loop.niter)
