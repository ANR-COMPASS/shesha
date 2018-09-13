#!/usr/bin/env python
"""script test to simulate a closed loop

Usage:
  closed_loop.py <parameters_filename> [options]

with 'parameters_filename' the path to the parameters file

Options:
  -h --help          Show this help message and exit
  --brahma           Distribute data with BRAHMA
  --rtcsim           COMPASS simulation linked to real RTC with Octopus
  --bench            For a timed call
  -i, --interactive  keep the script interactive
  -d, --devices devices      Specify the devices
"""

from docopt import docopt

if __name__ == "__main__":
    arguments = docopt(__doc__)
    param_file = arguments["<parameters_filename>"]

    # Get parameters from file
    if arguments["--bench"]:
        from shesha.supervisor.benchSupervisor import BenchSupervisor as Supervisor

    elif arguments["--brahma"]:
        from shesha.supervisor.canapassSupervisor import CanapassSupervisor as Supervisor
    elif arguments["--rtcsim"]:
        from shesha.supervisor.rtcSupervisor import RTCSupervisor as Supervisor
    else:
        from shesha.supervisor.compassSupervisor import CompassSupervisor as Supervisor

    supervisor = Supervisor(param_file)

    if arguments["--devices"]:
        supervisor.config.p_loop.set_devices([
                int(device) for device in arguments["--devices"].split(",")
        ])

    supervisor.initConfig()
    supervisor.loop(supervisor.config.p_loop.niter)

    if arguments["--interactive"]:
        from shesha.util.ipython_embed import embed
        from os.path import basename
        embed(basename(__file__), locals())
