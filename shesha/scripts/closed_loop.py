#!/usr/bin/env python
""" @package shesha.script.closed_loop
script test to simulate a closed loop

Usage:
  closed_loop.py <parameters_filename> [options]

with 'parameters_filename' the path to the parameters file

Options:
  -h --help          Show this help message and exit
  --brahma           Distribute data with BRAHMA
  --bench            For a timed call
  -i, --interactive  keep the script interactive
  -d, --devices devices      Specify the devices
  -n, --niter niter  Number of iterations
  --DB               Use database to skip init phase
  -g, --generic      Use generic controller
  -f, --fast         Compute PSF only during monitoring
"""

from docopt import docopt

if __name__ == "__main__":
    arguments = docopt(__doc__)
    param_file = arguments["<parameters_filename>"]
    use_DB = False
    compute_tar_psf = not arguments["--fast"]

    # Get parameters from file
    if arguments["--bench"]:
        from shesha.supervisor.benchSupervisor import BenchSupervisor as Supervisor
    elif arguments["--brahma"]:
        from shesha.supervisor.canapassSupervisor import CanapassSupervisor as Supervisor
    else:
        from shesha.supervisor.compassSupervisor import CompassSupervisor as Supervisor

    if arguments["--DB"]:
        use_DB = True

    supervisor = Supervisor(param_file, use_DB=use_DB)

    if arguments["--devices"]:
        supervisor.config.p_loop.set_devices([
                int(device) for device in arguments["--devices"].split(",")
        ])
    if arguments["--generic"]:
        supervisor.config.p_controllers[0].set_type("generic")
        print("Using GENERIC controller...")

    supervisor.initConfig()
    if arguments["--niter"]:
        supervisor.loop(int(arguments["--niter"]), compute_tar_psf=compute_tar_psf)
    else:
        supervisor.loop(supervisor.config.p_loop.niter, compute_tar_psf=compute_tar_psf)

    if arguments["--interactive"]:
        from shesha.util.ipython_embed import embed
        from os.path import basename
        embed(basename(__file__), locals())
