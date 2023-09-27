#!/usr/bin/env python
## @package   shesha.tests
## @brief     Runs a set of tests
## @author    COMPASS Team <https://github.com/ANR-COMPASS>
## @version   5.5.0
## @date      2022/01/24
## @copyright GNU Lesser General Public License
#
#  This file is part of COMPASS <https://anr-compass.github.io/compass/>
#
#  Copyright (C) 2011-2023 COMPASS Team <https://github.com/ANR-COMPASS>
#  All rights reserved.
#  Distributed under GNU - LGPL
#
#  COMPASS is free software: you can redistribute it and/or modify it under the terms of the GNU Lesser
#  General Public License as published by the Free Software Foundation, either version 3 of the License,
#  or any later version.
#
#  COMPASS: End-to-end AO simulation tool using GPU acceleration
#  The COMPASS platform was designed to meet the need of high-performance for the simulation of AO systems.
#
#  The final product includes a software package for simulating all the critical subcomponents of AO,
#  particularly in the context of the ELT and a real-time core based on several control approaches,
#  with performances consistent with its integration into an instrument. Taking advantage of the specific
#  hardware architecture of the GPU, the COMPASS tool allows to achieve adequate execution speeds to
#  conduct large simulation campaigns called to the ELT.
#
#  The COMPASS platform can be used to carry a wide variety of simulations to both testspecific components
#  of AO of the E-ELT (such as wavefront analysis device with a pyramid or elongated Laser star), and
#  various systems configurations such as multi-conjugate AO.
#
#  COMPASS is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the
#  implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
#  See the GNU Lesser General Public License for more details.
#
#  You should have received a copy of the GNU Lesser General Public License along with COMPASS.
#  If not, see <https://www.gnu.org/licenses/lgpl-3.0.txt>.

"""script test to simulate a closed loop

Usage:
  check.py <parameters_filename> [options]

where parameters_filename is the path to the parameters file

Options:
  -h --help                    Show this help message and exit
  -d, --devices devices        Specify the devices
  --displayResult              Just print the results of the check process
  --repportResult=<repport.md> Save the results of the check process into a md_file
"""

from docopt import docopt
import time

if __name__ == "__main__":
    import pandas
    from shesha.supervisor.compassSupervisor import CompassSupervisor
    from shesha.config import ParamConfig

    arguments = docopt(__doc__)

    if arguments["--displayResult"]:
        from os import remove
        from tabulate import tabulate
        from datetime import datetime
        df = pandas.read_hdf("check.h5")
        print(tabulate(df, tablefmt="pipe", headers="keys"))
        if arguments["--repportResult"]:
            with open(arguments["--repportResult"], 'w') as the_file:
                the_file.write('# E2E Test Report\n')
                the_file.write('\n')
                the_file.write(datetime.now().strftime(
                        '*Report generated on %d-%b-%Y %H:%M:%S by checkCompass.sh*\n'))
                the_file.write('\n')
                the_file.write('[Unit Tests report](report_unit_test.html)\n')
                the_file.write('\n')
                the_file.write('## Summary\n')
                the_file.write('\n')
                the_file.write(str(tabulate(df, tablefmt="pipe", headers="keys")))
        remove("check.h5")
    else:
        # Get parameters from file
        param_file = arguments["<parameters_filename>"]
        config = ParamConfig(param_file)

        if arguments["--devices"]:
            config.p_loop.set_devices([
                    int(device) for device in arguments["--devices"].split(",")
            ])

        try:
            t0 = time.perf_counter()
            supervisor = CompassSupervisor(config)
            t_init = time.perf_counter() - t0
            is_init = supervisor.is_init
        except:
            supervisor = None
            is_init = False
            t_init = 0
            SR = "N/A"
        try:
            t0 = time.perf_counter()
            supervisor.loop(supervisor.config.p_loop.niter)
            t_loop = time.perf_counter() - t0
            t_init = 0
            SR = supervisor.target.get_strehl(0)[1]
        except:
            SR = "N/A"
            t_loop = 0
        try:
            df = pandas.read_hdf("check.h5")
        except FileNotFoundError:
            columns = ["Test name", "Init", "SR@100iter"]
            df = pandas.DataFrame(columns=columns)

        idx = len(df.index)
        df.loc[idx, "Test name"] = param_file.split('/')[-1]
        df.loc[idx, "Init"] = str(is_init)
        df.loc[idx, "T Init"] = str(t_init)
        df.loc[idx, "SR@100iter"] = str(SR)
        df.loc[idx, "T Loop"] = str(t_loop)

        df.to_hdf("check.h5", "check")
