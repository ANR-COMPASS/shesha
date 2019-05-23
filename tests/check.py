#!/usr/bin/env python
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

if __name__ == "__main__":
    import pandas
    from shesha.supervisor.compassSupervisor import CompassSupervisor

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
        supervisor = CompassSupervisor(param_file)

        if arguments["--devices"]:
            supervisor.config.p_loop.set_devices([
                    int(device) for device in arguments["--devices"].split(",")
            ])
        try:
            supervisor.initConfig()
            isInit = supervisor.isInit()
        except:
            isInit = False
            SR = "N/A"
        try:
            supervisor.loop(supervisor.config.p_loop.niter)
            SR = supervisor.getStrehl(0)[1]
        except:
            SR = "N/A"

        try:
            df = pandas.read_hdf("check.h5")
        except FileNotFoundError:
            columns = ["Test name", "Init", "SR@100iter"]
            df = pandas.DataFrame(columns=columns)

        idx = len(df.index)
        df.loc[idx, "Test name"] = param_file.split('/')[-1]
        df.loc[idx, "Init"] = isInit
        df.loc[idx, "SR@100iter"] = SR

        df.to_hdf("check.h5", "check")
