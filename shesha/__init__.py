''' @package shesha
Documentation for shesha.

More details.
'''

import subprocess, sys

__version__ = "5.2.1"

def check_shesha_compass_versions():
    compass_package = subprocess.check_output('conda list compass | tail -n1',shell=True).decode(
            sys.stdout.encoding)
    if(compass_package.startswith('compass')):
        # using conda package version
        compass_version = compass_package.split()[1]
        assert(__version__ == compass_version), 'SHESHA and COMPASS versions are not matching : %r != %r ' %\
                                              (__version__, compass_version)

check_shesha_compass_versions()
