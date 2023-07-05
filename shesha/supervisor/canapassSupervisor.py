## @package   shesha.supervisor.canapassSupervisor
## @brief     Initialization and execution of a CANAPASS supervisor
## @author    COMPASS Team <https://github.com/ANR-COMPASS>
## @version   5.4.4
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
"""
Initialization and execution of a CANAPASS supervisor

Usage:
  canapassSupervisor.py <parameters_filename> [options]

with 'parameters_filename' the path to the parameters file

Options:
  -h, --help                Show this help message and exit
  -f, --freq freq           change the frequency of the loop
  -d, --delay delay         change the delay of the loop
  -s, --spiders spiders     change the spiders size
  -n, --nxsub nxsub         change the number of pixels in subap
  -p, --pupsep pupsep       change the distance between subap center and frame center
  -g, --gsmag gsmag         change guide star magnitude
  -r, --rmod rmod           change modulation radius
  -x, --offaxis offaxis     change all targets position along x axis
  -r0,--setr0 setr0        change the global r0
"""

import os, sys
import numpy as np
import time

from tqdm import tqdm
import astropy.io.fits as pfits
from threading import Thread
from subprocess import Popen, PIPE

import shesha.ao as ao
import shesha.constants as scons
from shesha.constants import CentroiderType, WFSType

from typing import Any, Dict, Tuple, Callable, List
from shesha.supervisor.compassSupervisor import CompassSupervisor

# from carmaWrap.obj import obj_Double2D
# from carmaWrap.magma import syevd_Double, svd_host_Double
# from carmaWrap.context import context as carmaWrap_context

# from carmaWrap.host_obj import host_obj_Double1D, host_obj_Double2D


class CanapassSupervisor(CompassSupervisor):

    def __init__(self, config, cacao: bool = True, silence_tqdm: bool = False) -> None:
        print("switching to a generic controller")
        config.p_controllers[0].type = scons.ControllerType.GENERIC
        CompassSupervisor.__init__(self, config, cacao=cacao, silence_tqdm=silence_tqdm)


########################## PROTO #############################

# def initModalGain(self, gain, cmatModal, modal_basis, control=0, reset_gain=True):
#     """
#     Given a gain, cmat and btt2v initialise the modal gain mode
#     """
#     print("TODO: A RECODER !!!!")
#     nmode_total = modal_basis.shape[1]
#     nactu_total = modal_basis.shape[0]
#     nfilt = nmode_total - cmatModal.shape[0]
#     ctrl = self._sim.rtc.d_control[control]
#     ctrl.set_commandlaw('modal_integrator')
#     cmat = np.zeros((nactu_total, cmatModal.shape[1]))
#     dec = cmat.shape[0] - cmatModal.shape[0]
#     cmat[:-dec, :] += cmatModal  # Fill the full Modal with all non-filtered modes
#     modes2V = np.zeros((nactu_total, nactu_total))
#     dec2 = modes2V.shape[1] - modal_basis.shape[1]
#     modes2V[:, :-dec2] += modal_basis
#     mgain = np.ones(len(modes2V)) * gain  # Initialize the gain
#     ctrl.set_matE(modes2V)
#     ctrl.set_cmat(cmat)
#     if reset_gain:
#         ctrl.set_modal_gains(mgain)

# def leaveModalGain(self, control=0):
#     ctrl = self._sim.rtc.d_control[control]
#     ctrl.set_commandlaw('integrator')

class loopHandler:

    def __init__(self):
        pass

    def start(self):
        pass

    def stop(self):
        pass

    def alive(self):
        return "alive"

if __name__ == '__main__':
    from docopt import docopt
    from shesha.config import ParamConfig
    arguments = docopt(__doc__)
    config = ParamConfig(arguments["<parameters_filename>"])
    if (arguments["--freq"]):
        print("Warning changed frequency loop to: ", arguments["--freq"])
        config.p_loop.set_ittime(1 / float(arguments["--freq"]))
    if (arguments["--delay"]):
        print("Warning changed delay loop to: ", arguments["--delay"])
        config.p_controllers[0].set_delay(float(arguments["--delay"]))
    if (arguments["--spiders"]):
        print("Warning changed spiders size to: ", arguments["--spiders"])
        config.p_tel.set_t_spiders(float(arguments["--spiders"]))
    if (arguments["--nxsub"]):
        print("Warning changed number of pixels per subaperture to: ", arguments["--nxsub"])
        config.p_wfss[0].set_nxsub(int(arguments["--nxsub"]))
    if (arguments["--pupsep"]):
        print("Warning changed distance between subaperture center and frame center to: ", arguments["--pupsep"])
        config.p_wfss[0].set_pyr_pup_sep(int(arguments["--pupsep"]))
    if (arguments["--gsmag"]):
        print("Warning changed guide star magnitude to: ", arguments["--gsmag"])
        config.p_wfss[0].set_gsmag(float(arguments["--gsmag"]))
    if (arguments["--setr0"]):
        print("Warning changed r0 to: ", arguments["--setr0"])
        config.p_atmos.set_r0(float(arguments["--setr0"]))
    if (arguments["--rmod"]):
        print("Warning changed modulation radius to: ", arguments["--rmod"])
        rMod = int(arguments["--rmod"])
        nbPtMod = int(np.ceil(int(rMod * 2 * 3.141592653589793) / 4.) * 4)
        config.p_wfss[0].set_pyr_npts(nbPtMod)
        config.p_wfss[0].set_pyr_ampl(rMod)
    if (arguments["--offaxis"]):
        print("Warning changed target x position: ", arguments["--offaxis"])
        config.p_targets[0].set_xpos(float(arguments["--offaxis"]))
        config.p_targets[1].set_xpos(float(arguments["--offaxis"]))
        config.p_targets[2].set_xpos(float(arguments["--offaxis"]))

    supervisor = CanapassSupervisor(config, cacao=True)

    try:
        from subprocess import Popen, PIPE
        from hraa.server.pyroServer import PyroServer
        import Pyro4
        Pyro4.config.REQUIRE_EXPOSE = False
        p = Popen("whoami", shell=True, stdout=PIPE, stderr=PIPE)
        out, err = p.communicate()
        if (err != b''):
            print(err)
            raise Exception("ERROR CANNOT RECOGNIZE USER")
        else:
            user = out.split(b"\n")[0].decode("utf-8")
            print("User is " + user)
        if (supervisor.corono == None):
            from shesha.util.pyroEmptyClass import PyroEmptyClass
            coro2pyro = PyroEmptyClass()
        else:
            coro2pyro = supervisor.corono
        devices = [
                supervisor, supervisor.rtc, supervisor.wfs, supervisor.target,
                supervisor.tel, supervisor.basis, supervisor.calibration,
                supervisor.atmos, supervisor.dms, supervisor.config, supervisor.modalgains, coro2pyro
        ]
        names = [
                "supervisor", "supervisor_rtc", "supervisor_wfs", "supervisor_target",
                "supervisor_tel", "supervisor_basis", "supervisor_calibration",
                "supervisor_atmos", "supervisor_dms", "supervisor_config", "supervisor_modalgains", "supervisor_corono",
        ]
        nname = []
        for name in names:
            nname.append(name + "_" + user)
        server = PyroServer(listDevices=devices, listNames=nname)
        #server.add_device(supervisor, "waoconfig_" + user)
        server.start()
    except:
        raise EnvironmentError(
                "Missing dependencies (code HRAA or Pyro4 or Dill Serializer)")
