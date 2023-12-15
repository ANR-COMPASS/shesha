## @package   shesha.supervisor.twoStagesManager
## @brief     Initialization and execution of a CANAPASS supervisor
## @author    COMPASS Team <https://github.com/ANR-COMPASS>
## @version   5.2.1
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
Initialization and execution of an AO 2-stages manager. 
It instanciates in one process two compass simulations: 
1 for first stage and 1 for the second stage (as defined in their relative .par files)

IMPORTANT:
The next method of this manager --superseeds-- the compass next method so that the loop is fully handled by the manager. 

Usage:
  twoStagesManager.py <parameters_filename1> <parameters_filename2> <freqratio> [options]

with 'parameters_filename1' the path to the parameters file the first stage 
with 'parameters_filename2' the path to the parameters file the second stage
with 'freqratio' the ratio of the frequencies of the two stages
Options:
  -a, --adopt       used to connect optional ADOPT client to the manager (via pyro + shm cacao)

Example: 
    ipython -i twoStagesManager.py ../../data/par/SPHERE+/sphere.py ../../data/par/SPHERE+/sphere+.py 3
    ipython -i twoStagesManager.py ../../data/par/SPHERE+/sphere.py ../../data/par/SPHERE+/sphere+.py 3 -- --adopt
"""

import numpy as np
import time
from shesha.supervisor.stageSupervisor import StageSupervisor

class TwoStagesManager(object):
    """
    Class handling both supervisors of first stage and second stage.

    Attributes:
        first_stage : (StageSupervisor) : first stage StageSupervisor instance

        second_stage : (StageSupervisor) : second stage StageSupervisor instance

        iterations : (int) : frame counter

        second_stage_input : (array) : input phase screen for second stage

        mpup_offset : (int) : number of padding pixel from first stage s_pupil to second stage m_pupil

        frequency_ratio : (int) : second stage simulated frequency over first stage simulated frequency
    """
    def __init__(self, first_stage : StageSupervisor, second_stage : StageSupervisor, frequency_ratio : int):
        """ 
        Init of the TwoStagesManager object

        Args:
            first_stage : (StageSupervisor) : first stage StageSupervisor instance

            second_stage : (StageSupervisor) : second stage StageSupervisor instance

            frequency_ratio : (int) : ratio between second stage frequency and first stage frequency. Only integers are accepted.
        """

        self.first_stage = first_stage
        self.second_stage = second_stage
        self.second_stage.atmos.enable_atmos(False) # second stage atmos is not used 
        self.iterations = 0
        mpup_shape = self.second_stage.config.p_geom._mpupil.shape
        self.second_stage_input = np.zeros((mpup_shape[0], mpup_shape[1], 1))
        residual_shape = self.first_stage.config.p_geom._spupil.shape
        self.mpup_offset = (mpup_shape[0] - residual_shape[0]) // 2
        self.frequency_ratio = int(frequency_ratio)

        # flags for enabling coronagraphic images computation
        self.compute_first_stage_corono = True
        self.compute_second_stage_corono = True

    def next(self, *, do_control: bool = True) -> None:
        """
        MAIN method that allows to manage properly the 2 AO stages of SAXO+ system. 
        The phase residuals (including turbulence + AO loop residuals) of the first stage simulation is sent to second stage simulation
        at each iteration of the manager. 
        The manager disable the seconds stage turbulence simulation (as it is propageated through the first stage residals if any). 
        This next method sould ALWAYS be called to perform a regular SAXO+ simulation 
        instead of the individual stage next methods to ensure the correct synchronisation of the 2 systems. 
        """
        # Iteration time of the first stage is set as the same as the second stage to
        # allow correct atmosphere movement for second stage integration. Then,
        # first stage is controlled only once every frequency_ratio times

        # Turbulence always disabled on 2nd instance of COMPASS                         
        self.second_stage.atmos.enable_atmos(False) 

        if do_control:
            # compute flags to specify which action need to be done in this first stage:
            # 1. check if go on stacking WFS image
            first_stage_stack_wfs = bool(self.iterations % self.frequency_ratio)
            # 2. check if centroids need to be computed  (end of WFS exposure)
            first_stage_centroids = not(bool((self.iterations + 1) % self.frequency_ratio))
            # 3. Check if a new command is computed (when new centroids appear)
            first_stage_control = first_stage_centroids
            self.first_stage.next(do_control = first_stage_control,
                                  apply_control = True,
                                  do_centroids = first_stage_centroids,
                                  compute_tar_psf = True,
                                  stack_wfs_image = first_stage_stack_wfs)
        else:
            self.first_stage.next(do_control=False,
                                  do_centroids=True,
                                  apply_control=True,
                                  compute_tar_psf = True)
        
        # FIRST STAGE IS DONE.

        # Get residual of first stage to put it into second stage
        # For now, involves GPU-CPU memory copies, can be improved later if speed is
        # a limiting factor here... 
        first_stage_residual = self.first_stage.target.get_tar_phase(0)
        self.second_stage_input[self.mpup_offset:-self.mpup_offset,
                                self.mpup_offset:-self.mpup_offset,:] = first_stage_residual[:,:,None]
        self.second_stage.tel.set_input_phase(self.second_stage_input) # 1st stage residuals sent to seconds stage simulation. 

        # SECOND STAGE LOOP STARTS...
        
        #"Updates the second stage simulation accordingly".
        # WFS exposure is always reset (default).
        self.second_stage.next(move_atmos=False, do_control=do_control) 
        # SECOND STAGE IS DONE.
        self.iterations += 1

    def enable_corono(self, stage=None):
        """ Enable coronagraphic image computation for both stages.

        Args:
            stage: (str, optional): If 'first', enable only first stage coronagrapic image computation.
                If 'second', enable only second stage coronagraphic image computation.
                Default = None.
        """
        if stage == 'first':
            self.compute_first_stage_corono = True
        elif stage == 'second':
            self.compute_second_stage_corono = True
        else:
            self.compute_first_stage_corono = True
            self.compute_second_stage_corono = True

    def disable_corono(self):
        """ Disable all coronagraphic image computation
        """
        self.compute_first_stage_corono = False
        self.compute_second_stage_corono = False
    
    def reset_exposure(self):
        """ Reset long exposure psf and coronagraphic images for both stages
        """
        self.first_stage.corono.reset()
        self.second_stage.corono.reset()
        self.first_stage.target.reset_strehl(0)
        self.second_stage.target.reset_strehl(0)

    def get_frame_counter(self):
        """ Returns the current iteration number of the manager
        
        Returns:
            iterations : (int) : Number of manager iterations already performed
        """
        return self.iterations

    def loop(self, number_of_iter: int, *, monitoring_freq: int = 100, **kwargs):
        """ Perform the AO loop for <number_of_iter> iterations

        Args:
            number_of_iter: (int) : Number of iteration that will be done

        Kwargs:
            monitoring_freq: (int) : Monitoring frequency [frames]. Default is 100
        """

        print("----------------------------------------------------")
        print("iter# | S.E. SR | L.E. SR | ETR (s) | Framerate (Hz)")
        print("----------------------------------------------------")
        # self.next(**kwargs)
        t0 = time.time()
        t1 = time.time()
        if number_of_iter == -1:  # Infinite loop
            while (True):
                self.next()
                if ((self.iterations + 1) % monitoring_freq == 0):
                    self.second_stage._print_strehl(monitoring_freq, time.time() - t1, number_of_iter)
                    t1 = time.time()

        for _ in range(number_of_iter):
            self.next()
            if ((self.iterations + 1) % monitoring_freq == 0):
                self.second_stage._print_strehl(monitoring_freq, time.time() - t1, number_of_iter)
                t1 = time.time()
        t1 = time.time()
        print(" loop execution time:", t1 - t0, "  (", number_of_iter, "iterations), ",
              (t1 - t0) / number_of_iter, "(mean)  ", number_of_iter / (t1 - t0), "Hz")

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
    adopt = arguments["--adopt"]

    config1 = ParamConfig(arguments["<parameters_filename1>"])
    config2 = ParamConfig(arguments["<parameters_filename2>"])
    frequency_ratio = arguments["<freqratio>"]

    first_stage = StageSupervisor(config1, cacao=adopt)
    second_stage = StageSupervisor(config2, cacao=adopt)
    manager = TwoStagesManager(first_stage, second_stage, frequency_ratio)

    if(adopt): 
        
        supervisor1 = manager.first_stage
        supervisor2 = manager.second_stage

        
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


            if (supervisor1.corono is None):
                from shesha.util.pyroEmptyClass import PyroEmptyClass
                coro2pyro1 = PyroEmptyClass()
            else:
                coro2pyro1 = supervisor1.corono

            if (supervisor2.corono is None):
                from shesha.util.pyroEmptyClass import PyroEmptyClass
                coro2pyro2 = PyroEmptyClass()
            else:
                coro2pyro2 = supervisor2.corono


            devices1 = [
                    supervisor1, supervisor1.rtc, supervisor1.wfs, supervisor1.target,
                    supervisor1.tel, supervisor1.basis, supervisor1.calibration,
                    supervisor1.atmos, supervisor1.dms, supervisor1.config, supervisor1.modalgains,
                    coro2pyro1
            ]
            devices2 = [
                    supervisor2, supervisor2.rtc, supervisor2.wfs, supervisor2.target,
                    supervisor2.tel, supervisor2.basis, supervisor2.calibration,
                    supervisor2.atmos, supervisor2.dms, supervisor2.config, supervisor2.modalgains,
                    coro2pyro2
            ]
            names = [
                    "supervisor", "supervisor_rtc", "supervisor_wfs", "supervisor_target",
                    "supervisor_tel", "supervisor_basis", "supervisor_calibration",
                    "supervisor_atmos", "supervisor_dms", "supervisor_config", "supervisor_modalgains",
                    "supervisor_corono"
            ]

            label = "firstStage"
            nname = []
            for name in names:
                nname.append(name + "_" + user + "_" +label)

            label = "secondStage"
            for name in names:
                nname.append(name + "_" + user + "_" +label)

            nname.append('twoStagesManager'+ "_" + user ) # Adding master next 2-stages loop
            devices = devices1 + devices2 + [manager]
            server = PyroServer(listDevices=devices, listNames=nname)
            #server.add_device(supervisor, "waoconfig_" + user)
            server.start()
        except BaseException:
            raise EnvironmentError(
                    "Missing dependencies (code HRAA or Pyro4 or Dill Serializer)")
        
