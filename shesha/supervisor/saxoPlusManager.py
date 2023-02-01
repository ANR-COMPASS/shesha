## @package   shesha.supervisor.canapassSupervisor
## @brief     Initialization and execution of a CANAPASS supervisor
## @author    COMPASS Team <https://github.com/ANR-COMPASS>
## @version   5.2.1
## @date      2022/01/24
## @copyright GNU Lesser General Public License
#
#  This file is part of COMPASS <https://anr-compass.github.io/compass/>
#
#  Copyright (C) 2011-2022 COMPASS Team <https://github.com/ANR-COMPASS>
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
Initialization and execution of a saxo+ manager. 
It instanciates in one process two compass simulations: 
1 for first stage and 1 for the second stage (as defined in their relative .par files)
The frequency ratio between the 2 stages (second stage is assumed to have higher frequency) must be specified, in order to properly call for each stage next method for their respective loops (accumulated integration on the first stage WFS and low frequency update of the first stage control).

IMPORTANT:
The next method of this manager --superseeds-- the compass next method so that the loop is fully handled by the saxoPlus manager. 

Usage:
  saxoPlusmanager.py <saxoparameters_filename> <saxoPlusparameters_filename> <frequency_ratio> [options]

with 'saxoparameters_filename' the path to the parameters file for SAXO+ First stage (I.e current SAXO system)
with 'saxoPlusparameters_filename' the path to the parameters file for SAXO+ Second stage
with 'frequency_ratio' the ratio of the frequencies of the two stages

Options:
  -a, --adopt       used to connect optional ADOPT client to the manager (via pyro + shm cacao)

Example: 
    ipython -i saxoPlusManager.py ../../data/par/SPHERE+/sphere.py ../../data/par/SPHERE+/sphere+.py 3
    ipython -i saxoPlusManager.py ../../data/par/SPHERE+/sphere.py ../../data/par/SPHERE+/sphere+.py 3 -- --adopt
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

import shesha.util.sphere_pupil as pup
try: 
    from Asterix.util import read_parameter_file
    from Asterix.optics import Pupil, Coronagraph, Testbed
    ASTERIX = True
except:
    print("Warning Asterix Package not found")
    ASTERIX = False

class SaxoPlusManager():
    """
    Class handling both supervisors of first stage and second stage.

    Attributes:
        first_stage : (stage1Supervisor) : first stage stage1Supervisor instance

        second_stage : (stage2Supervisor) : second stage stage2Supervisor instance

        iterations : (int) : frame counter

        second_stage_input : (array) : input phase screen for second stage

        mpup_offset : (int) : number of padding pixel from first stage s_pupil to second stage m_pupil

        frequency_ratio : (int) : second stage simulated frequency over first stage simulated frequency
    """
    def __init__(self, first_stage, second_stage, frequency_ratio):
        """ 
        Init of the saxoPlusManager object

        Args:
            first_stage : (stage1Supervisor) : first stage stage1Supervisor instance

            second_stage : (stage2Supervisor) : second stage stage2Supervisor instance

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

        # Must be an input to configure the manager
        self.frequency_ratio = frequency_ratio

        # flags for enabling asterix coronagraph
        self.first_stage.computeCoroImage = False
        self.second_stage.computeCoroImage = False

    def next(self, *, do_control: bool = True) -> None:
        """
        MAIN method that allows to manage properly the 2 AO stages of SAXO+ system. 
        The phase residuals (including turbulence + AO loop residuals) of the first stage simulation is sent to second stage simulation
        at each iteration of the manager. 
        The saxo+ manager disable the seconds stage turbulence simulation (as it is propageated through the first stage residals if any). 

        This next method sould ALWAYS be called to perform a regular SAXO+ simulation 
        instead of the individuals COMPASS next methods to ensure the correct synchronisation of the 2 systems. 
        """
        # Iteration time of the first stage is set as the same as the second stage to
        # allow correct atmosphere movement for second stage integration. Then,
        # first stage is controlled only once every frequency_ratio times

        # Turbulence always disabled on 2nd instance of COMPASS                         
        self.second_stage.atmos.enable_atmos(False) 
        
        if do_control:
            # system has already been calibrated and normal run can be done

            if not (self.iterations % self.frequency_ratio):
                # Time for first stage start of new WFS exposure. 
                self.first_stage.reset_wfs_exposure()
                # DM shape is updated, but no centroid, neither control is computed.
                self.first_stage.next(do_control=False, apply_control=True,
                                      do_centroids=False, compute_tar_psf=True) 
                
            elif not ((self.iterations+1) % self.frequency_ratio):
                # Finish integration of WFS image, then compute centroids,
                # do control to get the new commands.
                self.first_stage.next(do_control=True, apply_control=True,
                                      do_centroids=True, compute_tar_psf=True)

            else:
                # In the middle of a WFS integration frame. Only raytracing current
                # turbulence phase, new DMs phase (apply control), and accumulates
                # WFS image.
                self.first_stage.next(do_control=False, apply_control=True,
                                      do_centroids=False, compute_tar_psf=True)

        else: # probably calibration is ongoing and no control can be computed so far
            self.first_stage.next(do_control=False, do_centroids=True,
                                  apply_control=True, compute_tar_psf = True)
            

        # FIRST STAGE IS DONE.
        # Get residual of first stage to put it into second stage
        # For now, involves GPU-CPU memory copies, can be improved later if speed is
        # a limiting factor here... 
        first_stage_residual = self.first_stage.target.get_tar_phase(0)
        self.second_stage_input[self.mpup_offset:-self.mpup_offset,
                                self.mpup_offset:-self.mpup_offset,:] = first_stage_residual[:,:,None]
        self.second_stage.tel.set_input_phase(self.second_stage_input) # 1st stage residuals sent to seconds stage simulation. 

        # SECOND STAGE LOOP STARTS...
        if do_control:
            #"Updates the second stage simulation accordingly".                          
            self.second_stage.next(move_atmos=False) 
        else:
            self.second_stage.next(move_atmos=False, do_control=False)
        # SECOND STAGE IS DONE.
            
        if(ASTERIX): # Works only with Asterix installed in python path.
            if self.first_stage.computeCoroImage and not (self.iterations % self.frequency_ratio):
                targetOPD = self.first_stage.target.get_tar_phase(0)  # OPD [micron]
                targetPhase = targetOPD * 2 * np.pi * 1e-6 / self.irdisAsterix.wavelength_0  # phase [rad]
                targetEF = self.irdisAsterix.EF_from_phase_and_ampl(phase_abb = targetPhase, ampl_abb = 0)
                shortExp = self.irdisAsterix.todetector_intensity(entrance_EF = targetEF, noFPM = False, center_on_pixel = False)
                self.first_stage.coroImage += shortExp
            if self.second_stage.computeCoroImage:
                targetOPD = self.second_stage.target.get_tar_phase(0)
                targetPhase = targetOPD * 2 * np.pi * 1e-6 / self.irdisAsterix.wavelength_0
                targetEF = self.irdisAsterix.EF_from_phase_and_ampl(phase_abb = targetPhase, ampl_abb = 0)
                shortExp = self.irdisAsterix.todetector_intensity(entrance_EF = targetEF, noFPM = False, center_on_pixel = False)
                self.second_stage.coroImage += shortExp
        self.iterations += 1

    def waitSAXO(self):
        while (self.iterations % self.frequency_ratio) != 0:
            self.next()
        print('iterations modulo frequency_ratio =', self.iterations % self.frequency_ratio)

    def enableCoro(self):
        self.first_stage.computeCoroImage = True
        self.second_stage.computeCoroImage = True

    def disableCoro(self):
        self.first_stage.computeCoroImage = False
        self.second_stage.computeCoroImage = False

    def getCoroImage(self):
        """
        Return the long exposure coronagraphic images from both stage

        Returns
        -------
        coroImages : (array, array)
            tuple containing (first stage, second stage) coronagraphic images
        """
        coroImages = (self.first_stage.coroImage, self.second_stage.coroImage)
        return coroImages
    
    def getAsterixShortExpPSF1stStage(self):
        targetOPD = self.first_stage.target.get_tar_phase(0)
        targetPhase = targetOPD * 2 * np.pi * 1e-6 / self.irdisAsterix.wavelength_0
        targetEF = self.irdisAsterix.EF_from_phase_and_ampl(phase_abb = targetPhase, ampl_abb = 0)
        shortExpPSF = self.irdisAsterix.todetector_intensity(entrance_EF = targetEF, noFPM = True, center_on_pixel = True)
        return shortExpPSF

    def getAsterixShortExpPSF2ndStage(self):
        targetOPD = self.second_stage.target.get_tar_phase(0)
        targetPhase = targetOPD * 2 * np.pi * 1e-6 / self.irdisAsterix.wavelength_0
        targetEF = self.irdisAsterix.EF_from_phase_and_ampl(phase_abb = targetPhase, ampl_abb = 0)
        shortExpPSF = self.irdisAsterix.todetector_intensity(entrance_EF = targetEF, noFPM = True, center_on_pixel = True)
        return shortExpPSF

    def resetExposure(self):
        self.first_stage.coroImage = np.zeros((self.dimCoroImage, self.dimCoroImage))
        self.second_stage.coroImage = np.zeros((self.dimCoroImage, self.dimCoroImage))
        self.first_stage.target.reset_strehl(0)
        self.second_stage.target.reset_strehl(0)

    # Asterix coronagraph
    def initAsterix(self):
        self.parameterFileAsterix = '/home/fvidal/compass/shesha/data/par/SPHERE+/param_file_Asterix.ini'
        self.dataDirAsterix = '/home/fvidal/codes/ADOPT/projects/simusCompass/sphere+/data'

        self.H3_filter_central_wl = 1667e-9
        self.H3_filter_delta_wl = 54e-9
        self.sphere_optical_resolution_H3 = np.rad2deg(self.H3_filter_central_wl / 8.) * 3600 * 1000  # mas/(lambda/D)
        self.sphere_plate_scale = 12.25  # mas/pix
        self.dimCoroImage = 256  # [pixel]
        self.first_stage.coroImage = np.zeros((self.dimCoroImage, self.dimCoroImage))
        self.second_stage.coroImage = np.zeros((self.dimCoroImage, self.dimCoroImage))

        self.configAsterix = read_parameter_file(self.parameterFileAsterix,
            NewMODELconfig={
                'wavelength_0': self.H3_filter_central_wl,
                'Delta_wav': self.H3_filter_delta_wl,
                'nb_wav': 3,
                'dimScience': self.dimCoroImage,  # we crop sphere images to go faster since we are mostly interested in the center
                'Science_sampling': self.sphere_optical_resolution_H3 / self.sphere_plate_scale,
                'diam_pup_in_pix': self.first_stage.config.p_geom.pupdiam,
                'diam_pup_in_m': 180e-3
            },
            NewCoronaconfig={
                'rad_lyot_fpm': 185 / 2 / self.sphere_optical_resolution_H3,  # 185 mas in diameter correspond to 185/sphere_optical_resolution_H3 = 4.465 lambda/D
                'corona_type': "classiclyot"
            },
            NewSIMUconfig={'nb_photons': 1e8})
        
        self.apodizer = pup.make_sphere_apodizer(self.first_stage.config.p_geom.pupdiam)
        self.lyotStop = pup.make_sphere_lyot_stop(self.first_stage.config.p_geom.pupdiam)
        pfits.writeto(self.dataDirAsterix + 'pupVLT.fits', self.first_stage.get_s_pupil(), overwrite = True)
        pfits.writeto(self.dataDirAsterix + 'apodizerSphere.fits', self.apodizer, overwrite = True)
        pfits.writeto(self.dataDirAsterix + 'lyotStopSphere.fits', self.lyotStop, overwrite = True)

        modelconfig = self.configAsterix["modelconfig"]
        entrance_pupil = Pupil(modelconfig, PupType = self.dataDirAsterix + 'pupVLT.fits')
        Model_local_dir = self.dataDirAsterix + 'Model_local'
        Coronaconfig = self.configAsterix["Coronaconfig"]
        Coronaconfig['bool_overwrite_perfect_coro'] = False
        Coronaconfig['filename_instr_apod'] = self.dataDirAsterix + 'apodizerSphere.fits'
        Coronaconfig['filename_instr_lyot'] = self.dataDirAsterix + 'lyotStopSphere.fits'
        corono = Coronagraph(modelconfig, Coronaconfig, Model_local_dir=Model_local_dir)

        self.irdisAsterix = Testbed([entrance_pupil, corono], ["entrance_pupil", "corono"])


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

    config1 = ParamConfig(arguments["<saxoparameters_filename>"])
    config2 = ParamConfig(arguments["<saxoPlusparameters_filename>"])
    frequency_ratio = arguments["frequency_ratio"]

    """
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
    """


    first_stage = CompassSupervisor(config1, cacao=adopt)
    second_stage = CompassSupervisor(config2, cacao=adopt)
    manager = SaxoPlusManager(first_stage, second_stage, frequency_ratio)

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


            devices1 = [
                    supervisor1, supervisor1.rtc, supervisor1.wfs, supervisor1.target,
                    supervisor1.tel, supervisor1.basis, supervisor1.calibration,
                    supervisor1.atmos, supervisor1.dms, supervisor1.config, supervisor1.modalgains
            ]
            devices2 = [
                    supervisor2, supervisor2.rtc, supervisor2.wfs, supervisor2.target,
                    supervisor2.tel, supervisor2.basis, supervisor2.calibration,
                    supervisor2.atmos, supervisor2.dms, supervisor2.config, supervisor2.modalgains
            ]
            names = [
                    "supervisor", "supervisor_rtc", "supervisor_wfs", "supervisor_target",
                    "supervisor_tel", "supervisor_basis", "supervisor_calibration",
                    "supervisor_atmos", "supervisor_dms", "supervisor_config", "supervisor_modalgains"
            ]

            label = "firstStage"
            nname = []
            for name in names:
                nname.append(name + "_" + user + "_" +label)

            label = "secondStage"
            for name in names:
                nname.append(name + "_" + user + "_" +label)

            nname.append('supervisorSAXOPlus'+ "_" + user ) # Adding master next dedicated to trigger SAXO+ hybrid loop
            devices = devices1 + devices2 + [manager]
            server = PyroServer(listDevices=devices, listNames=nname)
            #server.add_device(supervisor, "waoconfig_" + user)
            server.start()
        except:
            raise EnvironmentError(
                    "Missing dependencies (code HRAA or Pyro4 or Dill Serializer)")
        
