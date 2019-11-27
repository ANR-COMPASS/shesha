## @package   shesha.supervisor.compassSupervisor
## @brief     Initialization and execution of a COMPASS supervisor
## @author    COMPASS Team <https://github.com/ANR-COMPASS>
## @version   4.3.2
## @date      2011/01/28
## @copyright GNU Lesser General Public License
#
#  This file is part of COMPASS <https://anr-compass.github.io/compass/>
#
#  Copyright (C) 2011-2019 COMPASS Team <https://github.com/ANR-COMPASS>
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
Initialization and execution of a COMPASS supervisor

Usage:
  compassSupervisor.py [<parameters_filename>]

with 'parameters_filename' the path to the parameters file

Options:
  -h --help          Show this help message and exit
"""
from .aoSupervisor import AoSupervisor
import numpy as np

import shesha.constants as scons
from shesha.constants import CONST

from tqdm import trange


class CompassSupervisor(AoSupervisor):

    #     _    _         _                  _
    #    / \  | |__  ___| |_ _ __ __ _  ___| |_
    #   / _ \ | '_ \/ __| __| '__/ _` |/ __| __|
    #  / ___ \| |_) \__ \ |_| | | (_| | (__| |_
    # /_/   \_\_.__/|___/\__|_|  \__,_|\___|\__|
    #
    #  __  __      _   _               _
    # |  \/  | ___| |_| |__   ___   __| |___
    # | |\/| |/ _ \ __| '_ \ / _ \ / _` / __|
    # | |  | |  __/ |_| | | | (_) | (_| \__ \
    # |_|  |_|\___|\__|_| |_|\___/ \__,_|___/

    def singleNext(self, moveAtmos: bool = True, showAtmos: bool = True,
                   getPSF: bool = False, getResidual: bool = False) -> None:
        '''
        Move atmos -> getSlope -> applyControl ; One integrator step
        '''
        self._sim.next(see_atmos=showAtmos)  # why not self._seeAtmos?
        self.iter += 1

    def getTarImage(self, tarID, expoType: str = "se") -> np.ndarray:
        '''
        Get an image from a target
        '''
        if (expoType == "se"):
            return np.fft.fftshift(np.array(self._sim.tar.d_targets[tarID].d_image_se))
        elif (expoType == "le"):
            return np.fft.fftshift(np.array(self._sim.tar.d_targets[tarID].d_image_le)
                                   ) / self._sim.tar.d_targets[tarID].strehl_counter
        else:
            raise ValueError("Unknown exposure type")

    def setCommand(self, nctrl: int, command: np.ndarray) -> None:
        '''
        Set the RTC command vector
        '''
        self._sim.rtc.d_control[nctrl].set_com(command, command.size)

    def getCom(self, nControl: int):
        '''
        Get command from nControl controller
        '''
        return np.array(self._sim.rtc.d_control[nControl].d_com)

    #  ____                  _ _   _        __  __      _   _               _
    # / ___| _ __   ___  ___(_) |_(_) ___  |  \/  | ___| |_| |__   ___   __| |___
    # \___ \| '_ \ / _ \/ __| | __| |/ __| | |\/| |/ _ \ __| '_ \ / _ \ / _` / __|
    #  ___) | |_) |  __/ (__| | |_| | (__  | |  | |  __/ |_| | | | (_) | (_| \__ \
    # |____/| .__/ \___|\___|_|\__|_|\___| |_|  |_|\___|\__|_| |_|\___/ \__,_|___/
    #       |_|

    def __init__(self, configFile: str = None, cacao: bool = False,
                 use_DB: bool = False):
        '''
        Init the COMPASS supervisor

        Parameters
        ------------
        configFile: (str): (optionnal) Path to the parameter file
        cacao: (bool): (optionnal) Flag to enable cacao
        use_DB: (bool): (optionnal) Flag to enable database
        '''
        self._sim = None
        self._seeAtmos = False
        self.config = None
        self.cacao = cacao
        self.use_DB = use_DB

        if configFile is not None:
            self.loadConfig(configFile=configFile)

    def __repr__(self):
        return object.__repr__(self) + str(self._sim)

    def setPyrModulation(self, pyrMod: float, numwfs=0) -> None:
        '''
        Set pyramid modulation value - in l/D units
        '''
        from shesha.ao.wfs import comp_new_pyr_ampl
        p_wfs = self._sim.config.p_wfss[numwfs]

        _, _, _, pyr_npts = comp_new_pyr_ampl(0, pyrMod, self._sim.wfs, self._sim.rtc,
                                              self._sim.config.p_wfss,
                                              self._sim.config.p_tel)
        if(len(p_wfs._halfxy.shape) == 2):
            print("PYR modulation set to: %f L/D using %d points" % (pyrMod, pyr_npts))
        elif(len(p_wfs._halfxy.shape) == 3):
            newhalfxy = np.tile(p_wfs._halfxy[0,:,:],(pyr_npts, 1, 1))
            print("Loading new modulation arrays")
            self._sim.wfs.d_wfs[numwfs].set_phalfxy(np.exp(1j*newhalfxy).astype(np.complex64).T) 
            print("Done. PYR modulation set to: %f L/D using %d points" % (pyrMod, pyr_npts))
        else:
            raise ValueError("Error unknown p_wfs._halfxy shape")


    def setNoise(self, noise, numwfs=0, seed=1234):
        '''
        Set noise value of WFS numwfs
        '''
        self._sim.wfs.d_wfs[numwfs].set_noise(noise, int(seed + numwfs))
        print("Noise set to: %f on WFS %d" % (noise, numwfs))

    def setDmShapeFrom(self, command: np.ndarray) -> None:
        '''
        Immediately sets provided command to DMs - does not affect integrator
        '''
        self._sim.dms.set_full_com(command)

    def setOneActu(self, ndm: int, nactu: int, ampli: float = 1) -> None:
        '''
        Push the selected actuator
        '''
        self._sim.dms.d_dms[ndm].comp_oneactu(nactu, ampli)

    def enableAtmos(self, enable) -> None:
        ''' TODO
        Set or unset whether atmos is enabled when running loop (see singleNext)
        '''
        self._seeAtmos = enable

    def setGSmag(self, mag, numwfs=0):
        numwfs = int(numwfs)
        sim = self._sim
        wfs = sim.wfs.d_wfs[numwfs]
        if (sim.config.p_wfs0.type == "pyrhr"):
            r = wfs.comp_nphot(sim.config.p_loop.ittime,
                               sim.config.p_wfss[numwfs].optthroughput,
                               sim.config.p_tel.diam, sim.config.p_tel.cobs,
                               sim.config.p_wfss[numwfs].zerop, mag)
        else:
            r = wfs.comp_nphot(sim.config.p_loop.ittime,
                               sim.config.p_wfss[numwfs].optthroughput,
                               sim.config.p_tel.diam, sim.config.p_wfss[numwfs].nxsub,
                               sim.config.p_wfss[numwfs].zerop, mag)
        if (r == 0):
            print("GS magnitude is now %f on WFS %d" % (mag, numwfs))

    def loop(self, n: int = 1, monitoring_freq: int = 100, **kwargs):
        """
        Perform the AO loop for n iterations

        :parameters:
            n: (int): (optional) Number of iteration that will be done
            monitoring_freq: (int): (optional) Monitoring frequency [frames]
        """
        self._sim.loop(n, monitoring_freq=monitoring_freq, **kwargs)

    def forceContext(self) -> None:
        '''
        Clear the initialization of the simulation
        '''
        self._sim.force_context()

    def computeImages(self):
        for w in self._sim.wfs.d_wfs:
            w.d_gs.comp_image()

    def resetDM(self, numdm: int = -1) -> None:
        '''
        Reset the DM number nDM or all DMs if  == -1
        '''
        if (numdm == -1):  # All Dms reset
            for dm in self._sim.dms.d_dms:
                dm.reset_shape()
        else:
            self._sim.dms.d_dms[numdm].reset_shape()

    def resetCommand(self, nctrl: int = -1) -> None:
        '''
        Reset the nctrl Controller command buffer, reset all controllers if nctrl  == -1
        '''
        if (nctrl == -1):  # All Dms reset
            for control in self._sim.rtc.d_control:
                control.d_com.reset()
        else:
            self._sim.rtc.d_control[nctrl].d_com.reset()

    def resetSimu(self, noiseList):
        self.resetTurbu()
        self.resetNoise(noiseList)

    def resetTurbu(self):
        ilayer = 0
        for k in range(self._sim.atm.nscreens):
            self._sim.atm.set_seed(k, 1234 + ilayer)
            self._sim.atm.refresh_screen(k)
            ilayer += 1

    def resetNoise(self, noiseList):
        for nwfs in range(len(self._sim.config.p_wfss)):
            self._sim.wfs.d_wfs[nwfs].set_noise(noiseList[nwfs], 1234 + nwfs)

    def resetStrehl(self, nTar: int) -> None:
        '''
        Reset the Strehl Ratio of the target nTar
        '''
        self._sim.tar.d_targets[nTar].reset_strehlmeter()

    def resetTarPhase(self, nTar: int) -> None:
        '''
        Reset the phase screen of the target nTar
        '''
        self._sim.tar.d_targets[nTar].d_phase.reset()

    def loadConfig(self, configFile: str = None, sim=None) -> None:
        '''
        Init the COMPASS simulator wih the configFile
        '''
        if self._sim is None:
            if sim is None:
                if self.cacao:
                    from shesha.sim.simulatorCacao import SimulatorCacao as Simulator
                else:
                    from shesha.sim.simulator import Simulator
                self._sim = Simulator(filepath=configFile, use_DB=self.use_DB)
            else:
                self._sim = sim
        else:
            self._sim.clear_init()
            self._sim.load_from_file(configFile)
        self.config = self._sim.config

    def isInit(self) -> bool:
        '''
        return the status on COMPASS init
        '''
        return self._sim.is_init

    def clearInitSim(self) -> None:
        '''
        Clear the initialization of the simulation
        '''
        self._sim.clear_init()

    def initConfig(self) -> None:
        '''
        Initialize the simulation
        '''
        self._sim.init_sim()
        self.rtc = self._sim.rtc
        self.iter = self._sim.iter
        self.enableAtmos(True)
        self.is_init = True

    def getNcpaWfs(self, wfsnum):
        return np.array(self._sim.wfs.d_wfs[wfsnum].d_gs.d_ncpa_phase)

    def getNcpaTar(self, tarnum):
        return np.array(self._sim.tar.d_targets[tarnum].d_ncpa_phase)

    def getAtmScreen(self, indx: int) -> np.ndarray:
        '''
        return the selected atmos screen
        '''
        return np.array(self._sim.atm.d_screens[indx].d_screen)

    def getWfsPhase(self, numWFS: int) -> np.ndarray:
        '''
        return the WFS screen of WFS number numWFS
        '''
        return np.array(self._sim.wfs.d_wfs[numWFS].d_gs.d_phase)

    def getDmShape(self, indx: int) -> np.ndarray:
        '''
        return the selected DM screen
        '''
        return np.array(self._sim.dms.d_dms[indx].d_shape)

    def getTarPhase(self, numTar: int) -> np.ndarray:
        '''
        return the target screen of target number numTar
        '''
        return np.array(self._sim.tar.d_targets[numTar].d_phase)

    def getPyrHRImage(self, numWFS: int = 0) -> np.ndarray:
        '''
        Get an HR image from the WFS
        '''
        return np.array(self._sim.wfs.d_wfs[numWFS].d_hrimg)

    def getSlopeGeom(self, numWFS: int, ncontrol : int =0) -> np.ndarray:
        '''
        return the slopes geom of WFS number numWFS
        '''
        self._sim.rtc.do_centroids_geom(ncontrol)
        slopesGeom = np.array(self._sim.rtc.d_control[ncontrol].d_centroids)
        self._sim.rtc.do_centroids(ncontrol)
        return slopesGeom

    def getStrehl(self, numTar: int, do_fit: bool = True) -> np.ndarray:
        '''
        return the Strehl Ratio of target number numTar
        '''
        src = self._sim.tar.d_targets[numTar]
        src.comp_strehl(do_fit)
        avgVar = 0
        if (src.phase_var_count > 0):
            avgVar = src.phase_var_avg / src.phase_var_count
        return [src.strehl_se, src.strehl_le, src.phase_var, avgVar]

    def getIFsparse(self, nControl: int):
        '''
        Return the IF of DM as a sparse matrix
        '''
        return self._sim.rtc.d_control[nControl].d_IFsparse.get_csr()

    def getIFtt(self, nControl: int):
        '''
        Return the IF of a TT DM as a sparse matrix
        '''
        return np.array(self._sim.rtc.d_control[nControl].d_TT)

    def getIFdm(self, nDM: int):
        '''
        Return the IF of a Petal DM made with M4
        '''

        from shesha.ao import basis
        if_sparse = basis.compute_DMbasis(self._sim.dms.d_dms[nDM],
                                          self._sim.config.p_dms[nDM],
                                          self._sim.config.p_geom)

        return if_sparse

    def setNcpaWfs(self, ncpa, wfsnum):
        self._sim.wfs.d_wfs[wfsnum].d_gs.set_ncpa(ncpa)

    def setNcpaTar(self, ncpa, tarnum):
        self._sim.tar.d_targets[tarnum].set_ncpa(ncpa)

    def setWfsPhase(self, numwfs, phase):
        self._sim.wfs.d_wfs[numwfs].d_gs.set_phase(phase)

    def setMpupil(self, mpupil, numwfs=0):
        oldmpup = self.getMpupil()
        dimx = oldmpup.shape[0]
        dimy = oldmpup.shape[1]
        if ((mpupil.shape[0] != dimx) or (mpupil.shape[1] != dimy)):
            print("Error mpupil shape on wfs %d must be: (%d,%d)" % (numwfs, dimx, dimy))
        else:
            self._sim.wfs.d_wfs[numwfs].set_pupil(mpupil.copy())

    def getIpupil(self):
        return self._sim.config.p_geom._ipupil

    def getSpupil(self):
        return self._sim.config.p_geom._spupil

    def getMpupil(self):
        return self._sim.config.p_geom._mpupil

    def getTarAmplipup(self, tarnum):
        return self._sim.config.tar.get_amplipup(tarnum)

    def getPyrFocalPlane(self, nwfs: int = 0):
        """
        No arguments
        Returns the psf in the focal plane of the pyramid.
        """
        return np.fft.fftshift(np.array(self._sim.wfs.d_wfs[nwfs].d_pyrfocalplane))

    def reset(self, tar=-1, rst=True):
        self.resetTurbu()
        if (tar < 0):
            for tar in range(self._sim.tar.ntargets):
                self.resetStrehl(tar)
        else:
            self.resetStrehl(tar)
        self.resetDM()
        self.openLoop(rst=rst)
        self.closeLoop()

    def setDMRegistration(self, indDM, dx=None, dy=None, theta=None, G=None):
        """Set the registration parameters for DM #indDM

        Parameters:
            indDM : (int) : DM index
            dx : (float, optionnal) : X axis registration parameter [meters]. If None, re-use the last one
            dy : (float, optionnal) : Y axis registration parameter [meters]. If None, re-use the last one
            theta : (float, optionnal) : Rotation angle parameter [rad]. If None, re-use the last one
            G : (float, optionnal) : Magnification factor. If None, re-use the last one

        """
        if dx is not None:
            self.config.p_dms[indDM].set_dx(dx)
        if dy is not None:
            self.config.p_dms[indDM].set_dy(dy)
        if theta is not None:
            self.config.p_dms[indDM].set_theta(theta)
        if G is not None:
            self.config.p_dms[indDM].set_G(G)

        self._sim.dms.d_dms[indDM].set_registration(
                self.config.p_dms[indDM].dx / self.config.p_geom._pixsize,
                self.config.p_dms[indDM].dy / self.config.p_geom._pixsize,
                self.config.p_dms[indDM].theta, self.config.p_dms[indDM].G)

    def getSelectedPix(self):
        """Return the pyramid image with only the selected pixels used by the full pixels centroider
        """
        if (self.config.p_centroiders[0].type != scons.CentroiderType.MASKEDPIX):
            raise TypeError("Centroider must be maskedPix")

        carma_centroids = self._sim.rtc.d_control[0].d_centroids
        self._sim.rtc.d_centro[0].fill_selected_pix(carma_centroids)

        return np.array(self._sim.rtc.d_centro[0].d_selected_pix)
