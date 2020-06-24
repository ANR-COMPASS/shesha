## @package   shesha.supervisor.compassSupervisor
## @brief     Initialization and execution of a COMPASS supervisor
## @author    COMPASS Team <https://github.com/ANR-COMPASS>
## @version   4.4.2
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
from shesha.supervisor.aoSupervisor import AoSupervisor
import numpy as np

import shesha.constants as scons
from shesha.constants import CONST
import shesha.ao.basis as basis
import astropy.io.fits as pfits
from tqdm import trange, tqdm
import time


class CompassSupervisor(AoSupervisor):

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
        self.P = None
        self.modalBasis = None
        if configFile is not None:
            self.loadConfig(configFile=configFile)

    def __repr__(self):
        return object.__repr__(self) + str(self._sim)

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

    def setPyrModulation(self, pyrMod: float, numwfs=0) -> None:
        '''
        Set pyramid modulation value - in l/D units
        '''
        from shesha.ao.wfs import comp_new_pyr_ampl
        p_wfs = self._sim.config.p_wfss[numwfs]

        _, _, _, pyr_npts = comp_new_pyr_ampl(0, pyrMod, self._sim.wfs, self._sim.rtc,
                                              self._sim.config.p_wfss,
                                              self._sim.config.p_tel)
        if (len(p_wfs._halfxy.shape) == 2):
            print("PYR modulation set to: %f L/D using %d points" % (pyrMod, pyr_npts))
        elif (len(p_wfs._halfxy.shape) == 3):
            newhalfxy = np.tile(p_wfs._halfxy[0, :, :], (pyr_npts, 1, 1))
            print("Loading new modulation arrays")
            self._sim.wfs.d_wfs[numwfs].set_phalfxy(
                    np.exp(1j * newhalfxy).astype(np.complex64).T)
            print("Done. PYR modulation set to: %f L/D using %d points" % (pyrMod,
                                                                           pyr_npts))
        else:
            raise ValueError("Error unknown p_wfs._halfxy shape")
        self._sim.rtc.do_centroids(0)  # To be ready for the next getSlopes

    def setFourierMask(self, newmask, wfsnum=0):
        """
        Set a mask in the Fourier Plane of the given WFS
        """
        if newmask.shape != self.config.p_wfss[wfsnum].get_halfxy().shape:
            print('Error : mask shape should be {}'.format(
                    self.config.p_wfss[wfsnum].get_halfxy().shape))
        else:
            self._sim.wfs.d_wfs[wfsnum].set_phalfxy(
                    np.exp(1j * np.fft.fftshift(newmask)).astype(np.complex64).T)

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

    def setGlobalR0(self, r0, reset_seed=-1):
        """
        Change the current global r0 of all layers
        :param r0 (float): r0 @ 0.5 µm
        :param reset_seed (int): if -1 keep same seed and same screen
                                if 0 random seed is applied and refresh screens
                                if (value) set the given seed and refresh screens
        """

        self._sim.atm.set_global_r0(r0)
        if reset_seed != -1:
            if reset_seed == 0:
                ilayer = np.random.randint(1e4)
            else:
                ilayer = reset_seed
            for k in range(self._sim.atm.nscreens):
                self._sim.atm.set_seed(k, 1234 + ilayer)
                self._sim.atm.refresh_screen(k)
                ilayer += 1

    def setGSmag(self, mag, numwfs=0):
        """
        Change the guide star magnitude for the given WFS.

        """
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

    def setWind(self, nScreen: int, windspeed: float = None, winddir: float = None):
        """ Set new wind information for the given screen

        Parameters:
            nScreen : (int) : Atmos screen to change

            windspeed : (float) [m/s] : new wind speed of the screen. If None, the wind speed is unchanged

            winddir : (float) [deg]: new wind direction of the screen. If None, the wind direction is unchanged


        Author: FF
        """
        if windspeed is not None:
            self.config.p_atmos.windspeed[nScreen] = windspeed
        if winddir is not None:
            self.config.p_atmos.winddir[nScreen] = winddir

        lin_delta = self.config.p_geom.pupdiam / self.config.p_tel.diam * self.config.p_atmos.windspeed[nScreen] * \
                    np.cos(CONST.DEG2RAD * self.config.p_geom.zenithangle) * self.config.p_loop.ittime
        oldx = self.config.p_atmos._deltax[nScreen]
        oldy = self.config.p_atmos._deltay[nScreen]
        self.config.p_atmos._deltax[nScreen] = lin_delta * np.sin(
                CONST.DEG2RAD * self.config.p_atmos.winddir[nScreen] + np.pi)
        self.config.p_atmos._deltay[nScreen] = lin_delta * np.cos(
                CONST.DEG2RAD * self.config.p_atmos.winddir[nScreen] + np.pi)
        self._sim.atm.d_screens[nScreen].set_deltax(self.config.p_atmos._deltax[nScreen])
        self._sim.atm.d_screens[nScreen].set_deltay(self.config.p_atmos._deltay[nScreen])
        if (oldx * self.config.p_atmos._deltax[nScreen] <
                    0):  #Sign has changed, must change the stencil
            stencilx = np.array(self._sim.atm.d_screens[nScreen].d_istencilx)
            n = self.config.p_atmos.dim_screens[nScreen]
            stencilx = (n * n - 1) - stencilx
            self._sim.atm.d_screens[nScreen].set_istencilx(stencilx)
        if (oldy * self.config.p_atmos._deltay[nScreen] <
                    0):  #Sign has changed, must change the stencil
            stencily = np.array(self._sim.atm.d_screens[nScreen].d_istencily)
            n = self.config.p_atmos.dim_screens[nScreen]
            stencily = (n * n - 1) - stencily
            self._sim.atm.d_screens[nScreen].set_istencily(stencily)

    def getInfluFunction(self, ndm):
        """
        returns the influence function cube for the given dm

        """
        return self._sim.config.p_dms[ndm]._influ

    def getInfluFunctionIpupilCoords(self, ndm):
        """
        returns the lower left coordinates of the influ function support in the ipupil coord system

        """
        i1 = self._sim.config.p_dm0._i1  # i1 is in the dmshape support coords
        j1 = self._sim.config.p_dm0._j1  # j1 is in the dmshape support coords
        ii1 = i1 + self._sim.config.p_dm0._n1  # in  ipupil coords
        jj1 = j1 + self._sim.config.p_dm0._n1  # in  ipupil coords
        return ii1, jj1

    def getTargetPhase(self, tarnum):
        """
        Returns the target phase
        """
        pup = self.getSpupil()
        ph = self.getTarPhase(tarnum) * pup
        return ph

    def getTarPhase(self, numTar: int) -> np.ndarray:
        '''
        returns the target screen of target number numTar
        '''
        return np.array(self._sim.tar.d_targets[numTar].d_phase)

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

    def getPyrHRImage(self, numWFS: int = 0) -> np.ndarray:
        '''
        Get an HR image from the WFS
        '''
        return np.array(self._sim.wfs.d_wfs[numWFS].d_hrimg)

    def getSlopeGeom(self, numWFS: int, ncontrol: int = 0) -> np.ndarray:
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
        """
        sets the additional fixed phase in the WFS path.

        ncpa must be of the same size of the Mpupil support (see also getMpupil())
        """
        self._sim.wfs.d_wfs[wfsnum].d_gs.set_ncpa(ncpa)

    def setNcpaTar(self, ncpa, tarnum):
        """
        sets the additional fixed phase in the TARGET path.

        ncpa must be of the same size of the Spupil support (see also getSpupil())
        """
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

    def setDmCommand(self, numdm, volts):
        """
        Allows to by-pass the RTC for sending a command to the
        specified DM <numdm>.
        This command comes in addition to the RTC computation.
        It allows a direct access the DM without using the RTC.

        <numdm> : number of the DM
        <volts> : voltage vector to be applied on the DM.


        Author: unknown
        """
        ntotDm = len(self._sim.config.p_dms)
        if (numdm < ntotDm):
            self._sim.dms.d_dms[numdm].set_com(volts)
        else:
            print("ERROR !!!!\nRequested DM (", numdm,
                  ") conflicts with number of available DMs (", ntotDm, ").")

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

    """
    ---------------------------------
    ---- Imported from Canapass -----
    ---------------------------------
    """
    """
    ____    _    ____ ___ ____
    | __ )  / \  / ___|_ _/ ___|
    |  _ \ / _ \ \___ \| |\___ \
    | |_) / ___ \ ___) | | ___) |
    |____/_/   \_\____/___|____/

     ____ ___  __  __ ____  _   _ _____  _  _____ ___ ___  _   _
    / ___/ _ \|  \/  |  _ \| | | |_   _|/ \|_   _|_ _/ _ \| \ | |
    | |  | | | | |\/| | |_) | | | | | | / _ \ | |  | | | | |  \| |
    | |__| |_| | |  | |  __/| |_| | | |/ ___ \| |  | | |_| | |\  |
    \____\___/|_|  |_|_|    \___/  |_/_/   \_\_| |___\___/|_| \_|

    """

    def first_nonzero(self, arr, axis, invalid_val=-1):
        """
        Find the first non zero element of an array.

        Author: Milan Rozel
        """
        mask = arr != 0
        return np.where(mask.any(axis=axis), mask.argmax(axis=axis), invalid_val)

    def getModes2VBasis(self, ModalBasisType, merged=False, nbpairs=None,
                        returnDelta=False):
        """
        Computes a given modal basis ("KL2V", "Btt", "Btt_petal")

        See also: returnkl2V, compute_Btt and compute_btt_petal


        Authors: FV, AB, EG
        """

        if (ModalBasisType == "KL2V"):
            print("Computing KL2V basis...")
            self.modalBasis, _ = self.returnkl2V()
            fnz = self.first_nonzero(self.modalBasis, axis=0)
            # Computing the sign of the first non zero element
            #sig = np.sign(self.modalBasis[[fnz, np.arange(self.modalBasis.shape[1])]])
            sig = np.sign(self.modalBasis[tuple([
                    fnz, np.arange(self.modalBasis.shape[1])
            ])])  # pour remove le future warning!
            self.modalBasis *= sig[None, :]
            return self.modalBasis, 0
        elif (ModalBasisType == "Btt"):
            print("Computing Btt basis...")
            self.modalBasis, self.P = self.compute_Btt(inv_method="cpu_svd",
                                                       merged=merged, nbpairs=nbpairs,
                                                       returnDelta=returnDelta)
            fnz = self.first_nonzero(self.modalBasis, axis=0)
            # Computing the sign of the first non zero element
            #sig = np.sign(self.modalBasis[[fnz, np.arange(self.modalBasis.shape[1])]])
            sig = np.sign(self.modalBasis[tuple([
                    fnz, np.arange(self.modalBasis.shape[1])
            ])])  # pour remove le future warning!
            self.modalBasis *= sig[None, :]
            return self.modalBasis, self.P
        elif (ModalBasisType == "Btt_petal"):
            print("Computing Btt with a Petal basis...")
            self.modalBasis, self.P = self.compute_btt_petal()
            return self.modalBasis, self.P

    def returnkl2V(self):
        """
        Compute the Karhunen-Loeve to Volt matrix
        (transfer matrix between the KL space and volt space for a pzt dm)

        Author: FF
        """
        if (self.KL2V is None):
            print("Computing KL2V...")
            KL2V = basis.compute_KL2V(self._sim.config.p_controllers[0], self._sim.dms,
                                      self._sim.config.p_dms, self._sim.config.p_geom,
                                      self._sim.config.p_atmos, self._sim.config.p_tel)
            print("KL2V Done!")
            self.KL2V = KL2V
            return KL2V, 0
        else:
            return self.KL2V, 0

    def compute_Btt(self, inv_method: str = "cpu_svd", merged=False, nbpairs=None,
                    returnDelta=False):
        """
        Computes the so-called Btt modal basis.

        if returnDelta = False: returns
        (Btt, projection matrix)
        if returnDelta = True: returns
        (Btt, Delta Matrix)
        If return Delta is True returns:

        if merged = True merges 2x2 the actuators IFs for actuators at each side of the spider (ELT case).


        Authors: FF, EG, FV, VD
        """

        IF = self.getIFsparse(1)
        if (merged):
            couplesActus, indUnderSpiders = self.computeMerged(nbpairs=nbpairs)
            IF2 = IF.copy()
            indremoveTmp = indUnderSpiders.copy()
            indremoveTmp += list(couplesActus[:, 1])
            print("Pairing Actuators...")
            for i in tqdm(range(couplesActus.shape[0])):
                IF2[couplesActus[i, 0], :] += IF2[couplesActus[i, 1], :]
            print("Pairing Done")
            boolarray = np.zeros(IF2.shape[0], dtype=np.bool)
            boolarray[indremoveTmp] = True
            self.slavedActus = boolarray
            self.selectedActus = ~boolarray
            self.couplesActus = couplesActus
            self.indUnderSpiders = indUnderSpiders
            IF2 = IF2[~boolarray, :]
            IF = IF2
        else:
            self.slavedActus = None
            self.selectedActus = None
            self.couplesActus = None
            self.indUnderSpiders = None
        n = IF.shape[0]
        N = IF.shape[1]
        T = self.getIFtt(1)

        delta = IF.dot(IF.T).toarray() / N

        # Tip-tilt + piston
        Tp = np.ones((T.shape[0], T.shape[1] + 1))
        Tp[:, :2] = T.copy()  #.toarray()
        deltaT = IF.dot(Tp) / N
        # Tip tilt projection on the pzt dm
        tau = np.linalg.inv(delta).dot(deltaT)

        # Famille generatrice sans tip tilt
        G = np.identity(n)
        tdt = tau.T.dot(delta).dot(tau)
        subTT = tau.dot(np.linalg.inv(tdt)).dot(tau.T).dot(delta)
        G -= subTT

        # Base orthonormee sans TT
        gdg = G.T.dot(delta).dot(G)

        startTimer = time.time()
        if inv_method == "cpu_svd":
            print("Doing SVD (CPU)")
            U, s, _ = np.linalg.svd(gdg)
        # elif inv_method == "gpu_svd":
        #     print("Doing SVD on CPU of a matrix...")
        #     m = gdg.shape[0]
        #     h_mat = host_obj_Double2D(data=gdg, mallocType="pagelock")
        #     h_eig = host_obj_Double1D(data=np.zeros([m], dtype=np.float64),
        #                               mallocType="pagelock")
        #     h_U = host_obj_Double2D(data=np.zeros((m, m), dtype=np.float64),
        #                             mallocType="pagelock")
        #     h_VT = host_obj_Double2D(data=np.zeros((m, m), dtype=np.float64),
        #                              mallocType="pagelock")
        #     svd_host_Double(h_mat, h_eig, h_U, h_VT)
        #     U = h_U.getData().T.copy()
        #     s = h_eig.getData()[::-1].copy()
        # elif inv_method == "gpu_evd":
        #     print("Doing EVD on GPU of a matrix...")
        #     c = carmaWrap_context()
        #     m = gdg.shape[0]
        #     d_mat = obj_Double2D(c, data=gdg)
        #     d_U = obj_Double2D(c, data=np.zeros([m, m], dtype=np.float64))
        #     h_s = np.zeros(m, dtype=np.float64)
        #     syevd_Double(d_mat, h_s, d_U)
        #     U = d_U.device2host().T.copy()
        #     s = h_s[::-1].copy()
        else:
            raise "ERROR cannot recognize inv_method"
        print("Done in %fs" % (time.time() - startTimer))
        U = U[:, :U.shape[1] - 3]
        s = s[:s.size - 3]
        L = np.identity(s.size) / np.sqrt(s)
        B = G.dot(U).dot(L)

        # Rajout du TT
        TT = T.T.dot(T) / N  #.toarray()/N
        Btt = np.zeros((n + 2, n - 1))
        Btt[:B.shape[0], :B.shape[1]] = B
        mini = 1. / np.sqrt(np.abs(TT))
        mini[0, 1] = 0
        mini[1, 0] = 0
        Btt[n:, n - 3:] = mini

        # Calcul du projecteur actus-->modes
        delta = np.zeros((n + T.shape[1], n + T.shape[1]))
        delta[:-2, :-2] = IF.dot(IF.T).toarray() / N
        delta[-2:, -2:] = T.T.dot(T) / N
        P = Btt.T.dot(delta)
        if (merged):
            Btt2 = np.zeros((len(boolarray) + 2, Btt.shape[1]))
            Btt2[np.r_[~boolarray, True, True], :] = Btt
            Btt2[couplesActus[:, 1], :] = Btt2[couplesActus[:, 0], :]

            P2 = np.zeros((Btt.shape[1], len(boolarray) + 2))
            P2[:, np.r_[~boolarray, True, True]] = P
            P2[:, couplesActus[:, 1]] = P2[:, couplesActus[:, 0]]
            Btt = Btt2
            P = P2
        if (returnDelta):
            P = delta
        return Btt, P

    def computeMerged(self, nbpairs=None):
        """
        Used to compute merged IF from each side of the spider for an ELT case (Petalling Effect)

        Authors: FV, VD
        """
        p_geom = self._sim.config.p_geom
        import shesha.util.make_pupil as mkP
        import shesha.util.utilities as util
        import scipy.ndimage

        cent = p_geom.pupdiam / 2. + 0.5
        p_tel = self._sim.config.p_tel
        p_tel.t_spiders = 0.51
        spup = mkP.make_pupil(p_geom.pupdiam, p_geom.pupdiam, p_tel, cent,
                              cent).astype(np.float32).T

        p_tel.t_spiders = 0.
        spup2 = mkP.make_pupil(p_geom.pupdiam, p_geom.pupdiam, p_tel, cent,
                               cent).astype(np.float32).T

        spiders = spup2 - spup

        (spidersID, k) = scipy.ndimage.label(spiders)
        spidersi = util.pad_array(spidersID, p_geom.ssize).astype(np.float32)
        pxListSpider = [np.where(spidersi == i) for i in range(1, k + 1)]

        # DM positions in iPupil:
        dmposx = self._sim.config.p_dm0._xpos - 0.5
        dmposy = self._sim.config.p_dm0._ypos - 0.5
        dmposMat = np.c_[dmposx, dmposy].T  # one actu per column

        pitch = self._sim.config.p_dm0._pitch
        DISCARD = np.zeros(len(dmposx), dtype=np.bool)
        PAIRS = []

        # For each of the k pieces of the spider
        for k, pxList in enumerate(pxListSpider):
            pts = np.c_[pxList[1], pxList[0]]  # x,y coord of pixels of the spider piece
            # lineEq = [a, b]
            # Which minimizes leqst squares of aa*x + bb*y = 1
            lineEq = np.linalg.pinv(pts).dot(np.ones(pts.shape[0]))
            aa, bb = lineEq[0], lineEq[1]

            # Find any point of the fitted line.
            # For simplicity, the intercept with one of the axes x = 0 / y = 0
            if np.abs(bb) < np.abs(aa):  # near vertical
                onePoint = np.array([1 / aa, 0.])
            else:  # otherwise
                onePoint = np.array([0., 1 / bb])

            # Rotation that aligns the spider piece to the horizontal
            rotation = np.array([[-bb, aa], [-aa, -bb]]) / (aa**2 + bb**2)**.5

            # Rotated the spider mask
            rotatedPx = rotation.dot(pts.T - onePoint[:, None])
            # Min and max coordinates along the spider length - to filter actuators that are on
            # 'This' side of the pupil and not the other side
            minU, maxU = rotatedPx[0].min() - 5. * pitch, rotatedPx[0].max() + 5. * pitch

            # Rotate the actuators
            rotatedActus = rotation.dot(dmposMat - onePoint[:, None])
            selGoodSide = (rotatedActus[0] > minU) & (rotatedActus[0] < maxU)
            seuil = 0.05
            # Actuators below this piece of spider
            selDiscard = (np.abs(rotatedActus[1]) < seuil * pitch) & selGoodSide
            DISCARD |= selDiscard

            # Actuator 'near' this piece of spider
            selPairable = (np.abs(rotatedActus[1]) > seuil  * pitch) & \
                            (np.abs(rotatedActus[1]) < 1. * pitch) & \
                            selGoodSide

            pairableIdx = np.where(selPairable)[0]  # Indices of these actuators
            uCoord = rotatedActus[
                    0, selPairable]  # Their linear coord along the spider major axis

            order = np.sort(uCoord)  # Sort by linear coordinate
            orderIdx = pairableIdx[np.argsort(
                    uCoord)]  # And keep track of original indexes

            # i = 0
            # while i < len(order) - 1:
            if (nbpairs is None):
                i = 0
                ii = len(order) - 1
            else:
                i = len(order) // 2 - nbpairs
                ii = len(order) // 2 + nbpairs
            while (i < ii):
                # Check if next actu in sorted order is very close
                # Some lonely actuators may be hanging in this list
                if np.abs(order[i] - order[i + 1]) < .2 * pitch:
                    PAIRS += [(orderIdx[i], orderIdx[i + 1])]
                    i += 2
                else:
                    i += 1
        print('To discard: %u actu' % np.sum(DISCARD))
        print('%u pairs to slave' % len(PAIRS))
        if np.sum(DISCARD) == 0:
            DISCARD = []
        else:
            list(np.where(DISCARD)[0])
        return np.asarray(PAIRS), list(np.where(DISCARD)[0])

    def compute_btt_petal(self):
        """
        Computes a Btt modal basis with Pistons filtered

        Author: AB
        """

        # Tip-tilt + piston + petal modes
        IF = self.getIFsparse(1)
        IFpetal = self.getIFdm(1)
        IFtt = self.getIFdm(2)

        n = IF.shape[0]  # number of points (pixels) over the pupil
        N = IF.shape[1]  # number of influence functions (nb of actuators)

        # Compute matrix delta (geometric covariance of actus)
        delta = IF.dot(IF.T).toarray() / N

        # Petal basis generation (orthogonal to global piston)
        nseg = IFpetal.toarray().shape[0]

        petal_modes = -1 / (nseg - 1) * np.ones((nseg, (nseg - 1)))
        petal_modes += nseg / (nseg - 1) * np.eye(nseg)[:, 0:(
                nseg - 1)]  # petal modes within the petal dm space
        tau_petal = np.dot(IF.toarray(), IFpetal.toarray().T).dot(petal_modes)

        Tp = np.concatenate((IFtt.toarray(), np.ones((1, N))),
                            axis=0)  # Matrice contenant Petal Basis + Tip/Tilt + Piston
        deltaT = IF.dot(Tp.T) / N

        # Tip tilt + petals projection on the pzt dm
        tau = np.concatenate((tau_petal, np.linalg.inv(delta).dot(deltaT)), axis=1)

        # Famille generatrice sans tip tilt ni pétales
        G = np.identity(n)
        tdt = tau.T.dot(delta).dot(tau)
        subTT = tau.dot(np.linalg.inv(tdt)).dot(tau.T).dot(delta)
        G -= subTT

        # Base orthonormee sans Tip, Tilp, Piston, Pétales
        gdg = G.T.dot(delta).dot(G)
        U, s, V = np.linalg.svd(gdg)
        U = U[:, :U.shape[1] - 8]
        s = s[:s.size - 8]
        L = np.identity(s.size) / np.sqrt(s)
        B = G.dot(U).dot(L)

        # Rajout du TT et Pétales
        TT = IFtt.toarray().dot(IFtt.toarray().T) / N  # .toarray()/N
        Btt = np.zeros((n + 2, n - 1))
        Btt[:n, :B.shape[1]] = B
        mini = 1. / np.sqrt(np.abs(TT))
        mini[0, 1] = 0
        mini[1, 0] = 0
        Btt[n:, -2:] = mini  # ajout du tip tilt sur le miroir tip tilt
        Btt[:n, -7:-2] = tau_petal  # ajout des modes pétales sur le miroir M4

        # Calcul du projecteur actus-->modes
        delta = np.zeros((n + IFtt.shape[0], n + IFtt.shape[0]))
        delta[:-2, :-2] = IF.dot(IF.T).toarray() / N
        delta[-2:, -2:] = IFtt.toarray().dot(IFtt.toarray().T) / N
        P = Btt.T.dot(delta)

        return Btt.astype(np.float32), P.astype(np.float32)

    def setModalBasis(self, modalBasis, P):
        """
        Function used to set the modal basis and projector in canapass
        """
        self.modalBasis = modalBasis
        self.P = P

    def computePh2Modes(self, modalBasis):
        """
        return the phase 2 modes matrix by using the modal basis

        Author: unknown
        """
        oldnoise = self._sim.config.p_wfs0.noise
        self.setNoise(-1)

        nbmode = modalBasis.shape[1]
        pup = self._sim.config.p_geom._spupil
        ph = self._sim.tar.get_phase(0)
        ph2KL = np.zeros((nbmode, ph.shape[0], ph.shape[1]))
        S = np.sum(pup)
        for i in range(nbmode):
            self.resetTarPhase(0)
            self._sim.dms.set_full_comm((modalBasis[:, i]).astype(np.float32).copy())
            self._sim.next(see_atmos=False)
            ph = self.getTarPhase(0) * pup
            # Normalisation pour les unites rms en microns !!!
            norm = np.sqrt(np.sum((ph)**2) / S)
            ph2KL[i] = ph / norm
        self.ph2modes = ph2KL
        self.setNoise(oldnoise)
        return ph2KL

    def computePh2ModesFits(self, modalBasis, fullpath):
        """
        computes the phase 2 modes matrix by using the modal basis and writes it in the user provided fits file

        Author: unknown
        """
        data = self.computePh2Modes(modalBasis)
        pfits.writeto(fullpath, data, overwrite=True)

    def applyVoltGetSlopes(self, noise=False, turbu=False, reset=1):
        """
        Force to apply the Voltages and update the new slopes

        Author: FV
        """
        self._sim.rtc.apply_control(0)
        for w in range(len(self._sim.wfs.d_wfs)):

            if (turbu):
                self._sim.raytraceWfs(w, "all")

            else:
                self._sim.raytraceWfs(w, ["dm", "ncpa"], rst=reset)
            self._sim.compWfsImage(w, noise=noise)
        self._sim.rtc.do_centroids(0)
        c = self.getCentroids(0)
        return c

    def doImatModal(self, ampliVec, modalbasis, Nslopes, noise=False, nmodesMax=0,
                    withTurbu=False, pushPull=False):
        """
        Computes an iteraction Matrix from the user provided modal Basis

        Author: FV
        """
        iMatKL = np.zeros((Nslopes, modalbasis.shape[1]))

        if (nmodesMax == 0):
            nmodesMax = modalbasis.shape[1]
        vold = self.getCom(0)
        self.openLoop(rst=False)
        for kl in range(nmodesMax):
            # v = ampliVec[kl] * modalbasis[:, kl:kl + 1].T.copy()
            v = ampliVec[kl] * modalbasis[:, kl]
            if ((pushPull is True) or
                (withTurbu is True)):  # with turbulence/aberrations => push/pull
                self.setPerturbationVoltage(
                        0, "imatModal",
                        vold + v)  # Adding Perturbation voltage on current iteration
                devpos = self.applyVoltGetSlopes(turbu=withTurbu, noise=noise)
                self.setPerturbationVoltage(0, "imatModal", vold - v)
                devmin = self.applyVoltGetSlopes(turbu=withTurbu, noise=noise)
                iMatKL[:, kl] = (devpos - devmin) / (2. * ampliVec[kl])
                #imat[:-2, :] /= pushDMMic
                #if(nmodesMax == 0):# i.e we measured all modes including TT
                #imat[-2:, :] /= pushTTArcsec
            else:  # No turbulence => push only
                self.openLoop()  # openLoop
                self.setPerturbationVoltage(0, "imatModal", v)
                iMatKL[:, kl] = self.applyVoltGetSlopes(noise=noise) / ampliVec[kl]
        self.removePerturbationVoltage(0, "imatModal")
        if ((pushPull is True) or (withTurbu is True)):
            self.closeLoop()  # We are supposed to be in close loop now
        return iMatKL

    def doImatPhase(self, cubePhase, Nslopes, noise=False, nmodesMax=0, withTurbu=False,
                    pushPull=False, wfsnum=0):
        """
        Computes an iteraction Matrix from the user provided cube phase [nphase, NFFT, NFFT]


        Author: FV
        """
        iMatPhase = np.zeros((cubePhase.shape[0], Nslopes))
        for nphase in range(cubePhase.shape[0]):
            if ((pushPull is True) or
                (withTurbu is True)):  # with turbulence/aberrations => push/pull
                self.setNcpaWfs(cubePhase[nphase, :, :], wfsnum=wfsnum)
                devpos = self.applyVoltGetSlopes(turbu=withTurbu, noise=noise)
                self.setNcpaWfs(-cubePhase[nphase, :, :], wfsnum=wfsnum)
                devmin = self.applyVoltGetSlopes(turbu=withTurbu, noise=noise)
                iMatPhase[nphase, :] = (devpos - devmin) / 2
            else:  # No turbulence => push only
                self.openLoop()  # openLoop
                self.setNcpaWfs(cubePhase[nphase, :, :], wfsnum=wfsnum)
                iMatPhase[nphase, :] = self.applyVoltGetSlopes(noise=noise)
        self.setNcpaWfs(cubePhase[nphase, :, :] * 0.,
                        wfsnum=wfsnum)  # Remove the Phase on WFS
        _ = self.applyVoltGetSlopes(turbu=withTurbu, noise=noise)

        return iMatPhase

    def computeModalResiduals(self):
        """
        Computes the modal residuals coefficients of the residual phase.

        Uses the P matrix computed from getModes2VBasis

        Requires to use Rocket
        Author: FV
        """
        try:
            self._sim.doControl(1, 0)
        except:
            return [0]
        v = self.getCom(
                1
        )  # We compute here the residual phase on the DM modes. Gives the Equivalent volts to apply/
        if (self.P is None):
            return [0]
            # self.modalBasis, self.P = self.getModes2VBasis("Btt")
        if (self.selectedActus is None):
            ai = self.P.dot(v) * 1000.  # np rms units
        else:  # Slaving actus case
            v2 = v[:-2][list(
                    self.selectedActus)]  # If actus are slaved then we select them.
            v3 = v[-2:]
            ai = self.P.dot(np.concatenate((v2, v3))) * 1000.
        return ai
        """
         ______   ______
        |  _ \ \ / /  _ \
        | |_) \ V /| |_) |
        |  __/ | | |  _ <
        |_|    |_| |_| \_\

        """

    def setPyrSourceArray(self, cx, cy, nwfs=0):
        """
        Sets the Pyramid source Array
        cx, cy must be in arcseconds units

        Author: MG
        """
        pyr_npts = len(cx)
        wfs = self._sim.wfs
        pwfs = self._sim.config.p_wfss[nwfs]
        pwfs.set_pyr_npts(pyr_npts)
        pwfs.set_pyr_cx(cx)
        pwfs.set_pyr_cy(cy)
        wfs.d_wfs[nwfs].set_pyr_modulation(cx, cy, pyr_npts)
        # RTC scale units to be updated ????
        #scale = pwfs.Lambda * 1e-6 / p_tel.diam * ampli * 180. / np.pi * 3600.
        #rtc.d_centro[nwfs].set_scale(scale)

    def setPyrMultipleStarsSource(self, coords, weights=None, pyrmod=3., niters=None,
                                  nwfs=0):
        """
        Sets the Pyramid source Array with a multiple star system
        coords is a list of couples of length n, coordinates of the n stars in lambda/D
        pyrmod is the modulation of the pyramid in lambda/D
        niters is the number of iteration

        Author: MG
        """
        if niters == None:
            perim = pyrmod * 2 * np.pi
            niters = int((perim // 4 + 1) * 4)
            print(niters)
        nstars = len(coords)
        pyr_npts = niters * nstars
        wfs = self._sim.wfs
        pwfs = self._sim.config.p_wfss[nwfs]
        ptel = self._sim.config.p_tel
        #Computes the positions of the stars during the modulation
        pyrsize = pwfs._Nfft
        pixsize = (np.pi * pwfs._qpixsize) / (3600 * 180)
        scale_circ = 2 * np.pi / pyrsize * \
            (pwfs.Lambda * 1e-6 / ptel.diam) / pixsize * pyrmod
        scale_pos = 2 * np.pi / pyrsize * \
            (pwfs.Lambda * 1e-6 / ptel.diam) / pixsize
        temp_cx = []
        temp_cy = []
        for k in coords:
            temp_cx.append(scale_circ * \
                np.sin((np.arange(niters)) * 2. * np.pi / niters) + \
                k[0] * scale_pos)
            temp_cy.append(scale_circ * \
                np.cos((np.arange(niters)) * 2. * np.pi / niters) + \
                k[1] * scale_pos)
        cx = np.concatenate(np.array(temp_cx))
        cy = np.concatenate(np.array(temp_cy))
        #Gives the arguments to the simulation
        pwfs.set_pyr_npts(pyr_npts)
        pwfs.set_pyr_cx(cx)
        pwfs.set_pyr_cy(cy)
        if weights == None:
            wfs.d_wfs[nwfs].set_pyr_modulation(cx, cy, pyr_npts)
        else:
            w = []
            for k in weights:
                w += niters * [k]
            wfs.d_wfs[nwfs].set_pyr_modulation(cx, cy, w, pyr_npts)

    def setPyrDiskSourceHP(self, radius, density=1., nwfs=0):
        """
        radius is the radius of the disk object in lambda/D
        density is the spacing between the packed PSF in the disk object, in lambda/D

        create disk object by packing PSF in a given radius, using hexagonal packing
        /!\ There is no modulation

        Author: MG
        """
        wfs = self._sim.wfs
        pwfs = self._sim.config.p_wfss[nwfs]
        ptel = self._sim.config.p_tel
        pyrsize = pwfs._Nfft
        pixsize = (np.pi * pwfs._qpixsize) / (3600 * 180)
        scale_pos = 2 * np.pi / pyrsize * \
            (pwfs.Lambda * 1e-6 / ptel.diam) / pixsize
        #Vectors used to generate the hexagonal paving
        gen_xp, gen_yp = np.array([1,
                                   0.]), np.array([np.cos(np.pi / 3),
                                                   np.sin(np.pi / 3)])
        n = 1 + int(1.2 * radius)
        mat_circ = []
        for k in range(-n, n):
            for l in range(-n, n):
                coord = k * gen_xp + l * gen_yp
                if np.sqrt(coord[0]**2 + coord[1]**2) <= radius:
                    mat_circ.append(coord)
        mat_circ = np.array(mat_circ)
        cx, cy = mat_circ[:, 0], mat_circ[:, 1]
        pyr_npts = len(cx)
        pwfs.set_pyr_npts(pyr_npts)
        pwfs.set_pyr_cx(cx)
        pwfs.set_pyr_cy(cy)
        wfs.d_wfs[nwfs].set_pyr_modulation(cx, cy, pyr_npts)

    def generate_square(self, radius, density=1.):
        """
        radius is half the length of a side in lambda/D
        density is the number of psf per lambda/D

        Author: MG
        """
        x = np.linspace(-radius, radius, 1 + 2 * int(radius / density))
        cx, cy = np.meshgrid(x, x, indexing='ij')
        cx = cx.flatten()
        cy = cy.flatten()
        return (cx, cy)

    def generate_square_circ(self, radius, density=1.):
        """
        Author: MG
        """
        x = np.linspace(-radius, radius, 1 + 2 * int(radius / density))
        cx, cy = np.meshgrid(x, x, indexing='ij')
        cx = cx.flatten()
        cy = cy.flatten()
        r = cx * cx + cy * cy <= radius**2
        return (cx[r], cy[r])

    def setPyrDiskSourceSP(self, radius, density=1., nwfs=0):
        """
        radius is the radius of the disk object in lambda/D
        density is the spacing between the packed PSF in the disk object, in lambda/D

        create disk object by packing PSF in a given radius, using square packing
        /!\ There is no modulation

        Author: MG
        """

        wfs = self._sim.wfs
        pwfs = self._sim.config.p_wfss[nwfs]
        ptel = self._sim.config.p_tel
        pyrsize = pwfs._Nfft
        pixsize = (np.pi * pwfs._qpixsize) / (3600 * 180)
        scale_pos = 2 * np.pi / pyrsize * \
            (pwfs.Lambda * 1e-6 / ptel.diam) / pixsize

        cx, cy = self.generate_square_circ(radius, density)
        cx = cx.flatten() * scale_pos
        cy = cy.flatten() * scale_pos
        pyr_npts = len(cx)
        pwfs.set_pyr_npts(pyr_npts)
        pwfs.set_pyr_cx(cx)
        pwfs.set_pyr_cy(cy)
        wfs.d_wfs[nwfs].set_pyr_modulation(cx, cy, pyr_npts)

    def setPyrSquareSource(self, radius, density=1., nwfs=0):
        """
        radius is half of the side of the object in lambda/D
        density is the spacing between the packed PSF in the square object, in lambda/D

        create square object by packing PSF in a given radius, using square packing
        /!\ There is no modulation

        Author: MG
        """
        wfs = self._sim.wfs
        pwfs = self._sim.config.p_wfss[nwfs]
        ptel = self._sim.config.p_tel
        pyrsize = pwfs._Nfft
        pixsize = (np.pi * pwfs._qpixsize) / (3600 * 180)
        scale_pos = 2 * np.pi / pyrsize * \
            (pwfs.Lambda * 1e-6 / ptel.diam) / pixsize
        x = np.linspace(-radius, radius, 1 + 2 * int(radius / density)) * scale_pos
        cx, cy = np.meshgrid(x, x, indexing='ij')
        cx = cx.flatten()
        cy = cy.flatten()
        pyr_npts = len(cx)
        pwfs.set_pyr_npts(pyr_npts)
        pwfs.set_pyr_cx(cx)
        pwfs.set_pyr_cy(cy)
        wfs.d_wfs[nwfs].set_pyr_modulation(cx, cy, pyr_npts)

    def generate_pseudo_source(self, radius, additional_psf=0, density=1.):
        """

        Used to generate a pseudo source for PYRWFS

        Author: MG
        """

        struct_size = (1 + 2 * additional_psf)**2
        center_x, center_y = self.generate_square(additional_psf, density)
        center_weight = (1 + 2 * int(additional_psf / density))**2 * [1]
        center_size = 1 + 2 * int(additional_psf / density)

        weight_edge = [(1 + 2 * int(radius / density) - center_size) // 2]
        xc, yc = self.generate_square_circ(radius, density)
        for k in range(additional_psf):
            line_length = np.sum(yc == (k + 1))
            print(line_length)
            weight_edge.append((line_length - center_size) // 2)

        edge_dist = (radius + additional_psf) // 2
        V_edge_x = []
        V_edge_y = []
        V_edge_weight = []
        for m in [-1, 1]:
            V_edge_x.append(0)
            V_edge_y.append(m * edge_dist)
            V_edge_weight.append(weight_edge[0])
        for k, val in enumerate(weight_edge[1:]):
            for l in [-1, 1]:
                for m in [-1, 1]:
                    V_edge_x.append(l * (k + 1) * density)
                    V_edge_y.append(m * edge_dist)
                    V_edge_weight.append(val)
        H_edge_x = []
        H_edge_y = []
        H_edge_weight = []
        for m in [-1, 1]:
            H_edge_x.append(m * edge_dist)
            H_edge_y.append(0)
            H_edge_weight.append(weight_edge[0])
        for k, val in enumerate(weight_edge[1:]):
            for l in [-1, 1]:
                for m in [-1, 1]:
                    H_edge_x.append(m * edge_dist)
                    H_edge_y.append(l * (k + 1) * density)
                    H_edge_weight.append(val)
        pup_cent_x = []
        pup_cent_y = []
        pup_cent_weight = 4 * [(len(xc) - 2 * np.sum(H_edge_weight) - struct_size) / 4]
        pup_cent_dist = int(edge_dist // np.sqrt(2))
        for l in [-1, 1]:
            for m in [-1, 1]:
                pup_cent_x.append(l * pup_cent_dist)
                pup_cent_y.append(m * pup_cent_dist)
        ox = np.concatenate((center_x, V_edge_x, H_edge_x, pup_cent_x))
        oy = np.concatenate((center_y, V_edge_y, H_edge_y, pup_cent_y))
        w = np.concatenate((center_weight, V_edge_weight, H_edge_weight,
                            pup_cent_weight))
        return (ox, oy, w, xc, yc)

    def setPyrPseudoSource(self, radius, additional_psf=0, density=1., nwfs=0):
        """

        Author: MG
        """
        cx, cy, w, _, _ = self.generate_pseudo_source(radius, additional_psf, density)

        wfs = self._sim.wfs
        pwfs = self._sim.config.p_wfss[nwfs]
        ptel = self._sim.config.p_tel
        pyrsize = pwfs._Nfft
        pixsize = (np.pi * pwfs._qpixsize) / (3600 * 180)
        scale_pos = 2 * np.pi / pyrsize * \
            (pwfs.Lambda * 1e-6 / ptel.diam) / pixsize

        cx = cx.flatten() * scale_pos
        cy = cy.flatten() * scale_pos
        pyr_npts = len(cx)
        pwfs.set_pyr_npts(pyr_npts)
        pwfs.set_pyr_cx(cx)
        pwfs.set_pyr_cy(cy)
        wfs.d_wfs[nwfs].set_pyr_modulation(cx, cy, w, pyr_npts)

    def recordCB(self, CBcount, subSample=1, tarnum=0, seeAtmos=True, cubeDataType=None,
                 cubeDataFilePath="", NCPA=False, ncpawfs=None, refSlopes=None,
                 ditchStrehl=True):
        """

        Used to record a synchronized circular buffer AO loop data.

        ----- Inputs ----- :

        CBcount: the number of iterations to record.
        subSample: sub sampling of the data (default=1, I.e no subsampling)
        tarnum: target number
        seeAtmos: used for the next function to enable or not the Atmos
        cubeDataType:  if  specified ("tarPhase" or "psfse") returns the target phase or short exposure PSF data cube in the output variable
        cubeDataFilePath: if specified it will also save the target phase cube data (full path on the server)

        NCPA: !!experimental!!!: Used only in the context of PYRWFS + NCPA compensation on the fly (with optical gain)
        defines how many iters the NCPA refslopes are updates with the proper optical gain. Ex: if NCPA=10 refslopes will be updates every 10 iters.
        ncpawfs: the ncpa phase as seen from the wfs array with dims = size of Mpupil
        refSlopes: the reference slopes to use.

        DitchStrehl: resets the long exposure SR computation at the beginning of the Circular buffer (default= True)

        ----- Outputs ----- :

        slopes: the slopes CB
        volts: the volts applied to the DM(s) CB
        ai: the modal coefficient of the residual phase projected on the currently used modal Basis
        psfLE: Long exposure PSF over the <CBcount> iterations (I.e SR is reset at the begining of the CB if ditchStrehl=True)
        srseList: The SR short exposure evolution during CB recording
        srleList: The SR long exposure evolution during CB recording
        gNPCAList: the gain applied to the NCPA (PYRWFS CASE) if NCPA is set to True
        cubeData: the tarPhase or psfse cube data (see cubeDataType)
        """
        slopesdata = None
        voltsdata = None
        cubeData = None
        aiData = None
        k = 0
        srseList = []
        srleList = []
        gNPCAList = []

        # Resets the target so that the PSF LE is synchro with the data
        # Doesn't reset it if DitchStrehl == False (used for real time gain computation)
        if ditchStrehl:
            for i in range(len(self._sim.config.p_targets)):
                self.resetStrehl(i)

        # Starting CB loop...
        for j in range(CBcount):
            print(j, end="\r")
            if (NCPA):
                if (j % NCPA == 0):
                    ncpaDiff = refSlopes[None, :]
                    ncpaturbu = self.doImatPhase(-ncpawfs[None, :, :],
                                                 refSlopes.shape[0], noise=False,
                                                 withTurbu=True)
                    gNCPA = float(
                            np.sqrt(
                                    np.dot(ncpaDiff, ncpaDiff.T) / np.dot(
                                            ncpaturbu, ncpaturbu.T)))
                    if (gNCPA > 1e18):
                        gNCPA = 0
                        print('Warning NCPA ref slopes gain too high!')
                        gNPCAList.append(gNCPA)
                        self.setRefSlopes(-refSlopes * gNCPA)
                    else:
                        gNPCAList.append(gNCPA)
                        print('NCPA ref slopes gain: %4.3f' % gNCPA)
                        self.setRefSlopes(-refSlopes / gNCPA)

            self._sim.next(see_atmos=seeAtmos)
            for t in range(len(self._sim.config.p_targets)):
                self._sim.compTarImage(t)

            srse, srle, _, _ = self.getStrehl(tarnum)
            srseList.append(srse)
            srleList.append(srle)
            if (j % subSample == 0):
                aiVector = self.computeModalResiduals()
                if (aiData is None):
                    aiData = np.zeros((len(aiVector), int(CBcount / subSample)))
                aiData[:, k] = aiVector

                slopesVector = self.getCentroids(0)
                if (slopesdata is None):
                    slopesdata = np.zeros((len(slopesVector), int(CBcount / subSample)))
                slopesdata[:, k] = slopesVector

                voltsVector = self.getCom(0)
                if (voltsdata is None):
                    voltsdata = np.zeros((len(voltsVector), int(CBcount / subSample)))
                voltsdata[:, k] = voltsVector

                if (cubeDataType):
                    if (cubeDataType == "tarPhase"):
                        dataArray = self.getTargetPhase(tarnum)
                    elif (cubeDataType == "psfse"):
                        dataArray = self.getTarImage(tarnum, "se")
                    else:
                        raise ValueError("unknown dataData" % cubeDataType)
                    if (cubeData is None):
                        cubeData = np.zeros((*dataArray.shape, int(CBcount / subSample)))
                    cubeData[:, :, k] = dataArray
                k += 1
        if (cubeDataFilePath != ""):
            print("Saving tarPhase cube at: ", cubeDataFilePath)
            pfits.writeto(cubeDataFilePath, cubeData, overwrite=True)

        psfLE = self.getTarImage(tarnum, "le")
        return slopesdata, voltsdata, aiData, psfLE, srseList, srleList, gNPCAList, cubeData
