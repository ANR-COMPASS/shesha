## @package   shesha.supervisor.aoSupervisor
## @brief     Abstract layer for initialization and execution of a AO supervisor
## @author    COMPASS Team <https://github.com/ANR-COMPASS>
## @version   4.4.1
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

from shesha.supervisor.abstractSupervisor import AbstractSupervisor
from shesha.constants import CentroiderType
import numpy as np
from tqdm import trange
import os
from collections import OrderedDict
import astropy.io.fits as pfits


class AoSupervisor(AbstractSupervisor):

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

    def __init__(self):
        self.CLOSE = False

    def getConfig(self):
        ''' Returns the configuration in use, in a supervisor specific format ? '''
        return self.config

    def loadFlat(self, flat: np.ndarray, numwfs: int = 0):
        """
        Load flat field for the given wfs

        """
        self.rtc.d_centro[numwfs].set_flat(flat, flat.shape[0])

    def loadBackground(self, background: np.ndarray, numwfs: int = 0):
        """
        Load background for the given wfs

        """
        self.rtc.d_centro[numwfs].set_dark(background, background.shape[0])

    def computeSlopes(self, nControl: int = 0):
        self.rtc.do_centroids(nControl)
        return self.getCentroids(nControl)

    def getWfsImage(self, numWFS: int = 0, calPix=False) -> np.ndarray:
        '''
        Get an image from the WFS (wfs[0] by default)
        '''
        if (calPix):
            if self.rtc.d_centro[numWFS].d_img is None:
                return np.array(self._sim.wfs.d_wfs[numWFS].d_binimg)
            if self.rtc.d_centro[numWFS].type == CentroiderType.MASKEDPIX:
                #     self.rtc.d_centro[numWFS].fill_selected_pix(
                #             self.rtc.d_control[0].d_centroids)
                #     return np.array(self.rtc.d_centro[numWFS].d_selected_pix)
                # else:
                self.rtc.d_centro[numWFS].fill_selected_pix(
                        self.rtc.d_control[0].d_centroids)
                mask = np.array(self.rtc.d_centro[numWFS].d_selected_pix)
                mask[np.where(mask)] = np.array(
                        self.rtc.d_centro[numWFS].d_img)[np.where(mask)]
                return mask
            return np.array(self.rtc.d_centro[numWFS].d_img)
        else:
            if self.rtc.d_centro[numWFS].d_img is None:
                return np.array(self._sim.wfs.d_wfs[numWFS].d_binimg)
            return np.array(self.rtc.d_centro[numWFS].d_img_raw)

    def getAllDataLoop(self, nIter: int, slope: bool, command: bool, target: bool,
                       intensity: bool, targetPhase: bool) -> np.ndarray:
        '''
        Returns a sequence of data at continuous loop steps.
        Requires loop to be asynchronously running
        '''
        raise NotImplementedError("Not implemented")
        # return np.empty(1)

    def getIntensities(self) -> np.ndarray:
        '''
        Return sum of intensities in subaps. Size nSubaps, same order as slopes
        '''
        raise NotImplementedError("Not implemented")
        # return np.empty(1)

    def computeIMatModal(self, M2V: np.ndarray, pushVector: np.ndarray,
                         refOffset: np.ndarray, noise: bool,
                         useAtmos: bool) -> np.ndarray:
        '''
        TODO
        Computes a modal interaction matrix for the given modal matrix
        with given push values (length = nModes)
        around an (optional) offset value
        optionally with noise
        with/without atmos shown to WFS
        '''
        raise NotImplementedError("Not implemented")
        # return np.empty(1)

    def setPerturbationVoltage(self, nControl: int, name: str,
                               command: np.ndarray) -> None:
        ''' Add this offset value to integrator (will be applied at the end of next iteration)'''
        if len(command.shape) == 1:
            self.rtc.d_control[nControl].set_perturb_voltage(name, command, 1)
        elif len(command.shape) == 2:
            self.rtc.d_control[nControl].set_perturb_voltage(name, command,
                                                             command.shape[0])
        else:
            raise AttributeError("command should be a 1D or 2D array")

    def resetPerturbationVoltage(self, nControl: int) -> None:
        '''
        Reset the perturbation voltage of the nControl controller
        (i.e. will remove ALL perturbation voltages.)
        If you want to reset just one, see the function removePerturbationVoltage().
        '''
        self.rtc.d_control[nControl].reset_perturb_voltage()

    def removePerturbationVoltage(self, nControl: int, name: str) -> None:
        '''
        Remove the perturbation voltage called <name>, from the controller number <nControl>.
        If you want to remove all of them, see function resetPerturbationVoltage().
        '''
        self.rtc.d_control[nControl].remove_perturb_voltage(name)

    def getCentroids(self, nControl: int = 0):
        '''
        Return the centroids of the nControl controller
        '''
        return np.array(self.rtc.d_control[nControl].d_centroids)

    def getErr(self, nControl: int = 0) -> np.ndarray:
        '''
        Get command increment from nControl controller
        '''
        return np.array(self.rtc.d_control[nControl].d_err)

    def getVoltage(self, nControl: int = 0) -> np.ndarray:
        '''
        Get voltages from nControl controller
        '''
        return np.array(self.rtc.d_control[nControl].d_voltage)

    def setIntegratorLaw(self, nControl: int = 0):
        """
        Set the control law to Integrator

        Parameters
        ------------
        nControl: (int): controller index
        """
        self.rtc.d_control[nControl].set_commandlaw("integrator")

    def set2MatricesLaw(self, nControl: int = 0):
        self.rtc.d_control[nControl].set_commandlaw("2matrices")

    def setModalIntegratorLaw(self, nControl: int = 0):
        self.rtc.d_control[nControl].set_commandlaw("modal_integrator")

    def setDecayFactor(self, decay, nControl: int = 0):
        """
        Set the decay factor

        Parameters
        ------------
        nControl: (int): controller index
        """
        self.rtc.d_control[nControl].set_decayFactor(decay)

    def setEMatrix(self, eMat, nControl: int = 0):
        self.rtc.d_control[nControl].set_matE(eMat)

    def doRefslopes(self, nControl: int = 0):
        print("Doing refslopes...")
        self.rtc.do_centroids_ref(nControl)
        print("refslopes done")

    def resetRefslopes(self, nControl: int = 0):
        for centro in self.rtc.d_centro:
            centro.d_centroids_ref.reset()

    def closeLoop(self) -> None:
        '''
        DM receives controller output + pertuVoltage
        '''
        self.rtc.d_control[0].set_openloop(0)  # closeLoop

    def openLoop(self, rst=True) -> None:
        '''
        Integrator computation goes to /dev/null but pertuVoltage still applied
        '''
        self.rtc.d_control[0].set_openloop(1, rst)  # openLoop

    def setRefSlopes(self, refSlopes: np.ndarray, numwfs=None) -> None:
        '''
        Set given ref slopes in controller
        '''
        if (numwfs is None):
            self.rtc.set_centroids_ref(refSlopes)
        else:
            self.rtc.d_centro[numwfs].set_centroids_ref(refSlopes)

    def getRefSlopes(self, numwfs=None) -> np.ndarray:
        '''
        Get the currently used reference slopes
        '''
        refSlopes = np.empty(0)
        if (numwfs is None):
            for centro in self.rtc.d_centro:
                refSlopes = np.append(refSlopes, np.array(centro.d_centroids_ref))
            return refSlopes
        else:
            return np.array(self.rtc.d_centro[numwfs].d_centroids_ref)

    def getImat(self, nControl: int = 0):
        """
        Return the interaction matrix of the controller

        Parameters
        ------------
        nControl: (int): controller index
        """
        return np.array(self.rtc.d_control[nControl].d_imat)

    def getCmat(self, nControl: int = 0):
        """
        Return the command matrix of the controller

        Parameters
        ------------
        nControl: (int): controller index
        """
        return np.array(self.rtc.d_control[nControl].d_cmat)

    # Warning: SH specific
    def setCentroThresh(self, nCentro: int = 0, thresh: float = 0.):
        """
        Set the threshold value of a thresholded COG

        Parameters
        ------------
        nCentro: (int): centroider index
        thresh: (float): new threshold value
        """
        self.rtc.d_centro[nCentro].set_threshold(thresh)

    # Warning: PWFS specific
    def getPyrMethod(self, nCentro):
        '''
        Get pyramid compute method
        '''
        return self.rtc.d_centro[nCentro].pyr_method

    # Warning: PWFS specific
    def setPyrMethod(self, pyrMethod, nCentro: int = 0):
        '''
        Set pyramid compute method
        '''
        self.rtc.d_centro[nCentro].set_pyr_method(pyrMethod)  # Sets the pyr method
        self.rtc.do_centroids(0)  # To be ready for the next getSlopes
        print("PYR method set to " + self.rtc.d_centro[nCentro].pyr_method)

    def getSlope(self) -> np.ndarray:
        '''
        Immediately gets one slope vector for all WFS at the current state of the system
        '''
        return self.computeSlopes()

    def next(self, nbiters, see_atmos=True):
        ''' Move atmos -> getSlope -> applyControl ; One integrator step '''
        for i in range(nbiters):
            self.singleNext(showAtmos=see_atmos)

    def setGain(self, gain) -> None:
        '''
        Set the scalar gain or mgain of feedback controller loop
        '''
        if self.rtc.d_control[0].command_law == 'integrator':  # Integrator law
            if np.isscalar(gain):
                self.rtc.d_control[0].set_gain(gain)
            else:
                raise ValueError("Cannot set array gain w/ generic + integrator law")
        else:  # E matrix mode
            if np.isscalar(gain):  # Automatic scalar expansion
                gain = np.ones(np.sum(self.rtc.d_control[0].nactu),
                               dtype=np.float32) * gain
            # Set array
            self.rtc.d_control[0].set_mgain(gain)

    def setCommandMatrix(self, cMat: np.ndarray) -> None:
        '''
        Set the cmat for the controller to use
        '''
        self.rtc.d_control[0].set_cmat(cMat)

    def getFrameCounter(self) -> int:
        '''
        return the current frame counter of the loop
        '''
        if not self.is_init:
            print('Warning - requesting frame counter of uninitialized BenchSupervisor.')
        return self.iter

    def getMaskedPix(self, nCentro: int = 0):
        """
        Return the mask of valid pixels used by a maskedpix centroider

        Parameters:
            nCentro : (int): Centroider index. Must be a maskedpix centroider
        """
        if (self.rtc.d_centro[nCentro].type != CentroiderType.MASKEDPIX):
            raise TypeError("Centroider must be a maskedpix one")
        self.rtc.d_centro[nCentro].fill_mask()
        return np.array(self.rtc.d_centro[nCentro].d_mask)

    def writeConfigOnFile(self, root=None):
        aodict = OrderedDict()
        dataDict = {}
        if (root is None):
            root = self

        if (root.config.p_tel is not None):
            aodict.update({"teldiam": root.config.p_tel.diam})
            aodict.update({"telobs": root.config.p_tel.cobs})
            aodict.update({"pixsize": root.config.p_geom._pixsize})
            # TURBU
            aodict.update({"r0": root.config.p_atmos.r0})
            aodict.update({"Fe": 1 / root.config.p_loop.ittime})
            aodict.update({"nbTargets": len(root.config.p_targets)})
        else:
            aodict.update({"nbTargets": 1})

        # WFS
        aodict.update({"nbWfs": len(root.config.p_wfss)})
        aodict.update({"nbCam": aodict["nbWfs"]})
        aodict.update({"nbOffaxis": 0})
        aodict.update({"nbNgsWFS": 1})
        aodict.update({"nbLgsWFS": 0})
        aodict.update({"nbFigSensor": 0})
        aodict.update({"nbSkyWfs": aodict["nbWfs"]})
        aodict.update({"nbOffNgs": 0})

        # DMS
        aodict.update({"nbDms": len(root.config.p_dms)})
        aodict.update({"Nactu": root.rtc.d_control[0].nactu})
        # List of things
        aodict.update({"list_NgsOffAxis": []})
        aodict.update({"list_Fig": []})
        aodict.update({"list_Cam": [0]})
        aodict.update({"list_SkyWfs": [0]})
        aodict.update({"list_ITS": []})
        aodict.update({"list_Woofer": []})
        aodict.update({"list_Tweeter": []})
        aodict.update({"list_Steering": []})

        listOfNstatesPerController = []
        listOfcontrolLawTypePerController = []
        for control in self.config.p_controllers:
            listOfNstatesPerController.append(control.nstates)
            listOfcontrolLawTypePerController.append(control.type)
        aodict.update({"list_nstatesPerController": listOfNstatesPerController})
        aodict.update({"list_controllerType": listOfcontrolLawTypePerController})

        # fct of Nb of wfss
        NslopesList = []
        NsubapList = []
        listWfsType = []
        listCentroType = []

        pyrModulationList = []
        pyr_npts = []
        pyr_pupsep = []
        pixsize = []
        xPosList = []
        yPosList = []
        fstopsize = []
        fstoptype = []
        npixPerSub = []
        nxsubList = []
        nysubList = []
        lambdaList = []
        dms_seen = []
        colTmpList = []
        noise = []
        #new_hduwfsl = pfits.HDUList()
        #new_hduwfsSubapXY = pfits.HDUList()
        for i in range(aodict["nbWfs"]):
            #new_hduwfsl.append(pfits.ImageHDU(root.config.p_wfss[i]._isvalid))  # Valid subap array
            #new_hduwfsl[i].header["DATATYPE"] = "valid_wfs%d" % i
            dataDict["wfsValid_" + str(i)] = root.config.p_wfss[i]._isvalid

            xytab = np.zeros((2, root.config.p_wfss[i]._validsubsx.shape[0]))
            xytab[0, :] = root.config.p_wfss[i]._validsubsx
            xytab[1, :] = root.config.p_wfss[i]._validsubsy
            dataDict["wfsValidXY_" + str(i)] = xytab

            #new_hduwfsSubapXY.append(pfits.ImageHDU(xytab))  # Valid subap array inXx Y on the detector
            #new_hduwfsSubapXY[i].header["DATATYPE"] = "validXY_wfs%d" % i
            pixsize.append(root.config.p_wfss[i].pixsize)
            """
            if (root.config.p_centroiders[i].type == "maskedpix"):
                factor = 4
            else:
                factor = 2
            NslopesList.append(
                    root.config.p_wfss[i]._nvalid * factor)  # slopes per wfs
            """
            listCentroType.append(
                    root.config.p_centroiders[i].
                    type)  # assumes that there is the same number of centroiders and wfs
            NsubapList.append(root.config.p_wfss[i]._nvalid)  # subap per wfs
            listWfsType.append(root.config.p_wfss[i].type)
            xPosList.append(root.config.p_wfss[i].xpos)
            yPosList.append(root.config.p_wfss[i].ypos)
            fstopsize.append(root.config.p_wfss[i].fssize)
            fstoptype.append(root.config.p_wfss[i].fstop)
            nxsubList.append(root.config.p_wfss[i].nxsub)
            nysubList.append(root.config.p_wfss[i].nxsub)
            lambdaList.append(root.config.p_wfss[i].Lambda)
            if (root.config.p_wfss[i].dms_seen is not None):
                dms_seen.append(list(root.config.p_wfss[i].dms_seen))
                noise.append(root.config.p_wfss[i].noise)

            if (root.config.p_centroiders[i].type == CentroiderType.MASKEDPIX):
                NslopesList.append(root.config.p_wfss[i]._nvalid * 4)  # slopes per wfs
            else:
                NslopesList.append(root.config.p_wfss[i]._nvalid * 2)  # slopes per wfs

            if (root.config.p_wfss[i].type == "pyrhr"):
                pyrModulationList.append(root.config.p_wfss[i].pyr_ampl)
                pyr_npts.append(root.config.p_wfss[i].pyr_npts)
                pyr_pupsep.append(root.config.p_wfss[i].pyr_pup_sep)
                npixPerSub.append(1)
            else:
                pyrModulationList.append(0)
                pyr_npts.append(0)
                pyr_pupsep.append(0)
                npixPerSub.append(root.config.p_wfss[i].npix)
        """
        confname = filepath.split("/")[-1].split('.conf')[0]
        print(filepath.split(".conf")[0] + '_wfsConfig.fits')
        new_hduwfsl.writeto(
                filepath.split(".conf")[0] + '_wfsConfig.fits', overwrite=True)
        new_hduwfsSubapXY.writeto(
                filepath.split(".conf")[0] + '_wfsValidXYConfig.fits', overwrite=True)
        """
        if (len(dms_seen) != 0):
            aodict.update({"listWFS_dms_seen": dms_seen})

        aodict.update({"listWFS_NslopesList": NslopesList})
        aodict.update({"listWFS_NsubapList": NsubapList})
        aodict.update({"listWFS_CentroType": listCentroType})
        aodict.update({"listWFS_WfsType": listWfsType})
        aodict.update({"listWFS_pixarc": pixsize})
        aodict.update({"listWFS_pyrModRadius": pyrModulationList})
        aodict.update({"listWFS_pyrModNPts": pyr_npts})
        aodict.update({"listWFS_pyrPupSep": pyr_pupsep})
        aodict.update({"listWFS_fstopsize": fstopsize})
        aodict.update({"listWFS_fstoptype": fstoptype})
        aodict.update({"listWFS_NsubX": nxsubList})
        aodict.update({"listWFS_NsubY": nysubList})
        aodict.update({"listWFS_Nsub": nysubList})
        aodict.update({"listWFS_NpixPerSub": npixPerSub})
        aodict.update({"listWFS_Lambda": lambdaList})
        if (len(noise) != 0):
            aodict.update({"listWFS_noise": noise})

        listDmsType = []
        NactuX = []
        Nactu = []
        unitPerVolt = []
        push4imat = []
        coupling = []
        push4iMatArcSec = []
        #new_hdudmsl = pfits.HDUList()

        for j in range(aodict["nbDms"]):
            listDmsType.append(root.config.p_dms[j].type)
            NactuX.append(
                    root.config.p_dms[j].nact)  # nb of actuators across the diameter !!
            Nactu.append(root.config.p_dms[j]._ntotact)  # nb of actuators in total
            unitPerVolt.append(root.config.p_dms[j].unitpervolt)
            push4imat.append(root.config.p_dms[j].push4imat)
            coupling.append(root.config.p_dms[j].coupling)
            tmp = []
            if (root.config.p_dms[j]._i1 is
                        not None):  # Simu Case where i1 j1 is known (simulated)
                if (root.config.p_dms[j].type != 'tt'):
                    tmpdata = np.zeros((4, len(root.config.p_dms[j]._i1)))
                    tmpdata[0, :] = root.config.p_dms[j]._j1
                    tmpdata[1, :] = root.config.p_dms[j]._i1
                    tmpdata[2, :] = root.config.p_dms[j]._xpos
                    tmpdata[3, :] = root.config.p_dms[j]._ypos
                else:
                    tmpdata = np.zeros((4, 2))

                dataDict["dmData" + str(j)] = tmpdata
                """
                new_hdudmsl.append(pfits.ImageHDU(tmpdata))  # Valid subap array
                new_hdudmsl[j].header["DATATYPE"] = "valid_dm%d" % j
                """
                #for k in range(aodict["nbWfs"]):
                #    tmp.append(root.computeDMrange(j, k))

                push4iMatArcSec.append(tmp)

        # new_hdudmsl.writeto(filepath.split(".conf")[0] + '_dmsConfig.fits', overwrite=True)
        if (len(push4iMatArcSec) != 0):
            aodict.update({"listDMS_push4iMat": push4imat})
            aodict.update({"listDMS_unitPerVolt": unitPerVolt})
        aodict.update({"listDMS_Nxactu": NactuX})
        aodict.update({"listDMS_Nyactu": NactuX})
        aodict.update({"listDMS_Nactu": Nactu})

        aodict.update({"listDMS_type": listDmsType})
        aodict.update({"listDMS_coupling": coupling})

        if (root.config.p_targets is not None):  # simu case
            listTargetsLambda = []
            listTargetsXpos = []
            listTargetsYpos = []
            listTargetsDmsSeen = []
            listTargetsMag = []
            listTARGETS_pixsize = []
            for k in range(aodict["nbTargets"]):
                listTargetsLambda.append(root.config.p_targets[k].Lambda)
                listTargetsXpos.append(root.config.p_targets[k].xpos)
                listTargetsYpos.append(root.config.p_targets[k].ypos)
                listTargetsMag.append(root.config.p_targets[k].mag)
                listTargetsDmsSeen.append(list(root.config.p_targets[k].dms_seen))
                PSFPixsize = (root.config.p_targets[k].Lambda * 1e-6) / (
                        root.config.p_geom._pixsize *
                        root.config.p_geom.get_ipupil().shape[0]) * 206265.
                listTARGETS_pixsize.append(PSFPixsize)

            aodict.update({"listTARGETS_Lambda": listTargetsLambda})
            aodict.update({"listTARGETS_Xpos": listTargetsXpos})
            aodict.update({"listTARGETS_Ypos": listTargetsYpos})
            aodict.update({"listTARGETS_Mag": listTargetsMag})
            aodict.update({"listTARGETS_DmsSeen": listTargetsDmsSeen})
            aodict.update({"listTARGETS_pixsize": listTARGETS_pixsize})

        listDmsType = []
        Nslopes = sum(NslopesList)
        Nsubap = sum(NsubapList)
        aodict.update({"Nslopes": Nslopes})
        aodict.update({"Nsubap": Nsubap})
        return aodict, dataDict
