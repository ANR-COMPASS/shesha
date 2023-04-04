## @package   shesha.config
## @brief     Parameters configuration class
## @author    COMPASS Team <https://github.com/ANR-COMPASS>
## @version   5.4.2
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
import importlib
import sys, os
from collections import OrderedDict
import numpy as np
import shesha.constants as scons
from typing import NoReturn, Dict

class ParamConfig(object):
    """ Shesha parameters configuration class. It embeds all the
    parameters classes instances needed for the simulation run.

    This class also exposes most useful getters from its components
    to allow an easier access and exposition through Pyro

    Attributes:
        simul_name : (str) : Simulation run name

        p_atmos : (Param_atmos) : A Param_atmos instance

        p_geom : (Param_geom) : A Param_geom instance

        p_tel : (Param_tel) : A Param_tel instance

        p_dms : (List of Param_dm) : List of Param_dm instance

        p_wfss : (List of Param_wfs) : List of Param_wfs instance

        p_targets : (List of Param_target) : List of Param_target instance

        p_loop : (Param_loop) : A Param_loop instance

        p_centroiders : (List of Param_centroider) : List of Param_centroider instance

        p_controllers : (List of Param_controller) : List of Param controller instance

        _config : (configuration module from parfile) : Raw parameter file module
    """
    def __init__(self, param_file : str):
        self._load_config_from_file(param_file)
        self.simul_name = self._config.simul_name
        self.p_atmos = self._config.p_atmos
        self.p_tel = self._config.p_tel
        self.p_geom = self._config.p_geom
        self.p_wfss = self._config.p_wfss
        self.p_dms = self._config.p_dms
        self.p_targets = self._config.p_targets
        self.p_loop = self._config.p_loop
        self.p_centroiders = self._config.p_centroiders
        self.p_controllers = self._config.p_controllers
        self.p_coronos = self._config.p_coronos

    def _load_config_from_file(self, filename_path: str) -> NoReturn:
        """ Load the parameters from the parameters file

        Args:
            filename_path: (str): path to the parameters file
        """
        path = os.path.dirname(os.path.abspath(filename_path))
        filename = os.path.basename(filename_path)
        name, ext = os.path.splitext(filename)

        if (ext == ".py"):
            if (path not in sys.path):
                sys.path.insert(0, path)

            return self._load_config_from_module(name)

            # exec("import %s as wao_config" % filename)
            sys.path.remove(path)
        elif importlib.util.find_spec(filename_path) is not None:
            return self._load_config_from_module(filename_path)
        else:
            raise ValueError("Config file must be .py or a module")


    def _load_config_from_module(self, filepath: str) -> NoReturn:
        """
        Load the parameters from the parameters module

        Args:
            filename_path: (str): path to the parameters file

        Returns:
            config : (config) : a config module
        """
        filename = filepath.split('.')[-1]
        print("loading: %s" % filename)

        config = importlib.import_module(filepath)
        del sys.modules[config.__name__]  # Forced reload
        self._config = importlib.import_module(filepath)

        if hasattr(config, 'par'):
            self._config = getattr("config.par.par4bench", filename)

        # Set missing config attributes to None
        if not hasattr(self._config, 'p_loop'):
            self._config.p_loop = None
        if not hasattr(self._config, 'p_geom'):
            self._config.p_geom = None
        if not hasattr(self._config, 'p_tel'):
            self._config.p_tel = None
        if not hasattr(self._config, 'p_atmos'):
            self._config.p_atmos = None
        if not hasattr(self._config, 'p_dms'):
            self._config.p_dms = None
        if not hasattr(self._config, 'p_targets'):
            self._config.p_targets = None
        if not hasattr(self._config, 'p_wfss'):
            self._config.p_wfss = None
        if not hasattr(self._config, 'p_centroiders'):
            self._config.p_centroiders = None
        if not hasattr(self._config, 'p_controllers'):
            self._config.p_controllers = None
        if not hasattr(self._config, 'p_coronos'):
            self._config.p_coronos = None

        if not hasattr(self._config, 'simul_name'):
            self._config.simul_name = None

    def get_pupil(self, pupil_type) -> np.ndarray:
        """ Returns the specified pupil of COMPASS.

        Possible args value are :
            - "i" or "ipupil" : returns the biggest pupil of size (Nfft x Nfft)
            - "m" or "mpupil" : returns the medium pupil, used for WFS computation
            - "s" or "spupil" : returns the smallest pupil of size (p_geom.pupdiam x p_geom.pupdiam)

        Returns:
            pupil : (np.ndarray) : pupil
        """
        if scons.PupilType(pupil_type) is scons.PupilType.SPUPIL:
            return self.p_geom.get_spupil()
        if scons.PupilType(pupil_type) is scons.PupilType.MPUPIL:
            return self.p_geom.get_mpupil()
        if scons.PupilType(pupil_type) is scons.PupilType.IPUPIL:
            return self.p_geom.get_ipupil()

    def export_config(self) -> [Dict, Dict]:
        """
        Extract and convert compass supervisor configuration parameters
        into 2 dictionnaries containing relevant AO parameters

        Returns : 2 dictionnaries
        """
        aodict = OrderedDict()
        dataDict = {}

        if (self.p_tel is not None):
            aodict.update({"teldiam": self.p_tel.diam})
            aodict.update({"telobs": self.p_tel.cobs})
            aodict.update({"pixsize": self.p_geom._pixsize})
            # TURBU
            aodict.update({"r0": self.p_atmos.r0})
            aodict.update({"Fe": 1 / self.p_loop.ittime})
            aodict.update({"nbTargets": len(self.p_targets)})
        else:
            aodict.update({"nbTargets": 1})

        # WFS
        aodict.update({"nbWfs": len(self.p_wfss)})
        aodict.update({"nbCam": aodict["nbWfs"]})
        aodict.update({"nbOffaxis": 0})
        aodict.update({"nbNgsWFS": 1})
        aodict.update({"nbLgsWFS": 0})
        aodict.update({"nbFigSensor": 0})
        aodict.update({"nbSkyWfs": aodict["nbWfs"]})
        aodict.update({"nbOffNgs": 0})

        # DMS
        aodict.update({"nbDms": len(self.p_dms)})
        aodict.update({"Nactu": self.p_controllers[0].nactu})
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
        listOfcommandLawTypePerController = []
        for control in self.p_controllers:
            listOfNstatesPerController.append(control.nstates)
            listOfcommandLawTypePerController.append(control.type)
        aodict.update({"list_nstatesPerController": listOfNstatesPerController})
        aodict.update({"list_controllerType": listOfcommandLawTypePerController})

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
            #new_hduwfsl.append(pfits.ImageHDU(self.p_wfss[i]._isvalid))  # Valid subap array
            #new_hduwfsl[i].header["DATATYPE"] = "valid_wfs%d" % i
            dataDict["wfsValid_" + str(i)] = self.p_wfss[i]._isvalid

            xytab = np.zeros((2, self.p_wfss[i]._validsubsx.shape[0]))
            xytab[0, :] = self.p_wfss[i]._validsubsx
            xytab[1, :] = self.p_wfss[i]._validsubsy
            dataDict["wfsValidXY_" + str(i)] = xytab

            #new_hduwfsSubapXY.append(pfits.ImageHDU(xytab))  # Valid subap array inXx Y on the detector
            #new_hduwfsSubapXY[i].header["DATATYPE"] = "validXY_wfs%d" % i
            pixsize.append(self.p_wfss[i].pixsize)
            """
            if (self.p_centroiders[i].type == "maskedpix"):
                factor = 4
            else:
                factor = 2
            NslopesList.append(
                    self.p_wfss[i]._nvalid * factor)  # slopes per wfs
            """
            listCentroType.append(
                    self.p_centroiders[i].
                    type)  # assumes that there is the same number of centroiders and wfs
            NsubapList.append(self.p_wfss[i]._nvalid)  # subap per wfs
            listWfsType.append(self.p_wfss[i].type)
            xPosList.append(self.p_wfss[i].xpos)
            yPosList.append(self.p_wfss[i].ypos)
            fstopsize.append(self.p_wfss[i].fssize)
            fstoptype.append(self.p_wfss[i].fstop)
            nxsubList.append(self.p_wfss[i].nxsub)
            nysubList.append(self.p_wfss[i].nxsub)
            lambdaList.append(self.p_wfss[i].Lambda)
            if (self.p_wfss[i].dms_seen is not None):
                dms_seen.append(list(self.p_wfss[i].dms_seen))
                noise.append(self.p_wfss[i].noise)

            if (self.p_centroiders[i].type == scons.CentroiderType.MASKEDPIX):
                NslopesList.append(self.p_wfss[i]._nvalid * 4)  # slopes per wfs
            else:
                NslopesList.append(self.p_wfss[i]._nvalid * 2)  # slopes per wfs

            if (self.p_wfss[i].type == "pyrhr"):
                pyrModulationList.append(self.p_wfss[i].pyr_ampl)
                pyr_npts.append(self.p_wfss[i].pyr_npts)
                pyr_pupsep.append(self.p_wfss[i].pyr_pup_sep)
                npixPerSub.append(1)
            else:
                pyrModulationList.append(0)
                pyr_npts.append(0)
                pyr_pupsep.append(0)
                npixPerSub.append(self.p_wfss[i].npix)
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
            listDmsType.append(self.p_dms[j].type)
            NactuX.append(
                    self.p_dms[j].nact)  # nb of actuators across the diameter !!
            Nactu.append(self.p_dms[j]._ntotact)  # nb of actuators in total
            unitPerVolt.append(self.p_dms[j].unitpervolt)
            push4imat.append(self.p_dms[j].push4imat)
            coupling.append(self.p_dms[j].coupling)
            tmp = []
            if (self.p_dms[j]._i1 is
                        not None):  # Simu Case where i1 j1 is known (simulated)
                if (self.p_dms[j].type != 'tt'):
                    tmpdata = np.zeros((4, len(self.p_dms[j]._i1)))
                    tmpdata[0, :] = self.p_dms[j]._j1
                    tmpdata[1, :] = self.p_dms[j]._i1
                    tmpdata[2, :] = self.p_dms[j]._xpos
                    tmpdata[3, :] = self.p_dms[j]._ypos
                else:
                    tmpdata = np.zeros((4, 2))

                dataDict["dmData" + str(j)] = tmpdata
                """
                new_hdudmsl.append(pfits.ImageHDU(tmpdata))  # Valid subap array
                new_hdudmsl[j].header["DATATYPE"] = "valid_dm%d" % j
                """
                #for k in range(aodict["nbWfs"]):
                #    tmp.append(supervisor.computeDMrange(j, k))

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

        if (self.p_targets is not None):  # simu case
            listTargetsLambda = []
            listTargetsXpos = []
            listTargetsYpos = []
            listTargetsDmsSeen = []
            listTargetsMag = []
            listTARGETS_pixsize = []
            for k in range(aodict["nbTargets"]):
                listTargetsLambda.append(self.p_targets[k].Lambda)
                listTargetsXpos.append(self.p_targets[k].xpos)
                listTargetsYpos.append(self.p_targets[k].ypos)
                listTargetsMag.append(self.p_targets[k].mag)
                listTargetsDmsSeen.append(list(self.p_targets[k].dms_seen))
                PSFPixsize = (self.p_targets[k].Lambda * 1e-6) / (
                        self.p_geom._pixsize *
                        self.p_geom.get_ipupil().shape[0]) * 206265.
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
