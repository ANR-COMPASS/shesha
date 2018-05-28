"""Widget expert
"""

import os, sys
import numpy as np

from PyQt5.uic import loadUiType

import shesha.ao as ao
import shesha.constants as scons
from shesha.constants import CONST

from .widget_base import uiLoader
ExpertWidgetTemplate, ExpertClassTemplate = uiLoader('widget_ao_expert')


class WidgetAOExpert(ExpertClassTemplate):

    def __init__(self) -> None:
        ExpertClassTemplate.__init__(self)
        self.sim = None

        self.uiExpert = ExpertWidgetTemplate()
        self.uiExpert.setupUi(self)

        #############################################################
        #                   ATTRIBUTES                              #
        #############################################################

        #############################################################
        #               PYQTGRAPH DockArea INIT                     #
        #############################################################

        #############################################################
        #                 CONNECTED BUTTONS                         #
        #############################################################
        # Default path for config files
        self.uiExpert.wao_setAtmos.clicked.connect(self.setAtmosParams)
        self.uiExpert.wao_setWfs.clicked.connect(self.setWfsParams)
        self.uiExpert.wao_setDM.clicked.connect(self.setDmParams)
        self.uiExpert.wao_setControl.clicked.connect(self.setRtcParams)
        self.uiExpert.wao_setCentro.clicked.connect(self.setRtcParams)
        self.uiExpert.wao_setTelescope.clicked.connect(self.setTelescopeParams)
        self.uiExpert.wao_resetDM.clicked.connect(self.resetDM)
        self.uiExpert.wao_update_gain.clicked.connect(self.updateGain)
        self.uiExpert.wao_update_pyr_ampl.clicked.connect(self.updatePyrAmpl)
        self.uiExpert.wao_selectRtcMatrix.currentIndexChanged.connect(
                self.displayRtcMatrix)
        self.uiExpert.wao_commandBtt.clicked.connect(self.BttCommand)
        self.uiExpert.wao_commandKL.clicked.connect(self.KLCommand)

        self.uiExpert.wao_dmUnitPerVolt.valueChanged.connect(self.updateDMrangeGUI)
        self.uiExpert.wao_dmpush4iMat.valueChanged.connect(self.updateDMrangeGUI)
        self.uiExpert.wao_pyr_ampl.valueChanged.connect(self.updateAmpliCompGUI)
        self.uiExpert.wao_dmActuPushArcSecNumWFS.currentIndexChanged.connect(
                self.updateDMrange)

    #############################################################
    #                       METHODS                             #
    #############################################################

    def setSim(self, sim) -> None:
        self.sim = sim

    def updateGain(self) -> None:
        if (self.sim.rtc):
            self.sim.rtc.set_gain(0, float(self.uiExpert.wao_controlGain.value()))
            print("Loop gain updated on GPU")

    def updateAmpliCompGUI(self) -> None:
        diffract = self.sim.config.p_wfss[0].Lambda * \
            1e-6 / self.sim.config.p_tel.diam * CONST.RAD2ARCSEC
        self.uiExpert.wao_pyr_ampl_arcsec.setValue(
                self.uiExpert.wao_pyr_ampl.value() * diffract)

    def updateAmpliComp(self) -> None:
        diffract = self.sim.config.p_wfss[0].Lambda * \
            1e-6 / self.sim.config.p_tel.diam * CONST.RAD2ARCSEC
        self.uiExpert.wao_pyr_ampl_arcsec.setValue(
                self.sim.config.p_wfss[0].pyr_ampl * diffract)

    def updatePyrAmpl(self) -> None:
        if (self.sim.rtc):
            self.sim.rtc.set_pyr_ampl(0,
                                      self.uiExpert.wao_pyr_ampl.value(),
                                      self.sim.config.p_wfss, self.sim.config.p_tel)
            print("Pyramid modulation updated on GPU")
            self.updatePlotWfs()

    def updateDMrangeGUI(self) -> None:
        push4imat = self.uiExpert.wao_dmpush4iMat.value()
        unitpervolt = self.uiExpert.wao_dmUnitPerVolt.value()
        self.updateDMrange(push4imat=push4imat, unitpervolt=unitpervolt)

    def updateDMrange(self, push4imat: float=None, unitpervolt: float=None) -> None:
        numdm = str(self.uiExpert.wao_selectDM.currentText())
        numwfs = str(self.uiExpert.wao_dmActuPushArcSecNumWFS.currentText())
        if ((numdm is not "") and (numwfs is not "") and (push4imat != 0) and
            (unitpervolt != 0)):
            arcsecDMrange = self.computeDMrange(
                    int(numdm), int(numwfs), push4imat=push4imat,
                    unitpervolt=unitpervolt)
            self.uiExpert.wao_dmActPushArcsec.setValue(arcsecDMrange)

    def updateTelescopePanel(self) -> None:
        self.uiExpert.wao_zenithAngle.setValue(self.sim.config.p_geom.zenithangle)
        self.uiExpert.wao_diamTel.setValue(self.sim.config.p_tel.diam)
        self.uiExpert.wao_cobs.setValue(self.sim.config.p_tel.cobs)

    def updateDmPanel(self) -> None:
        ndm = self.uiExpert.wao_selectDM.currentIndex()
        if (ndm < 0):
            ndm = 0
        self.uiExpert.wao_dmActuPushArcSecNumWFS.clear()
        self.uiExpert.wao_dmActuPushArcSecNumWFS.addItems([
                str(i) for i in range(len(self.sim.config.p_dms))
        ])
        self.uiExpert.wao_numberofDMs.setText(str(len(self.sim.config.p_dms)))
        self.uiExpert.wao_dmTypeSelector.setCurrentIndex(
                self.uiExpert.wao_dmTypeSelector.findText(
                        str(self.sim.config.p_dms[ndm].type)))
        self.uiExpert.wao_dmAlt.setValue(self.sim.config.p_dms[ndm].alt)
        if (self.sim.config.p_dms[ndm].type == scons.DmType.KL):
            self.uiExpert.wao_dmNactu.setValue(self.sim.config.p_dms[ndm].nkl)
        else:
            self.uiExpert.wao_dmNactu.setValue(self.sim.config.p_dms[ndm].nact)
        self.uiExpert.wao_dmUnitPerVolt.setValue(self.sim.config.p_dms[ndm].unitpervolt)
        self.uiExpert.wao_dmpush4iMat.setValue(self.sim.config.p_dms[ndm].push4imat)
        self.uiExpert.wao_dmCoupling.setValue(self.sim.config.p_dms[ndm].coupling)
        self.uiExpert.wao_dmThresh.setValue(self.sim.config.p_dms[ndm].thresh)
        self.updateDMrange()

    def updateWfsPanel(self) -> None:
        nwfs = self.uiExpert.wao_selectWfs.currentIndex()
        if (nwfs < 0):
            nwfs = 0
        self.uiExpert.wao_numberofWfs.setText(str(len(self.sim.config.p_dms)))
        self.uiExpert.wao_wfsType.setText(str(self.sim.config.p_wfss[nwfs].type))
        self.uiExpert.wao_wfsNxsub.setValue(self.sim.config.p_wfss[nwfs].nxsub)
        self.uiExpert.wao_wfsNpix.setValue(self.sim.config.p_wfss[nwfs].npix)
        self.uiExpert.wao_wfsPixSize.setValue(self.sim.config.p_wfss[nwfs].pixsize)
        self.uiExpert.wao_wfsXpos.setValue(self.sim.config.p_wfss[nwfs].xpos)
        self.uiExpert.wao_wfsYpos.setValue(self.sim.config.p_wfss[nwfs].ypos)
        self.uiExpert.wao_wfsFracsub.setValue(self.sim.config.p_wfss[nwfs].fracsub)
        self.uiExpert.wao_wfsLambda.setValue(self.sim.config.p_wfss[nwfs].Lambda)
        self.uiExpert.wao_wfsMagnitude.setValue(self.sim.config.p_wfss[nwfs].gsmag)
        self.uiExpert.wao_wfsZp.setValue(np.log10(self.sim.config.p_wfss[nwfs].zerop))
        self.uiExpert.wao_wfsThrough.setValue(self.sim.config.p_wfss[nwfs].optthroughput)
        self.uiExpert.wao_wfsNoise.setValue(self.sim.config.p_wfss[nwfs].noise)
        self.uiExpert.wao_pyr_ampl.setValue(self.sim.config.p_wfss[nwfs].pyr_ampl)

        # LGS panel
        if (self.sim.config.p_wfss[nwfs].gsalt > 0):
            self.uiExpert.wao_wfsIsLGS.setChecked(True)
            self.uiExpert.wao_wfsGsAlt.setValue(self.sim.config.p_wfss[nwfs].gsalt)
            self.uiExpert.wao_wfsLLTx.setValue(self.sim.config.p_wfss[nwfs].lltx)
            self.uiExpert.wao_wfsLLTy.setValue(self.sim.config.p_wfss[nwfs].llty)
            self.uiExpert.wao_wfsLGSpower.setValue(
                    self.sim.config.p_wfss[nwfs].laserpower)
            self.uiExpert.wao_wfsReturnPerWatt.setValue(
                    self.sim.config.p_wfss[nwfs].lgsreturnperwatt)
            self.uiExpert.wao_wfsBeamSize.setValue(self.sim.config.p_wfss[nwfs].beamsize)
            self.uiExpert.wao_selectLGSProfile.setCurrentIndex(
                    self.uiExpert.wao_selectLGSProfile.findText(
                            str(self.sim.config.p_wfss[nwfs].proftype)))

        else:
            self.uiExpert.wao_wfsIsLGS.setChecked(False)

        if (self.sim.config.p_wfss[nwfs].type == b"pyrhr" or
                    self.sim.config.p_wfss[nwfs].type == b"pyr"):
            self.uiExpert.wao_wfs_plotSelector.setCurrentIndex(3)
        self.updatePlotWfs()

    def updateAtmosPanel(self) -> None:
        nscreen = self.uiExpert.wao_selectAtmosLayer.currentIndex()
        if (nscreen < 0):
            nscreen = 0
        self.uiExpert.wao_r0.setValue(self.sim.config.p_atmos.r0)
        self.uiExpert.wao_atmosNlayers.setValue(self.sim.config.p_atmos.nscreens)
        self.uiExpert.wao_atmosAlt.setValue(self.sim.config.p_atmos.alt[nscreen])
        self.uiExpert.wao_atmosFrac.setValue(self.sim.config.p_atmos.frac[nscreen])
        self.uiExpert.wao_atmosL0.setValue(self.sim.config.p_atmos.L0[nscreen])
        self.uiExpert.wao_windSpeed.setValue(self.sim.config.p_atmos.windspeed[nscreen])
        self.uiExpert.wao_windDirection.setValue(
                self.sim.config.p_atmos.winddir[nscreen])
        if (self.sim.config.p_atmos.dim_screens is not None):
            self.uiExpert.wao_atmosDimScreen.setText(
                    str(self.sim.config.p_atmos.dim_screens[nscreen]))
        self.uiExpert.wao_atmosWindow.canvas.axes.cla()
        width = (self.sim.config.p_atmos.alt.max() / 20. + 0.1) / 1000.
        self.uiExpert.wao_atmosWindow.canvas.axes.barh(
                self.sim.config.p_atmos.alt / 1000. - width / 2.,
                self.sim.config.p_atmos.frac, width, color="blue")
        self.uiExpert.wao_atmosWindow.canvas.draw()

    def updateRtcPanel(self) -> None:
        # Centroider panel
        ncentro = self.uiExpert.wao_selectCentro.currentIndex()
        if (ncentro < 0):
            ncentro = 0
        type = self.sim.config.p_centroiders[ncentro].type
        self.uiExpert.wao_centroTypeSelector.setCurrentIndex(
                self.uiExpert.wao_centroTypeSelector.findText(str(type)))

        self.uiExpert.wao_centroThresh.setValue(
                self.sim.config.p_centroiders[ncentro].thresh)
        if type == scons.CentroiderType.BPCOG:
            self.uiExpert.wao_centroNbrightest.setValue(
                    self.sim.config.p_centroiders[ncentro].nmax)
        if type == scons.CentroiderType.TCOG:
            self.uiExpert.wao_centroThresh.setValue(
                    self.sim.config.p_centroiders[ncentro].thresh)
        if type in [scons.CentroiderType.CORR, scons.CentroiderType.WCOG]:
            self.uiExpert.wao_centroFunctionSelector.setCurrentIndex(
                    self.uiExpert.wao_centroFunctionSelector.findText(
                            str(self.sim.config.p_centroiders[ncentro].type_fct)))
            self.uiExpert.wao_centroWidth.setValue(
                    self.sim.config.p_centroiders[ncentro].width)

        # Controller panel
        type_contro = self.sim.config.p_controllers[0].type
        if (type_contro == scons.ControllerType.LS and
                    self.sim.config.p_controllers[0].modopti == 0):
            self.uiExpert.wao_controlTypeSelector.setCurrentIndex(0)
        elif (type_contro == scons.ControllerType.MV):
            self.uiExpert.wao_controlTypeSelector.setCurrentIndex(1)
        elif (type_contro == scons.ControllerType.GEO):
            self.uiExpert.wao_controlTypeSelector.setCurrentIndex(2)
        elif (type_contro == scons.ControllerType.LS and
              self.sim.config.p_controllers[0].modopti):
            self.uiExpert.wao_controlTypeSelector.setCurrentIndex(3)
        elif (type_contro == scons.ControllerType.CURED):
            self.uiExpert.wao_controlTypeSelector.setCurrentIndex(4)
        elif (type_contro == scons.ControllerType.GENERIC):
            self.uiExpert.wao_controlTypeSelector.setCurrentIndex(5)
        else:
            print("Controller type enumeration invalid.")

        self.uiExpert.wao_controlCond.setValue(self.sim.config.p_controllers[0].maxcond)
        self.uiExpert.wao_controlDelay.setValue(self.sim.config.p_controllers[0].delay)
        self.uiExpert.wao_controlGain.setValue(self.sim.config.p_controllers[0].gain)
        self.uiExpert.wao_controlTTcond.setValue(
                self.sim.config.p_controllers[0].maxcond)  # TODO : TTcond

    def updateTargetPanel(self) -> None:
        ntarget = self.uiExpert.wao_selectTarget.currentIndex()
        if (ntarget < 0):
            ntarget = 0
        self.uiExpert.wao_numberofTargets.setText(str(self.sim.config.p_target.ntargets))
        self.uiExpert.wao_targetMag.setValue(self.sim.config.p_target.mag[ntarget])
        self.uiExpert.wao_targetXpos.setValue(self.sim.config.p_target.xpos[ntarget])
        self.uiExpert.wao_targetYpos.setValue(self.sim.config.p_target.ypos[ntarget])
        self.uiExpert.wao_targetLambda.setValue(self.sim.config.p_target.Lambda[ntarget])

    def updatePanels(self) -> None:
        self.updateTelescopePanel()
        self.updateAtmosPanel()
        self.updateWfsPanel()
        self.updateDmPanel()
        self.updateRtcPanel()
        self.updateTargetPanel()

    def setTelescopeParams(self) -> None:
        self.sim.config.p_tel.set_diam(self.uiExpert.wao_diamTel.value())
        self.sim.config.p_tel.set_cobs(self.uiExpert.wao_cobs.value())
        self.sim.config.p_geom.set_zenithangle(self.uiExpert.wao_zenithAngle.value())
        print("New telescope parameters set")

    def setAtmosParams(self) -> None:
        nscreen = self.uiExpert.wao_selectAtmosLayer.currentIndex()
        if (nscreen < 0):
            nscreen = 0
        self.sim.config.p_atmos.alt[nscreen] = self.uiExpert.wao_atmosAlt.value()
        self.sim.config.p_atmos.frac[nscreen] = self.uiExpert.wao_atmosFrac.value()
        self.sim.config.p_atmos.L0[nscreen] = self.uiExpert.wao_atmosL0.value()
        self.sim.config.p_atmos.windspeed[nscreen] = self.uiExpert.wao_windSpeed.value()
        self.sim.config.p_atmos.winddir[
                nscreen] = self.uiExpert.wao_windDirection.value()
        print("New atmos parameters set")

    def setRtcParams(self) -> None:
        # Centroider params
        ncentro = self.uiExpert.wao_selectCentro.currentIndex()
        if (ncentro < 0):
            ncentro = 0
        self.sim.config.p_centroiders[ncentro].set_type(
                str(self.uiExpert.wao_centroTypeSelector.currentText()))
        self.sim.config.p_centroiders[ncentro].set_thresh(
                self.uiExpert.wao_centroThresh.value())
        self.sim.config.p_centroiders[ncentro].set_nmax(
                self.uiExpert.wao_centroNbrightest.value())
        self.sim.config.p_centroiders[ncentro].set_thresh(
                self.uiExpert.wao_centroThresh.value())
        self.sim.config.p_centroiders[ncentro].set_type_fct(
                str(self.uiExpert.wao_centroFunctionSelector.currentText()))
        self.sim.config.p_centroiders[ncentro].set_width(
                self.uiExpert.wao_centroWidth.value())

        # Controller panel
        type_contro = str(self.uiExpert.wao_controlTypeSelector.currentText())
        if (type_contro == "LS"):
            self.sim.config.p_controllers[0].set_type(scons.ControllerType.LS)
        elif (type_contro == "MV"):
            self.sim.config.p_controllers[0].set_type(scons.ControllerType.MV)
        elif (type_contro == "PROJ"):
            self.sim.config.p_controllers[0].set_type(scons.ControllerType.GEO)
        elif (type_contro == "OptiMods"):
            self.sim.config.p_controllers[0].set_type(scons.ControllerType.LS)
            self.sim.config.p_controllers[0].set_modopti(1)

        self.sim.config.p_controllers[0].set_maxcond(
                self.uiExpert.wao_controlCond.value())
        self.sim.config.p_controllers[0].set_delay(
                self.uiExpert.wao_controlDelay.value())
        self.sim.config.p_controllers[0].set_gain(self.uiExpert.wao_controlGain.value())
        # self.sim.config.p_controllers[0].set_TTcond(self.uiExpert.wao_controlTTcond.value())
        # # TODO : TTcond
        print("New RTC parameters set")

    def setWfsParams(self) -> None:
        nwfs = self.uiExpert.wao_selectWfs.currentIndex()
        if (nwfs < 0):
            nwfs = 0
        self.sim.config.p_wfss[nwfs].set_nxsub(self.uiExpert.wao_wfsNxsub.value())
        self.sim.config.p_wfss[nwfs].set_npix(self.uiExpert.wao_wfsNpix.value())
        self.sim.config.p_wfss[nwfs].set_pixsize(self.uiExpert.wao_wfsPixSize.value())
        self.sim.config.p_wfss[nwfs].set_xpos(self.uiExpert.wao_wfsXpos.value())
        self.sim.config.p_wfss[nwfs].set_ypos(self.uiExpert.wao_wfsYpos.value())
        self.sim.config.p_wfss[nwfs].set_fracsub(self.uiExpert.wao_wfsFracsub.value())
        self.sim.config.p_wfss[nwfs].set_Lambda(self.uiExpert.wao_wfsLambda.value())
        self.sim.config.p_wfss[nwfs].set_gsmag(self.uiExpert.wao_wfsMagnitude.value())
        # TODO: find a way to correctly set zerop (limited by the maximum value
        # allowed by the double spin box)
        self.sim.config.p_wfss[nwfs].set_zerop(10**(self.uiExpert.wao_wfsZp.value()))
        self.sim.config.p_wfss[nwfs].set_optthroughput(
                self.uiExpert.wao_wfsThrough.value())
        self.sim.config.p_wfss[nwfs].set_noise(self.uiExpert.wao_wfsNoise.value())

        # LGS params
        if (self.uiExpert.wao_wfsIsLGS.isChecked()):
            self.sim.config.p_wfss[nwfs].set_gsalt(self.uiExpert.wao_wfsGsAlt.value())
            self.sim.config.p_wfss[nwfs].set_lltx(self.uiExpert.wao_wfsLLTx.value())
            self.sim.config.p_wfss[nwfs].set_llty(self.uiExpert.wao_wfsLLTy.value())
            self.sim.config.p_wfss[nwfs].set_laserpower(
                    self.uiExpert.wao_wfsLGSpower.value())
            self.sim.config.p_wfss[nwfs].set_lgsreturnperwatt(
                    self.uiExpert.wao_wfsReturnPerWatt.value())
            self.sim.config.p_wfss[nwfs].set_beamsize(
                    self.uiExpert.wao_wfsBeamSize.value())
            self.sim.config.p_wfss[nwfs].set_proftype(
                    str(self.uiExpert.wao_selectLGSProfile.currentText()))
        print("New WFS parameters set")

    def setDmParams(self) -> None:
        ndm = self.uiExpert.wao_selectDM.currentIndex()
        if (ndm < 0):
            ndm = 0
        self.sim.config.p_dms[ndm].set_type(
                str(self.uiExpert.wao_dmTypeSelector.currentText()))
        self.sim.config.p_dms[ndm].set_alt(self.uiExpert.wao_dmAlt.value())
        self.sim.config.p_dms[ndm].set_nact(self.uiExpert.wao_dmNactu.value())
        self.sim.config.p_dms[ndm].set_unitpervolt(
                self.uiExpert.wao_dmUnitPerVolt.value())
        self.sim.config.p_dms[ndm].set_coupling(self.uiExpert.wao_dmCoupling.value())
        self.sim.config.p_dms[ndm].set_thresh(self.uiExpert.wao_dmThresh.value())
        print("New DM parameters set")

    def resetDM(self) -> None:
        if (self.sim.dms):
            ndm = self.uiExpert.wao_selectDM.currentIndex()
            if (ndm > -1):
                self.sim.dms.resetdm(
                        str(self.uiExpert.wao_dmTypeSelector.currentText()),
                        self.uiExpert.wao_dmAlt.value())
                self.updateDisplay()
                print("DM #%d reset" % ndm)
            else:
                print("Invalid DM : please select a DM to reset")
        else:
            print("There is no DM to reset")

    def BttCommand(self) -> None:
        if (self.sim.rtc):
            nfilt = int(self.uiExpert.wao_filterBtt.value())
            ao.command_on_Btt(self.sim.rtc, self.sim.dms, self.sim.config.p_dms,
                              self.sim.config.p_geom, nfilt)
            print("Loop is commanded from Btt basis now")

    def KLCommand(self) -> None:
        if (self.sim.rtc):
            nfilt = int(self.uiExpert.wao_filterBtt.value())
            cmat = ao.command_on_KL(
                    self.sim.rtc, self.sim.dms, self.sim.config.p_controllers[0],
                    self.sim.config.p_dms, self.sim.config.p_geom,
                    self.sim.config.p_atmos, self.sim.config.p_tel, nfilt)
            self.sim.rtc.set_cmat(0, cmat.astype(np.float32))
            print("Loop is commanded from KL basis now")

    def updatePlotWfs(self) -> None:
        typeText = str(self.uiExpert.wao_wfs_plotSelector.currentText())
        n = self.uiExpert.wao_selectWfs.currentIndex()
        self.uiExpert.wao_wfsWindow.canvas.axes.clear()
        ax = self.uiExpert.wao_wfsWindow.canvas.axes
        if (self.sim.config.p_wfss[n].type == scons.WFSType.PYRHR and
                    typeText == "Pyramid mod. pts" and self.sim.is_init):
            scale_fact = 2 * np.pi / self.sim.config.p_wfss[n]._Nfft * \
                self.sim.config.p_wfss[n].Lambda * \
                    1e-6 / self.sim.config.p_tel.diam / \
                    self.sim.config.p_wfss[n]._qpixsize * CONST.RAD2ARCSEC
            cx = self.sim.config.p_wfss[n]._pyr_cx / scale_fact
            cy = self.sim.config.p_wfss[n]._pyr_cy / scale_fact
            ax.scatter(cx, cy)

    def displayRtcMatrix(self) -> None:
        if not self.sim.is_init:
            # print(" widget not fully initialized")
            return

        data = None
        if (self.sim.rtc):
            type_matrix = str(self.uiExpert.wao_selectRtcMatrix.currentText())
            if (
                    type_matrix == "imat" and
                    self.sim.config.p_controllers[0].type != scons.ControllerType.GENERIC
                    and
                    self.sim.config.p_controllers[0].type != scons.ControllerType.GEO):
                data = self.sim.rtc.get_imat(0)
            elif (type_matrix == "cmat"):
                data = self.sim.rtc.get_cmat(0)
            elif (type_matrix == "Eigenvalues"):
                if (self.sim.config.p_controllers[0].type == scons.ControllerType.LS or
                            self.sim.config.p_controllers[0].type ==
                            scons.ControllerType.MV):
                    data = self.sim.rtc.getEigenvals(0)
            elif (type_matrix == "Cmm" and
                  self.sim.config.p_controllers[0].type == scons.ControllerType.MV):
                tmp = self.sim.rtc.get_cmm(0)
                ao.do_tomo_matrices(0, self.sim.rtc, self.sim.config.p_wfss,
                                    self.sim.dms, self.sim.atm, self.sim.wfs,
                                    self.sim.config.p_rtc, self.sim.config.p_geom,
                                    self.sim.config.p_dms, self.sim.config.p_tel,
                                    self.sim.config.p_atmos)
                data = self.sim.rtc.get_cmm(0)
                self.sim.rtc.set_cmm(0, tmp)
            elif (type_matrix == "Cmm inverse" and
                  self.sim.config.p_controllers[0].type == b"mv"):
                data = self.sim.rtc.get_cmm(0)
            elif (type_matrix == "Cmm eigen" and
                  self.sim.config.p_controllers[0].type == b"mv"):
                data = self.sim.rtc.getCmmEigenvals(0)
            elif (type_matrix == "Cphim" and
                  self.sim.config.p_controllers[0].type == b"mv"):
                data = self.sim.rtc.get_cphim(0)

            if (data is not None):
                self.uiExpert.wao_rtcWindow.canvas.axes.clear()
                ax = self.uiExpert.wao_rtcWindow.canvas.axes
                if (len(data.shape) == 2):
                    self.uiExpert.wao_rtcWindow.canvas.axes.matshow(
                            data, aspect="auto", origin="lower", cmap="gray")
                elif (len(data.shape) == 1):
                    # TODO : plot it properly, interactivity ?
                    self.uiExpert.wao_rtcWindow.canvas.axes.plot(
                            list(range(len(data))), data, color="black")
                    ax.set_yscale('log')
                    if (type_matrix == "Eigenvalues"):
                        #    major_ticks = np.arange(0, 101, 20)
                        #    minor_ticks = np.arange(0, 101, 5)

                        #    self.uiExpert.wao_rtcWindow.canvas.axes.set_xticks(major_ticks)
                        #    self.uiExpert.wao_rtcWindow.canvas.axes.set_xticks(minor_ticks, minor=True)
                        #    self.uiExpert.wao_rtcWindow.canvas.axes.set_yticks(major_ticks)
                        #    self.uiExpert.wao_rtcWindow.canvas.axes.set_yticks(minor_ticks, minor=True)

                        # and a corresponding grid

                        self.uiExpert.wao_rtcWindow.canvas.axes.grid(which='both')

                        # or if you want differnet settings for the grids:
                        self.uiExpert.wao_rtcWindow.canvas.axes.grid(
                                which='minor', alpha=0.2)
                        self.uiExpert.wao_rtcWindow.canvas.axes.grid(
                                which='major', alpha=0.5)
                        nfilt = self.sim.rtc.get_nfiltered(0, self.sim.config.p_rtc)
                        self.uiExpert.wao_rtcWindow.canvas.axes.plot([
                                nfilt - 0.5, nfilt - 0.5
                        ], [data.min(), data.max()], color="red", linestyle="dashed")
                        if (nfilt > 0):
                            self.uiExpert.wao_rtcWindow.canvas.axes.scatter(
                                    np.arange(0, nfilt, 1), data[0:nfilt], color="red"
                            )  # TODO : plot it properly, interactivity ?
                            self.uiExpert.wao_rtcWindow.canvas.axes.scatter(
                                    np.arange(nfilt, len(data),
                                              1), data[nfilt:], color="blue"
                            )  # TODO : plot it properly, interactivity ?
                            tt = "%d modes Filtered" % nfilt
                            # ax.text(nfilt + 2, data[nfilt-1], tt)
                            ax.text(0.5, 0.2, tt, horizontalalignment='center',
                                    verticalalignment='center', transform=ax.transAxes)

                self.uiExpert.wao_rtcWindow.canvas.draw()

    def computeDMrange(self, numdm: int, numwfs: int, push4imat: float=None,
                       unitpervolt: float=None) -> float:
        i = numdm
        if (push4imat is None or push4imat == 0):
            push4imat = self.sim.config.p_dms[i].push4imat
        if (unitpervolt is None or unitpervolt == 0):
            unitpervolt = self.sim.config.p_dms[i].unitpervolt

        actuPushInMicrons = push4imat * unitpervolt
        coupling = self.sim.config.p_dms[i].coupling
        a = coupling * actuPushInMicrons
        b = 0
        c = actuPushInMicrons
        d = coupling * actuPushInMicrons
        if (self.sim.config.p_dms[i].type is not scons.DmType.TT):
            dist = self.sim.config.p_tel.diam
        else:
            dist = self.sim.config.p_tel.diam / \
                self.sim.config.p_wfss[numwfs].nxsub
        Delta = (1e-6 * (((c + d) / 2) - ((a + b) / 2)))
        actuPushInArcsecs = CONST.RAD2ARCSEC * Delta / dist
        return actuPushInArcsecs
