"""Widget to simulate a closed loop

Usage:
  widget_ao.py [<parameters_filename>] [--expert] [--brama]

with 'parameters_filename' the path to the parameters file

Options:
  -h --help          Show this help message and exit
  --brama            Distribute data with BRAMA
"""

import os, sys
import numpy as np
import time

import pyqtgraph as pg

from tools import plsh, plpyr

import threading
import warnings

from PyQt5 import QtGui, QtWidgets
from PyQt5.uic import loadUiType
from PyQt5.QtCore import QThread, QObject, QTimer, pyqtSignal

from functools import partial
from subprocess import Popen, PIPE

import shesha_ao as ao
import shesha_sim
import shesha_constants as scons
from shesha_constants import CONST

from typing import Any, Dict, Tuple, Callable, List
"""
low levels debugs:
gdb --args python -i widget_ao.py
"""
from docopt import docopt

sys.path.insert(0, os.environ["SHESHA_ROOT"] + "/data/par/")
WindowTemplate, TemplateBaseClass = loadUiType(
        os.environ["SHESHA_ROOT"] + "/widgets/widget_ao.ui")  # type: type, type


class widgetAOWindow(TemplateBaseClass):

    def __init__(self, configFile: Any=None, BRAMA: bool=False,
                 expert: bool=False) -> None:
        TemplateBaseClass.__init__(self)

        self.BRAMA = BRAMA
        self.SRLE = []
        self.SRSE = []
        self.numiter = []

        self.ui = WindowTemplate()
        self.ui.setupUi(self)

        self.sim = None  # type: shesha_sim.simulator # Simulator object - initialized in addConfigFromFile

        #############################################################
        #                   ATTRIBUTES                              #
        #############################################################

        self.loopLock = threading.Lock(
        )  # type: Threading.Lock # Asynchronous loop / display safe-threading

        self.gui_timer = QTimer()  # type: QTimer
        self.gui_timer.timeout.connect(self.updateDisplay)
        if self.ui.wao_Display.isChecked():
            self.gui_timer.start(1000. / self.ui.wao_frameRate.value())

        self.stop = False  # type: bool  # Request quit

        self.ui.wao_nbiters.setValue(1000)  # Default GUI nIter box value
        self.nbiter = self.ui.wao_nbiters.value()
        self.refreshTime = 0  # type: float  # System time at last display refresh
        self.loopThread = None  # type: QThread
        self.assistant = None  # type: Any
        self.selector_init = None  # type: List[str]
        self.see_atmos = 1

        #############################################################
        #                 PYQTGRAPH WINDOW INIT                     #
        #############################################################

        self.img = pg.ImageItem(border='w')  # create image area
        self.img.setTransform(QtGui.QTransform(0, 1, 1, 0, 0, 0))  # flip X and Y
        # self.p1 = self.ui.wao_pgwindow.addPlot()  # create pyqtgraph plot
        # area
        self.p1 = self.ui.wao_pgwindow.addViewBox()
        self.p1.setAspectLocked(True)
        self.p1.addItem(self.img)  # Put image in plot area

        self.hist = pg.HistogramLUTItem()  # Create an histogram
        self.hist.setImageItem(self.img)  # Compute histogram from img
        self.ui.wao_pgwindow.addItem(self.hist)
        self.hist.autoHistogramRange()  # init levels
        self.hist.setMaximumWidth(100)

        #############################################################
        #                 CONNECTED BUTTONS                         #
        #############################################################
        # Default path for config files
        self.defaultParPath = os.environ["SHESHA_ROOT"] + "/data/par/par4bench/"
        self.ui.wao_loadConfig.clicked.connect(self.loadConfig)
        self.loadDefaultConfig()
        self.ui.wao_init.clicked.connect(self.InitConfig)
        self.ui.wao_run.setCheckable(True)
        self.ui.wao_run.clicked[bool].connect(self.aoLoopClicked)
        self.ui.wao_openLoop.setCheckable(True)
        self.ui.wao_openLoop.clicked[bool].connect(self.aoLoopOpen)
        self.ui.wao_next.clicked.connect(self.loopOnce)
        self.imgType = str(self.ui.wao_selectScreen.currentText())
        self.ui.wao_configFromFile.clicked.connect(self.addConfigFromFile)
        self.ui.wao_unzoom.clicked.connect(self.p1.autoRange)
        self.ui.wao_selectScreen.currentIndexChanged.connect(
                partial(self.updateNumberSelector, textType=None))
        self.ui.wao_selectNumber.currentIndexChanged.connect(self.setNumberSelection)
        self.ui.wao_wfs_plotSelector.currentIndexChanged.connect(self.updatePlotWfs)
        self.ui.wao_selectAtmosLayer.currentIndexChanged.connect(self.setLayerSelection)
        self.ui.wao_selectWfs.currentIndexChanged.connect(self.setWfsSelection)
        self.ui.wao_selectDM.currentIndexChanged.connect(self.setDmSelection)
        self.ui.wao_selectCentro.currentIndexChanged.connect(self.setCentroSelection)
        self.ui.wao_selectTarget.currentIndexChanged.connect(self.setTargetSelection)
        self.ui.wao_setAtmos.clicked.connect(self.setAtmosParams)
        self.ui.wao_setWfs.clicked.connect(self.setWfsParams)
        self.ui.wao_setDM.clicked.connect(self.setDmParams)
        self.ui.wao_setControl.clicked.connect(self.setRtcParams)
        self.ui.wao_setCentro.clicked.connect(self.setRtcParams)
        self.ui.wao_setTelescope.clicked.connect(self.setTelescopeParams)
        self.ui.wao_resetDM.clicked.connect(self.resetDM)
        self.ui.wao_update_gain.clicked.connect(self.updateGain)
        self.ui.wao_update_pyr_ampl.clicked.connect(self.updatePyrAmpl)
        self.ui.wao_selectRtcMatrix.currentIndexChanged.connect(self.displayRtcMatrix)
        self.ui.wao_rtcWindowMPL.hide()
        self.ui.wao_commandBtt.clicked.connect(self.BttCommand)
        self.ui.wao_commandKL.clicked.connect(self.KLCommand)
        self.ui.wao_resetSR.clicked.connect(self.resetSR)
        self.ui.wao_actionHelp_Contents.triggered.connect(self.on_help_triggered)

        self.ui.wao_allTarget.stateChanged.connect(self.updateAllTarget)
        self.ui.wao_forever.stateChanged.connect(self.updateForever)

        self.ui.wao_atmosphere.clicked[bool].connect(self.set_atmos)
        self.dispStatsInTerminal = False
        self.ui.wao_clearSR.clicked.connect(self.clearSR)
        self.ui.actionStats_in_Terminal.toggled.connect(self.updateStatsInTerminal)
        self.ui.actionQuit.triggered.connect(self.quitGUI)

        self.ui.wao_Display.stateChanged.connect(self.gui_timer_config)
        self.ui.wao_frameRate.setValue(2)

        self.ui.wao_dmUnitPerVolt.valueChanged.connect(self.updateDMrangeGUI)
        self.ui.wao_dmpush4iMat.valueChanged.connect(self.updateDMrangeGUI)
        self.ui.wao_pyr_ampl.valueChanged.connect(self.updateAmpliCompGUI)
        self.ui.wao_dmActuPushArcSecNumWFS.currentIndexChanged.connect(
                self.updateDMrange)

        self.SRcircleAtmos = {}  # type: Dict[int, pg.ScatterPlotItem]
        self.SRcircleWFS = {}  # type: Dict[int, pg.ScatterPlotItem]
        self.SRcircleDM = {}  # type: Dict[int, pg.ScatterPlotItem]
        self.SRcircleTarget = {}  # type: Dict[int, pg.ScatterPlotItem]

        # self.ui.splitter.setSizes([2000, 10])

        self.ui.wao_loadConfig.setDisabled(False)
        self.ui.wao_init.setDisabled(True)
        self.ui.wao_run.setDisabled(True)
        self.ui.wao_next.setDisabled(True)
        self.ui.wao_unzoom.setDisabled(True)
        self.ui.wao_resetSR.setDisabled(True)

        self.ui.wao_expertPanel.setVisible(expert)
        self.adjustSize()

        if configFile is not None:
            self.ui.wao_selectConfig.clear()
            self.ui.wao_selectConfig.addItem(configFile)
            self.loadConfig()
            self.InitConfig()

    def gui_timer_config(self, state) -> None:
        self.ui.wao_frameRate.setDisabled(state)
        if state:
            self.gui_timer.start(1000. / self.ui.wao_frameRate.value())
        else:
            self.gui_timer.stop()

    def on_help_triggered(self, i: Any=None) -> None:
        if i is None:
            return
        if not self.assistant or \
           not self.assistant.poll():

            helpcoll = os.environ["COMPASS_ROOT"] + "/doc/COMPASS.qhc"
            cmd = "assistant -enableRemoteControl -collectionFile %s" % helpcoll
            self.assistant = Popen(cmd, shell=True, stdin=PIPE)

    def closeEvent(self, event: Any) -> None:
        self.quitGUI(event)

    def quitGUI(self, event: Any=None) -> None:
        reply = QtWidgets.QMessageBox.question(self, 'Message', "Are you sure to quit?",
                                               QtWidgets.QMessageBox.Yes |
                                               QtWidgets.QMessageBox.No,
                                               QtWidgets.QMessageBox.No)

        if reply == QtWidgets.QMessageBox.Yes:
            self.stop = True
            if self.loopThread is not None:
                self.loopThread.join()
            quit()
            if event:
                event.accept()
        else:
            if event:
                event.ignore()

        #############################################################
        #                       METHODS                             #
        #############################################################
    def updateStatsInTerminal(self, state):
        self.dispStatsInTerminal = state

    def updateAllTarget(self, state):
        self.ui.wao_resetSR_tarNum.setDisabled(state)

    def updateForever(self, state):
        self.ui.wao_nbiters.setDisabled(state)

    def set_atmos(self, atmos):
        self.see_atmos = atmos

    def resetSR(self) -> None:
        if self.ui.wao_allTarget.isChecked():
            for t in range(self.sim.config.p_target.ntargets):
                self.sim.tar.reset_strehl(t)
        else:
            tarnum = self.ui.wao_resetSR_tarNum.value()
            print("Reset SR on target %d" % tarnum)
            self.sim.tar.reset_strehl(tarnum)

    def updateGain(self) -> None:
        if (self.sim.rtc):
            self.sim.rtc.set_gain(0, float(self.ui.wao_controlGain.value()))
            print("Loop gain updated on GPU")

    def updateAmpliCompGUI(self) -> None:
        diffract = self.sim.config.p_wfss[0].Lambda * \
            1e-6 / self.sim.config.p_tel.diam * CONST.RAD2ARCSEC
        self.ui.wao_pyr_ampl_arcsec.setValue(self.ui.wao_pyr_ampl.value() * diffract)

    def updateAmpliComp(self) -> None:
        diffract = self.sim.config.p_wfss[0].Lambda * \
            1e-6 / self.sim.config.p_tel.diam * CONST.RAD2ARCSEC
        self.ui.wao_pyr_ampl_arcsec.setValue(
                self.sim.config.p_wfss[0].pyr_ampl * diffract)

    def updatePyrAmpl(self) -> None:
        if (self.sim.rtc):
            self.sim.rtc.set_pyr_ampl(0,
                                      self.ui.wao_pyr_ampl.value(),
                                      self.sim.config.p_wfss, self.sim.config.p_tel)
            print("Pyramid modulation updated on GPU")
            self.updatePlotWfs()

    def updateDMrangeGUI(self) -> None:
        push4imat = self.ui.wao_dmpush4iMat.value()
        unitpervolt = self.ui.wao_dmUnitPerVolt.value()
        self.updateDMrange(push4imat=push4imat, unitpervolt=unitpervolt)

    def updateDMrange(self, push4imat: float=None, unitpervolt: float=None) -> None:
        numdm = str(self.ui.wao_selectDM.currentText())
        numwfs = str(self.ui.wao_dmActuPushArcSecNumWFS.currentText())
        if ((numdm is not "") and (numwfs is not "") and (push4imat != 0) and
            (unitpervolt != 0)):
            arcsecDMrange = self.computeDMrange(
                    int(numdm), int(numwfs), push4imat=push4imat,
                    unitpervolt=unitpervolt)
            self.ui.wao_dmActPushArcsec.setValue(arcsecDMrange)

    def updateTelescopePanel(self) -> None:
        self.ui.wao_zenithAngle.setValue(self.sim.config.p_geom.zenithangle)
        self.ui.wao_diamTel.setValue(self.sim.config.p_tel.diam)
        self.ui.wao_cobs.setValue(self.sim.config.p_tel.cobs)

    def updateDmPanel(self) -> None:
        ndm = self.ui.wao_selectDM.currentIndex()
        if (ndm < 0):
            ndm = 0
        self.ui.wao_dmActuPushArcSecNumWFS.clear()
        self.ui.wao_dmActuPushArcSecNumWFS.addItems([
                str(i) for i in range(len(self.sim.config.p_wfss))
        ])
        self.ui.wao_numberofDMs.setText(str(len(self.sim.config.p_dms)))
        self.ui.wao_dmTypeSelector.setCurrentIndex(
                self.ui.wao_dmTypeSelector.findText(
                        str(self.sim.config.p_dms[ndm].type)))
        self.ui.wao_dmAlt.setValue(self.sim.config.p_dms[ndm].alt)
        if (self.sim.config.p_dms[ndm].type == scons.DmType.KL):
            self.ui.wao_dmNactu.setValue(self.sim.config.p_dms[ndm].nkl)
        else:
            self.ui.wao_dmNactu.setValue(self.sim.config.p_dms[ndm].nact)
        self.ui.wao_dmUnitPerVolt.setValue(self.sim.config.p_dms[ndm].unitpervolt)
        self.ui.wao_dmpush4iMat.setValue(self.sim.config.p_dms[ndm].push4imat)
        self.ui.wao_dmCoupling.setValue(self.sim.config.p_dms[ndm].coupling)
        self.ui.wao_dmThresh.setValue(self.sim.config.p_dms[ndm].thresh)
        self.updateDMrange()

    def updateWfsPanel(self) -> None:
        nwfs = self.ui.wao_selectWfs.currentIndex()
        if (nwfs < 0):
            nwfs = 0
        self.ui.wao_numberofWfs.setText(str(len(self.sim.config.p_wfss)))
        self.ui.wao_wfsType.setText(str(self.sim.config.p_wfss[nwfs].type))
        self.ui.wao_wfsNxsub.setValue(self.sim.config.p_wfss[nwfs].nxsub)
        self.ui.wao_wfsNpix.setValue(self.sim.config.p_wfss[nwfs].npix)
        self.ui.wao_wfsPixSize.setValue(self.sim.config.p_wfss[nwfs].pixsize)
        self.ui.wao_wfsXpos.setValue(self.sim.config.p_wfss[nwfs].xpos)
        self.ui.wao_wfsYpos.setValue(self.sim.config.p_wfss[nwfs].ypos)
        self.ui.wao_wfsFracsub.setValue(self.sim.config.p_wfss[nwfs].fracsub)
        self.ui.wao_wfsLambda.setValue(self.sim.config.p_wfss[nwfs].Lambda)
        self.ui.wao_wfsMagnitude.setValue(self.sim.config.p_wfss[nwfs].gsmag)
        self.ui.wao_wfsZp.setValue(np.log10(self.sim.config.p_wfss[nwfs].zerop))
        self.ui.wao_wfsThrough.setValue(self.sim.config.p_wfss[nwfs].optthroughput)
        self.ui.wao_wfsNoise.setValue(self.sim.config.p_wfss[nwfs].noise)
        self.ui.wao_pyr_ampl.setValue(self.sim.config.p_wfss[nwfs].pyr_ampl)

        # LGS panel
        if (self.sim.config.p_wfss[nwfs].gsalt > 0):
            self.ui.wao_wfsIsLGS.setChecked(True)
            self.ui.wao_wfsGsAlt.setValue(self.sim.config.p_wfss[nwfs].gsalt)
            self.ui.wao_wfsLLTx.setValue(self.sim.config.p_wfss[nwfs].lltx)
            self.ui.wao_wfsLLTy.setValue(self.sim.config.p_wfss[nwfs].llty)
            self.ui.wao_wfsLGSpower.setValue(self.sim.config.p_wfss[nwfs].laserpower)
            self.ui.wao_wfsReturnPerWatt.setValue(
                    self.sim.config.p_wfss[nwfs].lgsreturnperwatt)
            self.ui.wao_wfsBeamSize.setValue(self.sim.config.p_wfss[nwfs].beamsize)
            self.ui.wao_selectLGSProfile.setCurrentIndex(
                    self.ui.wao_selectLGSProfile.findText(
                            str(self.sim.config.p_wfss[nwfs].proftype)))

        else:
            self.ui.wao_wfsIsLGS.setChecked(False)

        if (self.sim.config.p_wfss[nwfs].type == b"pyrhr" or
                    self.sim.config.p_wfss[nwfs].type == b"pyr"):
            self.ui.wao_wfs_plotSelector.setCurrentIndex(3)
        self.updatePlotWfs()

    def updateAtmosPanel(self) -> None:
        nscreen = self.ui.wao_selectAtmosLayer.currentIndex()
        if (nscreen < 0):
            nscreen = 0
        self.ui.wao_r0.setValue(self.sim.config.p_atmos.r0)
        self.ui.wao_atmosNlayers.setValue(self.sim.config.p_atmos.nscreens)
        self.ui.wao_atmosAlt.setValue(self.sim.config.p_atmos.alt[nscreen])
        self.ui.wao_atmosFrac.setValue(self.sim.config.p_atmos.frac[nscreen])
        self.ui.wao_atmosL0.setValue(self.sim.config.p_atmos.L0[nscreen])
        self.ui.wao_windSpeed.setValue(self.sim.config.p_atmos.windspeed[nscreen])
        self.ui.wao_windDirection.setValue(self.sim.config.p_atmos.winddir[nscreen])
        if (self.sim.config.p_atmos.dim_screens is not None):
            self.ui.wao_atmosDimScreen.setText(
                    str(self.sim.config.p_atmos.dim_screens[nscreen]))
        self.ui.wao_atmosWindow.canvas.axes.cla()
        width = (self.sim.config.p_atmos.alt.max() / 20. + 0.1) / 1000.
        self.ui.wao_atmosWindow.canvas.axes.barh(
                self.sim.config.p_atmos.alt / 1000. - width / 2.,
                self.sim.config.p_atmos.frac, width, color="blue")
        self.ui.wao_atmosWindow.canvas.draw()

    def updateRtcPanel(self) -> None:
        # Centroider panel
        ncentro = self.ui.wao_selectCentro.currentIndex()
        if (ncentro < 0):
            ncentro = 0
        type = self.sim.config.p_centroiders[ncentro].type
        self.ui.wao_centroTypeSelector.setCurrentIndex(
                self.ui.wao_centroTypeSelector.findText(str(type)))

        self.ui.wao_centroThresh.setValue(self.sim.config.p_centroiders[ncentro].thresh)
        if type == scons.CentroiderType.BPCOG:
            self.ui.wao_centroNbrightest.setValue(
                    self.sim.config.p_centroiders[ncentro].nmax)
        if type == scons.CentroiderType.TCOG:
            self.ui.wao_centroThresh.setValue(
                    self.sim.config.p_centroiders[ncentro].thresh)
        if type in [scons.CentroiderType.CORR, scons.CentroiderType.WCOG]:
            self.ui.wao_centroFunctionSelector.setCurrentIndex(
                    self.ui.wao_centroFunctionSelector.findText(
                            str(self.sim.config.p_centroiders[ncentro].type_fct)))
            self.ui.wao_centroWidth.setValue(
                    self.sim.config.p_centroiders[ncentro].width)

        # Controller panel
        type_contro = self.sim.config.p_controllers[0].type
        if (type_contro == scons.ControllerType.LS and
                    self.sim.config.p_controllers[0].modopti == 0):
            self.ui.wao_controlTypeSelector.setCurrentIndex(0)
        elif (type_contro == scons.ControllerType.MV):
            self.ui.wao_controlTypeSelector.setCurrentIndex(1)
        elif (type_contro == scons.ControllerType.GEO):
            self.ui.wao_controlTypeSelector.setCurrentIndex(2)
        elif (type_contro == scons.ControllerType.LS and
              self.sim.config.p_controllers[0].modopti):
            self.ui.wao_controlTypeSelector.setCurrentIndex(3)
        elif (type_contro == scons.ControllerType.CURED):
            self.ui.wao_controlTypeSelector.setCurrentIndex(4)
        elif (type_contro == scons.ControllerType.GENERIC):
            self.ui.wao_controlTypeSelector.setCurrentIndex(5)
        else:
            print("Controller type enumeration invalid.")

        self.ui.wao_controlCond.setValue(self.sim.config.p_controllers[0].maxcond)
        self.ui.wao_controlDelay.setValue(self.sim.config.p_controllers[0].delay)
        self.ui.wao_controlGain.setValue(self.sim.config.p_controllers[0].gain)
        self.ui.wao_controlTTcond.setValue(
                self.sim.config.p_controllers[0].maxcond)  # TODO : TTcond

    def updateTargetPanel(self) -> None:
        ntarget = self.ui.wao_selectTarget.currentIndex()
        if (ntarget < 0):
            ntarget = 0
        self.ui.wao_numberofTargets.setText(str(self.sim.config.p_target.ntargets))
        self.ui.wao_targetMag.setValue(self.sim.config.p_target.mag[ntarget])
        self.ui.wao_targetXpos.setValue(self.sim.config.p_target.xpos[ntarget])
        self.ui.wao_targetYpos.setValue(self.sim.config.p_target.ypos[ntarget])
        self.ui.wao_targetLambda.setValue(self.sim.config.p_target.Lambda[ntarget])

        self.ui.wao_targetWindow.canvas.axes.cla()
        xmax = np.max(np.abs(self.sim.config.p_target.xpos))
        ymax = np.max(np.abs(self.sim.config.p_target.ypos))
        if (self.sim.config.p_wfss):
            self.ui.wao_targetWindow.canvas.axes.plot([
                    w.xpos for w in self.sim.config.p_wfss
            ], [w.ypos for w in self.sim.config.p_wfss], 'o', color="green")
            xmax = np.max([
                    xmax, np.max(np.abs([w.xpos for w in self.sim.config.p_wfss]))
            ])
            ymax = np.max([
                    ymax, np.max(np.abs([w.ypos for w in self.sim.config.p_wfss]))
            ])
        self.ui.wao_targetWindow.canvas.axes.plot(self.sim.config.p_target.xpos,
                                                  self.sim.config.p_target.ypos, '*',
                                                  color="red")
        self.ui.wao_targetWindow.canvas.axes.set_xlim(-xmax - 10, xmax + 10)
        self.ui.wao_targetWindow.canvas.axes.set_ylim(-ymax - 10, ymax + 10)
        self.ui.wao_targetWindow.canvas.axes.grid()
        self.ui.wao_targetWindow.canvas.draw()

    def updatePanels(self) -> None:
        self.updateTelescopePanel()
        self.updateLayerSelection()
        self.updateAtmosPanel()
        self.updateWfsSelection()
        self.updateWfsPanel()
        self.updateDmSelection()
        self.updateDmPanel()
        self.updateCentroSelection()
        self.updateRtcPanel()
        self.updateTargetSelection()
        self.updateTargetPanel()

    def setTelescopeParams(self) -> None:
        self.sim.config.p_tel.set_diam(self.ui.wao_diamTel.value())
        self.sim.config.p_tel.set_cobs(self.ui.wao_cobs.value())
        self.sim.config.p_geom.set_zenithangle(self.ui.wao_zenithAngle.value())
        print("New telescope parameters set")

    def setAtmosParams(self) -> None:
        nscreen = self.ui.wao_selectAtmosLayer.currentIndex()
        if (nscreen < 0):
            nscreen = 0
        self.sim.config.p_atmos.alt[nscreen] = self.ui.wao_atmosAlt.value()
        self.sim.config.p_atmos.frac[nscreen] = self.ui.wao_atmosFrac.value()
        self.sim.config.p_atmos.L0[nscreen] = self.ui.wao_atmosL0.value()
        self.sim.config.p_atmos.windspeed[nscreen] = self.ui.wao_windSpeed.value()
        self.sim.config.p_atmos.winddir[nscreen] = self.ui.wao_windDirection.value()
        print("New atmos parameters set")

    def setRtcParams(self) -> None:
        # Centroider params
        ncentro = self.ui.wao_selectCentro.currentIndex()
        if (ncentro < 0):
            ncentro = 0
        self.sim.config.p_centroiders[ncentro].set_type(
                str(self.ui.wao_centroTypeSelector.currentText()))
        self.sim.config.p_centroiders[ncentro].set_thresh(
                self.ui.wao_centroThresh.value())
        self.sim.config.p_centroiders[ncentro].set_nmax(
                self.ui.wao_centroNbrightest.value())
        self.sim.config.p_centroiders[ncentro].set_thresh(
                self.ui.wao_centroThresh.value())
        self.sim.config.p_centroiders[ncentro].set_type_fct(
                str(self.ui.wao_centroFunctionSelector.currentText()))
        self.sim.config.p_centroiders[ncentro].set_width(self.ui.wao_centroWidth.value())

        # Controller panel
        type_contro = str(self.ui.wao_controlTypeSelector.currentText())
        if (type_contro == "LS"):
            self.sim.config.p_controllers[0].set_type(scons.ControllerType.LS)
        elif (type_contro == "MV"):
            self.sim.config.p_controllers[0].set_type(scons.ControllerType.MV)
        elif (type_contro == "PROJ"):
            self.sim.config.p_controllers[0].set_type(scons.ControllerType.GEO)
        elif (type_contro == "OptiMods"):
            self.sim.config.p_controllers[0].set_type(scons.ControllerType.LS)
            self.sim.config.p_controllers[0].set_modopti(1)

        self.sim.config.p_controllers[0].set_maxcond(self.ui.wao_controlCond.value())
        self.sim.config.p_controllers[0].set_delay(self.ui.wao_controlDelay.value())
        self.sim.config.p_controllers[0].set_gain(self.ui.wao_controlGain.value())
        # self.sim.config.p_controllers[0].set_TTcond(self.ui.wao_controlTTcond.value())
        # # TODO : TTcond
        print("New RTC parameters set")

    def setWfsParams(self) -> None:
        nwfs = self.ui.wao_selectWfs.currentIndex()
        if (nwfs < 0):
            nwfs = 0
        self.sim.config.p_wfss[nwfs].set_nxsub(self.ui.wao_wfsNxsub.value())
        self.sim.config.p_wfss[nwfs].set_npix(self.ui.wao_wfsNpix.value())
        self.sim.config.p_wfss[nwfs].set_pixsize(self.ui.wao_wfsPixSize.value())
        self.sim.config.p_wfss[nwfs].set_xpos(self.ui.wao_wfsXpos.value())
        self.sim.config.p_wfss[nwfs].set_ypos(self.ui.wao_wfsYpos.value())
        self.sim.config.p_wfss[nwfs].set_fracsub(self.ui.wao_wfsFracsub.value())
        self.sim.config.p_wfss[nwfs].set_Lambda(self.ui.wao_wfsLambda.value())
        self.sim.config.p_wfss[nwfs].set_gsmag(self.ui.wao_wfsMagnitude.value())
        # TODO: find a way to correctly set zerop (limited by the maximum value
        # allowed by the double spin box)
        self.sim.config.p_wfss[nwfs].set_zerop(10**(self.ui.wao_wfsZp.value()))
        self.sim.config.p_wfss[nwfs].set_optthroughput(self.ui.wao_wfsThrough.value())
        self.sim.config.p_wfss[nwfs].set_noise(self.ui.wao_wfsNoise.value())

        # LGS params
        if (self.ui.wao_wfsIsLGS.isChecked()):
            self.sim.config.p_wfss[nwfs].set_gsalt(self.ui.wao_wfsGsAlt.value())
            self.sim.config.p_wfss[nwfs].set_lltx(self.ui.wao_wfsLLTx.value())
            self.sim.config.p_wfss[nwfs].set_llty(self.ui.wao_wfsLLTy.value())
            self.sim.config.p_wfss[nwfs].set_laserpower(self.ui.wao_wfsLGSpower.value())
            self.sim.config.p_wfss[nwfs].set_lgsreturnperwatt(
                    self.ui.wao_wfsReturnPerWatt.value())
            self.sim.config.p_wfss[nwfs].set_beamsize(self.ui.wao_wfsBeamSize.value())
            self.sim.config.p_wfss[nwfs].set_proftype(
                    str(self.ui.wao_selectLGSProfile.currentText()))
        print("New WFS parameters set")

    def setDmParams(self) -> None:
        ndm = self.ui.wao_selectDM.currentIndex()
        if (ndm < 0):
            ndm = 0
        self.sim.config.p_dms[ndm].set_type(
                str(self.ui.wao_dmTypeSelector.currentText()))
        self.sim.config.p_dms[ndm].set_alt(self.ui.wao_dmAlt.value())
        self.sim.config.p_dms[ndm].set_nact(self.ui.wao_dmNactu.value())
        self.sim.config.p_dms[ndm].set_unitpervolt(self.ui.wao_dmUnitPerVolt.value())
        self.sim.config.p_dms[ndm].set_coupling(self.ui.wao_dmCoupling.value())
        self.sim.config.p_dms[ndm].set_thresh(self.ui.wao_dmThresh.value())
        print("New DM parameters set")

    def updateLayerSelection(self) -> None:
        self.ui.wao_selectAtmosLayer.clear()
        self.ui.wao_selectAtmosLayer.addItems([
                str(i) for i in range(self.sim.config.p_atmos.nscreens)
        ])

    def updateTargetSelection(self) -> None:
        self.ui.wao_selectTarget.clear()
        self.ui.wao_selectTarget.addItems([
                str(i) for i in range(self.sim.config.p_target.ntargets)
        ])

    def updateWfsSelection(self) -> None:
        self.ui.wao_selectWfs.clear()
        self.ui.wao_selectWfs.addItems([
                str(i) for i in range(len(self.sim.config.p_wfss))
        ])

    def updateDmSelection(self) -> None:
        self.ui.wao_selectDM.clear()
        self.ui.wao_selectDM.addItems([
                str(i) for i in range(len(self.sim.config.p_dms))
        ])

    def updateCentroSelection(self) -> None:
        self.ui.wao_selectCentro.clear()
        self.ui.wao_selectCentro.addItems([
                str(i) for i in range(len(self.sim.config.p_centroiders))
        ])

    def setCentroSelection(self) -> None:
        self.updateRtcPanel()

    def setLayerSelection(self) -> None:
        self.updateAtmosPanel()

    def setTargetSelection(self) -> None:
        self.updateTargetPanel()

    def setWfsSelection(self) -> None:
        self.updateWfsPanel()

    def setDmSelection(self) -> None:
        self.updateDmPanel()

    def addConfigFromFile(self) -> None:
        '''
            Callback when a config file is double clicked in the file browser
            Place the selected file name in the browsing drop-down menu,
            the call the self.loadConfig callback of the load button.
        '''
        filepath = QtWidgets.QFileDialog(directory=self.defaultParPath).getOpenFileName(
                self, "Select parameter file", "",
                "parameters file (*.py);;hdf5 file (*.h5);;all files (*)")

        self.ui.wao_selectConfig.clear()
        self.ui.wao_selectConfig.addItem(str(filepath[0]))

        self.loadConfig()

    def loadConfig(self) -> None:
        '''
            Callback when 'LOAD' button is hit
        '''
        configFile = str(self.ui.wao_selectConfig.currentText())
        sys.path.insert(0, self.defaultParPath)

        if self.sim is None:
            if self.BRAMA:
                self.sim = shesha_sim.SimulatorBrama(configFile)
            else:
                self.sim = shesha_sim.Simulator(configFile)
        else:
            self.sim.clear_init()
            self.sim.load_from_file(configFile)

        try:
            sys.path.remove(self.defaultParPath)
        except:
            pass

        self.ui.wao_selectScreen.clear()

        pyrSpecifics = [
                self.ui.ui_modradiusPanel, self.ui.ui_modradiusPanelarcesec,
                self.ui.wao_pyr_ampl, self.ui.wao_pyr_ampl_arcsec,
                self.ui.wao_update_pyr_ampl
        ]
        if (scons.WFSType.PYRHR in [p_wfs.type for p_wfs in self.sim.config.p_wfss]):
            self.selector_init = [
                    "Phase - Atmos", "Phase - WFS", "Pyrimg - LR", "Pyrimg - HR",
                    "Centroids - WFS", "Slopes - WFS", "Phase - Target", "Phase - DM",
                    "PSF LE", "PSF SE"
            ]
            [pane.show() for pane in pyrSpecifics]
        else:
            self.selector_init = [
                    "Phase - Atmos", "Phase - WFS", "Spots - WFS", "Centroids - WFS",
                    "Slopes - WFS", "Phase - Target", "Phase - DM", "PSF LE", "PSF SE"
            ]
            [pane.hide() for pane in pyrSpecifics]

        self.ui.wao_selectScreen.addItems(self.selector_init)
        self.ui.wao_selectScreen.setCurrentIndex(0)
        self.updateNumberSelector(textType=self.imgType)
        self.updatePanels()

        self.ui.wao_init.setDisabled(False)
        self.ui.wao_run.setDisabled(True)
        self.ui.wao_next.setDisabled(True)
        self.ui.wao_unzoom.setDisabled(True)
        self.ui.wao_resetSR.setDisabled(True)

    def aoLoopClicked(self, pressed: bool) -> None:
        if pressed:
            self.stop = False
            self.refreshTime = time.time()
            self.nbiter = self.ui.wao_nbiters.value()
            if self.dispStatsInTerminal:
                if self.ui.wao_forever.isChecked():
                    print("LOOP STARTED")
                else:
                    print("LOOP STARTED FOR %d iterations" % self.nbiter)
            self.run()
        else:
            self.stop = True

    def aoLoopOpen(self, pressed: bool) -> None:
        if (pressed):
            self.sim.rtc.set_openloop(0, 1)
        else:
            self.sim.rtc.set_openloop(0, 0)

    def setNumberSelection(self) -> None:
        if (self.ui.wao_selectNumber.currentIndex() > -1):
            self.numberSelected = self.ui.wao_selectNumber.currentIndex()
        else:
            self.numberSelected = 0
        self.updateDisplay()

    def updateNumberSelector(self, textType: str=None) -> None:
        if textType is None:
            textType = str(self.ui.wao_selectScreen.currentText())
        self.imgType = textType
        self.ui.wao_selectNumber.clear()
        if (textType == "Phase - Atmos"):
            n = self.sim.config.p_atmos.nscreens
        elif (textType == "Phase - WFS" or textType == "Spots - WFS" or
              textType == "Centroids - WFS" or textType == "Slopes - WFS" or
              textType == "Pyrimg - HR" or textType == "Pyrimg - LR"):
            n = len(self.sim.config.p_wfss)
        elif (textType == "Phase - Target" or textType == "PSF LE" or
              textType == "PSF SE"):
            n = self.sim.config.p_target.ntargets
        elif (textType == "Phase - DM"):
            n = len(self.sim.config.p_dms)
        else:
            n = 0
        self.ui.wao_selectNumber.addItems([str(i) for i in range(n)])
        self.updateDisplay()

    def loadDefaultConfig(self) -> None:
        import glob
        parlist = sorted(glob.glob(self.defaultParPath + "*.py"))
        self.ui.wao_selectConfig.clear()
        self.ui.wao_selectConfig.addItems([
                parlist[i].split('/')[-1] for i in range(len(parlist))
        ])

    def InitConfig(self) -> None:
        self.loopLock.acquire(True)
        self.sim.clear_init()

        self.ui.wao_loadConfig.setDisabled(True)
        self.ui.wao_init.setDisabled(True)
        thread = WorkerThread(self, self.InitConfigThread)
        thread.jobFinished['PyQt_PyObject'].connect(self.InitConfigFinished)
        thread.start()

    def InitConfigThread(self) -> None:
        self.ui.wao_deviceNumber.setDisabled(True)
        # self.sim.config.p_loop.devices = self.ui.wao_deviceNumber.value()  # using GUI value
        # gpudevice = "ALL"  # using all GPU avalaible
        # gpudevice = np.array([2, 3], dtype=np.int32)
        # gpudevice = np.arange(4, dtype=np.int32) # using 4 GPUs: 0-3
        # gpudevice = 0  # using 1 GPU : 0
        self.sim.init_sim()

    def InitConfigFinished(self) -> None:
        # Thread naga context reload:
        self.sim.force_context()

        self.ui.wao_atmosDimScreen.setText(str(self.sim.config.p_atmos.dim_screens[0]))
        self.ui.wao_loadConfig.setDisabled(False)

        self.currentViewSelected = None  # type: str
        self.SRCrossX = None  # type : pg.PlotCurveItem
        self.SRCrossY = None  # type : pg.PlotCurveItem

        for i in self.SRcircleAtmos:
            self.p1.removeItem(self.SRcircleAtmos[i])
        for i in self.SRcircleWFS:
            self.p1.removeItem(self.SRcircleWFS[i])
        for i in self.SRcircleDM:
            self.p1.removeItem(self.SRcircleDM[i])
        for i in self.SRcircleTarget:
            self.p1.removeItem(self.SRcircleTarget[i])

        for i in range(len(self.sim.config.p_atmos.alt)):
            data = self.sim.atm.get_screen(self.sim.config.p_atmos.alt[i])
            cx, cy = self.circleCoords(self.sim.config.p_geom.pupdiam / 2, 1000,
                                       data.shape[0], data.shape[1])
            self.SRcircleAtmos[i] = pg.ScatterPlotItem(cx, cy, pen='r', size=1)
            self.p1.addItem(self.SRcircleAtmos[i])
            self.SRcircleAtmos[i].setPoints(cx, cy)
            self.SRcircleAtmos[i].hide()

        for i in range(len(self.sim.config.p_wfss)):
            data = self.sim.wfs.get_phase(i)
            cx, cy = self.circleCoords(self.sim.config.p_geom.pupdiam / 2, 1000,
                                       data.shape[0], data.shape[1])
            self.SRcircleWFS[i] = pg.ScatterPlotItem(cx, cy, pen='r', size=1)
            self.p1.addItem(self.SRcircleWFS[i])
            self.SRcircleWFS[i].setPoints(cx, cy)
            self.SRcircleWFS[i].hide()

        for i in range(len(self.sim.config.p_dms)):
            dm_type = self.sim.config.p_dms[i].type
            alt = self.sim.config.p_dms[i].alt
            data = self.sim.dms.get_dm(dm_type, alt)
            cx, cy = self.circleCoords(self.sim.config.p_geom.pupdiam / 2, 1000,
                                       data.shape[0], data.shape[1])
            self.SRcircleDM[i] = pg.ScatterPlotItem(cx, cy, pen='r', size=1)
            self.p1.addItem(self.SRcircleDM[i])
            self.SRcircleDM[i].setPoints(cx, cy)
            self.SRcircleDM[i].hide()

        for i in range(self.sim.config.p_target.ntargets):
            data = self.sim.tar.get_phase(i)
            cx, cy = self.circleCoords(self.sim.config.p_geom.pupdiam / 2, 1000,
                                       data.shape[0], data.shape[1])
            self.SRcircleTarget[i] = pg.ScatterPlotItem(cx, cy, pen='r', size=1)
            self.p1.addItem(self.SRcircleTarget[i])
            self.SRcircleTarget[i].setPoints(cx, cy)
            self.SRcircleTarget[i].show()

        print(self.sim)

        self.updateDisplay()
        self.displayRtcMatrix()
        self.updatePlotWfs()
        self.p1.autoRange()

        self.ui.wao_init.setDisabled(False)
        self.ui.wao_run.setDisabled(False)
        self.ui.wao_next.setDisabled(False)
        self.ui.wao_openLoop.setDisabled(False)
        self.ui.wao_unzoom.setDisabled(False)
        self.ui.wao_resetSR.setDisabled(False)

        self.loopLock.release()

    def circleCoords(self, ampli: float, npts: int, datashape0: int,
                     datashape1: int) -> Tuple[float, float]:
        cx = ampli * np.sin((np.arange(npts) + 1) * 2. * np.pi / npts) + datashape0 / 2
        cy = ampli * np.cos((np.arange(npts) + 1) * 2. * np.pi / npts) + datashape1 / 2
        return cx, cy

    def resetDM(self) -> None:
        if (self.sim.dms):
            ndm = self.ui.wao_selectDM.currentIndex()
            if (ndm > -1):
                self.sim.dms.resetdm(
                        str(self.ui.wao_dmTypeSelector.currentText()),
                        self.ui.wao_dmAlt.value())
                self.updateDisplay()
                print("DM #%d reset" % ndm)
            else:
                print("Invalid DM : please select a DM to reset")
        else:
            print("There is no DM to reset")

    def BttCommand(self) -> None:
        if (self.sim.rtc):
            nfilt = int(self.ui.wao_filterBtt.value())
            ao.command_on_Btt(self.sim.rtc, self.sim.dms, self.sim.config.p_dms,
                              self.sim.config.p_geom, nfilt)
            print("Loop is commanded from Btt basis now")

    def KLCommand(self) -> None:
        if (self.sim.rtc):
            nfilt = int(self.ui.wao_filterBtt.value())
            cmat = ao.command_on_KL(
                    self.sim.rtc, self.sim.dms, self.sim.config.p_controllers[0],
                    self.sim.config.p_dms, self.sim.config.p_geom,
                    self.sim.config.p_atmos, self.sim.config.p_tel, nfilt)
            self.sim.rtc.set_cmat(0, cmat.astype(np.float32))
            print("Loop is commanded from KL basis now")

    def updatePlotWfs(self) -> None:
        typeText = str(self.ui.wao_wfs_plotSelector.currentText())
        n = self.ui.wao_selectWfs.currentIndex()
        self.ui.wao_wfsWindow.canvas.axes.clear()
        ax = self.ui.wao_wfsWindow.canvas.axes
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
            type_matrix = str(self.ui.wao_selectRtcMatrix.currentText())
            if (type_matrix == "imat" and
                        self.sim.config.p_controllers[0].type != "generic" and
                        self.sim.config.p_controllers[0].type != "geo"):
                data = self.sim.rtc.get_imat(0)
            elif (type_matrix == "cmat"):
                data = self.sim.rtc.get_cmat(0)
            elif (type_matrix == "Eigenvalues"):
                if (self.sim.config.p_controllers[0].type == "ls" or
                            self.sim.config.p_controllers[0].type == b"mv"):
                    data = self.sim.rtc.getEigenvals(0)
            elif (type_matrix == "Cmm" and
                  self.sim.config.p_controllers[0].type == b"mv"):
                tmp = self.sim.rtc.get_cmm(0)
                ao.doTomoMatrices(0, self.sim.rtc, self.sim.config.p_wfss, self.sim.dms,
                                  self.sim.atm, self.sim.wfs, self.sim.config.p_rtc,
                                  self.sim.config.p_geom, self.sim.config.p_dms,
                                  self.sim.config.p_tel, self.sim.config.p_atmos)
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
                self.ui.wao_rtcWindow.canvas.axes.clear()
                ax = self.ui.wao_rtcWindow.canvas.axes
                if (len(data.shape) == 2):
                    self.ui.wao_rtcWindow.canvas.axes.matshow(data, aspect="auto",
                                                              origin="lower",
                                                              cmap="gray")
                elif (len(data.shape) == 1):
                    # TODO : plot it properly, interactivity ?
                    self.ui.wao_rtcWindow.canvas.axes.plot(
                            list(range(len(data))), data, color="black")
                    ax.set_yscale('log')
                    if (type_matrix == "Eigenvalues"):
                        #    major_ticks = np.arange(0, 101, 20)
                        #    minor_ticks = np.arange(0, 101, 5)

                        #    self.ui.wao_rtcWindow.canvas.axes.set_xticks(major_ticks)
                        #    self.ui.wao_rtcWindow.canvas.axes.set_xticks(minor_ticks, minor=True)
                        #    self.ui.wao_rtcWindow.canvas.axes.set_yticks(major_ticks)
                        #    self.ui.wao_rtcWindow.canvas.axes.set_yticks(minor_ticks, minor=True)

                        # and a corresponding grid

                        self.ui.wao_rtcWindow.canvas.axes.grid(which='both')

                        # or if you want differnet settings for the grids:
                        self.ui.wao_rtcWindow.canvas.axes.grid(which='minor', alpha=0.2)
                        self.ui.wao_rtcWindow.canvas.axes.grid(which='major', alpha=0.5)
                        nfilt = self.sim.rtc.get_nfiltered(0, self.sim.config.p_rtc)
                        self.ui.wao_rtcWindow.canvas.axes.plot([
                                nfilt - 0.5, nfilt - 0.5
                        ], [data.min(), data.max()], color="red", linestyle="dashed")
                        if (nfilt > 0):
                            self.ui.wao_rtcWindow.canvas.axes.scatter(
                                    np.arange(0, nfilt, 1), data[0:nfilt], color="red"
                            )  # TODO : plot it properly, interactivity ?
                            self.ui.wao_rtcWindow.canvas.axes.scatter(
                                    np.arange(nfilt, len(data),
                                              1), data[nfilt:], color="blue"
                            )  # TODO : plot it properly, interactivity ?
                            tt = "%d modes Filtered" % nfilt
                            # ax.text(nfilt + 2, data[nfilt-1], tt)
                            ax.text(0.5, 0.2, tt, horizontalalignment='center',
                                    verticalalignment='center', transform=ax.transAxes)

                self.ui.wao_rtcWindow.canvas.draw()

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

    def setupDisp(self, fig: str="pg") -> None:
        if fig == "pg":
            widToShow = self.ui.wao_pgwindow
            widToHide = self.ui.wao_rtcWindowMPL
        elif fig == "MPL":
            widToShow = self.ui.wao_rtcWindowMPL
            widToHide = self.ui.wao_pgwindow
        else:
            return

        if (not widToShow.isVisible()):
            widToShow.show()
            widToHide.hide()

    def clearSR(self):
        self.SRLE = [self.SRLE[-1]]
        self.SRSE = [self.SRSE[-1]]
        self.numiter = [self.numiter[-1]]

    def updateSRDisplay(self, SRLE, SRSE, numiter):
        self.SRLE.append(SRLE)
        self.SRSE.append(SRSE)
        self.numiter.append(numiter)
        if (len(self.SRSE) > 100):  # Clipping last 100 points...
            self.SRLE = self.SRLE[-100:]
            self.SRSE = self.SRSE[-100:]
            self.numiter = self.numiter[-100:]
        self.ui.wao_SRPlotWindow.canvas.axes.clear()
        self.ui.wao_SRPlotWindow.canvas.axes.yaxis.set_label("SR")
        self.ui.wao_SRPlotWindow.canvas.axes.xaxis.set_label("num iter")
        self.ui.wao_SRPlotWindow.canvas.axes.plot(self.numiter, self.SRSE,
                                                  linestyle="--", color="red",
                                                  marker="o", label="SR SE")
        self.ui.wao_SRPlotWindow.canvas.axes.plot(self.numiter, self.SRLE,
                                                  linestyle="--", color="blue",
                                                  marker="o", label="SR LE")
        # self.ui.wao_SRPlotWindow.canvas.axes.grid()
        self.ui.wao_SRPlotWindow.canvas.draw()

    def updateDisplay(self) -> None:
        if (self.sim is None) or (not self.sim.is_init) or (
                not self.ui.wao_Display.isChecked()):
            # print("Widget not fully initialized")
            return
        data = None
        if not self.loopLock.acquire(False):
            return
        else:
            try:
                if (self.SRCrossX and (self.imgType in [
                        "Phase - Target", "Phase - DM", "Phase - Atmos", "Phase - WFS",
                        "Spots - WFS", "Centroids - WFS", "Slopes - WFS"
                ])):
                    self.SRCrossX.hide()
                    self.SRCrossY.hide()

                # if(self.SRcircle and (self.imgType in ["Spots - WFS",
                # "Centroids - WFS", "Slopes - WFS","PSF SE","PSF LE"])):
                for i in range(len(self.sim.config.p_atmos.alt)):
                    self.SRcircleAtmos[i].hide()
                for i in range(len(self.sim.config.p_wfss)):
                    self.SRcircleWFS[i].hide()
                for i in range(len(self.sim.config.p_dms)):
                    self.SRcircleDM[i].hide()
                for i in range(self.sim.config.p_target.ntargets):
                    self.SRcircleTarget[i].hide()

                if (self.sim.atm):
                    if (self.imgType == "Phase - Atmos"):
                        self.setupDisp("pg")
                        data = self.sim.atm.get_screen(
                                self.sim.config.p_atmos.alt[self.numberSelected])
                        if (self.imgType != self.currentViewSelected):
                            self.p1.setRange(xRange=(0, data.shape[0]),
                                             yRange=(0, data.shape[1]))
                        self.currentViewSelected = self.imgType
                        self.SRcircleAtmos[self.numberSelected].show()

                if (self.sim.wfs):
                    if (self.imgType == "Phase - WFS"):
                        self.setupDisp("pg")
                        data = self.sim.wfs.get_phase(self.numberSelected)
                        if (self.imgType != self.currentViewSelected):
                            self.p1.setRange(xRange=(0, data.shape[0]),
                                             yRange=(0, data.shape[1]))
                        self.currentViewSelected = self.imgType
                        self.SRcircleWFS[self.numberSelected].show()

                    if (self.imgType == "Spots - WFS"):
                        self.setupDisp("pg")
                        if (self.sim.config.p_wfss[self.numberSelected]
                                    .type == scons.WFSType.SH):
                            data = self.sim.wfs.get_binimg(self.numberSelected)
                        elif (self.sim.config.p_wfss[self.numberSelected]
                              .type == scons.WFSType.PYRHR):
                            data = self.sim.wfs.get_pyrimg(self.numberSelected)
                        if (self.imgType != self.currentViewSelected):
                            self.p1.setRange(xRange=(0, data.shape[0]),
                                             yRange=(0, data.shape[1]))
                        self.currentViewSelected = self.imgType

                    if (self.imgType == "Pyrimg - LR"):
                        self.setupDisp("pg")
                        if (self.sim.config.p_wfss[self.numberSelected]
                                    .type == scons.WFSType.PYRHR):
                            data = self.sim.wfs.get_pyrimg(self.numberSelected)
                        if (self.imgType != self.currentViewSelected):
                            self.p1.setRange(xRange=(0, data.shape[0]),
                                             yRange=(0, data.shape[1]))
                        self.currentViewSelected = self.imgType

                    if (self.imgType == "Pyrimg - HR"):
                        self.setupDisp("pg")
                        if (self.sim.config.p_wfss[self.numberSelected]
                                    .type == scons.WFSType.PYRHR):
                            data = self.sim.wfs.get_pyrimghr(self.numberSelected)
                        if (self.imgType != self.currentViewSelected):
                            self.p1.setRange(xRange=(0, data.shape[0]),
                                             yRange=(0, data.shape[1]))
                        self.currentViewSelected = self.imgType

                    if (self.imgType == "Centroids - WFS"):
                        self.setupDisp("MPL")
                        self.ui.wao_rtcWindowMPL.canvas.axes.clear()
                        # retrieving centroids
                        centroids = self.sim.rtc.get_centroids(0)
                        nvalid = [2 * o._nvalid for o in self.sim.config.p_wfss]
                        ind = np.sum(nvalid[:self.numberSelected], dtype=np.int32)
                        if (self.sim.config.p_wfss[self.numberSelected]
                                    .type == scons.WFSType.PYRHR):
                            plpyr(centroids[ind:ind + nvalid[self.numberSelected]],
                                  self.sim.config.p_wfs0._isvalid)
                        else:
                            x, y, vx, vy = plsh(
                                    centroids[ind:ind + nvalid[self.numberSelected]],
                                    self.sim.config.p_wfss[self.numberSelected].nxsub,
                                    self.sim.config.p_tel.cobs, returnquiver=True
                            )  # Preparing mesh and vector for display
                        self.ui.wao_rtcWindowMPL.canvas.axes.quiver(
                                x, y, vx, vy, pivot='mid')
                        self.ui.wao_rtcWindowMPL.canvas.draw()
                        self.currentViewSelected = self.imgType

                        return
                    if (self.imgType == "Slopes - WFS"):
                        self.setupDisp("MPL")
                        self.ui.wao_rtcWindowMPL.canvas.axes.clear()
                        self.sim.wfs.slopes_geom(self.numberSelected, 0)
                        slopes = self.sim.wfs.get_slopes(self.numberSelected)
                        x, y, vx, vy = plsh(
                                slopes,
                                self.sim.config.p_wfss[self.numberSelected].nxsub,
                                self.sim.config.p_tel.cobs, returnquiver=True
                        )  # Preparing mesh and vector for display
                        self.ui.wao_rtcWindowMPL.canvas.axes.quiver(
                                x, y, vx, vy, pivot='mid')
                        self.ui.wao_rtcWindowMPL.canvas.draw()
                        self.currentViewSelected = self.imgType

                        return

                if (self.sim.dms):
                    if (self.imgType == "Phase - DM"):
                        self.setupDisp("pg")
                        dm_type = self.sim.config.p_dms[self.numberSelected].type
                        alt = self.sim.config.p_dms[self.numberSelected].alt
                        data = self.sim.dms.get_dm(dm_type, alt)

                        if (self.imgType != self.currentViewSelected):
                            self.p1.setRange(xRange=(0, data.shape[0]),
                                             yRange=(0, data.shape[1]))
                        self.currentViewSelected = self.imgType
                        self.SRcircleDM[self.numberSelected].show()
                if (self.sim.tar):
                    if (self.imgType == "Phase - Target"):
                        self.setupDisp("pg")
                        data = self.sim.tar.get_phase(self.numberSelected)
                        if (self.imgType != self.currentViewSelected):
                            self.p1.setRange(xRange=(0, data.shape[0]),
                                             yRange=(0, data.shape[1]))
                        self.currentViewSelected = self.imgType
                        self.SRcircleTarget[self.numberSelected].show()

                    if (self.imgType == "PSF SE"):
                        self.setupDisp("pg")
                        data = self.sim.tar.get_image(self.numberSelected, b"se")
                        if (self.ui.actionPSF_Log_Scale.isChecked()):
                            if np.any(data <= 0):
                                warnings.warn(
                                        "\nZeros founds, filling with min nonzero value.\n"
                                )
                                data[data <= 0] = np.min(data[data > 0])
                            data = np.log10(data)

                        if (not self.SRCrossX):
                            Delta = 5
                            self.SRCrossX = pg.PlotCurveItem(
                                    np.array([
                                            data.shape[0] / 2 + 0.5 - Delta,
                                            data.shape[0] / 2 + 0.5 + Delta
                                    ]),
                                    np.array([
                                            data.shape[1] / 2 + 0.5,
                                            data.shape[1] / 2 + 0.5
                                    ]), pen='r')
                            self.SRCrossY = pg.PlotCurveItem(
                                    np.array([
                                            data.shape[0] / 2 + 0.5,
                                            data.shape[0] / 2 + 0.5
                                    ]),
                                    np.array([
                                            data.shape[1] / 2 + 0.5 - Delta,
                                            data.shape[1] / 2 + 0.5 + Delta
                                    ]), pen='r')
                            # Put image in plot area
                            self.p1.addItem(self.SRCrossX)
                            # Put image in plot area
                            self.p1.addItem(self.SRCrossY)

                        if (self.imgType != self.currentViewSelected):
                            zoom = 50
                            self.SRCrossX.show()
                            self.SRCrossY.show()
                            self.p1.setRange(
                                    xRange=(data.shape[0] / 2 + 0.5 - zoom,
                                            data.shape[0] / 2 + 0.5 + zoom),
                                    yRange=(data.shape[1] / 2 + 0.5 - zoom,
                                            data.shape[1] / 2 + 0.5 + zoom), )
                        self.currentViewSelected = self.imgType

                    if (self.imgType == "PSF LE"):
                        self.setupDisp("pg")
                        data = self.sim.tar.get_image(self.numberSelected, b"le")
                        if (self.ui.actionPSF_Log_Scale.isChecked()):
                            data = np.log10(data)
                        if (not self.SRCrossX):
                            Delta = 5
                            self.SRCrossX = pg.PlotCurveItem(
                                    np.array([
                                            data.shape[0] / 2 + 0.5 - Delta,
                                            data.shape[0] / 2 + 0.5 + Delta
                                    ]),
                                    np.array([
                                            data.shape[1] / 2 + 0.5,
                                            data.shape[1] / 2 + 0.5
                                    ]), pen='r')
                            self.SRCrossY = pg.PlotCurveItem(
                                    np.array([
                                            data.shape[0] / 2 + 0.5,
                                            data.shape[0] / 2 + 0.5
                                    ]),
                                    np.array([
                                            data.shape[1] / 2 + 0.5 - Delta,
                                            data.shape[1] / 2 + 0.5 + Delta
                                    ]), pen='r')

                            # Put image in plot area
                            self.p1.addItem(self.SRCrossX)
                            # Put image in plot area
                            self.p1.addItem(self.SRCrossY)
                        if (self.imgType != self.currentViewSelected):
                            zoom = 50
                            self.p1.setRange(xRange=(data.shape[0] / 2 + 0.5 - zoom,
                                                     data.shape[0] / 2 + 0.5 + zoom),
                                             yRange=(data.shape[1] / 2 + 0.5 - zoom,
                                                     data.shape[1] / 2 + 0.5 + zoom))
                            self.SRCrossX.show()
                            self.SRCrossY.show()

                        self.currentViewSelected = self.imgType

                if (data is not None):
                    autoscale = self.ui.actionAuto_Scale.isChecked()
                    if (autoscale):
                        # inits levels
                        self.hist.setLevels(data.min(), data.max())
                    self.img.setImage(data, autoLevels=autoscale)
                    # self.p1.autoRange()
            finally:
                self.loopLock.release()

    def loopOnce(self) -> None:
        if not self.loopLock.acquire(False):
            # print("Display locked")
            return
        else:
            try:
                start = time.time()
                self.sim.next(see_atmos=self.see_atmos)
                loopTime = time.time() - start

                refreshDisplayTime = 1. / self.ui.wao_frameRate.value()

                if (time.time() - self.refreshTime > refreshDisplayTime):
                    signal_le = ""
                    signal_se = ""
                    for t in range(self.sim.config.p_target.ntargets):
                        self.sim.tar.comp_image(t)
                        SR = self.sim.tar.get_strehl(t)
                        if (t == self.numberSelected):  # Plot on the wfs selected
                            self.updateSRDisplay(SR[1], SR[0], self.sim.iter)
                        signal_se += "%1.2f   " % SR[0]
                        signal_le += "%1.2f   " % SR[1]

                    currentFreq = 1 / loopTime
                    refreshFreq = 1 / (time.time() - self.refreshTime)

                    self.ui.wao_strehlSE.setText(signal_se)
                    self.ui.wao_strehlLE.setText(signal_le)
                    self.ui.wao_currentFreq.setValue(currentFreq)

                    if (self.dispStatsInTerminal):
                        self.printInPlace(
                                "iter #%d SR: (L.E, S.E.)= (%s, %s) running at %4.1fHz (real %4.1fHz)"
                                % (self.sim.iter, signal_le, signal_se, refreshFreq,
                                   currentFreq))

                    self.refreshTime = start

            finally:
                self.loopLock.release()

    def printInPlace(self, text: str) -> None:
        # This seems to trigger the GUI and keep it responsive
        print(text + "\r", end=' ')
        sys.stdout.flush()

    def run(self):
        self.loopOnce()
        if not self.ui.wao_forever.isChecked():
            self.nbiter -= 1
        if self.nbiter > 0 and not self.stop:
            QTimer.singleShot(0, self.run)  # Update loop
        else:
            self.ui.wao_run.setChecked(False)


class WorkerThread(QThread):
    jobFinished = pyqtSignal('PyQt_PyObject')

    def __init__(self, parentThread: QObject, parentLoop: Callable) -> None:
        QThread.__init__(self, parentThread)
        self.loopFunc = parentLoop

    def run(self) -> None:
        self.running = True
        self.loopFunc()
        success = True
        self.jobFinished.emit(success)

    def stop(self) -> None:
        self.running = False
        pass

    def cleanUp(self) -> None:
        pass


if __name__ == '__main__':
    arguments = docopt(__doc__)
    app = QtWidgets.QApplication(sys.argv)
    app.setStyle('cleanlooks')
    wao = widgetAOWindow(arguments["<parameters_filename>"], BRAMA=arguments["--brama"],
                         expert=arguments["--expert"])
    wao.show()
