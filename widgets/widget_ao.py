"""Widget to simulate a closed loop

Usage:
  widget_ao.py [<parameters_filename>] [options]

with 'parameters_filename' the path to the parameters file

Options:
  -h --help          Show this help message and exit
  --brahma            Distribute data with BRAHMA
  --expert           Display expert panel
  -d, --devices devices      Specify the devices
"""

import os, sys
import numpy as np
import time

import pyqtgraph as pg
from pyqtgraph.dockarea import Dock, DockArea

from tools import plsh, plpyr

import warnings

from PyQt5 import QtGui, QtWidgets
from PyQt5.uic import loadUiType
from PyQt5.QtCore import QThread, QTimer, Qt

from subprocess import Popen, PIPE

from typing import Any, Dict, Tuple, Callable, List

from docopt import docopt
from collections import deque

from matplotlibwidget import MatplotlibWidget

AOWindowTemplate, AOClassTemplate = loadUiType(
        os.environ["SHESHA_ROOT"] + "/widgets/widget_ao.ui")  # type: type, type

from widget_base import WidgetBase
from widget_ao_expert import WidgetAOExpert
from supervisor.compassSupervisor import CompassSupervisor, scons

# For debug
# from IPython.core.debugger import Pdb
# then add this line to create a breakpoint
# Pdb().set_trace()


class widgetAOWindow(AOClassTemplate, WidgetBase):

    def __init__(self, configFile: Any=None, BRAHMA: bool=False, expert: bool=False,
                 devices: str=None) -> None:
        WidgetBase.__init__(self)
        AOClassTemplate.__init__(self)

        self.BRAHMA = BRAHMA
        self.rollingWindow = 100
        self.SRLE = deque(maxlen=self.rollingWindow)
        self.SRSE = deque(maxlen=self.rollingWindow)
        self.numiter = deque(maxlen=self.rollingWindow)
        self.expert = expert
        self.devices = devices

        self.uiAO = AOWindowTemplate()
        self.uiAO.setupUi(self)

        #############################################################
        #                   ATTRIBUTES                              #
        #############################################################

        self.supervisor = None
        self.config = None
        self.stop = False  # type: bool  # Request quit

        self.uiAO.wao_nbiters.setValue(1000)  # Default GUI nIter box value
        self.nbiter = self.uiAO.wao_nbiters.value()
        self.refreshTime = 0  # type: float  # System time at last display refresh
        self.loopThread = None  # type: QThread
        self.assistant = None  # type: Any
        self.see_atmos = 1

        #############################################################
        #                 CONNECTED BUTTONS                         #
        #############################################################
        # Default path for config files
        self.defaultParPath = os.environ["SHESHA_ROOT"] + "/data/par/par4bench"
        self.defaultAreaPath = os.environ["SHESHA_ROOT"] + "/data/layouts"
        self.loadDefaultConfig()

        self.uiAO.wao_run.setCheckable(True)
        self.uiAO.wao_run.clicked[bool].connect(self.aoLoopClicked)
        self.uiAO.wao_openLoop.setCheckable(True)
        self.uiAO.wao_openLoop.clicked[bool].connect(self.aoLoopOpen)
        self.uiAO.wao_next.clicked.connect(self.loopOnce)
        self.uiAO.wao_resetSR.clicked.connect(self.resetSR)
        # self.uiAO.wao_actionHelp_Contents.triggered.connect(self.on_help_triggered)

        self.uiAO.wao_allTarget.stateChanged.connect(self.updateAllTarget)
        self.uiAO.wao_forever.stateChanged.connect(self.updateForever)

        self.uiAO.wao_atmosphere.clicked[bool].connect(self.set_see_atmos)
        self.dispStatsInTerminal = False
        self.uiAO.wao_clearSR.clicked.connect(self.clearSR)
        # self.uiAO.actionStats_in_Terminal.toggled.connect(self.updateStatsInTerminal)

        self.uiAO.wao_run.setDisabled(True)
        self.uiAO.wao_next.setDisabled(True)
        self.uiAO.wao_unzoom.setDisabled(True)
        self.uiAO.wao_resetSR.setDisabled(True)

        p1 = self.uiAO.wao_SRPlotWindow.addPlot(title='SR evolution')
        self.curveSRSE = p1.plot(pen=(255, 0, 0), symbolBrush=(255, 0, 0), name="SR SE")
        self.curveSRLE = p1.plot(pen=(0, 0, 255), symbolBrush=(0, 0, 255), name="SR LE")

        self.SRCrossX = {}  # type: Dict[str, pg.ScatterPlotItem]
        self.SRCrossY = {}  # type: Dict[str, pg.ScatterPlotItem]
        self.SRcircles = {}  # type: Dict[str, pg.ScatterPlotItem]

        self.natm = 0
        self.nwfs = 0
        self.ndm = 0
        self.ntar = 0
        self.PSFzoom = 50
        self.uiAO.wao_SRDock.setVisible(False)

        self.addDockWidget(Qt.DockWidgetArea(1), self.uiBase.wao_ConfigDock)
        self.addDockWidget(Qt.DockWidgetArea(1), self.uiBase.wao_DisplayDock)
        self.uiBase.wao_ConfigDock.setFloating(False)
        self.uiBase.wao_DisplayDock.setFloating(False)

        if expert:
            self.expertWidget = WidgetAOExpert()
            # self.expertWidget.setupUi(self)
            self.addDockWidget(
                    Qt.DockWidgetArea(1), self.expertWidget.uiExpert.wao_expertDock)
            self.expertWidget.uiExpert.wao_expertDock.setFloating(False)

        self.adjustSize()

        if configFile is not None:
            self.uiBase.wao_selectConfig.clear()
            self.uiBase.wao_selectConfig.addItem(configFile)
            self.loadConfig()
            self.initConfig()

    # def on_help_triggered(self, i: Any=None) -> None:
    #     if i is None:
    #         return
    #     if not self.assistant or \
    #        not self.assistant.poll():

    #         helpcoll = os.environ["COMPASS_ROOT"] + "/doc/COMPASS.qhc"
    #         cmd = "assistant -enableRemoteControl -collectionFile %s" % helpcoll
    #         self.assistant = Popen(cmd, shell=True, stdin=PIPE)

    #############################################################
    #                       METHODS                             #
    #############################################################

    # def updateStatsInTerminal(self, state):
    #     self.dispStatsInTerminal = state

    def updateAllTarget(self, state):
        self.uiAO.wao_resetSR_tarNum.setDisabled(state)

    def updateForever(self, state):
        self.uiAO.wao_nbiters.setDisabled(state)

    def set_see_atmos(self, atmos):
        self.see_atmos = atmos

    def resetSR(self) -> None:
        if self.uiAO.wao_allTarget.isChecked():
            for t in range(self.config.p_target.ntargets):
                self.supervisor.resetStrehl(t)
        else:
            tarnum = self.uiAO.wao_resetSR_tarNum.value()
            print("Reset SR on target %d" % tarnum)
            self.supervisor.resetStrehl(tarnum)

    def add_dispDock(self, name: str, parent, type: str="pg_image") -> None:
        d = super().add_dispDock(name, parent, type)
        if type == "SR":
            d.addWidget(self.uiAO.wao_Strehl)

    def loadConfig(self, *, ISupervisor=CompassSupervisor) -> None:
        '''
            Callback when 'LOAD' button is hit
            * required to catch positionals, as by default
            if a positional is allowed the QPushButton will send a boolean value
            and hence overwrite ISupervisor...
        '''
        super().loadConfig()
        for key, pgpl in self.SRcircles.items():
            self.viewboxes[key].removeItem(pgpl)

        for key, pgpl in self.SRCrossX.items():
            self.viewboxes[key].removeItem(pgpl)

        for key, pgpl in self.SRCrossY.items():
            self.viewboxes[key].removeItem(pgpl)

        configFile = str(self.uiBase.wao_selectConfig.currentText())
        sys.path.insert(0, self.defaultParPath)

        self.supervisor = ISupervisor(configFile, self.BRAHMA)
        self.config = self.supervisor.getConfig()

        if self.devices:
            self.config.p_loop.set_devices([
                    int(device) for device in self.devices.split(",")
            ])

        try:
            sys.path.remove(self.defaultParPath)
        except:
            pass

        self.SRcircles.clear()
        self.SRCrossX.clear()
        self.SRCrossY.clear()

        self.natm = len(self.config.p_atmos.alt)
        for atm in range(self.natm):
            name = 'atm_%d' % atm
            self.add_dispDock(name, self.wao_phasesgroup_cb)

        self.nwfs = len(self.config.p_wfss)
        for wfs in range(self.nwfs):
            name = 'wfs_%d' % wfs
            self.add_dispDock(name, self.wao_phasesgroup_cb)
            name = 'slpComp_%d' % wfs
            self.add_dispDock(name, self.wao_graphgroup_cb, "MPL")
            name = 'slpGeom_%d' % wfs
            self.add_dispDock(name, self.wao_graphgroup_cb, "MPL")
            if self.config.p_wfss[wfs].type == scons.WFSType.SH:
                name = 'SH_%d' % wfs
                self.add_dispDock(name, self.wao_imagesgroup_cb)
            elif self.config.p_wfss[wfs].type == scons.WFSType.PYRHR:
                name = 'pyrHR_%d' % wfs
                self.add_dispDock(name, self.wao_imagesgroup_cb)
                name = 'pyrLR_%d' % wfs
                self.add_dispDock(name, self.wao_imagesgroup_cb)
            else:
                raise "Analyser unknown"

        self.ndm = len(self.config.p_dms)
        for dm in range(self.ndm):
            name = 'dm_%d' % dm
            w = QtGui.QCheckBox(name)
            self.add_dispDock(name, self.wao_phasesgroup_cb)

        self.ntar = self.config.p_target.ntargets
        for tar in range(self.ntar):
            name = 'tar_%d' % tar
            self.add_dispDock(name, self.wao_phasesgroup_cb)
        for tar in range(self.ntar):
            name = 'psfSE_%d' % tar
            self.add_dispDock(name, self.wao_imagesgroup_cb)
        for tar in range(self.ntar):
            name = 'psfLE_%d' % tar
            self.add_dispDock(name, self.wao_imagesgroup_cb)

        self.add_dispDock("Strehl", self.wao_graphgroup_cb, "SR")

        self.uiAO.wao_resetSR_tarNum.setValue(0)
        self.uiAO.wao_resetSR_tarNum.setMaximum(self.config.p_target.ntargets - 1)

        self.uiAO.wao_dispSR_tar.setValue(0)
        self.uiAO.wao_dispSR_tar.setMaximum(self.config.p_target.ntargets - 1)

        self.uiAO.wao_run.setDisabled(True)
        self.uiAO.wao_next.setDisabled(True)
        self.uiAO.wao_unzoom.setDisabled(True)
        self.uiAO.wao_resetSR.setDisabled(True)

        self.uiBase.wao_init.setDisabled(False)

        if self.expert:
            self.expertWidget.setSupervisor(self.supervisor)
            self.expertWidget.updatePanels()

        if (hasattr(self.config, "layout")):
            area_filename = self.defaultAreaPath + "/" + self.config.layout + ".area"
            self.loadArea(filename=area_filename)

        self.adjustSize()

    def aoLoopClicked(self, pressed: bool) -> None:
        if pressed:
            self.stop = False
            self.refreshTime = time.time()
            self.nbiter = self.uiAO.wao_nbiters.value()
            if self.dispStatsInTerminal:
                if self.uiAO.wao_forever.isChecked():
                    print("LOOP STARTED")
                else:
                    print("LOOP STARTED FOR %d iterations" % self.nbiter)
            self.run()
        else:
            self.stop = True

    def aoLoopOpen(self, pressed: bool) -> None:
        if (pressed):
            self.supervisor.openLoop()
        else:
            self.supervisor.closeLoop()

    def initConfig(self) -> None:
        self.supervisor.clearInitSim()
        super().initConfig()

    def initConfigThread(self) -> None:
        self.uiAO.wao_deviceNumber.setDisabled(True)
        # self.config.p_loop.devices = self.uiAO.wao_deviceNumber.value()  # using GUI value
        # gpudevice = "ALL"  # using all GPU avalaible
        # gpudevice = np.array([2, 3], dtype=np.int32)
        # gpudevice = np.arange(4, dtype=np.int32) # using 4 GPUs: 0-3
        # gpudevice = 0  # using 1 GPU : 0
        self.supervisor.initConfig()

    def initConfigFinished(self) -> None:
        # Thread naga context reload:
        self.supervisor.forceContext()

        for i in range(self.natm):
            key = "atm_%d" % i
            data = self.supervisor.getAtmScreen(self.config.p_atmos.alt[i])
            cx, cy = self.circleCoords(self.config.p_geom.pupdiam / 2, 1000,
                                       data.shape[0], data.shape[1])
            self.SRcircles[key] = pg.ScatterPlotItem(cx, cy, pen='r', size=1)
            self.viewboxes[key].addItem(self.SRcircles[key])
            self.SRcircles[key].setPoints(cx, cy)

        for i in range(self.nwfs):
            key = "wfs_%d" % i
            data = self.supervisor.getWfsPhase(i)
            cx, cy = self.circleCoords(self.config.p_geom.pupdiam / 2, 1000,
                                       data.shape[0], data.shape[1])
            self.SRcircles[key] = pg.ScatterPlotItem(cx, cy, pen='r', size=1)
            self.viewboxes[key].addItem(self.SRcircles[key])
            self.SRcircles[key].setPoints(cx, cy)
            key = 'slpComp_%d' % i
            key = 'slpGeom_%d' % i

            if self.config.p_wfss[i].type == scons.WFSType.SH:
                key = "SH_%d" % i
                self.addSHGrid(self.docks[key].widgets[0],
                               self.config.p_wfss[i].get_validsub(), 8, 8)

        for i in range(self.ndm):
            key = "dm_%d" % i
            dm_type = self.config.p_dms[i].type
            alt = self.config.p_dms[i].alt
            data = self.supervisor.getDmPhase(dm_type, alt)
            cx, cy = self.circleCoords(self.config.p_geom.pupdiam / 2, 1000,
                                       data.shape[0], data.shape[1])
            self.SRcircles[key] = pg.ScatterPlotItem(cx, cy, pen='r', size=1)
            self.viewboxes[key].addItem(self.SRcircles[key])
            self.SRcircles[key].setPoints(cx, cy)

        for i in range(self.config.p_target.ntargets):
            key = "tar_%d" % i
            data = self.supervisor.getTarPhase(i)
            cx, cy = self.circleCoords(self.config.p_geom.pupdiam / 2, 1000,
                                       data.shape[0], data.shape[1])
            self.SRcircles[key] = pg.ScatterPlotItem(cx, cy, pen='r', size=1)
            self.viewboxes[key].addItem(self.SRcircles[key])
            self.SRcircles[key].setPoints(cx, cy)

            data = self.supervisor.getTarImage(i)
            for psf in ["psfSE_", "psfLE_"]:
                key = psf + str(i)
                Delta = 5
                self.SRCrossX[key] = pg.PlotCurveItem(
                        np.array([
                                data.shape[0] / 2 + 0.5 - Delta,
                                data.shape[0] / 2 + 0.5 + Delta
                        ]),
                        np.array([data.shape[1] / 2 + 0.5, data.shape[1] / 2 + 0.5]),
                        pen='r')
                self.SRCrossY[key] = pg.PlotCurveItem(
                        np.array([data.shape[0] / 2 + 0.5, data.shape[0] / 2 + 0.5]),
                        np.array([
                                data.shape[1] / 2 + 0.5 - Delta,
                                data.shape[1] / 2 + 0.5 + Delta
                        ]), pen='r')
                # Put image in plot area
                self.viewboxes[key].addItem(self.SRCrossX[key])
                # Put image in plot area
                self.viewboxes[key].addItem(self.SRCrossY[key])

        print(self.supervisor)

        if self.expert:
            self.expertWidget.displayRtcMatrix()

        self.updateDisplay()

        self.uiAO.wao_run.setDisabled(False)
        self.uiAO.wao_next.setDisabled(False)
        self.uiAO.wao_openLoop.setDisabled(False)
        self.uiAO.wao_unzoom.setDisabled(False)
        self.uiAO.wao_resetSR.setDisabled(False)

        super().initConfigFinished()

    def circleCoords(self, ampli: float, npts: int, datashape0: int,
                     datashape1: int) -> Tuple[float, float]:
        cx = ampli * np.sin((np.arange(npts) + 1) * 2. * np.pi / npts) + datashape0 / 2
        cy = ampli * np.cos((np.arange(npts) + 1) * 2. * np.pi / npts) + datashape1 / 2
        return cx, cy

    def clearSR(self):
        self.SRLE = deque(maxlen=20)
        self.SRSE = deque(maxlen=20)
        self.numiter = deque(maxlen=20)

    def updateSRDisplay(self, SRLE, SRSE, numiter):
        self.SRLE.append(SRLE)
        self.SRSE.append(SRSE)
        self.numiter.append(numiter)
        self.curveSRSE.setData(self.numiter, self.SRSE)
        self.curveSRLE.setData(self.numiter, self.SRLE)

    def updateDisplay(self) -> None:
        if (self.supervisor is None) or (not self.supervisor.isInit()) or (
                not self.uiBase.wao_Display.isChecked()):
            # print("Widget not fully initialized")
            return
        if not self.loopLock.acquire(False):
            return
        else:
            try:
                for key, dock in self.docks.items():
                    if key == "Strehl":
                        continue
                    elif dock.isVisible():
                        index = int(key.split("_")[-1])
                        data = None
                        if "atm" in key:
                            data = self.supervisor.getAtmScreen(
                                    self.config.p_atmos.alt[index])
                        if "wfs" in key:
                            data = self.supervisor.getWfsPhase(index)
                        if "dm" in key:
                            dm_type = self.config.p_dms[index].type
                            alt = self.config.p_dms[index].alt
                            data = self.supervisor.getDmPhase(dm_type, alt)
                        if "tar" in key:
                            data = self.supervisor.getTarPhase(index)
                        if "psfLE" in key:
                            data = self.supervisor.getTarImage(index, "le")
                        if "psfSE" in key:
                            data = self.supervisor.getTarImage(index, "se")

                        if "psf" in key:
                            if (self.uiAO.actionPSF_Log_Scale.isChecked()):
                                if np.any(data <= 0):
                                    warnings.warn(
                                            "\nZeros founds, filling with min nonzero value.\n"
                                    )
                                    data[data <= 0] = np.min(data[data > 0])
                                data = np.log10(data)

                            self.viewboxes[key].setRange(
                                    xRange=(data.shape[0] / 2 + 0.5 - self.PSFzoom,
                                            data.shape[0] / 2 + 0.5 + self.PSFzoom),
                                    yRange=(data.shape[1] / 2 + 0.5 - self.PSFzoom,
                                            data.shape[1] / 2 + 0.5 + self.PSFzoom), )
                        if "SH" in key:
                            data = self.supervisor.getRawWFSImage(index)
                        if "pyrLR" in key:
                            data = self.supervisor.getRawWFSImage(index)
                        if "pyrHR" in key:
                            data = self.supervisor.getPyrHRImage(index)

                        if (data is not None):
                            autoscale = True  # self.uiAO.actionAuto_Scale.isChecked()
                            # if (autoscale):
                            #     # inits levels
                            #     self.hist.setLevels(data.min(), data.max())
                            self.imgs[key].setImage(data, autoLevels=autoscale)
                            # self.p1.autoRange()
                        elif "slp" in key:  # Slope display
                            self.imgs[key].canvas.axes.clear()
                            if "Geom" in key:
                                slopes = self.supervisor.getSlopeGeom(index)
                                x, y, vx, vy = plsh(
                                        slopes, self.config.p_wfss[index].nxsub,
                                        self.config.p_tel.cobs, returnquiver=True
                                )  # Preparing mesh and vector for display
                                self.imgs[key].canvas.axes.quiver(
                                        x, y, vx, vy, pivot='mid')
                            if "Comp" in key:
                                centroids = self.supervisor.getSlope()
                                nmes = [
                                        2 * p_wfs._nvalid for p_wfs in self.config.p_wfss
                                ]
                                first_ind = np.sum(nmes[:index], dtype=np.int32)
                                if (self.config.p_wfss[index].type ==
                                            scons.WFSType.PYRHR):
                                    #TODO: DEBUG...
                                    plpyr(centroids[first_ind:first_ind + nmes[index]],
                                          np.stack([
                                                  wao.config.p_wfss[index]._validsubsx,
                                                  wao.config.p_wfss[index]._validsubsy
                                          ]))
                                else:
                                    x, y, vx, vy = plsh(
                                            centroids[first_ind:first_ind + nmes[index]],
                                            self.config.p_wfss[index].nxsub,
                                            self.config.p_tel.cobs, returnquiver=True
                                    )  # Preparing mesh and vector for display
                                self.imgs[key].canvas.axes.quiver(
                                        x, y, vx, vy, pivot='mid')
                            self.imgs[key].canvas.draw()

            finally:
                self.loopLock.release()

    def loopOnce(self) -> None:
        if not self.loopLock.acquire(False):
            print("Display locked")
            return
        else:
            try:
                start = time.time()
                self.supervisor.singleNext(showAtmos=self.see_atmos)
                loopTime = time.time() - start

                refreshDisplayTime = 1. / self.uiBase.wao_frameRate.value()

                if (time.time() - self.refreshTime > refreshDisplayTime):
                    signal_le = ""
                    signal_se = ""
                    for t in range(self.config.p_target.ntargets):
                        SR = self.supervisor.getStrehl(t)
                        # TODO: handle that !
                        if (t == self.uiAO.wao_dispSR_tar.value(
                        )):  # Plot on the wfs selected
                            self.updateSRDisplay(SR[1], SR[0],
                                                 self.supervisor.getFrameCounter())
                        signal_se += "%1.2f   " % SR[0]
                        signal_le += "%1.2f   " % SR[1]

                    currentFreq = 1 / loopTime
                    refreshFreq = 1 / (time.time() - self.refreshTime)

                    self.uiAO.wao_strehlSE.setText(signal_se)
                    self.uiAO.wao_strehlLE.setText(signal_le)
                    self.uiAO.wao_currentFreq.setValue(currentFreq)
                    if (self.dispStatsInTerminal):
                        self.printInPlace(
                                "iter #%d SR: (L.E, S.E.)= (%s, %s) running at %4.1fHz (real %4.1fHz)"
                                % (self.supervisor.getFrameCounter(), signal_le,
                                   signal_se, refreshFreq, currentFreq))

                    self.refreshTime = start
            except:
                print("error!!")
            finally:
                self.loopLock.release()

    def printInPlace(self, text: str) -> None:
        # This seems to trigger the GUI and keep it responsive
        print(text + "\r", end=' ')
        sys.stdout.flush()

    def run(self, stop=False):
        if stop:
            return
        self.loopOnce()
        if not self.uiAO.wao_forever.isChecked():
            self.nbiter -= 1
        if self.nbiter > 0 and not self.stop:
            QTimer.singleShot(0, self.run)  # Update loop
        else:
            self.uiAO.wao_run.setChecked(False)


if __name__ == '__main__':
    arguments = docopt(__doc__)
    app = QtWidgets.QApplication(sys.argv)
    app.setStyle('cleanlooks')
    wao = widgetAOWindow(arguments["<parameters_filename>"],
                         BRAHMA=arguments["--brahma"], expert=arguments["--expert"],
                         devices=arguments["--devices"])
    wao.show()
