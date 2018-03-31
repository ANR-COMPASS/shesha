"""Widget to simulate a closed loop

Usage:
  widget_ao.py [<parameters_filename>] [options]

with 'parameters_filename' the path to the parameters file

Options:
  -h --help          Show this help message and exit
  --brahma            Distribute data with BRAHMA
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

BenchWindowTemplate, BenchClassTemplate = loadUiType(
        os.environ["SHESHA_ROOT"] + "/widgets/widget_bench.ui")  # type: type, type

from widget_base import WidgetBase
from supervisor.benchSupervisor import BenchSupervisor
import matplotlib.pyplot as plt

# For debug
# from IPython.core.debugger import Pdb
# then add this line to create a breakpoint
# Pdb().set_trace()


class widgetBenchWindow(BenchClassTemplate, WidgetBase):

    def __init__(self, configFile: Any=None, BRAHMA: bool=False,
                 devices: str=None) -> None:
        WidgetBase.__init__(self)
        BenchClassTemplate.__init__(self)

        self.BRAHMA = BRAHMA
        self.devices = devices

        self.uiBench = BenchWindowTemplate()
        self.uiBench.setupUi(self)

        #############################################################
        #                   ATTRIBUTES                              #
        #############################################################

        self.supervisor = None
        self.config = None
        self.stop = False  # type: bool  # Request quit

        self.uiBench.wao_nbiters.setValue(1000)  # Default GUI nIter box value
        self.nbiter = self.uiBench.wao_nbiters.value()
        self.refreshTime = 0  # type: float  # System time at last display refresh
        self.loopThread = None  # type: QThread
        self.assistant = None  # type: Any

        #############################################################
        #                 CONNECTED BUTTONS                         #
        #############################################################
        # Default path for config files
        self.defaultParPath = os.environ["SHESHA_ROOT"] + "/data/par/bench-config"
        self.defaultAreaPath = os.environ["SHESHA_ROOT"] + "/data/layouts"
        self.loadDefaultConfig()

        self.uiBench.wao_run.setCheckable(True)
        self.uiBench.wao_run.clicked[bool].connect(self.aoLoopClicked)
        self.uiBench.wao_openLoop.setCheckable(True)
        self.uiBench.wao_openLoop.clicked[bool].connect(self.aoLoopOpen)
        self.uiBench.wao_next.clicked.connect(self.loopOnce)

        self.uiBench.wao_forever.stateChanged.connect(self.updateForever)

        self.dispStatsInTerminal = False

        self.uiBench.wao_run.setDisabled(True)
        self.uiBench.wao_next.setDisabled(True)
        self.uiBench.wao_unzoom.setDisabled(True)

        self.addDockWidget(Qt.DockWidgetArea(1), self.uiBase.wao_ConfigDock)
        self.addDockWidget(Qt.DockWidgetArea(1), self.uiBase.wao_DisplayDock)
        self.uiBase.wao_ConfigDock.setFloating(False)
        self.uiBase.wao_DisplayDock.setFloating(False)

        self.adjustSize()

        if configFile is not None:
            self.uiBase.wao_selectConfig.clear()
            self.uiBase.wao_selectConfig.addItem(configFile)
            self.loadConfig()
            self.initConfig()

    #############################################################
    #                       METHODS                             #
    #############################################################

    # def updateStatsInTerminal(self, state):
    #     self.dispStatsInTerminal = state

    def updateForever(self, state):
        self.uiBench.wao_nbiters.setDisabled(state)

    def add_dispDock(self, name: str, parent, type: str="pg_image") -> None:
        d = super().add_dispDock(name, parent, type)
        if type == "SR":
            d.addWidget(self.uiBench.wao_Strehl)

    def loadConfig(self) -> None:
        '''
            Callback when 'LOAD' button is hit
        '''
        super().loadConfig()
        configFile = str(self.uiBase.wao_selectConfig.currentText())
        sys.path.insert(0, self.defaultParPath)

        self.supervisor = BenchSupervisor(configFile, self.BRAHMA)
        self.config = self.supervisor.getConfig()

        # if self.devices:
        #     self.config.p_loop.set_devices([
        #             int(device) for device in self.devices.split(",")
        #     ])

        try:
            sys.path.remove(self.defaultParPath)
        except:
            pass

        self.nwfs = len(self.config.p_wfss)
        for wfs in range(self.nwfs):
            name = 'slpComp_%d' % wfs
            self.add_dispDock(name, self.wao_graphgroup_cb, "MPL")
            name = 'wfs_%d' % wfs
            self.add_dispDock(name, self.wao_imagesgroup_cb)

        self.uiBench.wao_run.setDisabled(True)
        self.uiBench.wao_next.setDisabled(True)
        self.uiBench.wao_unzoom.setDisabled(True)

        self.uiBase.wao_init.setDisabled(False)

        if (hasattr(self.config, "layout")):
            area_filename = self.defaultAreaPath + "/" + self.config.layout + ".area"
            self.loadArea(filename=area_filename)

        self.adjustSize()

    def aoLoopClicked(self, pressed: bool) -> None:
        if pressed:
            self.stop = False
            self.refreshTime = time.time()
            self.nbiter = self.uiBench.wao_nbiters.value()
            if self.dispStatsInTerminal:
                if self.uiBench.wao_forever.isChecked():
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
        # self.uiBench.wao_deviceNumber.setDisabled(True)
        # self.config.p_loop.devices = self.uiBench.wao_deviceNumber.value()  # using GUI value
        # gpudevice = "ALL"  # using all GPU avalaible
        # gpudevice = np.array([2, 3], dtype=np.int32)
        # gpudevice = np.arange(4, dtype=np.int32) # using 4 GPUs: 0-3
        # gpudevice = 0  # using 1 GPU : 0
        self.supervisor.initConfig()

    def initConfigFinished(self) -> None:
        # Thread naga context reload:
        self.supervisor.forceContext()

        print(self.supervisor)

        self.updateDisplay()

        self.uiBench.wao_run.setDisabled(False)
        self.uiBench.wao_next.setDisabled(False)
        self.uiBench.wao_openLoop.setDisabled(False)
        self.uiBench.wao_unzoom.setDisabled(False)

        super().initConfigFinished()

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
                        if "wfs" in key:
                            data = self.supervisor.getRawWFSImage(index)
                        if (data is not None):
                            autoscale = True  # self.uiBench.actionAuto_Scale.isChecked()
                            # if (autoscale):
                            #     # inits levels
                            #     self.hist.setLevels(data.min(), data.max())
                            self.imgs[key].setImage(data, autoLevels=autoscale)
                            # self.p1.autoRange()
                        elif "slp" in key:  # Slope display
                            self.imgs[key].canvas.axes.clear()
                            if "Comp" in key:
                                centroids = self.supervisor.getSlope()
                                nvalid = [2 * o._nvalid for o in self.config.p_wfss]
                                ind = np.sum(nvalid[:index], dtype=np.int32)
                                # if (self.config.p_wfss[index].type ==
                                #             scons.WFSType.PYRHR):
                                #     #TODO: DEBUG...
                                #     plpyr(centroids[ind:ind + nvalid[index]],
                                #           self.config.p_wfs0._isvalid)
                                # else:
                                #     x, y, vx, vy = plsh(
                                #             centroids[ind:ind + nvalid[index]],
                                #             self.config.p_wfss[index].nxsub,
                                #             self.config.p_tel.cobs, returnquiver=True
                                #     )  # Preparing mesh and vector for display
                                # self.imgs[key].canvas.axes.quiver(
                                #         x, y, vx, vy, pivot='mid')
                                # plt.plot(centroids)
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
                self.supervisor.singleNext()
                loopTime = time.time() - start

                refreshDisplayTime = 1. / self.uiBase.wao_frameRate.value()

                if (time.time() - self.refreshTime > refreshDisplayTime):
                    currentFreq = 1 / loopTime
                    refreshFreq = 1 / (time.time() - self.refreshTime)

                    self.uiBench.wao_currentFreq.setValue(currentFreq)

                    self.refreshTime = start
            except:
                print("error!!")
            finally:
                self.loopLock.release()

    def printInPlace(self, text: str) -> None:
        # This seems to trigger the GUI and keep it responsive
        print(text + "\r", end=' ')
        sys.stdout.flush()

    def run(self):
        self.loopOnce()
        if not self.uiBench.wao_forever.isChecked():
            self.nbiter -= 1
        if self.nbiter > 0 and not self.stop:
            QTimer.singleShot(0, self.run)  # Update loop
        else:
            self.uiBench.wao_run.setChecked(False)


if __name__ == '__main__':
    arguments = docopt(__doc__)
    app = QtWidgets.QApplication(sys.argv)
    app.setStyle('cleanlooks')
    wao = widgetBenchWindow(arguments["<parameters_filename>"],
                            BRAHMA=arguments["--brahma"], devices=arguments["--devices"])
    wao.show()
