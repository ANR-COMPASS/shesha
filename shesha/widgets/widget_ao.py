#!/usr/bin/env python
## @package   shesha.widgets.widget_ao
## @brief     Widget to simulate a closed loop
## @author    COMPASS Team <https://github.com/ANR-COMPASS>
## @version   5.4.3
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
Widget to simulate a closed loop

Usage:
  widget_ao.py [<parameters_filename>] [options]

with 'parameters_filename' the path to the parameters file

Options:
  -h --help          Show this help message and exit
  --cacao            Distribute data with cacao
  --expert           Display expert panel
  -d, --devices devices      Specify the devices
  -i, --interactive  keep the script interactive
"""

import os, sys

import numpy as np
import time

import pyqtgraph as pg

from shesha.util.tools import plsh, plpyr
from shesha.config import ParamConfig

try:
    from PyQt5 import QtWidgets
    from PyQt5.QtCore import Qt
except ModuleNotFoundError as e:
    try:    
        from PySide2 import QtWidgets
        from PySide2.QtCore import Qt
    except ModuleNotFoundError as e:
        raise ModuleNotFoundError("No module named 'PyQt5' or PySide2', please install one of them\nException raised: "+e.msg)

from typing import Any, Dict, Tuple

from docopt import docopt
from collections import deque

from shesha.widgets.widget_base import WidgetBase, uiLoader

AOWindowTemplate, AOClassTemplate = uiLoader('widget_ao')

from shesha.supervisor.compassSupervisor import CompassSupervisor, scons

# For debug
# from IPython.core.debugger import Pdb
# then add this line to create a breakpoint
# Pdb().set_trace()


class widgetAOWindow(AOClassTemplate, WidgetBase):

    def __init__(self, config_file: Any = None, cacao: bool = False,
                 expert: bool = False, devices: str = None,
                 hide_histograms: bool = False, twoStages: bool = False) -> None:
        WidgetBase.__init__(self, hide_histograms=hide_histograms)
        AOClassTemplate.__init__(self)
        self.twoStages = twoStages
        self.cacao = cacao
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
        self.assistant = None  # type: Any

        #############################################################
        #                 CONNECTED BUTTONS                         #
        #############################################################

        # Default path for config files
        self.defaultParPath = os.environ["SHESHA_ROOT"] + "/data/par/par4bench"
        self.defaultAreaPath = os.environ["SHESHA_ROOT"] + "/data/layouts"
        self.loadDefaultConfig()

        self.uiAO.wao_run.setCheckable(True)
        self.uiAO.wao_run.clicked[bool].connect(self.aoLoopClicked)
        self.uiAO.wao_open_loop.setCheckable(True)
        self.uiAO.wao_open_loop.clicked[bool].connect(self.aoLoopOpen)
        self.uiAO.wao_next.clicked.connect(self.loop_once)
        self.uiAO.wao_resetSR.clicked.connect(self.resetSR)
        self.uiAO.wao_resetCoro.clicked.connect(self.resetCoro)
        # self.uiAO.wao_actionHelp_Contents.triggered.connect(self.on_help_triggered)

        self.uiAO.wao_allTarget.stateChanged.connect(self.updateAllTarget)
        self.uiAO.wao_allCoro.stateChanged.connect(self.updateAllCoro)
        self.uiAO.wao_forever.stateChanged.connect(self.updateForever)

        self.uiAO.wao_atmosphere.clicked[bool].connect(self.enable_atmos)
        self.dispStatsInTerminal = False
        self.uiAO.wao_clearSR.clicked.connect(self.clearSR)
        # self.uiAO.actionStats_in_Terminal.toggled.connect(self.updateStatsInTerminal)

        self.uiAO.wao_run.setDisabled(True)
        self.uiAO.wao_next.setDisabled(True)
        self.uiAO.wao_unzoom.setDisabled(True)
        self.uiAO.wao_resetSR.setDisabled(True)
        self.uiAO.wao_resetCoro.setDisabled(True)
        self.uiAO.wao_allCoro.setDisabled(True)

        p1 = self.uiAO.wao_SRPlotWindow.addPlot(title='SR evolution')
        self.curveSRSE = p1.plot(pen=(255, 0, 0), symbolBrush=(255, 0, 0), name="SR SE")
        self.curveSRLE = p1.plot(pen=(0, 0, 255), symbolBrush=(0, 0, 255), name="SR LE")

        self.SRCrossX = {}  # type: Dict[str, pg.ScatterPlotItem]
        self.SRCrossY = {}  # type: Dict[str, pg.ScatterPlotItem]
        self.SRcircles = {}  # type: Dict[str, pg.ScatterPlotItem]
        self.PyrEdgeX = {}  # type: Dict[str, pg.ScatterPlotItem]
        self.PyrEdgeY = {}  # type: Dict[str, pg.ScatterPlotItem]

        self.natm = 0
        self.nwfs = 0
        self.ndm = 0
        self.ntar = 0
        self.PSFzoom = 50
        self.firstTime = 1
        self.uiAO.wao_SRDock.setVisible(False)

        self.addDockWidget(Qt.DockWidgetArea(1), self.uiBase.wao_ConfigDock)
        self.addDockWidget(Qt.DockWidgetArea(1), self.uiBase.wao_DisplayDock)
        self.uiBase.wao_ConfigDock.setFloating(False)
        self.uiBase.wao_DisplayDock.setFloating(False)

        if expert:
            from shesha.widgets.widget_ao_expert import WidgetAOExpert
            self.expertWidget = WidgetAOExpert()
            # self.expertWidget.setupUi(self)
            self.addDockWidget(
                    Qt.DockWidgetArea(1), self.expertWidget.uiExpert.wao_expertDock)
            self.expertWidget.uiExpert.wao_expertDock.setFloating(False)

        self.adjustSize()

        if config_file is not None:
            self.uiBase.wao_selectConfig.clear()
            self.uiBase.wao_selectConfig.addItem(config_file)
            self.load_config(config_file=config_file)
            self.init_config()

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

    def updateAllCoro(self, state):
        self.uiAO.wao_resetCoro_coroNum.setDisabled(state)

    def updateForever(self, state):
        self.uiAO.wao_nbiters.setDisabled(state)

    def enable_atmos(self, atmos):
        self.supervisor.atmos.enable_atmos(atmos)

    def resetSR(self) -> None:
        if self.uiAO.wao_allTarget.isChecked():
            for t in range(len(self.config.p_targets)):
                self.supervisor.target.reset_strehl(t)
        else:
            tarnum = self.uiAO.wao_resetSR_tarNum.value()
            print("Reset SR on target %d" % tarnum)
            self.supervisor.target.reset_strehl(tarnum)

    def resetCoro(self) -> None:
        # TODO Adapt for multiple corono 
        if self.uiAO.wao_allCoro.isChecked():
            for c in range(self.ncoro):
                self.supervisor.corono.reset()
        else:
            coroNum = self.uiAO.wao_resetCoro_coroNum.value()
            print("Reset Coro %d" % coroNum)
            self.supervisor.corono.reset()

    def add_dispDock(self, name: str, parent, type: str = "pg_image") -> None:
        d = WidgetBase.add_dispDock(self, name, parent, type)
        if type == "SR":
            d.addWidget(self.uiAO.wao_Strehl)

    def load_config(self, *args, config_file=None, supervisor=None, **kwargs) -> None:
        '''
            Callback when 'LOAD' button is hit
            * required to catch positionals, as by default
            if a positional is allowed the QPushButton will send a boolean value
            and hence overwrite supervisor...
        '''

        WidgetBase.load_config(self)
        for key, pgpl in self.SRcircles.items():
            self.viewboxes[key].removeItem(pgpl)

        for key, pgpl in self.SRCrossX.items():
            self.viewboxes[key].removeItem(pgpl)

        for key, pgpl in self.SRCrossY.items():
            self.viewboxes[key].removeItem(pgpl)

        for key, pgpl in self.PyrEdgeX.items():
            self.viewboxes[key].removeItem(pgpl)

        for key, pgpl in self.PyrEdgeY.items():
            self.viewboxes[key].removeItem(pgpl)

        if config_file is None:
            config_file = str(self.uiBase.wao_selectConfig.currentText())
            sys.path.insert(0, self.defaultParPath)

        if supervisor is None:
            self.config = ParamConfig(config_file)
        else:
            self.config = supervisor.get_config()

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
        self.PyrEdgeX.clear()
        self.PyrEdgeY.clear()
        self.nctrl = len(self.config.p_controllers)
        for ctrl in range(self.nctrl):
            self.add_dispDock("modalGains_"+str(ctrl), self.wao_graphgroup_cb, "MPL")

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
            elif self.config.p_wfss[
                    wfs].type == scons.WFSType.PYRHR or self.config.p_wfss[
                            wfs].type == scons.WFSType.PYRLR:
                name = 'pyrFocalPlane_%d' % wfs
                self.add_dispDock(name, self.wao_imagesgroup_cb)
                name = 'pyrHR_%d' % wfs
                self.add_dispDock(name, self.wao_imagesgroup_cb)
                name = 'pyrLR_%d' % wfs
                self.add_dispDock(name, self.wao_imagesgroup_cb)
            else:
                raise "Analyser unknown"

        self.ndm = len(self.config.p_dms)
        for dm in range(self.ndm):
            name = 'dm_%d' % dm
            self.add_dispDock(name, self.wao_phasesgroup_cb)

        self.ntar = len(self.config.p_targets)
        for tar in range(self.ntar):
            name = 'tar_%d' % tar
            self.add_dispDock(name, self.wao_phasesgroup_cb)
        for tar in range(self.ntar):
            name = 'psfSE_%d' % tar
            self.add_dispDock(name, self.wao_imagesgroup_cb)
        for tar in range(self.ntar):
            name = 'psfLE_%d' % tar
            self.add_dispDock(name, self.wao_imagesgroup_cb)
        if(self.config.p_coronos) is not None:
            self.ncoro = len(self.config.p_coronos)
        else:
            self.ncoro = 0
        for coro in range(self.ncoro):
            name = 'coroImageLE_%d' % coro
            self.add_dispDock(name, self.wao_imagesgroup_cb)
        for coro in range(self.ncoro):
            name = 'coroImageSE_%d' % coro
            self.add_dispDock(name, self.wao_imagesgroup_cb)
        for coro in range(self.ncoro):
            name = 'coroPSFLE_%d' % coro
            self.add_dispDock(name, self.wao_imagesgroup_cb)
        for coro in range(self.ncoro):
            name = 'coroPSFSE_%d' % coro
            self.add_dispDock(name, self.wao_imagesgroup_cb)
            
        self.add_dispDock("Strehl", self.wao_graphgroup_cb, "SR")
        for coro in range(self.ncoro):
            self.add_dispDock(f"ContrastLE_{coro}", self.wao_graphgroup_cb, "MPL")
            self.add_dispDock(f"ContrastSE_{coro}", self.wao_graphgroup_cb, "MPL")

        self.uiAO.wao_resetSR_tarNum.setValue(0)
        self.uiAO.wao_resetSR_tarNum.setMaximum(len(self.config.p_targets) - 1)

        self.uiAO.wao_resetCoro_coroNum.setValue(0)
        self.uiAO.wao_resetCoro_coroNum.setMaximum(self.ncoro - 1)

        self.uiAO.wao_dispSR_tar.setValue(0)
        self.uiAO.wao_dispSR_tar.setMaximum(len(self.config.p_targets) - 1)

        self.uiAO.wao_run.setDisabled(True)
        self.uiAO.wao_next.setDisabled(True)
        self.uiAO.wao_unzoom.setDisabled(True)
        self.uiAO.wao_resetSR.setDisabled(True)
        self.uiAO.wao_resetCoro.setDisabled(True)
        self.uiAO.wao_allCoro.setDisabled(True)

        self.uiBase.wao_init.setDisabled(False)

        if self.expert:
            self.expertWidget.setSupervisor(self.supervisor)
            self.expertWidget.updatePanels()

        if (hasattr(self.config._config, "layout")):
            area_filename = self.defaultAreaPath + "/" + self.config._config.layout + ".area"
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
            self.supervisor.rtc.close_loop()
            self.uiAO.wao_open_loop.setText("Open Loop")
        else:
            self.supervisor.rtc.open_loop()
            self.uiAO.wao_open_loop.setText("Close Loop")

    def init_config(self) -> None:
        if(self.twoStages):
            from shesha.supervisor.stageSupervisor import StageSupervisor, scons
            self.supervisor = StageSupervisor(self.config, cacao=self.cacao)
        else:
            self.supervisor = CompassSupervisor(self.config, cacao=self.cacao)
        WidgetBase.init_config(self)

    def init_configThread(self) -> None:
        self.uiAO.wao_deviceNumber.setDisabled(True)

    def init_configFinished(self) -> None:
        # Thread carmaWrap context reload:
        self.supervisor.force_context()

        for i in range(self.natm):
            key = "atm_%d" % i
            data = self.supervisor.atmos.get_atmos_layer(i)
            cx, cy = self.circleCoords(self.config.p_geom.pupdiam / 2, 1000,
                                       data.shape[0], data.shape[1])
            self.SRcircles[key] = pg.ScatterPlotItem(cx, cy, pen='r', size=1)
            self.viewboxes[key].addItem(self.SRcircles[key])
            self.SRcircles[key].setData(cx, cy)

        for i in range(self.nwfs):
            key = "wfs_%d" % i
            data = self.supervisor.wfs.get_wfs_phase(i)
            cx, cy = self.circleCoords(self.config.p_geom.pupdiam / 2, 1000,
                                       data.shape[0], data.shape[1])
            self.SRcircles[key] = pg.ScatterPlotItem(cx, cy, pen='r', size=1)
            self.viewboxes[key].addItem(self.SRcircles[key])
            self.SRcircles[key].setData(cx, cy)
            key = 'slpComp_%d' % i
            key = 'slpGeom_%d' % i

            # if self.config.p_wfss[i].type == scons.WFSType.SH:
            #     key = "SH_%d" % i
            #     self.addSHGrid(self.docks[key].widgets[0],
            #                    self.config.p_wfss[i].get_validsub(), 8, 8)

        for i in range(self.ndm):
            key = "dm_%d" % i
            dm_type = self.config.p_dms[i].type
            alt = self.config.p_dms[i].alt
            data = self.supervisor.dms.get_dm_shape(i)
            cx, cy = self.circleCoords(self.config.p_geom.pupdiam / 2, 1000,
                                       data.shape[0], data.shape[1])
            self.SRcircles[key] = pg.ScatterPlotItem(cx, cy, pen='r', size=1)
            self.viewboxes[key].addItem(self.SRcircles[key])
            self.SRcircles[key].setData(cx, cy)

        for i in range(len(self.config.p_targets)):
            key = "tar_%d" % i
            data = self.supervisor.target.get_tar_phase(i)
            cx, cy = self.circleCoords(self.config.p_geom.pupdiam / 2, 1000,
                                       data.shape[0], data.shape[1])
            self.SRcircles[key] = pg.ScatterPlotItem(cx, cy, pen='r', size=1)
            self.viewboxes[key].addItem(self.SRcircles[key])
            self.SRcircles[key].setData(cx, cy)

            data = self.supervisor.target.get_tar_image(i)
            for psf in ["psfSE_", "psfLE_"]:
                key = psf + str(i)
                Delta = 5
                self.SRCrossX[key] = pg.PlotCurveItem(
                        np.array([
                                data.shape[0] / 2 + 0.5 - Delta,
                                data.shape[0] / 2 + 0.5 + Delta
                        ]), np.array([data.shape[1] / 2 + 0.5, data.shape[1] / 2 + 0.5]),
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


        for i in range(self.ncoro):
            data = self.supervisor.corono.get_image(i, expo_type="se")
            for psf in ["coroImageSE_", "coroImageLE_", "coroPSFSE_", "coroPSFLE_"]:
                key = psf + str(i)
                Delta = 5
                if("Image" in key):
                    center = 0.
                else:
                    center = 0.5
                self.SRCrossX[key] = pg.PlotCurveItem(
                        np.array([
                                data.shape[0] / 2 + center - Delta,
                                data.shape[0] / 2 + center + Delta
                        ]), np.array([data.shape[1] / 2 + center, data.shape[1] / 2 + center]),
                        pen='r')
                self.SRCrossY[key] = pg.PlotCurveItem(
                        np.array([data.shape[0] / 2 + center, data.shape[0] / 2 + center]),
                        np.array([
                                data.shape[1] / 2 + center - Delta,
                                data.shape[1] / 2 + center + Delta
                        ]), pen='r')
                # Put image in plot area
                self.viewboxes[key].addItem(self.SRCrossX[key])
                # Put image in plot area
                self.viewboxes[key].addItem(self.SRCrossY[key])



        for i in range(len(self.config.p_wfss)):
            if (self.config.p_wfss[i].type == scons.WFSType.PYRHR or
                        self.config.p_wfss[i].type == scons.WFSType.PYRLR):
                key = "pyrFocalPlane_%d" % i
                data = self.supervisor.wfs.get_pyr_focal_plane(i)
                Delta = len(data) / 2
                self.PyrEdgeX[key] = pg.PlotCurveItem(
                        np.array([
                                data.shape[0] / 2 + 0.5 - Delta,
                                data.shape[0] / 2 + 0.5 + Delta
                        ]), np.array([data.shape[1] / 2 + 0.5, data.shape[1] / 2 + 0.5]),
                        pen='b')
                self.PyrEdgeY[key] = pg.PlotCurveItem(
                        np.array([data.shape[0] / 2 + 0.5, data.shape[0] / 2 + 0.5]),
                        np.array([
                                data.shape[1] / 2 + 0.5 - Delta,
                                data.shape[1] / 2 + 0.5 + Delta
                        ]), pen='b')
                # Put image in plot area
                self.viewboxes[key].addItem(self.PyrEdgeX[key])
                # Put image in plot area
                self.viewboxes[key].addItem(self.PyrEdgeY[key])

        print(self.supervisor)

        if self.expert:
            self.expertWidget.displayRtcMatrix()

        self.updateDisplay()

        self.uiAO.wao_run.setDisabled(False)
        self.uiAO.wao_next.setDisabled(False)
        self.uiAO.wao_open_loop.setDisabled(False)
        self.uiAO.wao_unzoom.setDisabled(False)
        self.uiAO.wao_resetSR.setDisabled(False)
        if(self.ncoro):
            self.uiAO.wao_resetCoro.setDisabled(False)
            self.uiAO.wao_allCoro.setDisabled(False)

        WidgetBase.init_configFinished(self)

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
        if (self.supervisor is None or self.supervisor.is_init is False):
            # print("Widget not fully initialized")
            return
        if not self.loopLock.acquire(False):
            return
        else:
            try:
                for key, dock in self.docks.items():
                    if key in ["Strehl"]:
                        continue
                    elif dock.isVisible():
                        index = int(key.split("_")[-1])
                        data = None
                        if "atm" in key:
                            data = self.supervisor.atmos.get_atmos_layer(index)
                        if "wfs" in key:
                            data = self.supervisor.wfs.get_wfs_phase(index)
                        if "dm" in key:
                            dm_type = self.config.p_dms[index].type
                            alt = self.config.p_dms[index].alt
                            data = self.supervisor.dms.get_dm_shape(index)
                        if "tar" in key:
                            data = self.supervisor.target.get_tar_phase(index)
                        if "psfLE" in key:
                            data = self.supervisor.target.get_tar_image(
                                    index, expo_type="le")
                        if "psfSE" in key:
                            data = self.supervisor.target.get_tar_image(
                                    index, expo_type="se")
                        if "coroImageLE" in key:
                            data = self.supervisor.corono.get_image(index, expo_type="le")
                        if "coroImageSE" in key:
                            data = self.supervisor.corono.get_image(index, expo_type="se")
                        if "coroPSFLE" in key:
                            data = self.supervisor.corono.get_psf(index, expo_type="le")
                        if "coroPSFSE" in key:
                            data = self.supervisor.corono.get_psf(index, expo_type="se")

                        if "psf" in key or "coro" in key:
                            if (self.uiAO.actionPSF_Log_Scale.isChecked()):
                                if np.any(data <= 0):
                                    # warnings.warn("\nZeros founds, filling with min nonzero value.\n")
                                    data[data <= 0] = np.min(data[data > 0])
                                data = np.log10(data)
                            if (self.supervisor.get_frame_counter() < 10):
                                self.viewboxes[key].setRange(
                                        xRange=(data.shape[0] / 2 + 0.5 - self.PSFzoom,
                                                data.shape[0] / 2 + 0.5 + self.PSFzoom),
                                        yRange=(data.shape[1] / 2 + 0.5 - self.PSFzoom,
                                                data.shape[1] / 2 + 0.5 + self.PSFzoom),
                                )

                        if "SH" in key:
                            data = self.supervisor.wfs.get_wfs_image(index)
                        if "pyrLR" in key:
                            data = self.supervisor.wfs.get_wfs_image(index)
                        if "pyrHR" in key:
                            data = self.supervisor.wfs.get_pyrhr_image(index)
                        if "pyrFocalPlane" in key:
                            data = self.supervisor.wfs.get_pyr_focal_plane(index)

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
                                slopes = self.supervisor.rtc.get_slopes_geom()
                                x, y, vx, vy = plsh(
                                        slopes, self.config.p_wfss[index].nxsub,
                                        self.config.p_tel.cobs, returnquiver=True
                                )  # Preparing mesh and vector for display
                            if "Comp" in key:
                                centroids = self.supervisor.rtc.get_slopes(index)
                                nmes = [
                                        2 * p_wfs._nvalid for p_wfs in self.config.p_wfss
                                ]
                                first_ind = np.sum(nmes[:index], dtype=np.int32)
                                if (self.config.p_wfss[index].type == scons.WFSType.PYRHR
                                            or self.config.p_wfss[index].type == scons.
                                            WFSType.PYRLR):
                                    #TODO: DEBUG... with full pixels (only works with classic slopes method)
                                    plpyr(
                                            centroids[first_ind:first_ind + nmes[index]],
                                            np.stack([
                                                    self.config.p_wfss[index]._validsubsx,
                                                    self.config.p_wfss[index]._validsubsy
                                            ]))
                                else:
                                    x, y, vx, vy = plsh(
                                            centroids[first_ind:first_ind + nmes[index]],
                                            self.config.p_wfss[index].nxsub,
                                            self.config.p_tel.cobs, returnquiver=True
                                    )  # Preparing mesh and vector for display
                                self.imgs[key].canvas.axes.quiver(x, y, vx, vy)
                            self.imgs[key].canvas.draw()
                        elif "modalGains" in key:
                            data = self.supervisor.rtc.get_modal_gains(index)
                            self.imgs[key].canvas.axes.clear()
                            self.imgs[key].canvas.axes.plot(data)
                            self.imgs[key].canvas.draw()
                        elif "ContrastLE" in key:     
                            distances, mean, std, mini, maxi = self.supervisor.corono.get_contrast(index, expo_type='le')
                            if(np.all(mean)): 
                                self.imgs[key].canvas.axes.clear()
                                self.imgs[key].canvas.axes.plot(distances, mean)
                                self.imgs[key].canvas.axes.set_yscale('log')
                                self.imgs[key].canvas.axes.set_xlabel("angular distance (Lambda/D)")
                                self.imgs[key].canvas.axes.set_ylabel("Raw contrast")

                                self.imgs[key].canvas.axes.grid()
                                self.imgs[key].canvas.draw()
                        elif "ContrastSE" in key:
                            distances, mean, std, mini, maxi = self.supervisor.corono.get_contrast(index, expo_type='se')
                            if(np.all(mean)): 
                                self.imgs[key].canvas.axes.clear()
                                self.imgs[key].canvas.axes.plot(distances, mean)
                                self.imgs[key].canvas.axes.set_yscale('log')
                                self.imgs[key].canvas.axes.set_xlabel("angular distance (Lambda/D)")
                                self.imgs[key].canvas.axes.set_ylabel("Raw contrast")
                                self.imgs[key].canvas.axes.grid()
                                self.imgs[key].canvas.draw()
                self.firstTime = 1

            finally:
                self.loopLock.release()

    def updateSRSE(self, SRSE):
        self.uiAO.wao_strehlSE.setText(SRSE)

    def updateSRLE(self, SRLE):
        self.uiAO.wao_strehlLE.setText(SRLE)

    def updateCurrentLoopFrequency(self, freq):
        self.uiAO.wao_currentFreq.setValue(freq)

    def loop_once(self) -> None:
        if not self.loopLock.acquire(False):
            print("Display locked")
            return
        else:
            try:
                start = time.time()
                self.supervisor.next()
                for t in range(len(self.supervisor.config.p_targets)):
                    self.supervisor.target.comp_tar_image(t)
                loopTime = time.time() - start

                refreshDisplayTime = 1. / self.uiBase.wao_frameRate.value()

                if (time.time() - self.refreshTime > refreshDisplayTime):
                    signal_le = ""
                    signal_se = ""
                    for t in range(len(self.config.p_targets)):
                        SR = self.supervisor.target.get_strehl(t)
                        # TODO: handle that !
                        if (t == self.uiAO.wao_dispSR_tar.value()
                            ):  # Plot on the wfs selected
                            self.updateSRDisplay(SR[1], SR[0],
                                                 self.supervisor.get_frame_counter())
                        signal_se += "%1.2f   " % SR[0]
                        signal_le += "%1.2f   " % SR[1]

                    currentFreq = 1 / loopTime
                    refreshFreq = 1 / (time.time() - self.refreshTime)

                    self.updateSRSE(signal_se)
                    self.updateSRLE(signal_le)
                    self.updateCurrentLoopFrequency(currentFreq)

                    if (self.dispStatsInTerminal):
                        self.printInPlace(
                                "iter #%d SR: (L.E, S.E.)= (%s, %s) running at %4.1fHz (real %4.1fHz)"
                                % (self.supervisor.get_frame_counter(), signal_le,
                                   signal_se, refreshFreq, currentFreq))

                    self.refreshTime = start
            except Exception as e:
                print(e)
                print("error!!")
                raise e
            finally:
                self.loopLock.release()

    def run(self):
        WidgetBase.run(self)
        if not self.uiAO.wao_forever.isChecked():
            self.nbiter -= 1

        if self.nbiter <= 0:
            self.stop = True
            self.uiAO.wao_run.setChecked(False)

import os
import socket

def tcp_connect_to_display():
        # get the display from the environment
        display_env = os.environ['DISPLAY']

        # parse the display string
        display_host, display_num = display_env.split(':')
        display_num_major = display_num.split('.')[0]

        # calculate the port number
        display_port = 6000 + int(display_num_major)

        # attempt a TCP connection
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
                sock.connect((display_host, display_port))
        except socket.error:
                return False
        finally:
            sock.close()
        return True

if __name__ == '__main__':
    # if(not tcp_connect_to_display()):
    #     raise RuntimeError("Cannot connect to display")
        
    arguments = docopt(__doc__)
    app = QtWidgets.QApplication(sys.argv)
    app.setStyle('cleanlooks')
    wao = widgetAOWindow(arguments["<parameters_filename>"], cacao=arguments["--cacao"],
                         expert=arguments["--expert"], devices=arguments["--devices"])
    wao.show()

    print("")
    print("If the GUI is black, you can:")
    print("    type %gui qt5 to unlock GUI")
    print(" or launch ipython with the option '--gui=qt' or '--matplotlib=qt'")
    print(" or edit ~/.ipython/profile_default/ipython_config.py to set c.TerminalIPythonApp.matplotlib = 'qt'")

    if arguments["--interactive"]:
        from shesha.util.ipython_embed import embed
        embed(os.path.basename(__file__), locals())
