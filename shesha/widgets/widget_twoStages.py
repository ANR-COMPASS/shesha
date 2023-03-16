## @package   shesha.widgets.widget_canapass
## @brief     Widget to simulate a closed loop using CANAPASS
## @author    COMPASS Team <https://github.com/ANR-COMPASS>
## @version   4.3.0
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
Widget built to simulate a 2 stage AO loop (1st stage = SH; second stage = pyramid)

Usage:
  widget_twoStages.py <parameters_filename1> <parameters_filename2> <freqratio> [options]

with 'parameters_filename1' the path to the parameters file for first stage
with 'parameters_filename2' the path to the parameters file for second stage
with 'freqratio' the ratio of the frequencies of the two stages

Options:
  -a, --adopt       used to connect ADOPT (via pyro + shm cacao)

Example: 
    ipython -i widget_twoStages.py ../../data/par/SPHERE+/sphere.py ../../data/par/SPHERE+/sphere+.py
    ipython -i widget_twoStages.py ../../data/par/SPHERE+/sphere.py ../../data/par/SPHERE+/sphere+.py -- --adopt
"""

import os, sys
import numpy as np
import time

import pyqtgraph as pg
from shesha.util.tools import plsh, plpyr
from tqdm import trange
import astropy.io.fits as pfits
from PyQt5 import QtWidgets
from shesha.supervisor.twoStagesManager import TwoStagesManager

from typing import Any, Dict, Tuple, Callable, List
from docopt import docopt

from shesha.widgets.widget_base import WidgetBase
from shesha.widgets.widget_ao import widgetAOWindow, widgetAOWindow

global server
server = None


class widgetTwoStagesWindowPyro():

    def __init__(self, config_file1: Any = None, config_file2: Any = None, freqratio : int = None, 
                 cacao: bool = False, expert: bool = False) -> None:
        self.config1 = config_file1
        self.config2 = config_file2
        self.freqratio = freqratio

        from shesha.config import ParamConfig


        self.wao2=widgetAOWindow(config_file2, cacao=cacao, hide_histograms=True, twoStages=True)
        self.wao1=widgetAOWindow(config_file1, cacao=cacao, hide_histograms=True, twoStages=True)
        pupdiam_first_stage = self.wao1.supervisor.config.p_geom.pupdiam
        pupdiam_second_stage = self.wao2.supervisor.config.p_geom.pupdiam
        if(pupdiam_first_stage != pupdiam_second_stage):
            print("---------------ERROR---------------")
            print("SECOND STAGE PUPDIAM IS SET TO %d" % pupdiam_second_stage)
            print("FIRST STAGE PUPDIAM IS SET TO %d" % pupdiam_first_stage)
            raise Exception('ERROR!!!! FIRST STAGE PUPDIAM MUST BE SET TO %d' % pupdiam_second_stage)

        #Pyro.core.ObjBase.__init__(self)
        self.CB = {}
        self.wpyr = None
        self.current_buffer = 1
        self.manager = None
        self.cacao = cacao
        #############################################################
        #                 CONNECTED BUTTONS                         #
        #############################################################
        # Default path for config files
        #self.wao1.uiAO.wao_open_loop.setChecked(False)
        #self.wao1.uiAO.wao_open_loop.setText("Close Loop")
        self.wao1.uiAO.actionShow_Pyramid_Tools.toggled.connect(self.show_pyr_tools)
        self.wao2.uiAO.actionShow_Pyramid_Tools.toggled.connect(self.show_pyr_tools)
        self.wpyrNbBuffer = 1
        #############################################################
        #                       METHODS                             #
        #############################################################

        self.manager = TwoStagesManager(self.wao1.supervisor, self.wao2.supervisor, self.freqratio)
        if(self.cacao):
            global server
            server = self.start_pyro_server()

    def loop_once(self) -> None:
        self.manager.next()

        for wao in [self.wao1, self.wao2]:
            start = time.time()
            for t in range(len(wao.supervisor.config.p_targets)):
                wao.supervisor.target.comp_tar_image(t)
            loopTime = time.time() - start

            refreshDisplayTime = 1. / wao.uiBase.wao_frameRate.value()

            if (time.time() - wao.refreshTime > refreshDisplayTime):
                signal_le = ""
                signal_se = ""
                for t in range(len(wao.config.p_targets)):
                    SR = wao.supervisor.target.get_strehl(t)
                    # TODO: handle that !
                    if (t == wao.uiAO.wao_dispSR_tar.value()
                        ):  # Plot on the wfs selected
                        wao.updateSRDisplay(SR[1], SR[0],
                                                wao.supervisor.get_frame_counter())
                    signal_se += "%1.2f   " % SR[0]
                    signal_le += "%1.2f   " % SR[1]

                currentFreq = 1 / loopTime
                refreshFreq = 1 / (time.time() - wao.refreshTime)

                wao.updateSRSE(signal_se)
                wao.updateSRLE(signal_le)
                wao.updateCurrentLoopFrequency(currentFreq)

                if (wao.dispStatsInTerminal):
                    wao.printInPlace(
                            "iter #%d SR: (L.E, S.E.)= (%s, %s) running at %4.1fHz (real %4.1fHz)"
                            % (wao.supervisor.get_frame_counter(), signal_le,
                                signal_se, refreshFreq, currentFreq))
        if (self.wao2.uiAO.actionShow_Pyramid_Tools.isChecked()):  # PYR only
            self.wpyr.Fe = 1 / self.config.p_loop.ittime  # needs Fe for PSD...
            if (self.wpyr.CBNumber == 1):
                self.ai = self.compute_modal_residuals()
                self.set_pyr_tools_params(self.ai)
            else:
                if (self.current_buffer == 1):  # First iter of the CB
                    aiVect = self.compute_modal_residuals()
                    self.ai = aiVect[np.newaxis, :]
                    self.current_buffer += 1  # Keep going

                else:  # Keep filling the CB
                    aiVect = self.compute_modal_residuals()
                    self.ai = np.concatenate((self.ai, aiVect[np.newaxis, :]))
                    if (self.current_buffer < self.wpyr.CBNumber):
                        self.current_buffer += 1  # Keep going
                    else:
                        self.current_buffer = 1  # reset buffer
                        self.set_pyr_tools_params(self.ai)  # display

    def next(self, nbIters):
        ''' Move atmos -> get_slopes -> applyControl ; One integrator step '''
        for i in trange(nbIters):
            self.manager.next()

    def initPyrTools(self):
        ADOPTPATH = os.getenv("ADOPTPATH")
        sys.path.append(ADOPTPATH + "/widgets")
        from pyrStats import widget_pyrStats
        print("OK Pyramid Tools Widget initialized")
        self.wpyr = widget_pyrStats()
        self.wpyrNbBuffer = self.wpyr.CBNumber
        self.wpyr.show()

    def set_pyr_tools_params(self, ai):
        self.wpyr.pup = self.manager.second_stage.config.p_geom._spupil
        self.wpyr.phase = self.supervisor.target.get_tar_phase(0, pupil=True)
        self.wpyr.updateResiduals(ai)
        if (self.phase_to_modes is None):
            print('computing phase 2 Modes basis')
            self.phase_to_modes = self.manager.second_stage.basis.compute_phase_to_modes(self.modal_basis)
        self.wpyr.phase_to_modes = self.phase_to_modes

    def show_pyr_tools(self):
        if (self.wpyr is None):
            try:
                print("Lauching pyramid widget...")
                self.initPyrTools()
                print("Done")
            except:
                raise ValueError("ERROR: ADOPT  not found. Cannot launch Pyramid tools")
        else:
            if (self.wao2.uiAO.actionShow_Pyramid_Tools.isChecked()):
                self.wpyr.show()
            else:
                self.wpyr.hide()

    def getAi(self):
        return self.wpyr.ai

    def start_pyro_server(self):
        try:
            wao_loop = loopHandler( self.wao1)

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
            supervisor  = self.manager
            supervisor1 = self.manager.first_stage
            supervisor2 = self.manager.second_stage

            if(supervisor1.corono == None):
                from shesha.util.pyroEmptyClass import PyroEmptyClass
                coro2pyro1 = PyroEmptyClass()
            else:
                coro2pyro1 = supervisor1.corono

            if(supervisor2.corono == None):
                from shesha.util.pyroEmptyClass import PyroEmptyClass
                coro2pyro2 = PyroEmptyClass()
            else:
                coro2pyro2 = supervisor2.corono

            devices1 = [
                    supervisor1, supervisor1.rtc, supervisor1.wfs, supervisor1.target,
                    supervisor1.tel, supervisor1.basis, supervisor1.calibration,
                    supervisor1.atmos, supervisor1.dms, supervisor1.config, supervisor1.modalgains,
                    coro2pyro1
            ]
            devices2 = [
                    supervisor2, supervisor2.rtc, supervisor2.wfs, supervisor2.target,
                    supervisor2.tel, supervisor2.basis, supervisor2.calibration,
                    supervisor2.atmos, supervisor2.dms, supervisor2.config, supervisor2.modalgains,
                    coro2pyro2
            ]
            names = [
                    "supervisor", "supervisor_rtc", "supervisor_wfs", "supervisor_target",
                    "supervisor_tel", "supervisor_basis", "supervisor_calibration",
                    "supervisor_atmos", "supervisor_dms", "supervisor_config", "supervisor_modalgains",
                    "supervisor_corono"
            ]

            label = "firstStage"
            nname = []
            for name in names:
                nname.append(name + "_" + user + "_" +label)

            label = "secondStage"
            for name in names:
                nname.append(name + "_" + user + "_" +label)

            nname.append('twoStagesManager'+ "_" + user ) # Adding master next dedicated to trigger 2-stages loop
            nname.append("wao_loop"+ "_" + user)
            devices = devices1 + devices2 + [supervisor, wao_loop]
            server = PyroServer(listDevices=devices, listNames=nname)
            #server.add_device(supervisor, "waoconfig_" + user)
            server.start()


        except:
            raise Exception("Error could not connect to Pyro server.\n It can  be:\n - Missing dependencies? (check if Pyro4 is installed)\n - pyro server not running")
        return server


class loopHandler:

    def __init__(self, wao):
        self.wao = wao

    def start(self):
        self.wao.aoLoopClicked(True)
        self.wao.uiAO.wao_run.setChecked(True)

    def stop(self):
        self.wao.aoLoopClicked(False)
        self.wao.uiAO.wao_run.setChecked(False)

    def alive(self):
        return "alive"


if __name__ == '__main__':
    arguments = docopt(__doc__)
    adopt = arguments["--adopt"]
    app = QtWidgets.QApplication(sys.argv)
    app.setStyle('cleanlooks')
    wao = widgetTwoStagesWindowPyro(arguments["<parameters_filename1>"], 
                                    arguments["<parameters_filename2>"], 
                                    arguments["<freqratio>"], cacao=adopt)

    wao.wao1.show()
    wao.wao2.show()
    wao.wao2.uiAO.wao_run.hide()
    wao.wao2.uiAO.wao_next.hide()
    wao.wao2.uiAO.wao_atmosphere.hide()
    wao.wao1.loop_once = wao.loop_once # very dirty (for some reason this does not work during class init...)
    wao.wao2.loop_once = wao.loop_once # very dirty bis

