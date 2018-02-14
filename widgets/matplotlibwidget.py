# -*- coding: utf-8 -*-
"""
Created on Tue Jun 24 00:27:01 2014

@author: fvidal
"""
"""
from PyQt4 import QtGui
from matplotlib.backends.backend_qt4agg import FigureCanvasQTAgg as FigureCanvas

from matplotlib.figure import Figure

class MplCanvas(FigureCanvas):

    def __init__(self):
        self.fig = Figure(facecolor='white')
        self.axes = self.fig.add_subplot(111)
        # We want the axes cleared every time plot() is called
        self.axes.hold(False)
        FigureCanvas.__init__(self, self.fig)
        FigureCanvas.setSizePolicy(self, QtGui.QSizePolicy.Expanding,QtGui.QSizePolicy.Expanding)
        FigureCanvas.updateGeometry(self)


class MatplotlibWidget(QtGui.QWidget):

    def __init__(self, parent = None):
        QtGui.QWidget.__init__(self, parent)
        self.canvas = MplCanvas()
        self.vbl = QtGui.QVBoxLayout()
        self.vbl.addWidget(self.canvas)
        self.setLayout(self.vbl)
"""

#!/usr/bin/python
import sys
import platform
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtCore import *
from PyQt5.QtGui import *

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as Navigationtoolbar
from matplotlib.figure import Figure
import matplotlib.gridspec as gridspec
import matplotlib
#matplotlib.use('Qt4Agg')
#matplotlib.rcParams['backend.qt4']='PySide'
#matplotlib.style.use('ggplot')
matplotlib.style.use('seaborn-muted')


#Embeddable matplotlib figure/canvas
class MplCanvas(FigureCanvas):

    def __init__(self):
        self.fig = Figure(frameon=True)
        self.gs1 = gridspec.GridSpec(1, 1)
        self.axes = self.fig.add_subplot(self.gs1[0], aspect="auto")

        FigureCanvas.__init__(self, self.fig)
        FigureCanvas.setSizePolicy(self, QtWidgets.QSizePolicy.Expanding,
                                   QtWidgets.QSizePolicy.Expanding)
        FigureCanvas.updateGeometry(self)


#creates embeddable matplotlib figure/canvas with toolbar
class MatplotlibWidget(QtWidgets.QWidget):

    def __init__(self, parent=None):
        QtWidgets.QWidget.__init__(self, parent)
        self.create_framentoolbar()

    def create_framentoolbar(self):
        self.frame = QtWidgets.QWidget()
        self.canvas = MplCanvas()
        self.canvas.setParent(self.frame)
        self.mpltoolbar = Navigationtoolbar(self.canvas, self.frame)
        self.vbl = QtWidgets.QVBoxLayout()
        self.vbl.addWidget(self.canvas)
        self.vbl.addWidget(self.mpltoolbar)
        self.setLayout(self.vbl)
