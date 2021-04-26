import numpy as np
from glob import glob
import os

import h5py

from bokeh.plotting import figure
from bokeh.models import ColumnDataSource
from bokeh.models.widgets import Panel, DataTable, TableColumn, Tabs, Button, RadioButtonGroup, Select, DataTable, DateFormatter, TableColumn, PreText
from bokeh.layouts import layout, widgetbox
from bokeh.io import curdoc, output_file, show

from guardians import drax


class Bokeh_roket:
    """
    Class that defines a bokeh layout and callback functions for ROKET
    Usage: see bokeh_roket.py which is the executable
    """

    def __init__(self):

        self.dataroot = os.getenv("DATA_GUARDIAN")
        self.datapath = self.dataroot
        self.files = [f.split('/')[-1] for f in glob(self.datapath + "roket_*.h5")]
        if self.files == []:
            self.files = ["No hdf5 files"]
        self.f = None
        self.Btt = None
        self.P = None
        self.cov = None
        self.cor = None
        self.url = "http://hippo6.obspm.fr/~fferreira/roket_display"
        self.old = None

        # Widgets Elements
        self.pretext = PreText(text=""" """, width=500, height=75)
        self.button_load = Button(label="Load", button_type="success")
        self.select_datapath = Select(
                title="Datapath", value=self.dataroot,
                options=[self.dataroot] + glob(self.dataroot + "*/"))
        self.select_files = Select(title="File", value=self.files[0], options=self.files)
        self.radioButton_basis = RadioButtonGroup(labels=["Actuators", "Btt"], active=1)

        self.colors = {
                "filtered modes": "green",
                "bandwidth": "orange",
                "noise": "red",
                "tomography": "purple",
                "non linearity": "cyan",
                "aliasing": "blue"
        }
        self.contributors = [c for c in self.colors.keys()]
        self.source_breakdown = ColumnDataSource(
                data=dict(n=[], a=[], b=[], t=[], nl=[], f=[], fm=[]))
        self.source_cov = ColumnDataSource(
                data=dict(Type=[], Noise=[], Trunc=[], Aliasing=[], FilteredModes=[],
                          Bandwidth=[], Tomography=[]))
        self.source_cor = ColumnDataSource(
                data=dict(Type=[], Noise=[], Trunc=[], Aliasing=[], FilteredModes=[],
                          Bandwidth=[], Tomography=[]))
        self.source_params = ColumnDataSource(data=dict(Parameter=[], Value=[]))
        columns = [
                TableColumn(field="n", title="noise"),
                TableColumn(field="a", title="aliasing"),
                TableColumn(field="b", title="bandwidth"),
                TableColumn(field="t", title="tomography"),
                TableColumn(field="nl", title="non lin."),
                TableColumn(field="f", title="fitting"),
                TableColumn(field="fm", title="filt. modes")
        ]

        self.table_breakdown = DataTable(source=self.source_breakdown, columns=columns,
                                         width=400, height=75)
        #self.update_breakdown()

        columns2 = [
                TableColumn(field="Type", title="Cov."),
                TableColumn(field="Noise", title="Noise"),
                TableColumn(field="Trunc", title="Non lin."),
                TableColumn(field="Aliasing", title="Alias."),
                TableColumn(field="FilteredModes", title="Filt."),
                TableColumn(field="Bandwidth", title="Band."),
                TableColumn(field="Tomography", title="Tomo"),
        ]
        self.table_cov = DataTable(source=self.source_cov, columns=columns2, width=400,
                                   height=200)
        columns2[0] = TableColumn(field="Type", title="Cor.")
        self.table_cor = DataTable(source=self.source_cor, columns=columns2, width=400,
                                   height=250)
        #self.update_covcor()

        tmp = [
                TableColumn(field="Parameter", title="Parameter"),
                TableColumn(field="Value", title="Value")
        ]
        self.table_params = DataTable(source=self.source_params, columns=tmp, width=600,
                                      height=500)
        #self.update_params()
        self.source_variances = {}
        for c in self.contributors:
            self.source_variances[c] = ColumnDataSource(data=dict(x=[], y=[]))
        self.p = figure(plot_height=600, plot_width=800, y_axis_type="log",
                        y_range=[1e-9, 1], title="Contibutors variance")
        for c in self.contributors:
            self.p.line(x="x", y="y", legend=c, color=self.colors[c],
                        muted_color=self.colors[c], muted_alpha=0.1,
                        source=self.source_variances[c])

        self.p.legend.click_policy = "mute"

        # Callback functions
        self.select_datapath.on_change(
                "value", lambda attr, old, new: self.update_files())
        self.button_load.on_click(self.load_file)
        self.radioButton_basis.on_change("active", lambda attr, old, new: self.update())

        # Layouts
        self.control_box = widgetbox(self.select_datapath, self.select_files,
                                     self.button_load, self.radioButton_basis)
        self.tab = Panel(
                child=layout([[
                        self.control_box, self.p,
                        widgetbox(self.pretext, self.table_breakdown, self.table_cov,
                                  self.table_cor)
                ], [self.table_params]]), title="ROKET")

    def update_files(self):
        """
        Update the select_files options following the current datapath
        """
        self.datapath = str(self.select_datapath.value)
        self.files = self.files = [
                f.split('/')[-1] for f in glob(self.datapath + "roket_*.h5")
        ]
        if self.files == []:
            self.files = ["No hdf5 files"]
        self.select_files.options = self.files
        self.select_files.value = self.files[0]

    def load_file(self):
        """
        Load the selected file and update the display
        """
        self.button_load.button_type = "danger"
        self.f = h5py.File(self.datapath + str(self.select_files.value), mode='r+')
        self.Btt = self.f["Btt"][:]
        self.P = self.f["P"][:]  #/np.sqrt(self.IF.shape[0])
        self.cov = self.f["cov"][:]
        self.cor = self.f["cor"][:]

        self.update()
        self.update_breakdown()
        self.update_covcor()
        self.update_params()
        self.button_load.button_type = "success"

        print("DB loaded")

    def update_breakdown(self):
        """
        Update the values of the error breakdown tables
        """
        self.pretext.text = """ Updating error breakdown... Please wait"""

        breakdown = drax.get_breakdown(self.datapath + str(self.select_files.value))
        self.source_breakdown.data = dict(
                n=[int(np.round(breakdown["noise"]))
                   ], a=[int(np.round(breakdown["aliasing"]))
                         ], b=[int(np.round(breakdown["bandwidth"]))
                               ], t=[int(np.round(breakdown["tomography"]))
                                     ], nl=[int(np.round(breakdown["non linearity"]))
                                            ], f=[int(np.round(breakdown["fitting"]))],
                fm=[int(np.round(breakdown["filtered modes"]))])
        self.pretext.text = """ """

    def update_covcor(self):
        """
        Update tables of covariances and correlations
        """
        self.pretext.text = """ Updating cov cor tables... Please wait"""

        self.source_cov.data = dict(
                Type=["Noise", "Trunc", "Alias.", "Filt.", "Band.",
                      "Tomo"], Noise=["%.2E" % v for v in self.cov[:, 0]
                                      ], Trunc=["%.2E" % v for v in self.cov[:, 1]],
                Aliasing=["%.2E" % v for v in self.cov[:, 2]
                          ], FilteredModes=["%.2E" % v for v in self.cov[:, 3]],
                Bandwidth=["%.2E" % v for v in self.cov[:, 4]
                           ], Tomography=["%.2E" % v for v in self.cov[:, 5]])
        self.source_cor.data = dict(
                Type=["Noise", "Trunc", "Alias.", "Filt.", "Band.",
                      "Tomo"], Noise=["%.2f" % v for v in self.cor[:, 0]
                                      ], Trunc=["%.2f" % v for v in self.cor[:, 1]],
                Aliasing=["%.2f" % v for v in self.cor[:, 2]
                          ], FilteredModes=["%.2f" % v for v in self.cor[:, 3]],
                Bandwidth=["%.2f" % v for v in self.cor[:, 4]
                           ], Tomography=["%.2f" % v for v in self.cor[:, 5]])

        self.pretext.text = """ """

    def update_params(self):
        """
        Update the simulation parameters table
        """
        self.pretext.text = """ Updating parameters table... Please wait"""
        params = list(self.f.attrs.keys())
        params.sort()
        values = []
        for k in params:
            values.append(str(self.f.attrs[k]))
        self.source_params.data = dict(Parameter=params, Value=values)
        self.pretext.text = """ """

    def update(self):
        """
        Main callback function that update the bokeh display
        """
        tmp = self.button_load.button_type
        self.button_load.button_type = "danger"
        self.pretext.text = """ Updating plot... Please wait"""

        basis_active = self.radioButton_basis.active
        xi = []
        yi = []
        coloris = []

        for c in self.contributors:
            self.source_variances[c].data = dict(x=[], y=[], color=[], legend=[])
            data = self.f[c][:]
            self.p.xaxis.axis_label = "Actuators"
            if (basis_active):
                data = self.P.dot(data)
                self.p.xaxis.axis_label = "Modes"
            self.source_variances[c].data = dict(
                    x=np.arange(len(data)).tolist(), y=np.var(data, axis=1).tolist())

        self.pretext.text = """ """
        self.button_load.button_type = tmp
