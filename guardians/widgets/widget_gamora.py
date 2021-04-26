import numpy as np
from glob import glob
import os
import datetime

import h5py
import matplotlib as mpl

from bokeh.plotting import figure
from bokeh.models import ColumnDataSource, Range1d
from bokeh.models.widgets import Panel, TextInput, Slider, CheckboxButtonGroup, DataTable, TableColumn, Tabs, Button, RadioButtonGroup, Select, DataTable, DateFormatter, TableColumn, PreText
from bokeh.layouts import layout, widgetbox
from bokeh.io import curdoc, output_file, show

from guardians import gamora, groot


class Bokeh_gamora:

    def __init__(self):

        self.dataroot = os.getenv("DATA_GUARDIAN")
        self.datapath = self.dataroot
        self.files = [f.split('/')[-1] for f in glob(self.datapath + "roket_*.h5")]
        if self.files == []:
            self.files = ["No hdf5 files"]

        self.f = None
        self.Btt = None
        self.P = None

        self.url = "http://" + os.uname()[1] + ".obspm.fr/~" + os.getlogin(
        ) + "/roket_display"
        self.old = None
        self.psf_compass = None
        self.psf_Vii = None

        # Widgets Elements
        self.pretext = PreText(text=""" """, width=500, height=75)
        self.SRcompass = TextInput(value=" ", title="SR compass:")
        self.SRVii = TextInput(value=" ", title="SR Vii:")

        self.button_psf = Button(label="PSF !", button_type="success")
        self.button_roll = Button(label="Roll", button_type="primary")

        self.select_datapath = Select(
                title="Datapath", value=self.dataroot,
                options=[self.dataroot] + glob(self.dataroot + "*/"))
        self.select_files = Select(title="File", value=self.files[0], options=self.files)

        self.xdr = Range1d(start=0, end=1024)
        self.ydr = Range1d(start=1024, end=0)
        self.image_compass = figure(x_range=self.xdr, y_range=self.ydr,
                                    x_axis_location="above", title="PSF COMPASS")
        self.image_Vii = figure(x_range=self.image_compass.x_range,
                                y_range=self.image_compass.y_range,
                                x_axis_location="above", title="PSF ROKET")
        self.plot_psf_cuts = figure(plot_height=600, plot_width=800, y_range=[1e-9, 1],
                                    x_range=self.image_compass.x_range,
                                    y_axis_type="log")
        self.source_psf_compass = ColumnDataSource(data=dict(x=[], y=[]))
        self.source_psf_Vii = ColumnDataSource(data=dict(x=[], y=[]))

        self.image_compass.image_url(url=[], x=0, y=0, w=1024, h=1024)
        self.image_Vii.image_url(url=[], x=0, y=0, w=1024, h=1024)
        self.plot_psf_cuts.line(x="x", y="y", legend="COMPASS", color="red",
                                muted_alpha=0.1, source=self.source_psf_compass)
        self.plot_psf_cuts.line(x="x", y="y", legend="Vii", color="blue",
                                muted_alpha=0.1, source=self.source_psf_Vii)
        self.plot_psf_cuts.legend.click_policy = "mute"

        # Callback functions
        self.select_datapath.on_change(
                "value", lambda attr, old, new: self.update_files())
        self.select_files.on_change("value", lambda attr, old, new: self.update())
        self.button_psf.on_click(self.comp_psf)
        self.button_roll.on_click(self.roll_psf)

        self.update()

        #layouts
        self.control_box = widgetbox(self.select_datapath, self.select_files,
                                     self.button_psf, self.button_roll, self.SRcompass,
                                     self.SRVii, self.pretext)
        self.tab = Panel(
                child=layout([[self.control_box, self.image_compass, self.image_Vii],
                              [self.plot_psf_cuts]]), title="GAMORA")

    def update(self):
        """
        Update the attributes based on the new selected filename
        """
        if os.path.exists(self.datapath + str(self.select_files.value)):
            self.f = h5py.File(self.datapath + str(self.select_files.value), mode='r+')
            self.psf_compass = self.f["psf"][:]
            self.psf_Vii = None
            self.Btt = self.f["Btt"][:]
            self.P = self.f["P"][:]

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

    def update_psf(self):
        """
        Update the PSF by ensquaring them
        """

        psfc = self.psf_compass
        time = str(datetime.datetime.now().strftime('%Y-%m-%d_%H_%M_%f'))
        self.old = "/home/" + os.getlogin() + "/public_html/roket_display" + time + ".png"
        mpl.image.imsave(self.old, np.log10(np.abs(psfc)))
        self.image_compass.image_url(
                url=dict(value=self.url + time + ".png"), x=0, y=0, w=psfc.shape[0],
                h=psfc.shape[0])
        self.image_compass.x_range.update(start=0, end=psfc.shape[0])
        self.image_compass.y_range.update(start=psfc.shape[0], end=0)

        self.SRcompass.value = "%.2f" % (self.psf_compass.max())

        if self.psf_Vii is not None:

            psfv = self.psf_Vii
            time = str(datetime.datetime.now().strftime('%Y-%m-%d_%H_%M_%f'))
            self.old = "/home/" + os.getlogin(
            ) + "/public_html/roket_display" + time + ".png"
            mpl.image.imsave(self.old, np.log10(np.abs(psfv)))
            self.image_Vii.image_url(
                    url=dict(value=self.url + time + ".png"), x=0, y=0, w=psfc.shape[0],
                    h=psfc.shape[0])
            self.SRVii.value = "%.2f" % (self.psf_Vii.max())

        self.update_cuts()

    def update_cuts(self):
        """
        Update the PSF cuts
        """
        x = np.arange(self.psf_compass.shape[0])
        self.source_psf_compass.data = dict(
                x=x, y=self.psf_compass[:, self.psf_compass.shape[0] // 2])
        if self.psf_Vii is not None:
            self.source_psf_Vii.data = dict(
                    x=x, y=self.psf_Vii[:, self.psf_Vii.shape[0] // 2])

    def comp_psf(self):
        """
        Compute the PSF using the Vii functions and display it
        """
        self.pretext.text = """ Computing PSF using Vii... Please wait"""
        self.button_psf.button_type = "danger"
        _, _, self.psf_Vii, _ = gamora.psf_rec_Vii(self.datapath +
                                                   str(self.select_files.value))
        self.psf_compass = self.f["psf"][:]
        self.update_psf()
        self.pretext.text = """ """
        self.button_psf.button_type = "success"

    def roll_psf(self):
        """
        Roll the COMPASS PSF (for retro-compatibility with old ROKET files)
        """
        self.psf_compass = np.fft.fftshift(self.psf_compass)
        self.update_psf()
