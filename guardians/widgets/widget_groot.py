import numpy as np
from glob import glob
import os
import datetime

import h5py
import matplotlib as mpl

from bokeh.plotting import figure
from bokeh.models import ColumnDataSource, Range1d
from bokeh.models.widgets import Panel, Toggle, TextInput, CheckboxButtonGroup, Button, RadioButtonGroup, Select, PreText
from bokeh.layouts import layout, widgetbox

from guardians import gamora, groot, drax


class Bokeh_groot:

    def __init__(self):

        self.dataroot = os.getenv("DATA_GUARDIAN")
        self.datapath = self.dataroot
        self.files = [f.split('/')[-1] for f in glob(self.datapath + "roket_*.h5")]
        if self.files == []:
            self.files = ["No hdf5 files"]
        self.f = None
        self.Btt = None
        self.P = None
        self.nactus = None
        self.nmodes = None
        self.nslopes = None

        self.url = "http://" + os.uname()[1] + ".obspm.fr/~" + os.getlogin(
        ) + "/roket_display"
        self.old = None
        self.psf_compass = None
        self.psf_roket = None
        self.psf_groot = None
        self.covmat_groot = None
        self.covmat_roket = None

        # Widgets Elements
        self.pretext = PreText(text=""" """, width=500, height=75)
        self.SRcompass = TextInput(value=" ", title="SR compass:")
        self.SRroket = TextInput(value=" ", title="SR roket:")
        self.SRgroot = TextInput(value=" ", title="SR groot:")

        self.button_covmat = Button(label="Covmat", button_type="success")
        self.button_psf = Button(label="PSF !", button_type="success")
        self.toggle_fit = Toggle(label="Fitting", button_type="primary")

        self.select_datapath = Select(
                title="Datapath", value=self.dataroot,
                options=[self.dataroot] + glob(self.dataroot + "*/"))
        self.select_files = Select(title="File", value=self.files[0], options=self.files)

        self.contributors = ["noise", "bandwidth & tomography", "aliasing"]
        self.checkboxButtonGroup_contributors = CheckboxButtonGroup(
                labels=self.contributors, active=[])
        self.radioButton_basis = RadioButtonGroup(labels=["Actus", "Btt", "Slopes"],
                                                  active=0)

        self.xdr = Range1d(start=0, end=1024)
        self.ydr = Range1d(start=1024, end=0)
        self.xdr2 = Range1d(start=0, end=1024)
        self.ydr2 = Range1d(start=1024, end=0)
        self.image_roket = figure(x_range=self.xdr, y_range=self.ydr,
                                  x_axis_location="above", title="PSF ROKET")
        self.image_groot = figure(x_range=self.image_roket.x_range,
                                  y_range=self.image_roket.y_range,
                                  x_axis_location="above", title="PSF GROOT")
        self.im_covmat_roket = figure(x_range=self.xdr2, y_range=self.ydr2,
                                      x_axis_location="above", title="Covmat ROKET")
        self.im_covmat_groot = figure(x_range=self.im_covmat_roket.x_range,
                                      y_range=self.im_covmat_roket.y_range,
                                      x_axis_location="above", title="Covmat GROOT")
        self.plot_psf_cuts = figure(plot_height=600, plot_width=800, y_range=[1e-9, 1],
                                    x_range=self.image_roket.x_range, y_axis_type="log")
        self.source_psf_roket = ColumnDataSource(data=dict(x=[], y=[]))
        self.source_psf_groot = ColumnDataSource(data=dict(x=[], y=[]))
        self.source_psf_compass = ColumnDataSource(data=dict(x=[], y=[]))
        self.source_covmat_roket = ColumnDataSource(data=dict(x=[], y=[]))
        self.source_covmat_groot = ColumnDataSource(data=dict(x=[], y=[]))

        self.image_roket.image_url(url=[], x=0, y=0, w=1024, h=1024)
        self.image_groot.image_url(url=[], x=0, y=0, w=1024, h=1024)
        self.im_covmat_roket.image_url(url=[], x=0, y=0, w=1024, h=1024)
        self.im_covmat_groot.image_url(url=[], x=0, y=0, w=1024, h=1024)
        self.plot_psf_cuts.line(x="x", y="y", legend="ROKET", color="blue",
                                muted_alpha=0.1, source=self.source_psf_roket)
        self.plot_psf_cuts.line(x="x", y="y", legend="COMPASS", color="red",
                                muted_alpha=0.1, source=self.source_psf_compass)
        self.plot_psf_cuts.line(x="x", y="y", legend="GROOT", color="green",
                                muted_alpha=0.1, source=self.source_psf_groot)
        self.plot_psf_cuts.legend.click_policy = "mute"

        # Callback functions
        self.select_datapath.on_change(
                "value", lambda attr, old, new: self.update_files())
        self.select_files.on_change("value", lambda attr, old, new: self.update())
        self.button_psf.on_click(self.comp_psf)
        self.button_covmat.on_click(self.comp_covmats)
        self.update()

        #layouts
        self.control_box = widgetbox(self.select_datapath, self.select_files,
                                     self.checkboxButtonGroup_contributors,
                                     self.radioButton_basis, self.button_covmat,
                                     self.button_psf, self.toggle_fit, self.SRcompass,
                                     self.SRroket, self.SRgroot, self.pretext)
        self.tab = Panel(
                child=layout([[
                        self.control_box, self.im_covmat_roket, self.im_covmat_groot
                ], [self.image_roket, self.image_groot], [self.plot_psf_cuts]]),
                title="GROOT")

    def update(self):
        """
        Update the attributes based on the new selected filename
        """
        if os.path.exists(self.datapath + str(self.select_files.value)):
            self.f = h5py.File(self.datapath + str(self.select_files.value), mode='r+')
            self.psf_compass = self.f["psf"][:]
            self.SRcompass.value = "%.2f" % (self.psf_compass.max())
            self.psf_groot = None
            self.psf_roket = None
            self.covmat_groot = None
            self.covmat_roket = None
            self.Btt = self.f["Btt"][:]
            self.P = self.f["P"][:]
            self.nactus = self.P.shape[1]
            self.nmodes = self.P.shape[0]
            self.nslopes = self.f["R"][:].shape[1]

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
        Update the PSF display
        """
        if self.psf_roket is not None:
            time = str(datetime.datetime.now().strftime('%Y-%m-%d_%H_%M_%f'))
            self.old = "/home/" + os.getlogin(
            ) + "/public_html/roket_display" + time + ".png"
            mpl.image.imsave(self.old, np.log10(np.abs(self.psf_roket)))
            self.image_roket.image_url(
                    url=dict(value=self.url + time + ".png"), x=0, y=0,
                    w=self.psf_roket.shape[0], h=self.psf_roket.shape[0])
            self.image_roket.x_range.update(start=0, end=self.psf_roket.shape[0])
            self.image_roket.y_range.update(start=self.psf_roket.shape[0], end=0)
            self.SRroket.value = "%.2f" % (self.psf_roket.max())

        if self.psf_groot is not None:
            time = str(datetime.datetime.now().strftime('%Y-%m-%d_%H_%M_%f'))
            self.old = "/home/" + os.getlogin(
            ) + "/public_html/roket_display" + time + ".png"
            mpl.image.imsave(self.old, np.log10(np.abs(self.psf_groot)))
            self.image_groot.image_url(
                    url=dict(value=self.url + time + ".png"), x=0, y=0,
                    w=self.psf_groot.shape[0], h=self.psf_groot.shape[0])
            self.SRgroot.value = "%.2f" % (self.psf_groot.max())

        self.update_cuts()

    def update_covmats(self):
        """
        Update the covmats
        """
        if self.covmat_roket is not None:
            time = str(datetime.datetime.now().strftime('%Y-%m-%d_%H_%M_%f'))
            self.old = "/home/" + os.getlogin(
            ) + "/public_html/roket_display" + time + ".png"
            mpl.image.imsave(self.old, self.covmat_roket)
            self.im_covmat_roket.image_url(
                    url=dict(value=self.url + time + ".png"), x=0, y=0,
                    w=self.covmat_roket.shape[0], h=self.covmat_roket.shape[0])
            self.im_covmat_roket.x_range.update(start=0, end=self.covmat_roket.shape[0])
            self.im_covmat_roket.y_range.update(start=self.covmat_roket.shape[0], end=0)

        if self.covmat_groot is not None:
            time = str(datetime.datetime.now().strftime('%Y-%m-%d_%H_%M_%f'))
            self.old = "/home/" + os.getlogin(
            ) + "/public_html/roket_display" + time + ".png"
            mpl.image.imsave(self.old, self.covmat_groot)
            self.im_covmat_groot.image_url(
                    url=dict(value=self.url + time + ".png"), x=0, y=0,
                    w=self.covmat_groot.shape[0], h=self.covmat_groot.shape[0])

    def update_cuts(self):
        """
        Update the PSF cuts
        """
        if self.psf_roket is not None:
            x = np.arange(self.psf_roket.shape[0])
            self.source_psf_roket.data = dict(
                    x=x, y=self.psf_roket[:, self.psf_roket.shape[0] // 2])
        if self.psf_groot is not None:
            x = np.arange(self.psf_groot.shape[0])
            self.source_psf_groot.data = dict(
                    x=x, y=self.psf_groot[:, self.psf_groot.shape[0] // 2])
        if self.psf_compass is not None:
            x = np.arange(self.psf_compass.shape[0])
            self.source_psf_compass.data = dict(
                    x=x, y=self.psf_compass[:, self.psf_compass.shape[0] // 2])

    def comp_covmats(self):
        """
        Compute the covmats using GROOT model and display it
        """
        self.pretext.text = """ Computing covmats... Please wait"""
        self.button_covmat.button_type = "danger"
        contrib = [
                self.contributors[c]
                for c in self.checkboxButtonGroup_contributors.active
        ]
        if contrib == []:
            contrib = self.contributors
        if "bandwidth & tomography" in contrib:
            contrib.remove("bandwidth & tomography")
            contrib.append("bandwidth")
            contrib.append("tomography")
        modal = self.radioButton_basis.active
        if modal == 1:
            self.covmat_groot = np.zeros((self.nmodes, self.nmodes))
        elif modal == 2:
            self.covmat_groot = np.zeros((self.nslopes, self.nslopes))
        else:
            self.covmat_groot = np.zeros((self.nactus, self.nactus))

        if modal != 2:
            if "noise" in contrib:
                self.covmat_groot += groot.compute_Cn_cpu(
                        self.datapath + str(self.select_files.value), modal=modal)
            if "aliasing" in contrib:
                self.covmat_groot += groot.compute_Ca_cpu(
                        self.datapath + str(self.select_files.value), modal=modal)
            if "tomography" in contrib or "bandwidth" in contrib:
                self.covmat_groot += groot.compute_Cerr(
                        self.datapath + str(self.select_files.value), modal=modal)

            err = drax.get_err_contributors(self.datapath + str(self.select_files.value),
                                            contrib)
            self.covmat_roket = err.dot(err.T) / err.shape[1]
            if modal:
                self.covmat_roket = self.P.dot(self.covmat_roket).dot(self.P.T)
        else:
            if "aliasing" in contrib:
                self.covmat_groot, self.covmat_roket = groot.compute_Calias(
                        self.datapath + str(self.select_files.value))

        self.update_covmats()

        self.pretext.text = """ """
        self.button_covmat.button_type = "success"

    def comp_psf(self):
        """
        Compute the PSF from the covmats
        """
        self.pretext.text = """ Computing PSFs... Please wait"""
        self.button_psf.button_type = "danger"

        fit = self.toggle_fit.active
        if self.covmat_groot.shape[0] != self.nmodes:
            self.covmat_groot = self.P.dot(self.covmat_groot).dot(self.P.T)
            self.covmat_roket = self.P.dot(self.covmat_roket).dot(self.P.T)

        otftel, otf2, self.psf_groot, _ = gamora.psf_rec_Vii(
                self.datapath + str(self.select_files.value),
                cov=self.covmat_groot.astype(np.float32), fitting=False)
        if fit:
            otffit, _ = groot.compute_OTF_fitting(
                    self.datapath + str(self.select_files.value), otftel)
            self.psf_groot = gamora.add_fitting_to_psf(
                    self.datapath + str(self.select_files.value), otf2 * otftel, otffit)

        _, _, self.psf_roket, _ = gamora.psf_rec_Vii(
                self.datapath + str(self.select_files.value),
                cov=self.covmat_roket.astype(np.float32), fitting=fit)

        self.update_psf()
        self.pretext.text = """ """
        self.button_psf.button_type = "success"
