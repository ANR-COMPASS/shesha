"""
Created on Tue Feb  2 09:39:35 2016

@author: fferreira

To launch it :

    - locally :
        bokeh serve --show bokeh_display.py
    - as a server :
        bokeh serve --port 8081 --host hippo6.obspm.fr:8081 bokeh_display.py
        then, open a web browser and connect to http://hippo6.obspm.fr:8081/bokeh_display.py
"""

import numpy as np
import glob
import os, sys

import h5py
import pandas
import datetime

from bokeh.plotting import Figure, figure
from bokeh.models import Range1d, ColumnDataSource, HoverTool
from bokeh.models.widgets import Select, Slider, CheckboxButtonGroup, Panel, Tabs, Button, Dialog, Paragraph, RadioButtonGroup, TextInput
from bokeh.io import curdoc
from bokeh.models.layouts import HBox, VBox
from bokeh.models.widgets import DataTable, DateFormatter, TableColumn
from bokeh.client import push_session

import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy.sparse import csr_matrix

sys.path.insert(0, os.environ["SHESHA_ROOT"] + "/test/gamora/")
import gamora


######################################################################################
#  _      _ _
# (_)_ _ (_) |_ ___
# | | ' \| |  _(_-<
# |_|_||_|_|\__/__/
######################################################################################
class html_display:

    def __del__(self):
        files = glob.glob("/home/fferreira/public_html/roket_display*")
        for f in files:
            os.remove(f)

    def __init__(self):

        self.datapath = "/home/fferreira/Data/correlation/"
        self.covmat = None
        self.files = glob.glob(self.datapath + "roket_*.h5")
        self.files.sort()
        self.f_list = []
        for f in self.files:
            self.f_list.append(f.split('/')[-1])

        self.f = h5py.File(self.files[0], mode='r+')

        self.Lambda_tar = self.f.attrs["target.Lambda"][0]
        self.Btt = self.f["Btt"][:]

        self.IF = csr_matrix((self.f["IF.data"][:], self.f["IF.indices"][:],
                              self.f["IF.indptr"][:]))
        self.IF = self.IF.T
        #self.TT = self.f["TT"][:]
        self.P = self.f["P"][:]  #/np.sqrt(self.IF.shape[0])

        self.indx_pup = self.f["indx_pup"][:]
        self.pup = np.zeros((self.f["dm_dim"].value, self.f["dm_dim"].value))

        self.niter = self.f["noise"][:].shape[1]
        self.nactus = self.f["noise"][:].shape[0]
        self.nmodes = self.P.shape[0]
        self.swap = np.arange(self.nmodes) - 2
        self.swap[0:2] = [self.nmodes - 2, self.nmodes - 1]

        self.plot_type = ["Commands", "Variance"]
        self.coms_list = [
                "noise", "aliasing", "tomography", "filtered modes", "bandwidth",
                "non linearity"
        ]

        self.cov = self.f["cov"][:]
        self.cor = self.f["cor"][:]
        self.psf_compass = np.fft.fftshift(self.f["psf"][:])
        self.psf_fitting = np.fft.fftshift(self.f["psfortho"][:])
        self.psf = None
        self.otftel = None
        self.otf2 = None
        self.gamora = None
        self.basis = ["Actuators", "Btt"]
        self.url = "http://hippo6.obspm.fr/~fferreira/roket_display"
        self.old = None

        ######################################################################################
        #         _    _          _
        # __ __ _(_)__| |__ _ ___| |_ ___
        # \ V  V / / _` / _` / -_)  _(_-<
        #  \_/\_/|_\__,_\__, \___|\__/__/
        #               |___/
        ######################################################################################
        self.dialog = Dialog(closable=False, visible=False, title="Dialog Box",
                             content="")

        # Tab 1
        self.comsTags = Paragraph(text="Commands type", height=25)
        self.coms = CheckboxButtonGroup(labels=self.coms_list, active=[0])
        self.DB_select = Select(title="Database", value=self.f_list[0],
                                options=self.f_list)
        self.DB_button = Button(label="Load DB", type="success")
        self.plot_select = Select(title="Plot type", value=self.plot_type[1],
                                  options=self.plot_type)
        self.basis_select1 = Select(title="Basis", value=self.basis[0],
                                    options=self.basis)
        self.iter_select = Slider(title="Iteration number", start=1, end=self.niter,
                                  step=1)
        self.plusTag = Paragraph(text="Add :", height=25)
        self.plus_select = CheckboxButtonGroup(
                labels=self.coms_list + ["fitting", "CORRECT"],
                active=[0, 1, 2, 3, 4, 5, 6])
        self.moinsTag = Paragraph(text="Substract :", height=25)
        self.moins_select = CheckboxButtonGroup(labels=self.coms_list + ["fitting"],
                                                active=[])
        self.diff_button = Button(label="Sum !", type="success")
        # Tab 2
        self.A = Select(title="Commands A", value=self.coms_list[0],
                        options=self.coms_list)
        self.B = Select(title="Commands B", value=self.coms_list[0],
                        options=self.coms_list)
        self.basis_select2 = Select(title="Basis", value=self.basis[0],
                                    options=self.basis)
        self.power = Slider(title="Abs(covmat)**X", start=0.1, end=1., step=0.1,
                            value=1.)
        self.cmin = Slider(title="vmin", start=1, end=10, step=1)
        self.cmax = Slider(title="vmax", start=1, end=10, step=1)
        self.rescale = Button(label="Rescale !", type="primary")
        self.draw = Button(label="Draw !", type="success")
        self.diag = Button(label="Plot diag !", type="primary")
        self.cut = Button(label="Cut !", type="primary")
        self.axiscut = Slider(title="X/Y cut", start=0, end=1, step=1)
        self.XY = RadioButtonGroup(labels=["X", "Y"], active=0)
        self.DataTableItems = [
                "Type", "Noise", "Truncature", "Aliasing", "FilteredModes", "Bandwidth",
                "Tomography"
        ]
        self.ParamTableItems = list(self.f.attrs.keys())

        self.table_cov_source = ColumnDataSource(
                data=dict(Type=[], Noise=[], Truncature=[], Aliasing=[],
                          FilteredModes=[], Bandwidth=[], Tomography=[]))
        self.table_cor_source = ColumnDataSource(
                data=dict(Type=[], Noise=[], Truncature=[], Aliasing=[],
                          FilteredModes=[], Bandwidth=[], Tomography=[]))

        self.table_param_source = ColumnDataSource(data=dict(Parameter=[], Value=[]))

        self.cov_table, self.cor_table, self.param_table = self.createDataTables()
        self.pcov_source = ColumnDataSource(
                data=dict(image=[], x=[], y=[], dw=[], dh=[]))
        self.pcor_source = ColumnDataSource(
                data=dict(image=[], x=[], y=[], dw=[], dh=[]))
        self.xdr4 = Range1d(start=0, end=6)
        self.ydr4 = Range1d(start=0, end=6)
        self.pcov = figure(x_range=self.xdr4, y_range=self.ydr4, x_axis_location="above")
        self.pcov.image("image", "x", "y", "dw", "dh", palette="Spectral11",
                        source=self.pcov_source)
        self.pcor = figure(x_range=self.xdr4, y_range=self.ydr4, x_axis_location="above")
        self.pcor.image("image", "x", "y", "dw", "dh", palette="Spectral11",
                        source=self.pcor_source)

        self.updateDataTables()
        # Tab 3
        self.basis_select3 = Select(title="Basis", value=self.basis[0],
                                    options=self.basis)
        self.modes_select = Slider(title="Mode #", value=0, start=0, end=self.nmodes,
                                   step=1)
        #self.modes_select = TextInput(value="0:"+str(self.nmodes-1),title="Enter a mode to display")
        self.draw_mode = Button(label="Draw !", type="success")
        self.inc_mode = Button(label="+", type="primary")
        self.desinc_mode = Button(label="-", type="primary")
        # Tab 4
        self.independence = CheckboxButtonGroup(labels=["Independence"], active=[])
        self.psf_display_select = Select(
                title="PSF display", value="COMPASS", options=[
                        "COMPASS", "ROKET", "Vii", "Fitting", "OTF Telescope", "OTF res"
                ])
        self.psf_rec_methods_select = Select(title="Reconstruction method", value="Vii",
                                             options=["Vii", "ROKET"])
        self.gamora_tag = Paragraph(text="PSF reconstruction :", height=25)
        self.psf_display_tag = Paragraph(text="PSF display :", height=25)
        self.error_select = CheckboxButtonGroup(labels=self.coms_list + ["fitting"],
                                                active=[0, 1, 2, 3, 4, 5, 6])
        self.gamora_comp = Button(label="Reconstruct !", type="primary")
        self.psf_display = Button(label="Display", type="primary")
        self.colors = {
                "filtered modes": "green",
                "bandwidth": "orange",
                "noise": "red",
                "tomography": "purple",
                "non linearity": "cyan",
                "aliasing": "blue"
        }

        self.source1 = ColumnDataSource(data=dict(x=[], y=[], color=[], typec=[]))
        self.source2 = ColumnDataSource(data=dict(x=[], y=[], color=[]))
        self.source3 = ColumnDataSource(data=dict(x=[], y=[], color=[]))
        self.sourcepsf = ColumnDataSource(data=dict(x=[], y=[], color=[]))

        self.hover = HoverTool(tooltips=[("x", "@x"), ("y", "@y"), ("type", "@typec")])
        self.hoverlog = HoverTool(tooltips=[("x", "@x"), ("y", "@y"), ("type",
                                                                       "@typec")])
        TOOLS = "resize,save,pan,box_zoom,tap, box_select, wheel_zoom, lasso_select,reset"

        self.plog = Figure(plot_height=600, plot_width=800, y_range=[1e-6, 10],
                           y_axis_type="log", tools=[TOOLS, self.hoverlog])
        self.psum = Figure(plot_height=600, plot_width=800)
        for c in self.colors:
            self.plog.line(legend=c, line_color=self.colors[c])

        self.plog.multi_line("x", "y", color="color", source=self.source1)
        self.psum.line(legend="Image SR", line_color="red")
        self.psum.line(legend="Phase SR ", line_color="purple")
        self.psum.line(legend="Var(X+Y)", line_color="blue")
        self.psum.line(legend="Var(X)+var(Y)", line_color="green")

        self.psum.multi_line("x", "y", color="color", source=self.source3)
        self.psum.yaxis.axis_label = "Strehl Ratio"

        self.xdr = Range1d(start=0, end=self.nactus)
        self.ydr = Range1d(start=self.nactus, end=0)
        self.p2 = figure(x_range=self.xdr, y_range=self.ydr, x_axis_location="above")
        self.p2.image_url(url=[], x=0, y=0, w=self.nactus, h=self.nactus)
        self.p3 = Figure(plot_height=600, plot_width=800)
        self.p3.line(x="x", y="y", source=self.source2)

        self.xdr2 = Range1d(start=0, end=self.pup.shape[0])
        self.ydr2 = Range1d(start=self.pup.shape[1], end=0)
        self.pmodes = figure(x_range=self.xdr2, y_range=self.ydr2,
                             x_axis_location="above")
        self.pmodes.image_url(url=[], x=0, y=0, w=self.pup.shape[0], h=self.pup.shape[1])

        self.control_plot = [self.plot_select, self.iter_select, self.basis_select1]

        self.xdr3 = Range1d(start=0, end=self.psf_compass.shape[0])
        self.ydr3 = Range1d(start=self.psf_compass.shape[1], end=0)
        self.ppsf = figure(x_range=self.xdr3, y_range=self.ydr3, x_axis_location="above")
        self.ppsf.image_url(url=[], x=0, y=0, w=self.psf_compass.shape[0],
                            h=self.psf_compass.shape[1])
        self.pcutpsf = Figure(plot_height=600, plot_width=800, y_range=[1e-9, 1],
                              y_axis_type="log")
        self.pcutpsf.line(legend="COMPASS", line_color="blue")
        self.pcutpsf.line(legend="PSF rec", line_color="red")
        self.pcutpsf.multi_line("x", "y", color="color", source=self.sourcepsf)

        self.buttons = [self.coms]
        for control in self.control_plot:
            control.on_change('value', self.update)
        for button in self.buttons:
            button.on_change('active', self.update)

        self.draw.on_click(self.update_matrix2)
        self.draw_mode.on_click(self.update_mode)
        self.rescale.on_click(self.rescale_matrix)
        self.diag.on_click(self.get_diag)
        self.cut.on_click(self.cut_matrix)
        self.inc_mode.on_click(self.mode_increment)
        self.desinc_mode.on_click(self.mode_desincrement)
        self.diff_button.on_click(self.plot_sum)
        self.DB_button.on_click(self.loadDB)
        self.gamora_comp.on_click(self.gamora_call)
        self.psf_display.on_click(self.update_psf)

        self.inputs = HBox(
                VBox(self.DB_select, self.DB_button, self.comsTags, self.coms,
                     self.plot_select, self.basis_select1, self.iter_select,
                     self.plusTag, self.plus_select, self.moinsTag, self.moins_select,
                     self.diff_button), width=350)
        self.inputs2 = HBox(
                VBox(self.DB_select, self.DB_button, self.basis_select2, self.A, self.B,
                     self.power, self.draw, self.cmax, self.cmin, self.rescale,
                     self.axiscut, self.XY, self.cut, self.diag))  #, width=350)
        self.inputs3 = HBox(
                VBox(
                        self.DB_select, self.DB_button, self.basis_select3,
                        VBox(
                                VBox(
                                        HBox(
                                                self.modes_select,
                                                HBox(self.desinc_mode, self.inc_mode,
                                                     height=40))), self.draw_mode)))
        self.inputs4 = HBox(
                VBox(
                        HBox(self.DB_select, self.DB_button), self.gamora_tag,
                        self.psf_rec_methods_select, self.error_select,
                        self.independence, self.gamora_comp, self.psf_display_tag,
                        self.psf_display_select, self.psf_display), width=350)
        self.tab1 = Panel(
                child=HBox(self.inputs, VBox(self.plog, self.psum)), title="Breakdown")
        self.tab2 = Panel(
                child=HBox(
                        VBox(
                                HBox(self.inputs2, self.p2, self.p3),
                                HBox(self.cov_table, self.pcov),
                                HBox(self.cor_table, self.pcor))), title="Cov/cor")
        self.tab3 = Panel(child=HBox(self.inputs3, self.pmodes), title="Basis")
        self.tab4 = Panel(
                child=HBox(self.inputs4, VBox(self.ppsf, self.pcutpsf)), title="PSF")
        self.tab5 = Panel(
                child=HBox(VBox(HBox(self.DB_select, self.DB_button), self.param_table)),
                title="Parameters")
        self.tabs = Tabs(tabs=[self.tab1, self.tab2, self.tab4, self.tab3, self.tab5])

        curdoc().clear()
        self.update(None, None, None)

        curdoc().add_root(self.tabs)  #hplot(inputs,p))#, p, p2)
        curdoc().add_root(self.dialog)

    ######################################################################################
    #   ___      _ _ _             _
    #  / __|__ _| | | |__  __ _ __| |__ ___
    # | (__/ _` | | | '_ \/ _` / _| / /(_-<
    #  \___\__,_|_|_|_.__/\__,_\__|_\_\/__/
    #
    ######################################################################################
    def loadDB(self):
        self.dialog.visible = False
        self.dialog.content = "Loading database..."
        self.dialog.visible = True

        self.f = h5py.File(self.datapath + str(self.DB_select.value), mode='r+')
        self.Lambda_tar = self.f.attrs["target.Lambda"][0]
        self.Btt = self.f["Btt"][:]

        self.IF = csr_matrix((self.f["IF.data"][:], self.f["IF.indices"][:],
                              self.f["IF.indptr"][:]))
        self.IF = self.IF.T
        #self.TT = self.f["TT"][:]
        self.P = self.f["P"][:]  #/np.sqrt(self.IF.shape[0])
        #self.modes = self.IF.dot(self.Btt)#np.dot(self.f["IF"][:],self.Btt)
        #        self.modes = self.modes[:,self.swap]

        self.indx_pup = self.f["indx_pup"][:]
        self.pup = np.zeros((self.f["dm_dim"].value, self.f["dm_dim"].value))

        self.niter = self.f["noise"][:].shape[1]
        self.nactus = self.f["noise"][:].shape[0]
        self.nmodes = self.P.shape[0]
        self.cov = self.f["cov"][:]
        self.cor = self.f["cor"][:]
        self.psf_compass = np.fft.fftshift(self.f["psf"][:])
        self.psf_fitting = np.fft.fftshift(self.f["psfortho"][:])
        self.psf = None
        self.otftel = None
        self.otf2 = None
        self.gamora = None

        self.plot_type = ["Commands", "Variance"]

        #self.cov_table, self.cor_table = self.createDataTables()
        self.updateDataTables()
        self.update(None, None, None)

        print("DB loaded")
        self.dialog.visible = False

    def update(self, attrname, old, new):
        # plot_val = plot_type.value
        self.source1.data = dict(x=[], y=[], color=[], typec=[])

        coms_active = self.coms.active
        plot_val = self.plot_select.value
        basis_val = self.basis_select1.value
        iteration = int(self.iter_select.value)

        yi = []
        xi = []
        typec = []
        coloris = []
        for jj in coms_active:
            j = self.coms_list[jj]
            data = self.f[j][:]
            if (plot_val == b"Commands"):
                if (basis_val == b"Actuators"):
                    yi.append(data[:, iteration].tolist())
                    self.plog.xaxis.axis_label = "Actuators"
                elif (basis_val == b"Btt"):
                    yi.append(np.dot(self.P, data[:, iteration])[self.swap].tolist())
                    self.plog.xaxis.axis_label = "Modes"
                xi.append(list(range(len(data[:, iteration]))))
                typec.append([j] * len(data[:, iteration]))
                coloris.append(self.colors[j])
                self.plog.yaxis.axis_label = "Volts"

            elif (plot_val == b"Variance"):
                if (basis_val == b"Actuators"):
                    yi.append(np.var(data, axis=1).tolist())
                    self.plog.xaxis.axis_label = "Actuators"
                elif (basis_val == b"Btt"):
                    yi.append(np.var(np.dot(self.P, data), axis=1)[self.swap].tolist())
                    self.plog.xaxis.axis_label = "Modes"
                xi.append(list(range(len(np.var(data, axis=1)))))
                typec.append([j] * len(np.var(data, axis=1)))
                coloris.append(self.colors[j])
                self.plog.yaxis.axis_label = "Variance"

        self.source1.data = dict(x=xi, y=yi, color=coloris, typec=typec)

        print("Plots updated")

    def gamora_call(self):
        self.dialog.visible = False
        psf_type = self.psf_rec_methods_select.value
        err_active = self.error_select.active
        err = self.f["noise"][:] * 0.
        covmodes = err.dot(err.T)
        independence = self.independence.active
        fiterr = False
        self.dialog.content = "Computing covariance matrix..."
        self.dialog.visible = True
        for k in err_active:
            if (self.error_select.labels[k] == b"fitting"):
                fiterr = True
            else:
                if (independence):
                    data = self.f[self.error_select.labels[k]][:]
                    covmodes += data.dot(data.T) / err.shape[1]
                else:
                    err += self.f[self.error_select.labels[k]][:]

        if (psf_type == b"Vii"):
            self.dialog.content = "Reconstructing PSF with Vii (may take a while)..."

            if (independence):
                self.otftel, self.otf2, self.psf, self.gamora = gamora.psf_rec_Vii(
                        self.datapath + str(self.DB_select.value), fitting=fiterr,
                        covmodes=covmodes)
            else:
                self.otftel, self.otf2, self.psf, self.gamora = gamora.psf_rec_Vii(
                        self.datapath + str(self.DB_select.value), err=err,
                        fitting=fiterr)
        if (psf_type == b"ROKET"):
            self.dialog.content = "Reconstructing PSF from ROKET file (may take a while)..."
            self.dialog.visible = True
            self.psf, self.gamora = gamora.psf_rec_roket_file(
                    self.datapath + str(self.DB_select.value), err=err)
        else:
            self.dialog.content = "PSF reconstruction is available with Vii or ROKET methods only"
            self.dialog.visible = True

        self.update_psf()
        self.sourcepsf.data = dict(
                x=[
                        list(range(self.psf_compass.shape[0])),
                        list(range(self.psf.shape[0]))
                ], y=[
                        self.psf_compass[self.psf_compass.shape[0] / 2, :],
                        self.psf[self.psf.shape[0] / 2, :]
                ], color=["blue", "red"])
        self.dialog.visible = False

    def update_psf(self):
        self.dialog.visible = False
        self.dialog.content = "Updating PSF display..."
        self.dialog.visible = True
        psf_type = self.psf_display_select.value
        image = None
        if (psf_type == b"COMPASS"):
            image = np.log10(self.psf_compass)
        if (psf_type == b"Vii" or psf_type == b"ROKET"):
            image = np.log10(self.psf)
        if (psf_type == b"Fitting"):
            image = np.log10(self.psf_fitting)
        if (psf_type == b"OTF Telescope"):
            image = np.fft.fftshift(self.otftel)
        if (psf_type == b"OTF res"):
            image = np.fft.fftshift(self.otf2)

        if (image is not None):
            if (self.old):
                os.remove(self.old)

            time = str(datetime.datetime.now().strftime('%Y-%m-%d_%H_%M_%f'))
            self.old = "/home/fferreira/public_html/roket_display" + time + ".png"
            mpl.image.imsave(self.old, image)
            self.ppsf.image_url(
                    url=dict(value=self.url + time + ".png"), x=0, y=0, w=image.shape[0],
                    h=image.shape[0])

        self.dialog.visible = False

    def rescale_matrix(self):
        self.dialog.visible = False
        vmin = self.cmin.value
        vmax = self.cmax.value
        self.dialog.content = "Updating matrix..."
        self.dialog.visible = True
        if (self.old):
            os.remove(self.old)
        time = str(datetime.datetime.now().strftime('%Y-%m-%d_%H_%M_%f'))
        self.old = "/home/fferreira/public_html/roket_display" + time + ".png"
        mpl.image.imsave(self.old, self.covmat, vmin=vmin, vmax=vmax)
        self.p2.image_url(
                url=dict(value=self.url + time + ".png"), x=0, y=0,
                w=self.covmat.shape[0], h=self.covmat.shape[0])
        self.dialog.visible = False

    def get_diag(self):
        x = np.arange(self.covmat.shape[0])
        y = np.diag(self.covmat)
        self.source2.data = dict(x=x, y=y)

    def cut_matrix(self):
        XorY = self.XY.labels[self.XY.active]
        ax = self.axiscut.value
        if (XorY == b"X"):
            data = self.covmat[ax, :]
        else:
            data = self.covmat[:, ax]
        x = np.arange(data.size)
        self.source2.data = dict(x=x, y=data)

    def update_matrix2(self):
        self.dialog.visible = False
        if (self.old):
            os.remove(self.old)
        #self.draw.disabled = True
        A_val = self.A.value
        B_val = self.B.value
        basis = self.basis_select2.value
        powa = self.power.value
        self.dialog.content = "Computing and loading matrix..."
        self.dialog.visible = True

        A_cov = self.f[A_val][:]
        B_cov = self.f[B_val][:]
        A_cov -= np.tile(np.mean(A_cov, axis=1), (A_cov.shape[1], 1)).T
        B_cov -= np.tile(np.mean(B_cov, axis=1), (B_cov.shape[1], 1)).T
        if (basis == b"Btt"):
            A_cov = np.dot(self.P, A_cov)
            B_cov = np.dot(self.P, B_cov)
        print("Values ok")
        self.covmat = (np.dot(A_cov, B_cov.T) / B_cov.shape[1])
        print("dot product ok")
        if (powa != 1):
            self.covmat = np.abs(self.covmat)**powa * np.sign(self.covmat)
            print("scale adjusted")
        self.cmin.start = self.covmat.min()
        self.cmin.end = self.covmat.max()
        self.cmin.value = self.cmin.start
        self.cmin.step = (self.cmin.end - self.cmin.start) / 100.
        self.cmax.start = self.covmat.min()
        self.cmax.end = self.covmat.max()
        self.cmax.value = self.cmax.end
        self.cmax.step = self.cmin.step
        self.axiscut.end = self.covmat.shape[0]
        time = str(datetime.datetime.now().strftime('%Y-%m-%d_%H_%M_%f'))
        self.old = "/home/fferreira/public_html/roket_display" + time + ".png"
        mpl.image.imsave(self.old, self.covmat)
        self.p2.image_url(
                url=dict(value=self.url + time + ".png"), x=0, y=0,
                w=self.covmat.shape[0], h=self.covmat.shape[0])

        #self.sourceC.data = dict(url=[self.url],x=0,y=covmat.shape[0],dw=covmat.shape[0],dh=covmat.shape[0])
        #self.draw.disabled = False
        print("Matrix updated2")
        self.dialog.visible = False

    def update_mode(self):
        self.dialog.visible = False
        if (self.old):
            os.remove(self.old)
        N = self.modes_select.value
        if (N >= self.nmodes):
            N = self.nmodes - 1
            self.modes_select.value = N
        basis = self.basis_select3.value
        self.dialog.content = "Loading..."
        self.dialog.visible = True

        if (basis == b"Actuators"):
            pup = self.pup.flatten()
            pup[self.indx_pup] = self.IF[:, N].toarray()  #self.f["IF"][:][:,N]
            self.pup = pup.reshape(self.pup.shape)
        elif (basis == b"Btt"):
            pup = self.pup.flatten()
            pup[self.indx_pup] = self.IF[:, N - 2].dot(self.Btt)
            self.pup = pup.reshape(self.pup.shape)
        time = str(datetime.datetime.now().strftime('%Y-%m-%d_%H_%M_%f'))
        self.old = "/home/fferreira/public_html/roket_display" + time + ".png"
        mpl.image.imsave(self.old, self.pup)
        self.pmodes.image_url(
                url=dict(value=self.url + time + ".png"), x=0, y=0, w=self.pup.shape[0],
                h=self.pup.shape[0])

        #self.sourceC.data = dict(url=[self.url],x=0,y=covmat.shape[0],dw=covmat.shape[0],dh=covmat.shape[0])
        #self.draw.disabled = False
        print("Mode updated")
        self.dialog.visible = False

    def mode_increment(self):
        if (self.modes_select.value < self.nmodes - 1):
            self.modes_select.value = self.modes_select.value + 1
        else:
            self.modes_select.value = self.nmodes - 1

    def mode_desincrement(self):
        if (self.modes_select.value > 0):
            self.modes_select.value = self.modes_select.value - 1
        else:
            self.modes_select.value = 0

    def plot_sum(self):

        self.dialog.visible = False
        self.dialog.content = "Computing..."
        self.dialog.visible = True

        plus = self.plus_select.active
        moins = self.moins_select.active
        basis_val = self.basis_select1.value
        plot_val = self.plot_select.value
        iteration = int(self.iter_select.value)

        if (plot_val == b"Commands"):
            data = np.zeros(self.nactus)
            x = list(range(self.nactus))
        elif (plot_val == b"Variance"):
            data = np.zeros((self.nmodes, self.niter))  #self.nmodes)
            data2 = np.zeros(self.nmodes)
            x = list(range(self.nmodes))
        fitp = False
        fitm = False
        for i in plus:
            self.dialog.content = "Computing " + self.plus_select.labels[i]
            if (self.plus_select.labels[i] != "CORRECT"):
                if (self.plus_select.labels[i] == b"fitting"):
                    fitp = True
                else:
                    if (plot_val == b"Commands"):
                        data += np.dot(self.P,
                                       self.f[self.coms_list[i]][:][:, iteration])
                    elif (plot_val == b"Variance"):
                        data += np.dot(self.P, self.f[self.coms_list[i]][:])
                        data2 += np.var(
                                np.dot(self.P, self.f[self.coms_list[i]][:]), axis=1)
            else:
                theta = np.arctan(
                        self.f.attrs["wfs.ypos"][0] / self.f.attrs["wfs.xpos"][0])
                theta -= (self.f.attrs["winddir"][0] * np.pi / 180.)
                r0 = self.f.attrs["r0"] * (self.f.attrs["target.Lambda"][0] /
                                           self.f.attrs["wfs.Lambda"][0])**(6. / 5.)
                RASC = 180 / np.pi * 3600.
                Dtomo = 0
                Dbp = 0
                Dcov = 0
                dt = self.f.attrs["ittime"]
                g = self.f.attrs["gain"][0]
                for k in range(self.f.attrs["nscreens"]):
                    H = self.f.attrs["atm.alt"][k]
                    v = self.f.attrs["windspeed"][k]
                    frac = self.f.attrs["frac"][k]
                    htheta = np.sqrt(self.f.attrs["wfs.xpos"][0]**2 +
                                     self.f.attrs["wfs.ypos"][0]**2) / RASC * H
                    Dtomo += 6.88 * (htheta / r0)**(5. / 3.)
                    Dbp += 6.88 * (v * dt / g / r0)**(5. / 3.)
                    rho = np.sqrt(htheta**2 + (v * dt / g)**2 -
                                  2 * htheta * v * dt / g * np.cos(np.pi - theta))
                    Dcov += 6.88 * (rho / r0)**(5. / 3.)
                covar = (Dbp + Dtomo - Dcov) * 0.5 * frac

                data2 += 2 * covar / (2 * np.pi / self.Lambda_tar)**2 / self.nmodes
                #data2 += 2*np.sqrt(np.var(np.dot(self.P,self.f["tomography"]),axis=1))*np.sqrt(np.var(np.dot(self.P,self.f["bandwidth"]),axis=1))*np.cos(theta)
        for i in moins:
            if (self.plus_select.labels[i] == b"fitting"):
                fitm = True
            else:
                if (plot_val == b"Commands"):
                    data -= np.dot(self.P, self.f[self.coms_list[i]][:][:, iteration])
                elif (plot_val == b"Variance"):
                    data -= np.dot(self.P, self.f[self.coms_list[i]][:])
                    data2 -= np.var(np.dot(self.P, self.f[self.coms_list[i]][:]), axis=1)


#        if(basis_val == b"Btt"):
#            data = np.dot(self.P,data)
#            data2 = np.dot(self.P,data2)
        if (plot_val == b"Variance"):
            data = np.var(data, axis=1)
            data = np.cumsum(data[self.swap])
            # theta = np.arctan(self.f.attrs["wfs.ypos"][0]/self.f.attrs["wfs.xpos"][0])
            # if(np.sign(self.f.attrs["wfs.ypos"][0])<0):
            #     theta += np.pi*0.
            # theta -= (self.f.attrs["winddir"][0] * np.pi/180.)
            # data2 += 2*np.sqrt(np.var(np.dot(self.P,self.f["tomography"]),axis=1))*np.sqrt(np.var(np.dot(self.P,self.f["bandwidth"]),axis=1))*np.cos(theta)
            data2 = np.cumsum(data2[self.swap])
            data2 = np.exp(-data2 * (2 * np.pi / self.Lambda_tar)**2)
            print("data2 : ", data2)
            data = np.exp(-data * (2 * np.pi / self.Lambda_tar)**2)
            if (fitp and list(self.f.keys()).count("fitting")):
                data *= np.exp(-self.f["fitting"].value)
                data2 *= np.exp(-self.f["fitting"].value)
                print("data2 : ", data2)
            if (fitm and list(self.f.keys()).count("fitting")):
                data /= np.exp(-self.f["fitting"].value)
                data2 /= np.exp(-self.f["fitting"].value)
        if (list(self.f.keys()).count("SR2")):
            self.source3.data = dict(
                    x=[x, x, x, x], y=[
                            data,
                            np.ones(len(x)) * self.f["SR"].value,
                            np.ones(len(x)) * self.f["SR2"].value, data2
                    ], color=["blue", "red", "purple", "green"])
        else:
            if (list(self.f.keys()).count("SR")):
                self.source3.data = dict(
                        x=[x, x,
                           x], y=[data,
                                  np.ones(len(x)) * self.f["SR"].value, data2],
                        color=["blue", "red", "green"])
            else:
                self.source3.data = dict(x=x, y=data)
        print("Sum plotted")
        self.dialog.visible = False

    def cov_cor(self):
        cov = np.zeros((6, 6))
        bufdict = {
                "0": self.f["noise"][:],
                "1": self.f["non linearity"][:],
                "2": self.f["aliasing"][:],
                "3": self.f["filtered modes"][:],
                "4": self.f["bandwidth"][:],
                "5": self.f["tomography"][:]
        }
        for i in range(cov.shape[0]):
            for j in range(cov.shape[1]):
                if (j >= i):
                    tmpi = self.P.dot(bufdict[str(i)])
                    tmpj = self.P.dot(bufdict[str(j)])
                    cov[i, j] = np.sum(
                            np.mean(tmpi * tmpj, axis=1) -
                            np.mean(tmpi, axis=1) * np.mean(tmpj, axis=1))
                else:
                    cov[i, j] = cov[j, i]

        s = np.reshape(np.diag(cov), (cov.shape[0], 1))
        sst = np.dot(s, s.T)
        cor = cov / np.sqrt(sst)

        return cov, cor

    def createDataTables(self):

        tmp = [TableColumn(field="Type", title="Covariance")]
        for item in self.DataTableItems[1:]:
            tmp.append(TableColumn(field=item, title=item))
        columns = tmp

        cov_table = DataTable(source=self.table_cov_source, columns=columns, width=1200,
                              height=280)
        tmp[0] = TableColumn(field="Type", title="Correlation")
        cor_table = DataTable(source=self.table_cor_source, columns=columns, width=1200,
                              height=280)

        tmp = [
                TableColumn(field="Parameter", title="Parameter"),
                TableColumn(field="Value", title="Value")
        ]
        param_table = DataTable(source=self.table_param_source, columns=tmp, width=700,
                                height=500)

        return cov_table, cor_table, param_table

    def updateDataTables(self):
        self.table_cov_source.data = dict(
                Type=self.DataTableItems[1:], Noise=self.cov[:, 0],
                Truncature=self.cov[:, 1], Aliasing=self.cov[:, 2],
                FilteredModes=self.cov[:, 3], Bandwidth=self.cov[:, 4],
                Tomography=self.cov[:, 5])
        self.table_cor_source.data = dict(
                Type=self.DataTableItems[1:], Noise=self.cor[:, 0],
                Truncature=self.cor[:, 1], Aliasing=self.cor[:, 2],
                FilteredModes=self.cor[:, 3], Bandwidth=self.cor[:, 4],
                Tomography=self.cor[:, 5])
        params = list(self.f.attrs.keys())
        params.sort()
        values = []
        for k in params:
            values.append(self.f.attrs[k])
        self.table_param_source.data = dict(Parameter=params, Value=values)

        self.pcov_source.data = dict(image=[self.cov], x=[0], y=[0], dw=[6], dh=[6],
                                     palette="Spectral11")
        self.pcor_source.data = dict(image=[self.cor], x=[0], y=[0], dw=[6], dh=[6],
                                     palette="Spectral11")

files = glob.glob("/home/fferreira/public_html/roket_display*")
for f in files:
    os.remove(f)

disp = html_display()

# initial load of the data
