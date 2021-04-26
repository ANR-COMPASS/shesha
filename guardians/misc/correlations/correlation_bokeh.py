"""
Created on Wed Oct 5 14:28:23 2016

@author: fferreira
"""

import numpy as np
import h5py
import pandas
from bokeh.plotting import Figure, figure
from bokeh.models import Range1d, ColumnDataSource, HoverTool
from bokeh.models.widgets import Select, Slider, CheckboxButtonGroup, Panel, Tabs, Button, Dialog, Paragraph, RadioButtonGroup, TextInput
from bokeh.io import curdoc
from bokeh.models.layouts import HBox, VBox
from bokeh.models.widgets import DataTable, DateFormatter, TableColumn
from bokeh.client import push_session
import glob
import matplotlib.pyplot as plt
from scipy.special import jv  # Bessel function
plt.ion()


def dphi_highpass(r, x0, tabx, taby):
    return (r**
            (5. / 3.)) * (1.1183343328701949 - Ij0t83(r * (np.pi / x0), tabx, taby)) * (
                    2 * (2 * np.pi)**(8 / 3.) * 0.0228956)


def dphi_lowpass(r, x0, L0, tabx, taby):
    return rodconan(r, L0) - dphi_highpass(r, x0, tabx, taby)
    #return (r**(5./3.)) * Ij0t83(r*(np.pi/x0), tabx, taby) * (2*(2*np.pi)**(8/3.)*0.0228956)


def Ij0t83(x, tabx, taby):
    if (x < np.exp(-3.0)):
        return 0.75 * x**(1. / 3) * (1 - x**2 / 112.)
    else:
        return np.interp(x, tabx, taby)


def unMoinsJ0(x):
    # if(x<0.1):
    #     x22 = (x/2.)**2
    #     return (1-x22/4.)*x22
    # else:
    return 1 - jv(0, x)


def tabulateIj0(L0):
    n = 10000
    t = np.linspace(-4, 10, n)
    dt = (t[-1] - t[0]) / (n - 1)
    smallx = np.exp(-4.0)
    A = 0.75 * smallx**(1. / 3) * (1 - smallx**2 / 112.)
    X = np.exp(t)
    #Y = np.exp(-t*(5./3.))*unMoinsJ0(X)
    Y = (np.exp(2 * t) + (1. / L0)**2)**(-8. / 6.) * unMoinsJ0(X) * np.exp(t)
    Y[1:] = np.cumsum(Y[:-1] + np.diff(Y) / 2.)
    Y[0] = 0.
    Y = Y * dt + A

    return X, Y


def asymp_macdo(x):
    k2 = 1.00563491799858928388289314170833
    k3 = 1.25331413731550012081
    a1 = 0.22222222222222222222
    a2 = -0.08641975308641974829
    a3 = 0.08001828989483310284

    x_1 = 1. / x
    res = k2 - k3 * np.exp(-x) * x**(1. / 3.) * (1.0 + x_1 * (a1 + x_1 *
                                                              (a2 + x_1 * a3)))
    return res


def macdo(x):
    a = 5. / 6.
    x2a = x**(2. * a)
    x22 = x * x / 4.
    s = 0.0

    Ga = [
            0, 12.067619015983075, 5.17183672113560444, 0.795667187867016068,
            0.0628158306210802181, 0.00301515986981185091, 9.72632216068338833e-05,
            2.25320204494595251e-06, 3.93000356676612095e-08, 5.34694362825451923e-10,
            5.83302941264329804e-12
    ]

    Gma = [
            -3.74878707653729304, -2.04479295083852408, -0.360845814853857083,
            -0.0313778969438136685, -0.001622994669507603, -5.56455315259749673e-05,
            -1.35720808599938951e-06, -2.47515152461894642e-08, -3.50257291219662472e-10,
            -3.95770950530691961e-12, -3.65327031259100284e-14
    ]

    x2n = 0.5

    s = Gma[0] * x2a
    s *= x2n

    x2n *= x22

    for n in np.arange(10) + 1:

        s += (Gma[n] * x2a + Ga[n]) * x2n
        x2n *= x22

    return s


def rodconan(r, L0):
    res = 0
    k1 = 0.1716613621245709486
    dprf0 = (2 * np.pi / L0) * r
    if (dprf0 > 4.71239):
        res = asymp_macdo(dprf0)
    else:
        res = -macdo(dprf0)

    res *= k1 * L0**(5. / 3.)

    return res


def variance(f, contributors, method="Default"):
    """ Return the error variance of specified contributors
    params:
        f : (h5py.File) : roket hdf5 file opened with h5py
        contributors : (list of string) : list of the contributors
        method : (optional, default="Default") : if "Independence", the
                    function returns ths sum of the contributors variances.
                    If "Default", it returns the variance of the contributors sum
    """
    P = f["P"][:]
    nmodes = P.shape[0]
    swap = np.arange(nmodes) - 2
    swap[0:2] = [nmodes - 2, nmodes - 1]
    if (method == b"Default"):
        err = f[contributors[0]][:] * 0.
        for c in contributors:
            err += f[c][:]
        return np.var(P.dot(err), axis=1)[swap], np.var(
                P.dot(f["tomography"][:]),
                axis=1)[swap], np.var(P.dot(f["bandwidth"][:]), axis=1)[swap]

    elif (method == b"Independence"):
        nmodes = P.shape[0]
        v = np.zeros(nmodes)
        for c in contributors:
            v += np.var(P.dot(f[c][:]), axis=1)
        return v[swap]

    else:
        raise TypeError("Wrong method input")


def varianceMultiFiles(fs, frac_per_layer, contributors):
    """ Return the variance computed from the sum of contributors of roket
    files fs, ponderated by frac
    params:
        fs : (list) : list of hdf5 files opened with h5py
        frac_per_layer : (dict) : frac for each layer
        contributors : (list of string) : list of the contributors
    return:
        v : (np.array(dim=1)) : variance vector
    """
    f = fs[0]
    P = f["P"][:]
    nmodes = P.shape[0]
    swap = np.arange(nmodes) - 2
    swap[0:2] = [nmodes - 2, nmodes - 1]
    err = f[contributors[0]][:] * 0.
    for f in fs:
        frac = frac_per_layer[f.attrs["atm.alt"][0]]
        for c in contributors:
            err += np.sqrt(frac) * f[c][:]

    return np.var(P.dot(err), axis=1)[swap]


def cumulativeSR(v, Lambda_tar):
    """ Returns the cumulative Strehl ratio over the modes from the variance
    on each mode
    params:
        v : (np.array(dim=1)) : variance vector
    return:
        s : (np.array(dim=1)) : cumulative SR
    """
    s = np.cumsum(v)
    s = np.exp(-s * (2 * np.pi / Lambda_tar)**2)

    return s


def update(attrs, old, new):
    speed = speed_select.value
    direction = dir_select.value
    g = gain_select.value
    xname = xaxis_select.value
    yname = yaxis_select.value

    ydata = ymap[yname]
    x = xmap[xname]

    ind = np.ones(ydata.shape[0])
    if (direction != "All"):
        ind *= (xmap["Winddir"] == float(direction))
    if (speed != "All"):
        ind *= (xmap["Windspeed"] == float(speed))
    if (g != "All"):
        ind *= (xmap["Gain"] == float(g))

    ind = np.where(ind)
    if (yname == b"Var(t)"):
        Hthetak = Htheta / xmap["Gain"]
        y_model = np.ones(ind[0].shape[0])
        #y_model = y_model * 6.88 * (Htheta/r0)**(5./3.) * 0.5
        for k in range(ind[0].shape[0]):
            y_model[k] = dphi_lowpass(Htheta, 0.2, L0, tabx, taby) * (1 / r0)**(
                    5. / 3.) * 0.5  #* xmap["Gain"][ind][k]**2
    if (yname == b"Var(bp)"):
        vdt = xmap["Windspeed"] * dt / xmap["Gain"]
        y_model = np.zeros(vdt[ind].shape[0])
        for k in range(vdt[ind].shape[0]):
            y_model[k] = dphi_lowpass(vdt[ind][k], 0.2, L0, tabx,
                                      taby) * (1. / r0)**(5. / 3.) * 0.5
    if (yname == b"Covar"):
        vdt = xmap["Windspeed"] * dt / xmap["Gain"]
        Hthetak = Htheta / xmap["Gain"]
        gamma = np.arctan2(ypos, xpos) - xmap["Winddir"] * np.pi / 180.
        rho = np.sqrt(Htheta**2 + (vdt)**2 - 2 * Htheta * vdt * np.cos(gamma))
        Drho = np.zeros(rho[ind].shape[0])
        Dt = Drho.copy()
        for k in range(rho[ind].shape[0]):
            Drho[k] = dphi_lowpass(rho[ind][k], 0.2, L0, tabx,
                                   taby) * (1 / r0)**(5. / 3.)
        #Drho = 6.88 * (rho[ind]/r0)**(5./3.)
        for k in range(Dt.shape[0]):
            Dt[k] = dphi_lowpass(Htheta, 0.2, L0, tabx, taby) * (1 / r0)**(
                    5. / 3.)  # * xmap["Gain"][ind][k]**2
        #Dt =  6.88 * (Htheta/r0)**(5./3.)
        Dbp = np.zeros(vdt[ind].shape[0])
        for k in range(vdt[ind].shape[0]):
            Dbp[k] = dphi_lowpass(vdt[ind][k], 0.2, L0, tabx, taby) * (1 / r0)**(5. / 3.)
        #Dbp = 6.88 * (vdt[ind]/r0) ** (5./3.)
        y_model = 0.5 * (Dt + Dbp - Drho)

    source.data = dict(x=x[ind], y=ydata[ind], speed=xmap["Windspeed"][ind],
                       theta=xmap["Winddir"][ind], gain=xmap["Gain"][ind])
    source_model.data = dict(x=x[ind], y=y_model, speed=xmap["Windspeed"][ind],
                             theta=xmap["Winddir"][ind], gain=xmap["Gain"][ind])


datapath = "/home/fferreira/Data/correlation/"
filenames = glob.glob(datapath + "roket_8m_1layer_dir*_cpu.h5")

files = []
for f in filenames:
    ff = h5py.File(f, mode='r')
    #if(ff.attrs["validity"]):
    files.append(ff)

nmodes = (files[0])["P"][:].shape[0]
xpos = files[0].attrs["wfs.xpos"][0]
ypos = files[0].attrs["wfs.ypos"][0]
contributors = ["tomography", "bandwidth"]
Lambda_tar = files[0].attrs["target.Lambda"][0]
Lambda_wfs = files[0].attrs["wfs.Lambda"][0]
L0 = files[0].attrs["L0"][0]
dt = files[0].attrs["ittime"]
H = files[0].attrs["atm.alt"][0]
RASC = 180 / np.pi * 3600.
Htheta = np.linalg.norm(
        [xpos, ypos]
) / RASC * H  # np.sqrt(2)*4/RASC*H # Hardcoded for angular separation of sqrt(2)*4 arcsec
r0 = files[0].attrs["r0"] * (Lambda_tar / Lambda_wfs)**(6. / 5.)
nfiles = len(files)
data = np.zeros((nmodes, 4, nfiles))
theta = np.zeros(nfiles)
speeds = np.zeros(nfiles)
gain = np.zeros(nfiles)

tabx, taby = tabulateIj0(L0)

# data[:,0,i] = var(tomo+bp) for file #i
# data[:,1,i] = var(tomo) for file #i
# data[:,2,i] = var(bp) for file #i
# data[:,3,i] = var(tomo)+var(bp) for file #i
ind = 0
for f in files:
    data[:, 0, ind], data[:, 1, ind], data[:, 2, ind] = variance(f, contributors)
    data[:, 3, ind] = variance(f, contributors, method="Independence")
    theta[ind] = f.attrs["winddir"][0]
    speeds[ind] = f.attrs["windspeed"][0]
    gain[ind] = float('%.1f' % f.attrs["gain"][0])
    ind += 1
data = data * ((2 * np.pi / Lambda_tar)**2)
covar = (data[:, 0, :] - data[:, 3, :]) / 2.

xaxis_select = Select(title="X-axis", value="Windspeed",
                      options=["Windspeed", "Winddir", "Gain"])
yaxis_select = Select(
        title="Y-axis", value="Covar",
        options=["Covar", "Var(t+bp)", "Var(t)", "Var(bp)", "Var(t)+Var(bp)"])

speed_select = Select(title="Windspeeds", value="All",
                      options=["All"] + [str(s) for s in np.unique(speeds)])
dir_select = Select(title="Winddirs", value="All",
                    options=["All"] + [str(s) for s in np.unique(theta)])
gain_select = Select(title="Gain", value="All",
                     options=["All"] + [str(s)[:3] for s in np.unique(gain)])
source = ColumnDataSource(data=dict(x=[], y=[], speed=[], theta=[], gain=[]))
source_model = ColumnDataSource(data=dict(x=[], y=[], speed=[], theta=[], gain=[]))
hover = HoverTool(tooltips=[("Speed", "@speed"), ("Winddir", "@theta"), ("Gain",
                                                                         "@gain")])
TOOLS = "resize,save,pan,box_zoom,tap, box_select, wheel_zoom, lasso_select,reset"

p = figure(plot_height=600, plot_width=700, title="", tools=[hover, TOOLS])
p.circle(x="x", y="y", source=source, size=7, color="blue")
p.circle(x="x", y="y", source=source_model, size=7, color="red")

xmap = {"Windspeed": speeds, "Winddir": theta, "Gain": gain}
ymap = {
        "Covar": np.sum(covar, axis=0),
        "Var(t+bp)": np.sum(data[:, 0, :], axis=0),
        "Var(t)": np.sum(data[:, 1, :], axis=0),
        "Var(bp)": np.sum(data[:, 2, :], axis=0),
        "Var(t)+Var(bp)": np.sum(data[:, 3, :], axis=0)
}

buttons = [xaxis_select, speed_select, dir_select, yaxis_select, gain_select]
for b in buttons:
    b.on_change('value', update)

curdoc().clear()
update(None, None, None)
curdoc().add_root(
        HBox(VBox(xaxis_select, yaxis_select, speed_select, dir_select, gain_select), p))
