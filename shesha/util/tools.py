## @package   shesha.util.tools
## @brief     Imported from CANARY
## @author    COMPASS Team <https://github.com/ANR-COMPASS>
## @version   5.1.0
## @date      2020/05/18
## @copyright GNU Lesser General Public License
#
#  This file is part of COMPASS <https://anr-compass.github.io/compass/>
#
#  Copyright (C) 2011-2019 COMPASS Team <https://github.com/ANR-COMPASS>
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

import numpy as np

import shlex
from subprocess import Popen, PIPE  # , call
from sys import stdout
from time import sleep

# from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import matplotlib.pyplot as plt


def clr(*figs):
    """
    THE Fab function

    clears the current figure (no arg) or specified window

    """
    if (figs):
        for fig in figs:
            # fig = fig[i]
            plt.figure(num=fig)
            plt.clf()
    else:
        plt.clf()


def system(cmd, output=False):
    """
    Execute the external command
    system("ls")


    out = system("ls", out=True)
    out = system("ls -l", out=True)




    and get its stdout exitcode and stderr.
    """
    args = shlex.split(cmd)
    proc = Popen(args, stdout=PIPE, stderr=PIPE)
    out, err = proc.communicate()
    exitcode = proc.returncode
    #
    if ('\n' in out):
        out = out.split('\n')[:-1]

    for i in range(len(out)):
        print((out[i]))

    if (output):
        # print("here")
        return out, exitcode, err


def pli(data, color='gist_earth', cmin=9998, cmax=9998, win=1, origin=None,
        aspect='equal'):
    """
    plots the transpose of the data

    color maps keywords can be found in
    http://wiki.scipy.org/Cookbook/Matplotlib/Show_colormaps

    """
    options = ''
    if (cmin != 9998):
        exec('options += ",vmin=cmin"')

    if (cmax != 9998):
        exec('options += ",vmax=cmax"')

    if (color == b'yorick'):
        color = 'gist_earth'
    if (origin is None):
        origin = ""
    if (aspect != 'auto'):
        aspect = "\'" + aspect + "\'"
    else:
        aspect = "\'auto\'"

    exec('plt.matshow(data, aspect=' + aspect + ', fignum=win, cmap=color' + options +
         origin + ")")


def binning(w, footprint):

    # the averaging block
    # prelocate memory
    binned = np.zeros(w.shape[0] * w.shape[1]).reshape(w.shape[0], w.shape[1])
    # print(w)
    for i in range(w.shape[0]):
        for j in range(w.shape[1]):
            binned[i, j] = w[i, j].sum() / (footprint * footprint + 0.0)

    return binned


def minmax(tab):
    tabVect = np.reshape(tab, tab.size)
    return [np.min(tabVect), np.max(tabVect)]


def plg(
        data,
        x="",
        win=1,
        xlog=0,
        ylog=0,
        color="black",
):
    """


    color = "green"
    color = "0.71" [0-1] gray scale
    color = '#eeefff'
    See also:

    http://matplotlib.org/api/colors_api.html

    """
    fig = plt.figure(win)
    ax = fig.add_subplot(1, 1, 1)
    try:
        data.ndim
        if (data.ndim > 1):
            print(("Warning %dD dimensions. Cannot plot data. Use pli instead. " %
                   data.ndim))
    except:
        return
    if (x == ""):
        ax.plot(data, color=color)
    else:
        ax.plot(x, data, color=color)

    if (xlog == 1):
        ax.set_xscale('log')
    else:
        ax.set_xscale('linear')

    if (ylog == 1):
        ax.set_yscale('log')
    else:
        ax.set_yscale('linear')
    fig.show()
    return fig, ax


def zcen(data):
    data = np.array(data)
    if (len(data.shape) > 1):
        print("oups zcen with dims > 1 not coded yet...")
        return 0
    tmp = tmp2 = []
    for i in range(len(data) - 1):
        tmp = (float(data[i]) + float(data[i + 1])) / 2.
        tmp2 = np.append(tmp2, tmp)
    return tmp2


def getValidSubapArray(nssp, rext, rint, return2d=False):
    # The Grata case, tip-tilt sensor only.
    if (nssp == 1):
        return [1]
    # to avoid some bug that eliminates useful central subapertures when
    # obs=0.286
    if ((nssp == 7) and (rint > 0.285 and rint < 0.29)):
        rint = 0.285
        print("cas particulier")
    x = zcen(np.linspace(-1, 1, num=nssp + 1))
    xx = []
    for i in range(nssp):
        xx = np.hstack((xx, x))
    x = np.reshape(xx, (nssp, nssp))
    y = np.transpose(x)
    r = np.sqrt(x * x + y * y)
    valid2dext = ((r < rext)) * 1
    valid2dint = ((r >= rint)) * 1
    valid2d = valid2dint * valid2dext

    if (return2d):
        return valid2d
    else:
        valid = np.reshape(valid2d, [nssp * nssp])

    return valid.tolist()


"""
def plsh(slopesvector,  nssp=14,  rmax=0.98, obs=0, win=1, invertxy=False):

    tmp = getValidSubapArray( nssp, rmax, obs);
    X,Y = meshgrid(np.linspace(-1, 1, nssp), np.linspace(-1, 1, nssp))
    vx = np.zeros([nssp*nssp])
    vy = np.zeros([nssp*nssp])
    hart = where(tmp)[0]
    vx.flat[hart] = slopesvector.flat[0:len(slopesvector)/2]
    vy.flat[hart] = slopesvector.flat[len(slopesvector)/2+1:]
    vx = vx.reshape([nssp, nssp])
    vy = vy.reshape([nssp, nssp])

    figure(num=win)
    if(invertxy):
        Q = quiver(X,Y, vy, vx)
    else:
        Q = quiver(X,Y, vx, vy)
    #qk = quiverkey(Q, 0.5, 0.92, 2, r'$2 \frac{m}{s}$', labelpos='W', fontproperties={'weight': 'bold'})
    l,r,b,t = axis()
    dx, dy = r-l, t-b
    axis([l-0.05*dx, r+0.05*dx, b-0.05*dy, t+0.05*dy]) # MUST DO OTHERWISE THE AUTOSCALE CAN MISS SOME ARROWS
    #title('Minimal arguments, no kwargs')
"""


def plpyr(slopesvector, validArray):
    """
    wao.config.p_wfss[0]._isvalid
    """
    nslopes = slopesvector.shape[0] / 2
    x, y = np.where(validArray.T)
    plt.quiver(x, y, slopesvector[0:nslopes], slopesvector[nslopes:])


def plsh(slopesvector, nssp, validint, sparta=False, invertxy=False, returnquiver=False):
    """
    <slopesvector> is the input vector of slopes
    <nssp> is the number of subapertures in the diameter of the pupil
    <validint> is the normalized diameter of central obscuration (between 0 and 1.00)
    <sparta> when==1, slopes are ordered xyxyxyxy...
             when==0, slopes are xxxxxxyyyyyyy
    <xy> when==1, swap x and y. Does nothing special when xy==0.

    The routine plots a field vector of subaperture gradients defined in
    vector <slopesvector>.
    The routine automatically adjusts/finds what are the valid subapertures
    for plotting, depending on the number of elements in <slopesvector>. Only the
    devalidated subapertures inside the central obscuration cannot be
    known, that’s why <validint> has to be passed in the argument list.

    """
    nsub = slopesvector.shape[0] // 2
    x = np.linspace(-1, 1, nssp)
    x, y = np.meshgrid(x, x)
    r = np.sqrt(x * x + y * y)
    # defines outer and inner radiuses that will decide of validity of subapertures
    # inner radius <validint> is passed as an argument.
    # outer one will be computed so that it will match the number of
    # subapertures in slopesvector
    rorder = np.sort(r.reshape(nssp * nssp))
    # number of subapertures not valid due to central obscuration
    ncentral = nssp * nssp - np.sum(r >= validint, dtype=np.int32)
    # determine value of external radius so that the test (validint < r < validext)
    # leads to the correct number of subapertures
    validext = rorder[ncentral + nsub]
    # get the indexes of valid subapertures in the nsspxnssp map
    valid = (r < validext) & (r >= validint)
    ivalid = np.where(valid)
    # feeding data <slopesvector> into <vv>
    vx = np.zeros([nssp, nssp])
    vy = np.zeros([nssp, nssp])
    if (sparta is False):
        # Canary, compass, etc..  slopes ordered xxxxxxxyyyyyyy
        vy[ivalid] = slopesvector[0:nsub]
        vx[ivalid] = slopesvector[nsub:]
    else:
        # SPARTA case, slopes ordered xyxyxyxyxyxy...
        vx[ivalid] = slopesvector[0::2]
        vy[ivalid] = slopesvector[1::2]
    if (invertxy is True):
        # swaps X and Y
        tmp = vx
        vx = vy
        vy = tmp
    if (returnquiver):
        return x, y, vx, vy
    else:
        plt.quiver(x, y, vx, vy, pivot='mid')


def pl3d(im):
    """
    ir = pyfits.get_data("/home/fvidal/data/Run2015/June2015_27_onsky/ir/ir_2015-06-28_06h27m40s_script44_gain.fits")

    JAMAIS TESTEE !!!!!!!!!!!!!!

    """
    X = np.arange(-5, 5, 0.25)
    Y = np.arange(-5, 5, 0.25)
    X, Y = np.meshgrid(X, Y)
    Z = im
    plt.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap=cm.jet)
    plt.show()


def FFThz(signal, fe, freq=0):
    """ PSD = FFThz( signal, fe )   OU  f = FFThz( 1024, fe, freq=1 )
    On the first form, returns the power spectral density of signal.
    If signal has units 'u', the PSD has units 'u^2/Hz'.
    The frequency axis can be get by using the keyword freq=1."""
    if freq == 1:
        n = signal.size
        d = np.linspace(0, fe, n + 1)[0:n / 2 + 1]
        return d[1:]
    else:
        n = signal.size
        d = np.abs(np.fft.fft(signal))[0:n / 2 + 1]
        d = d**2 / (fe * n / 2)
        d[n / 2] /= 2
        return d[1:]


def computePSD(zerall, fe, izerNum, wfsNum):
    if np.isscalar(wfsNum):
        wfsNum = [wfsNum]

    for ii in wfsNum:
        PSD = FFThz(zerall[ii][izerNum, :], fe)

    PSD /= len(wfsNum)
    if (len(wfsNum) > 1):
        ff = FFThz(zerall[wfsNum][izerNum, :], fe, freq=1)
    else:
        ff = FFThz(zerall[wfsNum[0]][izerNum, :], fe, freq=1)

    return PSD, ff


def countExample(seconds):
    for i in range(1, int(seconds)):
        stdout.write("\r%d" % i)
        stdout.flush()
        sleep(1)
    stdout.write("\n")


def plotSubapRectangles(pup, isvalid, istart, jstart):
    fig = plt.matshow(pup)
    pdiam = istart[1] - istart[0]
    for i in istart:
        for j in jstart:
            if (isvalid[i // pdiam, j // pdiam]):
                color = "green"
            else:
                color = "red"
            fig.axes.add_patch(
                    plt.Rectangle((i - 0.5, j - 0.5), pdiam, pdiam, fill=False,
                                  color=color))
