# -*- coding: utf-8 -*-
"""
Created on Wed Oct 9 14:03:29 2017

@author: sdurand
"""
# import cProfile
# import pstats as ps
#@profile
import sys
import os
# import numpy as np
import carmaWrap as ch
import shesha as ao
import time
import matplotlib.pyplot as plt
import hdf5_util as h5u
import numpy as np
plt.ion()
sys.path.append('/home/sdurand/hracode/codes/PYRCADO/Python')
import PYRCADOCALIB as pyrcalib
from astropy.io import fits
from numba import autojit
#import gnumpy as gpu

print("TEST SHESHA\n closed loop: call loop(int niter)")

if (len(sys.argv) != 2):
    error = 'command line should be:"python -i test.py parameters_filename"\n with "parameters_filename" the path to the parameters file'
    raise StandardError(error)

# get parameters from file
param_file = sys.argv[1]
if (param_file.split('.')[-1] == "py"):
    filename = param_file.split('/')[-1]
    param_path = param_file.split(filename)[0]
    sys.path.insert(0, param_path)
    exec("import %s as config" % filename.split(".py")[0])
    sys.path.remove(param_path)
# elif (param_file.split('.')[-1] == "h5"):
#     sys.path.insert(0, os.environ["SHESHA_ROOT"] + "/data/par/par4bench/")
#     import scao_16x16_8pix as config
#     sys.path.remove(os.environ["SHESHA_ROOT"] + "/data/par/par4bench/")
#     h5u.configFromH5(param_file, config)
else:
    raise ValueError("Parameter file extension must be .py or .h5")

print("param_file is", param_file)

if (hasattr(config, "simul_name")):
    if (config.simul_name is None):
        simul_name = ""
    else:
        simul_name = config.simul_name
        print("simul name is", simul_name)
else:
    simul_name = ""

clean = 1
matricesToLoad = {}
if (simul_name != ""):
    clean = 0
    param_dict = h5u.params_dictionary(config)
    matricesToLoad = h5u.checkMatricesDataBase(os.environ["SHESHA_ROOT"] + "/data/",
                                               config, param_dict)

# initialisation:

#   context
# c = ch.carmaWrap_context(0)
# c = ch.carmaWrap_context(devices=np.array([0,1], dtype=np.int32))
# c.set_active_device(0) #useful only if you use ch.carmaWrap_context()
c = ch.carmaWrap_context(devices=config.p_loop.devices)
#    wfs
print("->wfs")
wfs, tel = ao.wfs_init(config.p_wfss, config.p_atmos, config.p_tel, config.p_geom,
                       config.p_target, config.p_loop, config.p_dms)

#   atmos
print("->atmos")
atm = ao.atmos_init(c, config.p_atmos, config.p_tel, config.p_geom, config.p_loop,
                    config.p_wfss, wfs, config.p_target, clean=clean,
                    load=matricesToLoad)

#   dm
print("->dm")
dms = ao.dm_init(config.p_dms, config.p_wfss, wfs, config.p_geom, config.p_tel)

#   target
print("->target")
tar = ao.target_init(c, tel, config.p_target, config.p_atmos, config.p_geom,
                     config.p_tel, config.p_dms, config.p_wfss)

print("->rtc")
#   rtc
rtc = ao.rtc_init(tel, wfs, config.p_wfss, dms, config.p_dms, config.p_geom,
                  config.p_rtc, config.p_atmos, atm, config.p_tel, config.p_loop,
                  clean=clean, simul_name=simul_name, load=matricesToLoad)

if not clean:
    h5u.validDataBase(os.environ["SHESHA_ROOT"] + "/data/", matricesToLoad)

print("====================")
print("init done")
print("====================")
print("objects initialzed on GPU:")
print("--------------------------------------------------------")
print(atm)
print(wfs)
print(dms)
print(tar)
print(rtc)


def import_im(nb_im, path):
    im = fits.open(path)
    size = im[0].data.shape[0]
    pyr_im_cube = np.zeros((nb_im, size, size), dtype=np.float32)
    for i in range(nb_im):
        pyr_im_cube[i] = im[i].data
    im.close()
    return pyr_im_cube


def create_P(bin_factor, size):
    return np.repeat(
            np.identity(size / bin_factor, dtype=np.float32), bin_factor, axis=0)


def calib_pyr(centers, wfs_numbers, bin_factor=1, crop_factor=0):

    #initialisation
    #offset 4 roi :
    offset = np.zeros((2, 4))
    j = [2, 1, 0, 3]
    npup = config.p_wfss[wfs_numbers]._validsubsx.shape[0]
    #decoupage 4 roi
    for i in range(4):
        #decoupage coordonnee
        #x :
        subx = config.p_wfss[wfs_numbers]._validsubsx[npup * (i) / 4:npup * (i + 1) / 4]
        #y :
        suby = config.p_wfss[wfs_numbers]._validsubsy[npup * (i) / 4:npup * (i + 1) / 4]
        # calcul des 4 centres
        center_compass = [((np.max(subx) - np.min(subx)) / 2.) + np.min(subx),
                          ((np.max(suby) - np.min(suby)) / 2.) + np.min(suby)]
        # calcul des offsets
        offset[:, i] = [
                np.int32((centers[j[i]][0] - crop_factor / 2.) / bin_factor) -
                center_compass[0],
                np.int32((centers[j[i]][1] - crop_factor / 2.) / bin_factor) -
                center_compass[1]
        ]

    return offset


def pyr_aquisition(n=0):

    #fonction d'aquisition d'image pour la pyramide
    #lib sesame python
    # cam 10gbit.py
    # ten gb class
    # get_image(1, num_cam) -->
    #im_path = ['pyrimgforSeb1.fits','pyrimgforSeb2.fits','pyrimgforSeb3.fits','pyrimgforSeb4.fits','pyrimgforSeb5.fits','pyrimgforSeb6.fits']
    #im = fits.open('/home/sdurand/im_pyr_banc/'+ im_path[n])
    #pyr_im = im[0].data
    path = '/home/sdurand/RecordPyrImages_2017_06_06_07h49/pyrImageCube.fits'
    im = fits.open(path)
    pyr_im = im[n].data
    im.close()
    return pyr_im


def get_slope_pyrhr(npup, valid_pixel):

    pup = np.zeros((npup / 4, 4))
    j = [0, 2, 3, 1]
    for i in range(4):
        pup[:, i] = valid_pixel[(npup / 4) * j[i]:(npup / 4) * (j[i] + 1)]
    tot = np.sum(pup, axis=1)
    t = np.average(tot)

    gx = (pup[:, 0] + pup[:, 2] - (pup[:, 1] + pup[:, 3])) / t
    gy = (pup[:, 0] + pup[:, 1] - (pup[:, 2] + pup[:, 3])) / t
    #gz = (pup[:,0] - pup[:,1] - pup[:,2] + pup[:,3]) / t

    slope = np.append(gx, gy) * (
            (config.p_wfss[0].pyr_ampl * config.p_wfss[0].Lambda * 1e-6) /
            config.p_tel.diam) * (180 / np.pi) * 3600

    return slope


def crop_im(im, taille_sortie):

    #im_crop = np.zeros((taille_sortie,taille_sortie),dtype=np.float32)
    size = im.shape[0]
    im_crop = im[np.int32((size / 2.) - (taille_sortie / 2.)):np.int32(
            (size / 2.) + (taille_sortie / 2)),
                 np.int32((size / 2.) - (taille_sortie / 2.)):np.int32(
                         (size / 2.) + (taille_sortie / 2.))]

    return im_crop


@autojit
def binning_im(im, bin_factor):

    bin_factor = np.int32(bin_factor)
    size = im.shape[0]
    size_bin = size / bin_factor
    binimage = np.zeros((size_bin, size_bin), dtype=np.float32)

    a = np.arange(size)
    xx, yy = np.meshgrid(a, a)
    xx = xx / bin_factor
    yy = yy / bin_factor
    for i in range(size):
        for j in range(size):
            binimage[xx[i, j], yy[i, j]] += im[i, j]
    return binimage / (bin_factor**2)


#@autojit
def binning_im_2(im, bin_factor):
    size = im.shape[0]
    bin_factor = np.int32(bin_factor)
    P = create_P(bin_factor, size)  #
    #GP = gpu.garray(P)
    #Gim  = gpu.garray(im)
    binimage = ((P.T).dot(im)).dot(P)
    #binimage = ((GP.T).dot(Gim)).dot(GP)

    return binimage / (bin_factor**2)


def loop(n, d_valid_pix=[], d_P=[], offset=[],
         bool_fake_wfs=np.zeros(len(config.p_wfss)), bin_factor=[], crop_factor=[],
         cube_im=[]):
    print("----------------------------------------------------")
    print("iter# | S.E. SR | L.E. SR | Est. Rem. | framerate")
    print("----------------------------------------------------")
    t0 = time.time()
    #fake_pos = np.where(bool_fake_wfs==1)

    for i in range(n):
        atm.move_atmos()
        if (config.p_controllers[0].type_control == "geo"):
            for t in range(config.p_target.ntargets):
                tar.atmos_trace(t, atm, tel)
                rtc.docontrol_geo(0, dms, tar, 0)
                rtc.applycontrol(0, dms)
                tar.dmtrace(0, dms)

        else:
            for t in range(config.p_target.ntargets):
                tar.atmos_trace(t, atm, tel)
                tar.dmtrace(t, dms)

            fake_it = 0
            for w in range(len(config.p_wfss)):
                wfs.sensors_trace(w, "all", tel, atm, dms)
                if bool_fake_wfs[w]:  #verif fake_wfs
                    if (config.p_wfss[w].type_wfs == 'pyrhr'):  # verif type_wfs = pyrhr
                        if (bin_factor[fake_it] > 1):  # verif bining
                            if (cube_im == []):  # verif bincube not here
                                pyr_im = pyr_aquisition(i)  # aquistion image
                                pyr_im_crop = crop_im(
                                        pyr_im,
                                        pyr_im.shape[0] - crop_factor[w])  # crop image

                            else:
                                pyr_im_crop = crop_im(cube_im[i],
                                                      cube_im[i].shape[0] - 2).astype(
                                                              np.float32)  # crop image

                            d_imhr = ch.carmaWrap_obj_Float2D(
                                    ch.carmaWrap_context(), data=pyr_im_crop /
                                    (bin_factor[fake_it]**2))  # inject pyr_image in GPU
                            d_imlr = d_P[fake_it].gemm(d_imhr, 't', 'n').gemm(
                                    d_P[fake_it])  # bining GPU
                        else:
                            if (cube_im == []):
                                pyr_im = pyr_aquisition(i)  # aquistion image
                                d_imlr = ch.carmaWrap_obj_Float2D(
                                        ch.carmaWrap_context(),
                                        data=pyr_im)  # inject pyr_image in GPU
                            else:
                                d_imlr = ch.carmaWrap_obj_Float2D(
                                        ch.carmaWrap_context(),
                                        data=cube_im[i])  # inject pyr_image in GPU
                        # valable seulmement pour wf0 :
                        wfs.copy_pyrimg(
                                w, d_imlr, d_valid_pix[fake_it][0],
                                d_valid_pix[fake_it][1])  # envoie de l image pyramide

                    elif (config.p_wfss[w].type_wfs == 'sh'):  # verif type_wfs = pyrhr
                        print("TODO SH")
                    else:
                        print("error")
                    fake_it += 1  # increment for fake_wfs
                else:
                    wfs.sensors_compimg(w)  # normal wfs

            rtc.docentroids(0)
            #slope_compass_0[:,i] = rtc.get_centroids(0)
            rtc.docontrol(0)

            rtc.applycontrol(0, dms)

        if ((i + 1) % 100 == 0):
            strehltmp = tar.get_strehl(0)
            print(i + 1, "\t", strehltmp[0], "\t", strehltmp[1])
    t1 = time.time()
    print(" loop execution time:", t1 - t0, "  (", n, "iterations), ", (t1 - t0) / n,
          "(mean)  ", n / (t1 - t0), "Hz")


#____________________________________________________________
# lib sesam
# sesam_class
# init vector fake_wfs -->
bool_fake_wfs = np.zeros(len(config.p_wfss), dtype=np.int32)

bool_fake_wfs[0] = 1

# init wfs
crop_factor = np.zeros(sum(bool_fake_wfs))
size_c = np.zeros(sum(bool_fake_wfs))
centers_fake_wfs = []
d_P = []
offset = np.zeros((2, 4, sum(bool_fake_wfs)))
size = np.zeros(sum(bool_fake_wfs))
bin_factor = np.zeros(sum(bool_fake_wfs))
d_valid_pix = []
#____________________________________________________________

# pour le wfs_fake = 0
w = 0
# rebin param
bin_factor[w] = 3
#fake_wfs_param
bool_fake_wfs[w] = 1
size[w] = 800
#____________________________________________________________

# import calibration and image
if (bool_fake_wfs[w] == 1):
    if (config.p_wfss[w].type_wfs == 'pyrhr'):
        centers_fake_wfs.append(pyrcalib.giveMeTheCalibs()[1]['centers'])

        nb_im = 100  # nombre d'image
        path = '/home/sdurand/RecordPyrImages_2017_06_06_07h49/pyrImageCube.fits'
        pyr_im_cube = import_im(nb_im, path)
#_____________________________________________________________

# initialisation fake wfs
fake_pos = np.where(bool_fake_wfs == 1)
for f in range(sum(bool_fake_wfs)):
    crop_factor[f] = size[f] - ((size[f] / bin_factor[f]) * bin_factor[f])
    size_c[f] = size[f] - crop_factor[f]

    if (config.p_wfss[fake_pos[f]].type_wfs == 'pyrhr'):
        offset[:, :, f] = calib_pyr(centers_fake_wfs[f], fake_pos[f],
                                    bin_factor=bin_factor[f],
                                    crop_factor=crop_factor[f])  # calcul offset for wfs
        d_P.append(
                ch.carmaWrap_obj_Float2D(ch.carmaWrap_context(), data=create_P(
                        bin_factor[f], size_c[f])))  # add wfs offset on GPU
    else:
        d_P.append([])

#_____________________________________________________________
# valable seulmement pour wf0_fake :
w = 0

if (bool_fake_wfs[w] == 1):  # verif fake_wfs
    if (config.p_wfss[w].type_wfs == 'pyrhr'):  # verif fake_pyrhr
        npup = config.p_wfss[w]._validsubsx.shape[0]
        valid_pix = np.zeros((2, npup), dtype=np.int32)
        d_P.append(
                ch.carmaWrap_obj_Float2D(ch.carmaWrap_context(),
                                         data=create_P(bin_factor[w], size_c[w])))
        valid_pix[0, :] = np.int32(config.p_wfss[w]._validsubsx + offset[0, :, w].repeat(
                config.p_wfss[w]._nvalid))  # cacul new X  new validsubx
        valid_pix[1, :] = np.int32(config.p_wfss[w]._validsubsy + offset[1, :, w].repeat(
                config.p_wfss[w]._nvalid))  # cacul new Y  new validsuby

        #d_valid_pix =  ch.carmaWrap_obj_Float2D(ch.carmaWrap_context(), data=valid_pix)
        d_valid_pix.append([
                ch.carmaWrap_obj_Int1D(ch.carmaWrap_context(), data=valid_pix[0, :]),
                ch.carmaWrap_obj_Int1D(ch.carmaWrap_context(), data=valid_pix[1, :])
        ])  # add valid subpix coord in GPU
        loop(100, d_valid_pix, d_P, offset=offset, bool_fake_wfs=bool_fake_wfs,
             cube_im=pyr_im_cube, bin_factor=bin_factor,
             crop_factor=crop_factor)  # Run loop
#_______________________________________________________________
    elif (config.p_wfss[w].type_wfs == 'sh'):

        print("TODO SH")
    else:
        print("Error")
else:
    loop(100)
