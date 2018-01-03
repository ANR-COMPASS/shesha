'''
Initialization of a Dms object
'''

import shesha_config as conf
import shesha_constants as scons

from shesha_constants import CONST

from shesha_util import dm_util, influ_util, kl_util
from shesha_util import hdf5_utils as h5u

import numpy as np

import pandas as pd
from scipy import interpolate
from sutra_bind.wrap import naga_context, Dms

from typing import List

from tqdm import tqdm


def dm_init(context: naga_context, p_dms: List[conf.Param_dm], p_tel: conf.Param_tel,
            p_geom: conf.Param_geom, p_wfss: List[conf.Param_wfs]=None) -> Dms:
    """Create and initialize a Dms object on the gpu

    :parameters:
        context: (naga_context): context
        p_dms: (list of Param_dms) : dms settings
        p_tel: (Param_tel) : telescope settings
        p_geom: (Param_geom) : geom settings
        p_wfss: (list of Param_wfs) : wfs settings
    :return:
        Dms: (Dms): Dms object
    """
    max_extent = [0]
    xpos_wfs = []
    ypos_wfs = []
    for i in range(len(p_wfss)):
        xpos_wfs.append(p_wfss[i].xpos)
        ypos_wfs.append(p_wfss[i].ypos)

    if (len(p_dms) != 0):
        dms = Dms(context, len(p_dms))
        for i in range(len(p_dms)):
            # max_extent
            #_dm_init(dms, p_dms[i], p_wfss, p_geom, p_tel, & max_extent)
            _dm_init(dms, p_dms[i], xpos_wfs, ypos_wfs, p_geom, p_tel.diam, p_tel.cobs,
                     max_extent)

    return dms


def _dm_init(dms: Dms, p_dm: conf.Param_dm, xpos_wfs: list, ypos_wfs: list,
             p_geom: conf.Param_geom, diam: float, cobs: float, max_extent: list):
    """ inits a Dms object on the gpu

    :parameters:
        dms: (Dms) : dm object

        p_dm: (Param_dms) : dm settings

        xpos_wfs: (list) : list of wfs xpos

        ypos_wfs: (list) : list of wfs ypos

        p_geom: (Param_geom) : geom settings

        diam: (float) : diameter of telescope

        cobs: (float) : cobs of telescope

        max_extent: (list) : maximum dimension of all dms

    """

    if (p_dm.pupoffset is not None):
        p_dm._puppixoffset = p_dm.pupoffset / diam * p_geom.pupdiam

    # For patchDiam
    patchDiam = dm_util.dim_dm_patch(p_geom.pupdiam, diam, p_dm.type, p_dm.alt, xpos_wfs,
                                     ypos_wfs)

    if (p_dm.type == scons.DmType.PZT):
        if p_dm.file_influ_hdf5 == None:
            p_dm._pitch = patchDiam / float(p_dm.nact - 1)
            # + 2.5 pitch each side
            extent = p_dm._pitch * (p_dm.nact + p_dm.pzt_extent)
            p_dm._n1, p_dm._n2 = dm_util.dim_dm_support(p_geom.cent, extent,
                                                        p_geom.ssize)

            # calcul defaut influsize
            make_pzt_dm(p_dm, p_geom, cobs)
        else:
            init_pzt_from_hdf5(p_dm, p_geom, diam)

        # max_extent
        max_extent[0] = max(max_extent[0], p_dm._n2 - p_dm._n1 + 1)

        dim = max(p_dm._n2 - p_dm._n1 + 1, p_geom._mpupil.shape[0])
        ninflupos = p_dm._influpos.size
        n_npts = p_dm._ninflu.size

        dms.add_dm(p_dm.type, p_dm.alt, dim, p_dm._ntotact, p_dm._influsize, ninflupos,
                   n_npts, p_dm.push4imat)
        dms.load_pzt(p_dm.alt, p_dm._influ,
                     p_dm._influpos.astype(np.int32), p_dm._ninflu, p_dm._influstart,
                     p_dm._i1, p_dm._j1)

    elif (p_dm.type == scons.DmType.TT):

        if (p_dm.alt == 0):
            extent = int(max_extent[0] * 1.05)
            if (extent % 2 != 0):
                extent += 1
        else:
            extent = p_geom.pupdiam + 16
        p_dm._n1, p_dm._n2 = dm_util.dim_dm_support(p_geom.cent, extent, p_geom.ssize)
        # max_extent
        max_extent[0] = max(max_extent[0], p_dm._n2 - p_dm._n1 + 1)

        dim = p_dm._n2 - p_dm._n1 + 1
        make_tiptilt_dm(p_dm, patchDiam, p_geom, diam)
        dms.add_dm(p_dm.type, p_dm.alt, dim, 2, dim, 1, 1, p_dm.push4imat)
        dms.load_tt(p_dm.alt, p_dm._influ)

    elif (p_dm.type == scons.DmType.KL):

        extent = p_geom.pupdiam + 16
        p_dm._n1, p_dm._n2 = dm_util.dim_dm_support(p_geom.cent, extent, p_geom.ssize)
        # max_extent
        max_extent[0] = max(max_extent[0], p_dm._n2 - p_dm._n1 + 1)

        dim = p_dm._n2 - p_dm._n1 + 1

        make_kl_dm(p_dm, patchDiam, p_geom, cobs)

        ninflu = p_dm.nkl

        dms.add_dm(p_dm.type, p_dm.alt, dim, p_dm.nkl, p_dm._ncp, p_dm._nr, p_dm._npp,
                   p_dm.push4imat, nord=p_dm._ord.max())

        dms.load_kl(p_dm.alt, p_dm._rabas, p_dm._azbas, p_dm._ord, p_dm._cr, p_dm._cp)

    else:

        raise TypeError("This type of DM doesn't exist ")
        # Verif
        # res1 = pol2car(*y_dm(n)._klbas,gkl_sfi(*y_dm(n)._klbas, 1));
        # res2 = yoga_getkl(g_dm,0.,1);


def dm_init_standalone(p_dms: list, p_geom: conf.Param_geom, diam=1., cobs=0.,
                       wfs_xpos=[0], wfs_ypos=[0]):
    """Create and initialize a Dms object on the gpu

    :parameters:
        p_dms: (list of Param_dms) : dms settings

        p_geom: (Param_geom) : geom settings

        diam: (float) : diameter of telescope (default 1.)

        cobs: (float) : cobs of telescope (default 0.)

        wfs_xpos: (array) : guide star x position on sky (in arcsec).

        wfs_ypos: (array) : guide star y position on sky (in arcsec).

    """
    max_extent = [0]
    if (len(p_dms) != 0):
        dms = Dms(len(p_dms))
        for i in range(len(p_dms)):
            _dm_init(dms, p_dms[i], wfs_xpos, wfs_ypos, p_geom, diam, cobs, max_extent)
    return dms


def make_pzt_dm(p_dm: conf.Param_dm, p_geom: conf.Param_geom, cobs: float):
    """Compute the actuators positions and the influence functions for a pzt DM

    :parameters:
        p_dm: (Param_dm) : dm settings

        p_geom: (Param_geom) : geom settings

        cobs: (float) : tel cobs

    :return:
        influ: (np.ndarray(dims=3,dtype=np.float64)) : cube of the IF for each actuator

    """
    # best parameters, as determined by a multi-dimensional fit
    #(see coupling3.i)
    coupling = p_dm.coupling

    # prepare to compute IF on partial (local) support of size <smallsize>
    pitch = p_dm._pitch
    smallsize = 0

    if (p_dm.influType == scons.InfluType.RADIALSCHWARTZ):
        smallsize = influ_util.makeRadialSchwartz(pitch, coupling)
    elif (p_dm.influType == scons.InfluType.SQUARESCHWARTZ):
        smallsize = influ_util.makeSquareSchwartz(pitch, coupling)
    elif (p_dm.influType == scons.InfluType.BLACKNUTT):
        smallsize = influ_util.makeBlacknutt(pitch, coupling)
    elif (p_dm.influType == scons.InfluType.GAUSSIAN):
        smallsize = influ_util.makeGaussian(pitch, coupling)
    elif (p_dm.influType == scons.InfluType.BESSEL):
        smallsize = influ_util.makeBessel(pitch, coupling, p_dm.type_pattern)
    elif (p_dm.influType == scons.InfluType.DEFAULT):
        smallsize = influ_util.makeRigaut(pitch, coupling)
    else:
        print("ERROR influtype not recognized ")
    p_dm._influsize = smallsize

    # compute location (x,y and i,j) of each actuator:
    nxact = p_dm.nact

    if p_dm.type_pattern is None:
        p_dm.type_pattern = scons.PatternType.SQUARE

    if p_dm.type_pattern == scons.PatternType.HEXA:
        print("Pattern type : hexa")
        cub = dm_util.createHexaPattern(pitch, p_geom.pupdiam * 1.1)
    elif p_dm.type_pattern == scons.PatternType.HEXAM4:
        print("Pattern type : hexaM4")
        cub = dm_util.createDoubleHexaPattern(pitch, p_geom.pupdiam * 1.1)
    elif p_dm.type_pattern == scons.PatternType.SQUARE:
        print("Pattern type : square")
        cub = dm_util.createSquarePattern(pitch, nxact + 4)
    else:
        raise ValueError("This pattern does not exist for pzt dm")

    inbigcirc = dm_util.select_actuators(cub[0, :], cub[1, :], p_dm.nact, p_dm._pitch,
                                         cobs, p_dm.margin_in, p_dm.margin_out,
                                         p_dm._ntotact)
    p_dm._ntotact = inbigcirc.size

    # print(('inbigcirc',inbigcirc.shape))

    # converting to array coordinates:
    cub += p_geom.cent

    # filtering actuators outside of a disk radius = rad (see above)
    cubval = cub[:, inbigcirc]
    ntotact = cubval.shape[1]
    #pfits.writeto("cubeval.fits", cubval)
    xpos = cubval[0, :]
    ypos = cubval[1, :]
    i1t = (cubval[0, :] - smallsize / 2 + 0.5 - p_dm._n1).astype(np.int32)
    j1t = (cubval[1, :] - smallsize / 2 + 0.5 - p_dm._n1).astype(np.int32)

    p_dm._xpos = xpos
    p_dm._ypos = ypos
    p_dm._i1 = i1t
    p_dm._j1 = j1t

    # Allocate array of influence functions

    influ = np.zeros((smallsize, smallsize, ntotact), dtype=np.float32)
    # Computation of influence function for each actuator

    print("Computing Influence Function type : ", p_dm.influType)

    for i in tqdm(range(ntotact)):

        i1 = i1t[i]
        x = np.tile(np.arange(i1, i1 + smallsize, dtype=np.float32),
                    (smallsize, 1)).T  # pixel coords in ref frame "dm support"
        # pixel coords in ref frame "pupil support"
        x += p_dm._n1
        # pixel coords in local ref frame
        x -= xpos[i]

        j1 = j1t[i]
        y = np.tile(np.arange(j1, j1 + smallsize, dtype=np.float32),
                    (smallsize, 1))  # idem as X, in Y
        y += p_dm._n1
        y -= ypos[i]
        # print("Computing Influence Function #%d/%d \r" % (i, ntotact), end=' ')

        if (p_dm.influType == scons.InfluType.RADIALSCHWARTZ):
            influ[:, :, i] = influ_util.makeRadialSchwartz(pitch, coupling, x=x, y=y)
        elif (p_dm.influType == scons.InfluType.SQUARESCHWARTZ):
            influ[:, :, i] = influ_util.makeSquareSchwartz(pitch, coupling, x=x, y=y)
        elif (p_dm.influType == scons.InfluType.BLACKNUTT):
            influ[:, :, i] = influ_util.makeBlacknutt(pitch, coupling, x=x, y=y)
        elif (p_dm.influType == scons.InfluType.GAUSSIAN):
            influ[:, :, i] = influ_util.makeGaussian(pitch, coupling, x=x, y=y)
        elif (p_dm.influType == scons.InfluType.BESSEL):
            influ[:, :, i] = influ_util.makeBessel(pitch, coupling, x=x, y=y,
                                                   patternType=p_dm.type_pattern)
        elif (p_dm.influType == scons.InfluType.DEFAULT):
            influ[:, :, i] = influ_util.makeRigaut(pitch, coupling, x=x, y=y)
        else:
            print("ERROR influtype not recognized (defaut or gaussian or bessel)")

    if (p_dm._puppixoffset is not None):
        xpos += p_dm._puppixoffset[0]
        ypos += p_dm._puppixoffset[1]
    influ = influ * float(p_dm.unitpervolt / np.max(influ))

    p_dm._influ = influ

    comp_dmgeom(p_dm, p_geom)

    dim = max(p_geom._mpupil.shape[0], p_dm._n2 - p_dm._n1 + 1)
    off = (dim - p_dm._influsize) // 2


def init_pzt_from_hdf5(p_dm: conf.Param_dm, p_geom: conf.Param_geom, diam: float):
    """Read HDF for influence pzt fonction and form

    :parameters:
        p_dm: (Param_dm) : dm settings

        p_geom: (Param_geom) : geom settings

        diam: (float) : tel diameter

    """
    # read h5 file for influence fonction
    h5_tp = pd.read_hdf(p_dm.file_influ_hdf5, 'resAll')
    print("Read Ifluence fonction in h5 : ", p_dm.file_influ_hdf5)

    # cube_name
    influ_h5 = h5_tp[p_dm.cube_name][0]

    # x_name
    xpos_h5 = h5_tp[p_dm.x_name][0]

    # y_name
    ypos_h5 = h5_tp[p_dm.y_name][0]

    # center_name
    center_h5 = h5_tp[p_dm.center_name][0]

    # influ_res
    res_h5 = h5_tp[p_dm.influ_res][0]
    res_h5_m = (res_h5[0] + res_h5[1]) / 2.

    # a introduire dm diameter
    diam_dm_h5 = h5_tp[p_dm.diam_dm][0]
    # diam_dm_h5 = [2.54,2.54] # metre
    diam_dm_pup_h5 = h5_tp[p_dm.diam_dm_proj][0]
    # diam_dm_pup_h5 = [43.73,43.73] #metre

    # soustraction du centre introduit
    xpos_h5_0 = xpos_h5 - center_h5[0]
    ypos_h5_0 = ypos_h5 - center_h5[1]

    # interpolation du centre (ajout du nouveau centre)
    center = p_geom.cent

    # calcul de la resolution de la pupille
    res_compass = diam / p_geom.pupdiam

    # interpolation des coordonnées en pixel avec ajout du centre
    xpos = (xpos_h5_0 * (diam_dm_pup_h5[0] / diam_dm_h5[0])) / res_compass + center
    ypos = (ypos_h5_0 * (diam_dm_pup_h5[1] / diam_dm_h5[1])) / res_compass + center

    # interpolation des fonction d'influence

    influ_size_h5 = influ_h5.shape[0]
    ninflu = influ_h5.shape[2]
    # number of actuator
    print("Actuator number in H5 data : ", ninflu)
    p_dm._ntotact = np.int(ninflu)

    x = np.arange(influ_size_h5) * res_h5_m * (diam / diam_dm_h5[0])
    y = np.arange(influ_size_h5) * res_h5_m * (diam / diam_dm_h5[1])
    xmax = max(x)
    ymax = max(y)
    xnew = np.arange(0, xmax, res_compass)
    xnew = xnew + (xmax - max(xnew)) / 2.
    ynew = np.arange(0, ymax, res_compass)
    ynew = ynew + (ymax - max(ynew)) / 2.
    influ_size = xnew.shape[0]

    # creation du ouveaux cube d'influance
    influ_new = np.zeros((influ_size, influ_size, ninflu))

    for i in range(ninflu):

        influ = influ_h5[:, :, i]
        f = interpolate.interp2d(x, y, influ, kind='cubic')
        influ_new[:, :, i] = f(xnew, ynew).T

    p_dm._xpos = np.float32(xpos)
    p_dm._ypos = np.float32(ypos)

    # def influence size
    print("influence size in pupil : ", np.int(influ_size), "pixel")
    p_dm._influsize = np.int(influ_size)

    # def influente fonction normalize by unitpervolt
    p_dm._influ = np.float32(influ_new) * p_dm.unitpervolt

    # Def dm limite (n1 and n2)
    extent = (max(xpos) - min(xpos)) + (influ_size * 2)
    p_dm._n1, p_dm._n2 = dm_util.dim_dm_support(p_geom.cent, extent, p_geom.ssize)
    # refaire la definition du pitch pour n_actuator
    #inbigcirc = n_actuator_select(p_dm,p_tel,xpos-center[0],ypos-center[1])
    #print('nb = ',np.size(inbigcirc))
    #p_dm._ntotact = np.size(inbigcirc)

    # i1, j1 calc :

    p_dm._i1 = (p_dm._xpos - p_dm._influsize / 2. + 0.5 - p_dm._n1).astype(np.int32)
    p_dm._j1 = (p_dm._ypos - p_dm._influsize / 2. + 0.5 - p_dm._n1).astype(np.int32)

    comp_dmgeom(p_dm, p_geom)

    dim = max(p_geom._mpupil.shape[0], p_dm._n2 - p_dm._n1 + 1)
    off = (dim - p_dm._influsize) // 2


def make_tiptilt_dm(p_dm: conf.Param_dm, patchDiam: int, p_geom: conf.Param_geom,
                    diam: float):
    """Compute the influence functions for a tip-tilt DM

    :parameters:
        p_dm: (Param_dm) : dm settings

        patchDiam: (int) : patchDiam for dm size

        p_geom: (Param_geom) : geom settings

        diam: (float) : telescope diameter
    :return:
        influ: (np.ndarray(dims=3,dtype=np.float64)) : cube of the IF

    """
    dim = max(p_dm._n2 - p_dm._n1 + 1, p_geom._mpupil.shape[0])
    #norms = [np.linalg.norm([w.xpos, w.ypos]) for w in p_wfs]

    nzer = 2
    influ = dm_util.make_zernike(nzer + 1, dim, patchDiam, p_geom.cent - p_dm._n1 + 1,
                                 p_geom.cent - p_dm._n1 + 1, 1)[:, :, 1:]

    # normalization factor: one unit of tilt gives 1 arcsec:
    current = influ[dim // 2 - 1, dim // 2 - 1, 0] - \
        influ[dim // 2 - 2, dim // 2 - 2, 0]
    fact = p_dm.unitpervolt * diam / p_geom.pupdiam * 4.848 / current

    influ = influ * fact
    p_dm._ntotact = influ.shape[2]
    p_dm._influsize = influ.shape[0]
    p_dm._influ = influ

    return influ


def make_kl_dm(p_dm: conf.Param_dm, patchDiam: int, p_geom: conf.Param_geom,
               cobs: float) -> None:
    """Compute the influence function for a Karhunen-Loeve DM

    :parameters:
        p_dm: (Param_dm) : dm settings

        patchDiam: (int) : patchDiam for dm size

        p_geom: (Param_geom) : geom settings

        cobs: (float) : telescope cobs

    """
    dim = p_geom._mpupil.shape[0]

    print("KL type: ", p_dm.type_kl)

    if (p_dm.nkl < 13):
        nr = np.long(5.0 * np.sqrt(52))  # one point per degree
        npp = np.long(10.0 * nr)
    else:
        nr = np.long(5.0 * np.sqrt(p_dm.nkl))
        npp = np.long(10.0 * nr)

    radp = kl_util.make_radii(cobs, nr)

    kers = kl_util.make_kernels(cobs, nr, radp, p_dm.type_kl, p_dm.outscl)

    evals, nord, npo, ordd, rabas = kl_util.gkl_fcom(kers, cobs, p_dm.nkl)

    azbas = kl_util.make_azimuth(nord, npp)

    ncp, ncmar, px, py, cr, cp, pincx, pincy, pincw, ap = kl_util.set_pctr(
            patchDiam, nr, npp, p_dm.nkl, cobs, nord)

    p_dm._ntotact = p_dm.nkl
    p_dm._nr = nr  # number of radial points
    p_dm._npp = npp  # number of elements
    p_dm._ord = ordd  # the radial orders of the basis
    p_dm._rabas = rabas  # the radial array of the basis
    p_dm._azbas = azbas  # the azimuthal array of the basis
    p_dm._ncp = ncp  # dim of grid
    p_dm._cr = cr  # radial coord in cartesien grid
    p_dm._cp = cp  # phi coord in cartesien grid
    p_dm._i1 = np.zeros((p_dm.nkl), dtype=np.int32) + \
        (dim - patchDiam) // 2
    p_dm._j1 = np.zeros((p_dm.nkl), dtype=np.int32) + \
        (dim - patchDiam) // 2
    p_dm._ntotact = p_dm.nkl
    p_dm.ap = ap


def comp_dmgeom(p_dm: conf.Param_dm, p_geom: conf.Param_geom):
    """Compute the geometry of a DM : positions of actuators and influence functions

    :parameters:
        dm: (Param_dm) : dm settings

        geom: (Param_geom) : geom settings
    """
    smallsize = p_dm._influsize
    nact = p_dm._ntotact
    dm_dim = int(p_dm._n2 - p_dm._n1 + 1)
    mpup_dim = p_geom._mpupil.shape[0]

    if (dm_dim < mpup_dim):
        offs = (mpup_dim - dm_dim) // 2
    else:
        offs = 0
        mpup_dim = dm_dim

    indgen = np.tile(np.arange(smallsize, dtype=np.int32), (smallsize, 1))

    tmpx = np.tile(indgen, (nact, 1, 1))
    tmpy = np.tile(indgen.T, (nact, 1, 1))

    tmpx += offs + p_dm._i1[:, None, None]
    tmpy += offs + p_dm._j1[:, None, None]

    tmp = tmpx + mpup_dim * tmpy

    # bug in limit of def zone -10 destoe influpos for all actuator
    tmp[tmpx < 0] = mpup_dim * mpup_dim + 10  # -10
    tmp[tmpy < 0] = mpup_dim * mpup_dim + 10  # -10
    tmp[tmpx > dm_dim - 1] = mpup_dim * mpup_dim + 10  # -10
    tmp[tmpy > dm_dim - 1] = mpup_dim * mpup_dim + 10  # -10
    itmps = np.argsort(tmp.flatten()).astype(np.int32)
    tmps = tmp.flatten()[itmps].astype(np.int32)
    itmps = itmps[np.where(itmps > -1)]

    istart = np.zeros((mpup_dim * mpup_dim), dtype=np.int32)
    npts = np.zeros((mpup_dim * mpup_dim), dtype=np.int32)

    tmps_unique, cpt = np.unique(tmps, return_counts=True)
    if (tmps_unique > npts.size - 1).any():
        tmps_unique = tmps_unique[:-1]
        cpt = cpt[:-1]

    for i in range(tmps_unique.size):
        npts[tmps_unique[i]] = cpt[i]
    istart[1:] = np.cumsum(npts[:-1])

    p_dm._influpos = itmps[:np.sum(npts)].astype(np.int32)
    p_dm._ninflu = npts.astype(np.int32)
    p_dm._influstart = istart.astype(np.int32)

    p_dm._i1 += offs
    p_dm._j1 += offs


def correct_dm(dms: Dms, p_dms: list, p_controller: conf.Param_controller,
               p_geom: conf.Param_geom, imat: np.ndarray=None, dataBase: dict={},
               use_DB: bool=False):
    """Correct the geometry of the DMs using the imat (filter unseen actuators)

    :parameters:
        dms: (Dms) : Dms object
        p_dms: (list of Param_dm) : dms settings
        p_controller: (Param_controller) : controller settings
        p_geom: (Param_geom) : geom settings
        imat: (np.ndarray) : interaction matrix
        dataBase: (dict): dictionary containing paths to files to load
        use_DB: (bool): dataBase use flag
    """
    print("Filtering unseen actuators... ")
    ndm = p_controller.ndm.size
    for i in range(ndm):
        nm = p_controller.ndm[i]
        dms.remove_dm(p_dms[nm].type, p_dms[nm].alt)

    if imat is not None:
        resp = np.sqrt(np.sum(imat**2, axis=0))

    inds = 0

    for nmc in range(ndm):
        nm = p_controller.ndm[nmc]
        nactu_nm = p_dms[nm]._ntotact
        # filter actuators only in stackarray mirrors:
        if (p_dms[nm].type == scons.DmType.PZT):
            if "dm" in dataBase:
                influpos, ninflu, influstart, i1, j1, ok = h5u.load_dm_geom_from_dataBase(
                        dataBase, nmc)
                p_dms[nm].set_ntotact(ok.shape[0])
                p_dms[nm].set_influ(p_dms[nm]._influ[:, :, ok.tolist()])
                p_dms[nm].set_xpos(p_dms[nm]._xpos[ok])
                p_dms[nm].set_ypos(p_dms[nm]._ypos[ok])
                p_dms[nm]._influpos = influpos
                p_dms[nm]._ninflu = ninflu
                p_dms[nm]._influstart = influstart
                p_dms[nm]._i1 = i1
                p_dms[nm]._j1 = j1
            else:
                tmp = resp[inds:inds + p_dms[nm]._ntotact]
                ok = np.where(tmp > p_dms[nm].thresh * np.max(tmp))[0]
                nok = np.where(tmp <= p_dms[nm].thresh * np.max(tmp))[0]

                p_dms[nm].set_ntotact(ok.shape[0])
                p_dms[nm].set_influ(p_dms[nm]._influ[:, :, ok.tolist()])
                p_dms[nm].set_xpos(p_dms[nm]._xpos[ok])
                p_dms[nm].set_ypos(p_dms[nm]._ypos[ok])
                p_dms[nm].set_i1(p_dms[nm]._i1[ok])
                p_dms[nm].set_j1(p_dms[nm]._j1[ok])

                comp_dmgeom(p_dms[nm], p_geom)
                if use_DB:
                    h5u.save_dm_geom_in_dataBase(nmc, p_dms[nm]._influpos,
                                                 p_dms[nm]._ninflu,
                                                 p_dms[nm]._influstart, p_dms[nm]._i1,
                                                 p_dms[nm]._j1, ok)

            dim = max(p_dms[nm]._n2 - p_dms[nm]._n1 + 1, p_geom._mpupil.shape[0])
            ninflupos = p_dms[nm]._influpos.size
            n_npts = p_dms[nm]._ninflu.size

            dms.add_dm(p_dms[nm].type, p_dms[nm].alt, dim, p_dms[nm]._ntotact,
                       p_dms[nm]._influsize, ninflupos, n_npts, p_dms[nm].push4imat)
            dms.load_pzt(p_dms[nm].alt, p_dms[nm]._influ,
                         p_dms[nm]._influpos.astype(np.int32), p_dms[nm]._ninflu,
                         p_dms[nm]._influstart, p_dms[nm]._i1, p_dms[nm]._j1)

        elif (p_dms[nm].type == scons.DmType.TT):
            dim = p_dms[nm]._n2 - p_dms[nm]._n1 + 1
            dms.add_dm(p_dms[nm].type, p_dms[nm].alt, dim, 2, dim, 1, 1,
                       p_dms[nm].push4imat)
            dms.load_tt(p_dms[nm].alt, p_dms[nm]._influ)

        elif (p_dms[nm].type == scons.DmType.KL):
            dim = int(p_dms[nm]._n2 - p_dms[nm]._n1 + 1)

            dms.add_dm(p_dms[nm].type, p_dms[nm].alt, dim, p_dms[nm].nkl, p_dms[nm]._ncp,
                       p_dms[nm]._nr, p_dms[nm]._npp, p_dms[nm].push4imat)
            dms.load_kl(p_dms[nm].alt, p_dms[nm]._rabas, p_dms[nm]._azbas,
                        p_dms[nm]._ord, p_dms[nm]._cr, p_dms[nm]._cp)
        else:
            raise ValueError("Screwed up.")

        inds += nactu_nm
    print("Done")
