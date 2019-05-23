''' @package shesha.init.dm_init

Initialization of a Dms object

'''

import shesha.config as conf
import shesha.constants as scons

from shesha.constants import CONST

from shesha.util import dm_util, influ_util, kl_util
from shesha.util import hdf5_util as h5u

import numpy as np

import pandas as pd
from scipy import interpolate
from shesha.sutra_wrap import carmaWrap_context, Dms

from typing import List

from tqdm import tqdm


def dm_init(context: carmaWrap_context, p_dms: List[conf.Param_dm],
            p_tel: conf.Param_tel, p_geom: conf.Param_geom,
            p_wfss: List[conf.Param_wfs] = None, keepAllActu: bool = False) -> Dms:
    """Create and initialize a Dms object on the gpu

    :parameters:
        context: (carmaWrap_context): context
        p_dms: (list of Param_dms) : dms settings
        p_tel: (Param_tel) : telescope settings
        p_geom: (Param_geom) : geom settings
        p_wfss: (list of Param_wfs) : wfs settings
    :return:
        Dms: (Dms): Dms object
    """
    max_extent = 0
    if (p_wfss is not None):
        xpos_wfs = []
        ypos_wfs = []
        for i in range(len(p_wfss)):
            xpos_wfs.append(p_wfss[i].xpos)
            ypos_wfs.append(p_wfss[i].ypos)
    else:
        xpos_wfs = [0]
        ypos_wfs = [0]
    if (len(p_dms) != 0):
        dms = Dms()
        types_dm = [p_dm.type for p_dm in p_dms]
        if scons.DmType.TT in types_dm:
            first_TT = types_dm.index(scons.DmType.TT)
            if np.any(np.array(types_dm[first_TT:]) != scons.DmType.TT):
                raise RuntimeError("TT must be defined at the end of the dms parameters")

        for i in range(len(p_dms)):
            max_extent = _dm_init(context, dms, p_dms[i], xpos_wfs, ypos_wfs, p_geom,
                                  p_tel.diam, p_tel.cobs, p_tel.pupangle, max_extent,
                                  keepAllActu=keepAllActu)

    return dms


def _dm_init(context: carmaWrap_context, dms: Dms, p_dm: conf.Param_dm, xpos_wfs: list,
             ypos_wfs: list, p_geom: conf.Param_geom, diam: float, cobs: float,
             pupAngle: float, max_extent: int, keepAllActu: bool = False):
    """ inits a Dms object on the gpu

    :parameters:
        context: (carmaWrap_context): context
        dms: (Dms) : dm object

        p_dm: (Param_dms) : dm settings

        xpos_wfs: (list) : list of wfs xpos

        ypos_wfs: (list) : list of wfs ypos

        p_geom: (Param_geom) : geom settings

        diam: (float) : diameter of telescope

        cobs: (float) : cobs of telescope

        max_extent: (int) : maximum dimension of all dms

    :return:
        max_extent: (int) : new maximum dimension of all dms

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
            make_pzt_dm(p_dm, p_geom, cobs, pupAngle, keepAllActu=keepAllActu)
        else:
            init_pzt_from_hdf5(p_dm, p_geom, diam)

        # max_extent
        max_extent = max(max_extent, p_dm._n2 - p_dm._n1 + 1)

        dim = max(p_dm._n2 - p_dm._n1 + 1, p_geom._mpupil.shape[0])
        ninflupos = p_dm._influpos.size
        n_npts = p_dm._ninflu.size  #// 2
        dms.add_dm(context, p_dm.type, p_dm.alt, dim, p_dm._ntotact, p_dm._influsize,
                   ninflupos, n_npts, p_dm.push4imat, 0, context.activeDevice)
        #infludata = p_dm._influ.flatten()[p_dm._influpos]
        dms.d_dms[-1].pzt_loadarrays(p_dm._influ, p_dm._influpos.astype(np.int32),
                                     p_dm._ninflu, p_dm._influstart, p_dm._i1, p_dm._j1)

    elif (p_dm.type == scons.DmType.TT):

        if (p_dm.alt == 0) and (max_extent != 0):
            extent = int(max_extent * 1.05)
            if (extent % 2 != 0):
                extent += 1
        else:
            extent = p_geom.pupdiam + 16
        p_dm._n1, p_dm._n2 = dm_util.dim_dm_support(p_geom.cent, extent, p_geom.ssize)
        # max_extent
        max_extent = max(max_extent, p_dm._n2 - p_dm._n1 + 1)

        dim = p_dm._n2 - p_dm._n1 + 1
        make_tiptilt_dm(p_dm, patchDiam, p_geom, diam)
        dms.add_dm(context, p_dm.type, p_dm.alt, dim, 2, dim, 1, 1, p_dm.push4imat, 0,
                   context.activeDevice)
        dms.d_dms[-1].tt_loadarrays(p_dm._influ)

    elif (p_dm.type == scons.DmType.KL):

        extent = p_geom.pupdiam + 16
        p_dm._n1, p_dm._n2 = dm_util.dim_dm_support(p_geom.cent, extent, p_geom.ssize)
        # max_extent
        max_extent = max(max_extent, p_dm._n2 - p_dm._n1 + 1)

        dim = p_dm._n2 - p_dm._n1 + 1

        make_kl_dm(p_dm, patchDiam, p_geom, cobs)

        ninflu = p_dm.nkl

        dms.add_dm(context, p_dm.type, p_dm.alt, dim, p_dm.nkl, p_dm._ncp, p_dm._nr,
                   p_dm._npp, p_dm.push4imat, p_dm._ord.max(), context.activeDevice)

        dms.d_dms[-1].kl_loadarrays(p_dm._rabas, p_dm._azbas, p_dm._ord, p_dm._cr,
                                    p_dm._cp)

    else:

        raise TypeError("This type of DM doesn't exist ")
        # Verif
        # res1 = pol2car(*y_dm(n)._klbas,gkl_sfi(*y_dm(n)._klbas, 1));
        # res2 = yoga_getkl(g_dm,0.,1);

    return max_extent


def _dm_init_factorized(context: carmaWrap_context, dms: Dms, p_dm: conf.Param_dm,
                        xpos_wfs: list, ypos_wfs: list, p_geom: conf.Param_geom,
                        diam: float, cobs: float, pupAngle: float, max_extent: int,
                        keepAllActu: bool = False):
    """ inits a Dms object on the gpu
    NOTE: This is the

    :parameters:
        context: (carmaWrap_context): context
        dms: (Dms) : dm object

        p_dm: (Param_dms) : dm settings

        xpos_wfs: (list) : list of wfs xpos

        ypos_wfs: (list) : list of wfs ypos

        p_geom: (Param_geom) : geom settings

        diam: (float) : diameter of telescope

        cobs: (float) : cobs of telescope

        max_extent: (int) : maximum dimension of all dms

    :return:
        max_extent: (int) : new maximum dimension of all dms

    """

    if (p_dm.pupoffset is not None):
        p_dm._puppixoffset = p_dm.pupoffset / diam * p_geom.pupdiam
    # For patchDiam
    patchDiam = dm_util.dim_dm_patch(p_geom.pupdiam, diam, p_dm.type, p_dm.alt, xpos_wfs,
                                     ypos_wfs)

    if (p_dm.type == scons.DmType.PZT) and p_dm.file_influ_hdf5 is not None:
        init_pzt_from_hdf5(p_dm, p_geom, diam)
    else:
        if (p_dm.type == scons.DmType.PZT):
            p_dm._pitch = patchDiam / float(p_dm.nact - 1)
            # + 2.5 pitch each side
            extent = p_dm._pitch * (p_dm.nact + p_dm.pzt_extent)

            # calcul defaut influsize
            make_pzt_dm(p_dm, p_geom, cobs, pupAngle, keepAllActu=keepAllActu)

        elif (p_dm.type == scons.DmType.TT):
            if (p_dm.alt == 0) and (max_extent != 0):
                extent = int(max_extent * 1.05)
                if (extent % 2 != 0):
                    extent += 1
            else:
                extent = p_geom.pupdiam + 16

        elif (p_dm.type == scons.DmType.KL):
            extent = p_geom.pupdiam + 16
        else:
            raise TypeError("This type of DM doesn't exist ")

        # Verif
        # res1 = pol2car(*y_dm(n)._klbas,gkl_sfi(*y_dm(n)._klbas, 1));
        # res2 = yoga_getkl(g_dm,0.,1);

        p_dm._n1, p_dm._n2 = dm_util.dim_dm_support(p_geom.cent, extent, p_geom.ssize)

    # max_extent
    max_extent = max(max_extent, p_dm._n2 - p_dm._n1 + 1)

    dim = max(p_dm._n2 - p_dm._n1 + 1, p_geom._mpupil.shape[0])

    if (p_dm.type == scons.DmType.PZT):
        ninflupos = p_dm._influpos.size
        n_npts = p_dm._ninflu.size  #// 2
        dms.add_dm(context, p_dm.type, p_dm.alt, dim, p_dm._ntotact, p_dm._influsize,
                   ninflupos, n_npts, p_dm.push4imat, 0, context.activeDevice)
        #infludata = p_dm._influ.flatten()[p_dm._influpos]
        dms.d_dms[-1].pzt_loadarrays(p_dm._influ, p_dm._influpos.astype(np.int32),
                                     p_dm._ninflu, p_dm._influstart, p_dm._i1, p_dm._j1)
    elif (p_dm.type == scons.DmType.TT):
        make_tiptilt_dm(p_dm, patchDiam, p_geom, diam)
        dms.add_dm(context, p_dm.type, p_dm.alt, dim, 2, dim, 1, 1, p_dm.push4imat, 0,
                   context.activeDevice)
        dms.d_dms[-1].tt_loadarrays(p_dm._influ)
    elif (p_dm.type == scons.DmType.KL):
        make_kl_dm(p_dm, patchDiam, p_geom, cobs)
        ninflu = p_dm.nkl

        dms.add_dm(context, p_dm.type, p_dm.alt, dim, p_dm.nkl, p_dm._ncp, p_dm._nr,
                   p_dm._npp, p_dm.push4imat, p_dm._ord.max(), context.activeDevice)
        dms.d_dms[-1].kl_loadarrays(p_dm._rabas, p_dm._azbas, p_dm._ord, p_dm._cr,
                                    p_dm._cp)

    return max_extent


def dm_init_standalone(context: carmaWrap_context, p_dms: list, p_geom: conf.Param_geom,
                       diam=1., cobs=0., pupAngle=0., wfs_xpos=[0], wfs_ypos=[0]):
    """Create and initialize a Dms object on the gpu

    :parameters:
        p_dms: (list of Param_dms) : dms settings

        p_geom: (Param_geom) : geom settings

        diam: (float) : diameter of telescope (default 1.)

        cobs: (float) : cobs of telescope (default 0.)

        pupAngle: (float) : pupil rotation angle (degrees, default 0.)

        wfs_xpos: (array) : guide star x position on sky (in arcsec).

        wfs_ypos: (array) : guide star y position on sky (in arcsec).

    """
    max_extent = [0]
    if (len(p_dms) != 0):
        dms = Dms()
        for i in range(len(p_dms)):
            _dm_init(context, dms, p_dms[i], wfs_xpos, wfs_ypos, p_geom, diam, cobs,
                     pupAngle, max_extent)
    return dms


def make_pzt_dm(p_dm: conf.Param_dm, p_geom: conf.Param_geom, cobs: float,
                pupAngle: float, keepAllActu: bool = False):
    """Compute the actuators positions and the influence functions for a pzt DM.
    NOTE: if the DM is in altitude, central obstruction is forced to 0

    :parameters:
        p_dm: (Param_dm) : dm parameters

        p_geom: (Param_geom) : geometry parameters

        cobs: (float) : telescope central obstruction

    :return:
        influ: (np.ndarray(dims=3, dtype=np.float64)) : cube of the IF for each actuator

    """
    # best parameters, as determined by a multi-dimensional fit
    #(see coupling3.i)
    coupling = p_dm.coupling

    # prepare to compute IF on partial (local) support of size <smallsize>
    pitch = p_dm._pitch
    smallsize = 0

    # Petal DM (segmentation of M4)
    if (p_dm.influType == scons.InfluType.PETAL):
        makePetalDm(p_dm, p_geom, pupAngle)
        return

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
        keepAllActu = True
    elif p_dm.type_pattern == scons.PatternType.HEXAM4:
        print("Pattern type : hexaM4")
        keepAllActu = True
        cub = dm_util.createDoubleHexaPattern(pitch, p_geom.pupdiam * 1.1, pupAngle)
        if p_dm.margin_out is not None:
            pup_side = p_geom._ipupil.shape[0]
            cub_off = dm_util.filterActuWithPupil(cub + pup_side // 2 - 0.5,
                                                  p_geom._ipupil,
                                                  p_dm.margin_out * p_dm.get_pitch())
            cub = cub_off - pup_side // 2 + 0.5
            p_dm.set_ntotact(cub.shape[1])
    elif p_dm.type_pattern == scons.PatternType.SQUARE:
        print("Pattern type : square")
        cub = dm_util.createSquarePattern(pitch, nxact + 4)
    else:
        raise ValueError("This pattern does not exist for pzt dm")

    if keepAllActu:
        inbigcirc = np.arange(cub.shape[1])
    else:
        if (p_dm.alt > 0):
            cobs = 0
        inbigcirc = dm_util.select_actuators(cub[0, :], cub[1, :], p_dm.nact,
                                             p_dm._pitch, cobs, p_dm.margin_in,
                                             p_dm.margin_out, p_dm._ntotact)
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
    i1t = (cubval[0, :] - smallsize / 2 - 0.5 - p_dm._n1).astype(np.int32)
    j1t = (cubval[1, :] - smallsize / 2 - 0.5 - p_dm._n1).astype(np.int32)

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
        influ_new[:, :, i] = f(xnew, ynew)

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

    p_dm._i1 = (p_dm._xpos - p_dm._influsize / 2. - 0.5 - p_dm._n1).astype(np.int32)
    p_dm._j1 = (p_dm._ypos - p_dm._influsize / 2. - 0.5 - p_dm._n1).astype(np.int32)

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
    # infludata = p_dm._influ.flatten()[p_dm._influpos]
    # p_dm._influ = infludata[:,None,None]
    # p_dm._influpos = p_dm._influpos / (smallsize * smallsize)
    p_dm._ninflu = npts.astype(np.int32)
    p_dm._influstart = istart.astype(np.int32)

    p_dm._i1 += offs
    p_dm._j1 += offs

    # ninflu = np.zeros((istart.size * 2))
    # ninflu[::2] = istart.astype(np.int32)
    # ninflu[1::2] = npts.astype(np.int32)

    # p_dm._ninflu = ninflu


def correct_dm(context, dms: Dms, p_dms: list, p_controller: conf.Param_controller,
               p_geom: conf.Param_geom, imat: np.ndarray = None, dataBase: dict = {},
               use_DB: bool = False):
    """Correct the geometry of the DMs using the imat (filter unseen actuators)

    :parameters:
        context: (carmaWrap_context): context
        dms: (Dms) : Dms object
        p_dms: (list of Param_dm) : dms settings
        p_controller: (Param_controller) : controller settings
        p_geom: (Param_geom) : geom settings
        imat: (np.ndarray) : interaction matrix
        dataBase: (dict): dictionary containing paths to files to load
        use_DB: (bool): dataBase use flag
    """
    print("Filtering unseen actuators... ")
    if imat is not None:
        resp = np.sqrt(np.sum(imat**2, axis=0))

    ndm = p_controller.ndm.size
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
            dms.remove_dm(nm)
            dms.insert_dm(context, p_dms[nm].type, p_dms[nm].alt, dim,
                          p_dms[nm]._ntotact, p_dms[nm]._influsize, ninflupos, n_npts,
                          p_dms[nm].push4imat, 0, context.activeDevice, nm)
            dms.d_dms[nm].pzt_loadarrays(p_dms[nm]._influ, p_dms[nm]._influpos.astype(
                    np.int32), p_dms[nm]._ninflu, p_dms[nm]._influstart, p_dms[nm]._i1,
                                         p_dms[nm]._j1)

        inds += nactu_nm
    print("Done")


def makePetalDm(p_dm, p_geom, pupAngleDegree):
    '''
    makePetalDm(p_dm, p_geom, pupAngleDegree)

    The function builds a DM, segmented in petals according to the pupil
    shape. The petals will be adapted to the EELT case only.

    <p_geom> : compass object p_geom. The function requires the object p_geom
               in order to know what is the pupil mask, and what is the mpupil.
    <p_dm>   : compass petal dm object p_dm to be created. The function will
               transform/modify in place the attributes of the object p_dm.


    '''
    p_dm._n1 = p_geom._n1
    p_dm._n2 = p_geom._n2
    influ, i1, j1, smallsize, nbSeg = make_petal_dm_core(p_geom._mpupil, pupAngleDegree)
    p_dm._influsize = smallsize
    p_dm.set_ntotact(nbSeg)
    p_dm._i1 = i1
    p_dm._j1 = j1
    p_dm._xpos = i1 + smallsize / 2 + p_dm._n1
    p_dm._ypos = j1 + smallsize / 2 + p_dm._n1
    p_dm._influ = influ

    # generates the arrays of indexes for the GPUs
    comp_dmgeom(p_dm, p_geom)


def make_petal_dm_core(pupImage, pupAngleDegree):
    """
    <pupImage> : image of the pupil

    La fonction renvoie des fn d'influence en forme de petale d'apres
    une image de la pupille, qui est supposee etre segmentee.


    influ, i1, j1, smallsize, nbSeg = make_petal_dm_core(pupImage, 0.0)
    """
    # Splits the pupil into connex areas.
    # <segments> is the map of the segments, <nbSeg> in their number.
    # binary_opening() allows us to suppress individual pixels that could
    # be identified as relevant connex areas
    from scipy.ndimage.measurements import label
    from scipy.ndimage.morphology import binary_opening
    s = np.ones((2, 2), dtype=np.bool)
    segments, nbSeg = label(binary_opening(pupImage, s))

    # Faut trouver le plus petit support commun a tous les
    # petales : on determine <smallsize>
    smallsize = 0
    i1t = []  # list of starting indexes of influ functions
    j1t = []
    i2t = []  # list of ending indexes of influ functions
    j2t = []
    for i in range(nbSeg):
        petal = segments == (i + 1)  # identification (boolean) of a given segment
        profil = np.sum(petal, axis=1) != 0
        extent = np.sum(profil).astype(np.int32)
        i1t.append(np.min(np.where(profil)[0]))
        i2t.append(np.max(np.where(profil)[0]))
        if extent > smallsize:
            smallsize = extent

        profil = np.sum(petal, axis=0) != 0
        extent = np.sum(profil).astype(np.int32)
        j1t.append(np.min(np.where(profil)[0]))
        j2t.append(np.max(np.where(profil)[0]))
        if extent > smallsize:
            smallsize = extent

    # extension de la zone minimale pour avoir un peu de marge
    smallsize += 2

    # Allocate array of influence functions
    influ = np.zeros((smallsize, smallsize, nbSeg), dtype=np.float32)

    npt = pupImage.shape[0]
    i0 = j0 = npt / 2 - 0.5
    print('CORRIGER CETTE MERDE !!!!!')
    petalMap = build_petals(nbSeg, pupAngleDegree, i0, j0, npt)
    ii1 = np.zeros(nbSeg)
    jj1 = np.zeros(nbSeg)
    for i in range(nbSeg):
        ip = (smallsize - i2t[i] + i1t[i] - 1) // 2
        jp = (smallsize - j2t[i] + j1t[i] - 1) // 2
        i1 = np.maximum(i1t[i] - ip, 0)
        j1 = np.maximum(j1t[i] - jp, 0)
        if (j1 + smallsize) > npt:
            j1 = npt - smallsize
        if (i1 + smallsize) > npt:
            i1 = npt - smallsize
        #petal = segments==(i+1) # determine le segment pupille veritable
        k = petalMap[i1 + smallsize // 2, j1 + smallsize // 2]
        petal = (petalMap == k)
        influ[:, :, k] = petal[i1:i1 + smallsize, j1:j1 + smallsize]
        ii1[k] = i1
        jj1[k] = j1

    return influ, ii1, jj1, int(smallsize), nbSeg


def build_petals(nbSeg, pupAngleDegree, i0, j0, npt):
    """
    Makes an image npt x npt of <nbSeg> regularly spaced angular segments
    centred on (i0, j0).
    Origin of angles is set by <pupAngleDegree>.

    The segments are oriented as defined in document "Standard Coordinates
    and Basic Conventions", ESO-193058.
    This document states that the X axis lies in the middle of a petal, i.e.
    that the axis Y is along the spider.
    The separation angle between segments are [-30, 30, 90, 150, -150, -90].
    For this reason, an <esoOffsetAngle> = -pi/6 is introduced in the code.

    nbSeg = 6
    pupAngleDegree = 5.0
    i0 = j0 = 112.3
    npt = 222
    p = build_petals(nbSeg, pupAngleDegree, i0, j0, npt)
    """
    # conversion to radians
    rot = pupAngleDegree * np.pi / 180.0

    # building coordinate maps
    esoOffsetAngle = -np.pi / 6  # -30°, ESO definition.
    x = np.arange(npt) - i0
    y = np.arange(npt) - j0
    X, Y = np.meshgrid(x, y, indexing='ij')
    theta = (np.arctan2(Y, X) - rot + 2 * np.pi - esoOffsetAngle) % (2 * np.pi)

    # Compute separation angle between segments: start and end.
    angleStep = 2 * np.pi / nbSeg
    startAngle = np.arange(nbSeg) * angleStep
    endAngle = np.roll(startAngle, -1)
    endAngle[-1] = 2 * np.pi  # last angle is 0.00 and must be replaced by 2.pi
    petalMap = np.zeros((npt, npt), dtype=int)
    for i in range(nbSeg):
        nn = np.where(np.logical_and(theta >= startAngle[i], theta < endAngle[i]))
        petalMap[nn] = i
    return petalMap
