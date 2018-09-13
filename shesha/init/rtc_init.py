"""
Initialization of a Rtc object
"""

import shesha.config as conf
import shesha.constants as scons
from shesha.constants import CONST

from shesha.ao import imats, cmats, tomo, basis, modopti

from shesha.util import utilities, rtc_util
from shesha.init import dm_init
from typing import List

import numpy as np
from shesha.sutra_wrap import naga_context, Sensors, Dms, Target, Rtc, Rtc_brahma, Atmos, Telescope


def rtc_init(context: naga_context, tel: Telescope, wfs: Sensors, dms: Dms, atmos: Atmos,
             p_wfss: list, p_tel: conf.Param_tel, p_geom: conf.Param_geom,
             p_atmos: conf.Param_atmos, ittime: float, p_centroiders=None,
             p_controllers=None, p_dms=None, do_refslp=False, brahma=False, tar=None,
             dataBase={}, use_DB=False):
    """Initialize all the sutra_rtc objects : centroiders and controllers

    :parameters:
        context: (naga_context): context
        tel: (Telescope) : Telescope object
        wfs: (Sensors) : Sensors object
        dms: (Dms) : Dms object
        atmos: (Atmos) : Atmos object
        p_wfss: (list of Param_wfs) : wfs settings
        p_tel: (Param_tel) : telescope settings
        p_geom: (Param_geom) : geom settings
        p_atmos: (Param_atmos) : atmos settings
        ittime: (float) : iteration time [s]
        p_centroiders : (list of Param_centroider): (optional) centroiders settings
        p_controllers : (list of Param_controller): (optional) controllers settings
        p_dms: (list of Param_dms) : (optional) dms settings
        do_refslp : (bool): (optional) do ref slopes flag, default=False
        brahma: (bool) : (optional) BRAHMA flag
        tar: (Target) : (optional)
        dataBase: (dict): (optional) dict containig paths to files to load
        use_DB: (bool): use dataBase flag
    :return:
        Rtc : (Rtc) : Rtc object
    """
    # initialisation var
    # ________________________________________________
    if brahma:
        print(wfs)
        print(tar)
        rtc = Rtc_brahma(context, wfs, tar, "rtc_brahma")
    else:
        rtc = Rtc()

    if p_wfss is None:
        return rtc

    if p_centroiders:
        ncentro = len(p_centroiders)
    else:
        ncentro = 0

    if p_controllers:
        ncontrol = len(p_controllers)
    else:
        ncontrol = 0

    if p_centroiders is not None:
        for i in range(ncentro):
            nwfs = p_centroiders[i].nwfs
            init_centroider(context, nwfs, p_wfss[nwfs], p_centroiders[i], p_tel,
                            p_atmos, wfs, rtc)

    if p_controllers is not None:
        if (p_wfss is not None and p_dms is not None):
            for i in range(ncontrol):
                if not "dm" in dataBase:
                    imat = imats.imat_geom(wfs, dms, p_wfss, p_dms, p_controllers[i],
                                           meth=0)
                else:
                    imat = None

                if p_dms[0].type == scons.DmType.PZT:
                    dm_init.correct_dm(context, dms, p_dms, p_controllers[i], p_geom,
                                       imat, dataBase=dataBase, use_DB=use_DB)

                init_controller(context, i, p_controllers[i], p_wfss, p_geom, p_dms,
                                p_atmos, ittime, p_tel, rtc, dms, wfs, tel, atmos,
                                p_centroiders, do_refslp, dataBase=dataBase,
                                use_DB=use_DB)

            # add a geometric controller for processing error breakdown
            roket_flag = True in [w.roket for w in p_wfss]
            if (roket_flag):
                p_controller = p_controllers[0]
                Nphi = np.where(p_geom._spupil)[0].size

                list_dmseen = [p_dms[j].type for j in p_controller.ndm]
                nactu = np.sum([p_dms[j]._ntotact for j in p_controller.ndm])

                rtc.add_controller(context, p_controller.nvalid, p_controller.nslope,
                                   p_controller.nactu, p_controller.delay,
                                   context.activeDevice, scons.ControllerType.GEO, dms,
                                   p_controller.ndm, p_controller.ndm.size, Nphi, True)

                # rtc.add_controller_geo(context, nactu, Nphi, p_controller.delay,
                #                        context.activeDevice, p_controller.type, dms,
                #                        list_dmseen, p_controller.ndm.size, True)

                # list_dmseen,alt,p_controller.ndm.size
                init_controller_geo(ncontrol, rtc, dms, p_geom, p_controller, p_dms,
                                    roket=True)

    return rtc


def rtc_standalone(context: naga_context, nwfs: int, nvalid, nactu: int,
                   centroider_type: bytes, delay: float, offset: float, scale: float,
                   brahma: bool=False):
    """
    TODO docstring
    """
    if brahma:
        rtc = Rtc_brahma(context)
    else:
        rtc = Rtc()

    for k in range(nwfs):
        rtc.add_centroider(context, nvalid[k], offset, scale, context.activeDevice,
                           centroider_type)

    nslopes = sum([c.nslopes for c in rtc.d_centro])
    rtc.add_controller(context,
                       sum(nvalid), nslopes, nactu, delay, context.activeDevice,
                       "generic")

    return rtc


def init_centroider(context, nwfs: int, p_wfs: conf.Param_wfs,
                    p_centroider: conf.Param_centroider, p_tel: conf.Param_tel,
                    p_atmos: conf.Param_atmos, wfs: Sensors, rtc: Rtc):
    """ Initialize a centroider object in Rtc

    :parameters:
        context: (naga_context): context
        nwfs : (int) : index of wfs
        p_wfs : (Param_wfs): wfs settings
        p_centroider : (Param_centroider) : centroider settings
        wfs: (Sensors): Sensor object
        rtc : (Rtc) : Rtc object
    """
    if (p_wfs.type == scons.WFSType.SH):
        if (p_centroider.type != scons.CentroiderType.CORR):
            s_offset = p_wfs.npix // 2. - 0.5
        else:
            if (p_centroider.type_fct == scons.CentroiderFctType.MODEL):
                if (p_wfs.npix % 2 == 0):
                    s_offset = p_wfs.npix // 2 - 0.5
                else:
                    s_offset = p_wfs.npix // 2
            else:
                s_offset = p_wfs.npix // 2 - 0.5
        s_scale = p_wfs.pixsize

    elif (p_wfs.type == scons.WFSType.PYRHR or p_wfs.type == scons.WFSType.PYRLR):
        s_offset = 0.
        s_scale = (p_wfs.Lambda * 1e-6 / p_tel.diam) * \
            p_wfs.pyr_ampl * CONST.RAD2ARCSEC

    rtc.add_centroider(context, p_wfs._nvalid, s_offset, s_scale, context.activeDevice,
                       p_centroider.type, wfs.d_wfs[nwfs])

    if (p_centroider.type != scons.CentroiderType.MASKEDPIX):
        p_centroider._nslope = 2 * p_wfs._nvalid
    else:
        p_centroider._nslope = p_wfs._validsubsx.size

    if (p_centroider.type == scons.CentroiderType.PYR):
        # FIXME SIGNATURE CHANGES
        rtc.d_centro[nwfs].set_pyr_method(p_centroider.method)
        rtc.d_centro[nwfs].set_pyr_thresh(p_centroider.thresh)

    elif (p_wfs.type == scons.WFSType.SH):
        if (p_centroider.type == scons.CentroiderType.TCOG):
            rtc.d_centro[nwfs].set_threshold(p_centroider.thresh)
        elif (p_centroider.type == scons.CentroiderType.BPCOG):
            rtc.d_centro[nwfs].set_nmax(p_centroider.nmax)
        elif (p_centroider.type == scons.CentroiderType.WCOG or
              p_centroider.type == scons.CentroiderType.CORR):
            r0 = p_atmos.r0 * (p_wfs.Lambda / 0.5)**(6 / 5.)
            seeing = CONST.RAD2ARCSEC * (p_wfs.Lambda * 1.e-6) / r0
            npix = seeing // p_wfs.pixsize
            comp_weights(p_centroider, p_wfs, npix)
            if p_centroider.type == scons.CentroiderType.WCOG:
                rtc.d_centro[nwfs].init_weights()
                rtc.d_centro[nwfs].load_weights(p_centroider.weights,
                                                p_centroider.weights.ndim)
            else:
                corrnorm = np.ones((2 * p_wfs.npix, 2 * p_wfs.npix), dtype=np.float32)
                p_centroider.sizex = 3
                p_centroider.sizey = 3
                p_centroider.interpmat = rtc_util.create_interp_mat(
                        p_centroider.sizex, p_centroider.sizey).astype(np.float32)

                if (p_centroider.weights is None):
                    raise ValueError("p_centroider.weights is None")
                rtc.d_centro[nwfs].init_bincube(p_wfs.npix)
                rtc.d_centro[nwfs].init_corr(p_centroider.sizex, p_centroider.sizey,
                                             p_centroider.interpmat)
                rtc.d_centro[nwfs].load_corr(p_centroider.weights, corrnorm,
                                             p_centroider.weights.ndim)


def comp_weights(p_centroider: conf.Param_centroider, p_wfs: conf.Param_wfs, npix: int):
    """
        Compute the weights used by centroider wcog and corr

    :parameters:
        p_centroider : (Param_centroider) : centroider settings
        p_wfs : (Param_wfs) : wfs settings
        npix: (int):
    """
    if (p_centroider.type_fct == scons.CentroiderFctType.MODEL):

        if (p_wfs.gsalt > 0):
            tmp = p_wfs._lgskern
            tmp2 = utilities.makegaussian(tmp.shape[1],
                                          npix * p_wfs._nrebin).astype(np.float32)
            tmp3 = np.zeros((tmp.shape[1], tmp.shape[1], p_wfs._nvalid),
                            dtype=np.float32)

            for j in range(p_wfs._nvalid):
                tmp3[:, :, j] = np.fft.ifft2(
                        np.fft.fft2(tmp[:, :, j]) * np.fft.fft2(tmp2.T)).real
                tmp3[:, :, j] *= tmp3.shape[0] * tmp3.shape[1]
                tmp3[:, :, j] = np.fft.fftshift(tmp3[:, :, j])

            offset = (p_wfs._Ntot - p_wfs._nrebin * p_wfs.npix) // 2
            j = offset + p_wfs._nrebin * p_wfs.npix
            tmp = np.zeros((j - offset + 1, j - offset + 1, tmp3.shape[2]),
                           dtype=np.float32)
            tmp3 = np.cumsum(tmp3[offset:j, offset:j, :], axis=0)
            tmp[1:, 1:, :] = np.cumsum(tmp3, axis=1)
            tmp = np.diff(tmp[::p_wfs._nrebin, ::p_wfs._nrebin, :], axis=0)
            tmp = np.diff(tmp, axis=1)

            p_centroider.weights = tmp
        else:
            p_centroider.type_fct = scons.CentroiderFctType.GAUSS
            print("No LGS found, centroider weighting function becomes gaussian")

    if (p_centroider.type_fct == scons.CentroiderFctType.GAUSS):
        if p_centroider.width is None:
            p_centroider.width = npix
        if (p_wfs.npix % 2 == 1):
            p_centroider.weights = utilities.makegaussian(
                    p_wfs.npix, p_centroider.width, p_wfs.npix // 2,
                    p_wfs.npix // 2).astype(np.float32)
        elif (p_centroider.type == scons.CentroiderType.CORR):
            p_centroider.weights = utilities.makegaussian(
                    p_wfs.npix, p_centroider.width, p_wfs.npix // 2,
                    p_wfs.npix // 2).astype(np.float32)
        else:
            p_centroider.weights = utilities.makegaussian(
                    p_wfs.npix, p_centroider.width, p_wfs.npix // 2 - 0.5,
                    p_wfs.npix // 2 - 0.5).astype(np.float32)


def init_controller(context, i: int, p_controller: conf.Param_controller, p_wfss: list,
                    p_geom: conf.Param_geom, p_dms: list, p_atmos: conf.Param_atmos,
                    ittime: float, p_tel: conf.Param_tel, rtc: Rtc, dms: Dms,
                    wfs: Sensors, tel: Telescope, atmos: Atmos,
                    p_centroiders: List[conf.Param_centroider], do_refslp=False,
                    dataBase={}, use_DB=False):
    """
        Initialize the controller part of rtc

    :parameters:
        context: (naga_context): context
        i : (int) : controller index
        p_controller: (Param_controller) : controller settings
        p_wfss: (list of Param_wfs) : wfs settings
        p_geom: (Param_geom) : geom settings
        p_dms: (list of Param_dms) : dms settings
        p_atmos: (Param_atmos) : atmos settings
        ittime: (float) : iteration time [s]
        p_tel: (Param_tel) : telescope settings
        rtc: (Rtc) : Rtc objet
        dms: (Dms) : Dms object
        wfs: (Sensors) : Sensors object
        tel: (Telescope) : Telescope object
        atmos: (Atmos) : Atmos object
        p_centroiders: (list of Param_centroider): centroiders settings
    """
    if (p_controller.type != scons.ControllerType.GEO):
        nwfs = p_controller.nwfs
        if (len(p_wfss) == 1):
            nwfs = p_controller.nwfs
            # TODO fixing a bug ... still not understood
        nvalid = sum([p_wfss[k]._nvalid for k in nwfs])
        p_controller.set_nvalid(int(np.sum([p_wfss[k]._nvalid for k in nwfs])))
    # parameter for add_controller(_geo)
    ndms = p_controller.ndm.tolist()
    nactu = np.sum([p_dms[j]._ntotact for j in ndms])
    p_controller.set_nactu(int(nactu))

    alt = np.array([p_dms[j].alt for j in p_controller.ndm], dtype=np.float32)

    list_dmseen = [p_dms[j].type for j in p_controller.ndm]
    if (p_controller.type == scons.ControllerType.GEO):
        Nphi = np.where(p_geom._spupil)[0].size
    else:
        Nphi = -1

    nslope = np.sum([c._nslope for c in p_centroiders])
    p_controller.set_nslope(int(nslope))

    #TODO : find a proper way to set the number of slope (other than 2 times nvalid)
    rtc.add_controller(context, p_controller.nvalid, p_controller.nslope,
                       p_controller.nactu, p_controller.delay, context.activeDevice,
                       p_controller.type, dms, p_controller.ndm, p_controller.ndm.size,
                       Nphi, False)

    if (p_wfss is not None and do_refslp):
        rtc.do_centroids_ref(i)

    if (p_controller.type == scons.ControllerType.GEO):
        init_controller_geo(i, rtc, dms, p_geom, p_controller, p_dms)

    if (p_controller.type == scons.ControllerType.LS):
        init_controller_ls(i, p_controller, p_wfss, p_geom, p_dms, p_atmos, ittime,
                           p_tel, rtc, dms, wfs, tel, atmos, dataBase=dataBase,
                           use_DB=use_DB)

    if (p_controller.type == scons.ControllerType.CURED):
        init_controller_cured(i, rtc, p_controller, p_dms, p_wfss)

    if (p_controller.type == scons.ControllerType.MV):
        init_controller_mv(i, p_controller, p_wfss, p_geom, p_dms, p_atmos, p_tel, rtc,
                           dms, wfs, atmos)

    elif (p_controller.type == scons.ControllerType.GENERIC):
        init_controller_generic(i, p_controller, p_dms, rtc)


def init_controller_geo(i: int, rtc: Rtc, dms: Dms, p_geom: conf.Param_geom,
                        p_controller: conf.Param_controller, p_dms: list, roket=False):
    """
        Initialize geometric controller

    :parameters:
        i: (int): controller index
        rtc: (Rtc): rtc object
        dms: (Dms): Dms object
        p_geom: (Param_geom): geometry settings
        p_controller: (Param_controller): controller settings
        p_dms: (list of Param_dms): dms settings
        roket: (bool): Flag to initialize ROKET
    """
    indx_pup = np.where(p_geom._spupil.flatten('F'))[0].astype(np.int32)
    indx_mpup = np.where(p_geom._mpupil.flatten('F'))[0].astype(np.int32)
    cpt = 0
    indx_dm = np.zeros((p_controller.ndm.size * indx_pup.size), dtype=np.int32)
    for dmn in range(p_controller.ndm.size):
        tmp_s = (p_geom._ipupil.shape[0] - (p_dms[dmn]._n2 - p_dms[dmn]._n1 + 1)) // 2
        tmp_e0 = p_geom._ipupil.shape[0] - tmp_s
        tmp_e1 = p_geom._ipupil.shape[1] - tmp_s
        pup_dm = p_geom._ipupil[tmp_s:tmp_e0, tmp_s:tmp_e1]
        indx_dm[cpt:cpt + np.where(pup_dm)[0].size] = np.where(pup_dm.flatten('F'))[0]
        cpt += np.where(pup_dm)[0].size
    # convert unitpervolt list to a np.ndarray
    unitpervolt = np.array([p_dms[j].unitpervolt
                            for j in range(len(p_dms))], dtype=np.float32)

    rtc.d_control[i].init_proj_sparse(dms, indx_dm, unitpervolt, indx_pup, indx_mpup,
                                      roket=roket)


def init_controller_ls(i: int, p_controller: conf.Param_controller, p_wfss: list,
                       p_geom: conf.Param_geom, p_dms: list, p_atmos: conf.Param_atmos,
                       ittime: float, p_tel: conf.Param_tel, rtc: Rtc, dms: Dms,
                       wfs: Sensors, tel: Telescope, atmos: Atmos, dataBase: dict={},
                       use_DB: bool=False):
    """
        Initialize the least square controller
    :parameters:
        i : (int) : controller index
        p_controller: (Param_controller) : controller settings
        p_wfss: (list of Param_wfs) : wfs settings
        p_geom: (Param_geom) : geom settings
        p_dms: (list of Param_dms) : dms settings
        p_atmos: (Param_atmos) : atmos settings
        ittime: (float) : iteration time [s]
        p_tel: (Param_tel) : telescope settings
        rtc: (Rtc) : Rtc objet
        dms: (Dms) : Dms object
        wfs: (Sensors) : Sensors object
        tel: (Telescope) : Telescope object
        atmos: (Atmos) : Atmos object
    """
    M2V = None
    if p_controller.do_kl_imat:
        IF = basis.compute_IFsparse(dms, p_dms, p_geom).T
        M2V, _ = basis.compute_Btt(IF[:, :-2], IF[:, -2:].toarray())
        print("Filtering ", p_controller.nModesFilt, " modes based on mode ordering")
        M2V = M2V[:, list(range(M2V.shape[1] - 2 - p_controller.nModesFilt)) + [-2, -1]]

        if len(p_controller.klpush) == 1:  # Scalar allowed, now we expand
            p_controller.klpush = p_controller.klpush[0] * np.ones(M2V.shape[1])
    imats.imat_init(i, rtc, dms, p_dms, wfs, p_wfss, p_tel, p_controller, M2V,
                    dataBase=dataBase, use_DB=use_DB)

    if p_controller.modopti:
        print("Initializing Modal Optimization : ")
        p_controller.nrec = int(2**np.ceil(np.log2(p_controller.nrec)))
        if p_controller.nmodes is None:
            p_controller.nmodes = sum([p_dms[j]._ntotact for j in range(len(p_dms))])

        IF = basis.compute_IFsparse(dms, p_dms, p_geom).T
        M2V, _ = basis.compute_Btt(IF[:, :-2], IF[:, -2:].toarray())

        rtc.d_control[i].init_modalOpti(p_controller.nmodes, p_controller.nrec, M2V,
                                        p_controller.gmin, p_controller.gmax,
                                        p_controller.ngain, 1. / ittime)
        ol_slopes = modopti.openLoopSlp(tel, atmos, wfs, rtc, p_controller.nrec, i,
                                        p_wfss)
        rtc.d_control[i].loadOpenLoopSlp(ol_slopes)
        rtc.d_control[i].modalControlOptimization()
    else:
        cmats.cmat_init(i, rtc, p_controller, p_wfss, p_atmos, p_tel, p_dms,
                        nmodes=p_controller.nmodes)

        rtc.d_control[i].set_gain(p_controller.gain)
        mgain = np.ones(
                sum([p_dms[j]._ntotact for j in range(len(p_dms))]), dtype=np.float32)
        cc = 0
        for ndm in p_dms:
            mgain[cc:cc + ndm._ntotact] = ndm.gain
            cc += ndm._ntotact
        rtc.d_control[i].set_mgain(mgain)


def init_controller_cured(i: int, rtc: Rtc, p_controller: conf.Param_controller,
                          p_dms: list, p_wfss: list):
    """
        Initialize the CURED controller
    :parameters:
        i : (int) : controller index
        rtc: (Rtc) : Rtc objet
        p_controller: (Param_controller) : controller settings
        p_dms: (list of Param_dms) : dms settings
        p_wfss: (list of Param_wfs) : wfs settings
    """

    print("initializing cured controller")
    if (scons.DmType.TT in [p_dms[j].type for j in range(len(p_dms))]):
        tt_flag = True
    else:
        tt_flag = False
    rtc.d_control[i].init_cured(p_wfss[0].nxsub, p_wfss[0]._isvalid,
                                p_controller.cured_ndivs, tt_flag)
    rtc.d_control[i].set_gain(p_controller.gain)


def init_controller_mv(i: int, p_controller: conf.Param_controller, p_wfss: list,
                       p_geom: conf.Param_geom, p_dms: list, p_atmos: conf.Param_atmos,
                       p_tel: conf.Param_tel, rtc: Rtc, dms: Dms, wfs: Sensors,
                       atmos: Atmos):
    """
        Initialize the MV controller

    :parameters:
        i : (int) : controller index
        p_controller: (Param_controller) : controller settings
        p_wfss: (list of Param_wfs) : wfs settings
        p_geom: (Param_geom) : geom settings
        p_dms: (list of Param_dms) : dms settings
        p_atmos: (Param_atmos) : atmos settings
        p_tel: (Param_tel) : telescope settings
        rtc: (Rtc) : Rtc objet
        dms: (Dms) : Dms object
        wfs: (Sensors) : Sensors object
        atmos: (Atmos) : Atmos object
    """
    p_controller._imat = imats.imat_geom(wfs, dms, p_wfss, p_dms, p_controller)
    # imat_init(i,rtc,p_rtc,dms,wfs,p_wfss,p_tel,clean=1,simul_name=simul_name)
    rtc.d_control[i].set_imat(p_controller._imat)
    rtc.d_control[i].set_gain(p_controller.gain)
    size = sum([p_dms[j]._ntotact for j in range(len(p_dms))])
    mgain = np.ones(size, dtype=np.float32)
    rtc.d_control[i].set_mgain(mgain)
    tomo.do_tomo_matrices(i, rtc, p_wfss, dms, atmos, wfs, p_controller, p_geom, p_dms,
                          p_tel, p_atmos)
    cmats.cmat_init(i, rtc, p_controller, p_wfss, p_atmos, p_tel, p_dms)


def init_controller_generic(i: int, p_controller: conf.Param_controller, p_dms: list,
                            rtc: Rtc):
    """
        Initialize the generic controller

    :parameters:
        i: (int): controller index
        p_controller: (Param_controller): controller settings
        p_dms: (list of Param_dm): dms settings
        rtc: (Rtc): Rtc object
    """
    size = sum([p_dms[j]._ntotact for j in range(len(p_dms))])
    decayFactor = np.ones(size, dtype=np.float32)
    mgain = np.ones(size, dtype=np.float32) * p_controller.gain
    matE = np.identity(size, dtype=np.float32)
    cmat = np.zeros((size, p_controller.nvalid * 2), dtype=np.float32)

    rtc.d_control[i].set_decayFactor(decayFactor)
    rtc.d_control[i].set_mgain(mgain)
    rtc.d_control[i].set_cmat(cmat)
    rtc.d_control[i].set_matE(matE)
