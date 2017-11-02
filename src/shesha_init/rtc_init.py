"""
Initialization of a Rtc object
"""
from naga import naga_context

import shesha_config as conf
import shesha_constants as scons
from shesha_constants import CONST

import shesha_ao as shao

from shesha_util import utilities as util
from shesha_util import rtc_util
from shesha_init import dm_init as dmi

import numpy as np

from Dms import Dms
from Sensors import Sensors
from Telescope import Telescope
from Atmos import Atmos
from Target import Target
from Rtc import Rtc, Rtc_brama


def rtc_init(context: naga_context, tel: Telescope, wfs: Sensors, dms: Dms, atmos: Atmos,
             p_wfss: list, p_tel: conf.Param_tel, p_geom: conf.Param_geom,
             p_atmos: conf.Param_atmos, ittime: float, p_centroiders=None,
             p_controllers=None, p_dms=None, do_refslp=False, brama=False, tar=None,
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
        brama: (bool) : (optional) BRAMA flag
        tar: (Target) : (optional)
        dataBase: (dict): (optional) dict containig paths to files to load
        use_DB: (bool): use dataBase flag
    :return:
        Rtc : (Rtc) : Rtc object
    """
    # initialisation var
    # ________________________________________________
    if brama:
        rtc = Rtc_brama(context, wfs, tar)
    else:
        rtc = Rtc(context)

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
            init_centroider(nwfs, p_wfss[nwfs], p_centroiders[i], p_tel, p_atmos, wfs,
                            rtc)

    if p_controllers is not None:
        if (p_wfss is not None and p_dms is not None):
            for i in range(ncontrol):
                if not "dm" in dataBase:
                    imat = shao.imat_geom(wfs, dms, p_wfss, p_dms, p_controllers[i],
                                          meth=0)
                else:
                    imat = None

                if p_dms[0].type == scons.DmType.PZT:
                    dmi.correct_dm(dms, p_dms, p_controllers[i], p_geom, imat,
                                   dataBase=dataBase, use_DB=use_DB)

                init_controller(i, p_controllers[i], p_wfss, p_geom, p_dms, p_atmos,
                                ittime, p_tel, rtc, dms, wfs, tel, atmos, do_refslp,
                                dataBase=dataBase, use_DB=use_DB)

            # add a geometric controller for processing error breakdown
            error_budget_flag = True in [w.error_budget for w in p_wfss]
            if (error_budget_flag):
                p_controller = p_controllers[0]
                Nphi = np.where(p_geom._spupil)[0].size

                list_dmseen = [p_dms[j].type for j in p_controller.ndm]
                nactu = np.sum([p_dms[j]._ntotact for j in p_controller.ndm])
                alt = np.array([p_dms[j].alt
                                for j in p_controller.ndm], dtype=np.float32)

                rtc.add_controller(nactu, p_controller.delay, p_controller.type, dms,
                                   list_dmseen, alt, p_controller.ndm.size, Nphi, True)

                # list_dmseen,alt,p_controller.ndm.size
                init_controller_geo(ncontrol, rtc, dms, p_geom, p_controller, p_dms,
                                    roket=True)

    return rtc


def init_centroider(nwfs: int, p_wfs: conf.Param_wfs,
                    p_centroider: conf.Param_centroider, p_tel: conf.Param_tel,
                    p_atmos: conf.Param_atmos, wfs: Sensors, rtc: Rtc):
    """ Initialize a centroider object in Rtc

    :parameters:
        nwfs : (int) : index of wfs
        p_wfs : (Param_wfs): wfs settings
        p_centroider : (Param_centroider) : centroider settings
        wfs: (Sensors): Sensor object
        rtc : (Rtc) : Rtc object
    """
    if (p_wfs.type == scons.WFSType.SH):
        if (p_centroider.type != scons.CentroiderType.CORR):
            s_offset = p_wfs.npix // 2. + 0.5
        else:
            if (p_centroider.type_fct == scons.CentroiderFctType.MODEL):
                if (p_wfs.npix % 2 == 0):
                    s_offset = p_wfs.npix // 2 + 0.5
                else:
                    s_offset = p_wfs.npix // 2
            else:
                s_offset = p_wfs.npix // 2 + 0.5
        s_scale = p_wfs.pixsize

    elif (p_wfs.type == scons.WFSType.PYRHR):
        s_offset = 0.
        s_scale = (p_wfs.Lambda * 1e-6 / p_tel.diam) * \
            p_wfs.pyr_ampl * CONST.RAD2ARCSEC

    rtc.add_centroider(wfs, nwfs, p_wfs._nvalid, p_centroider.type, s_offset, s_scale)

    if (p_wfs.type == scons.WFSType.PYRHR):
        # FIXME SIGNATURE CHANGES
        rtc.set_pyr_method(nwfs, p_centroider.method)
        rtc.set_pyr_thresh(nwfs, p_centroider.thresh)

    elif (p_wfs.type == scons.WFSType.SH):
        if (p_centroider.type == scons.CentroiderType.TCOG):
            rtc.set_thresh(nwfs, p_centroider.thresh)
        elif (p_centroider.type == scons.CentroiderType.BPCOG):
            rtc.set_nmax(nwfs, p_centroider.nmax)
        elif (p_centroider.type == scons.CentroiderType.WCOG or
              p_centroider.type == scons.CentroiderType.CORR):
            r0 = p_atmos.r0 * (p_wfs.Lambda / 0.5)**(6 / 5.)
            seeing = CONST.RAD2ARCSEC * (p_wfs.Lambda * 1.e-6) / r0
            npix = seeing // p_wfs.pixsize
            comp_weights(p_centroider, p_wfs, npix)
            if p_centroider.type == scons.CentroiderType.WCOG:
                rtc.init_weights(nwfs, p_centroider.weights)
            else:
                corrnorm = np.ones((2 * p_wfs.npix, 2 * p_wfs.npix), dtype=np.float32)
                p_centroider.sizex = 3
                p_centroider.sizey = 3
                p_centroider.interpmat = rtc_util.create_interp_mat(
                        p_centroider.sizex, p_centroider.sizey).astype(np.float32)

                if (p_centroider.weights is None):
                    raise ValueError("p_centroider.weights is None")
                rtc.init_npix(nwfs)
                rtc.init_corr(nwfs, p_centroider.weights, corrnorm, p_centroider.sizex,
                              p_centroider.sizey, p_centroider.interpmat)


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
            tmp2 = util.makegaussian(tmp.shape[1],
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
            p_centroider.weights = util.makegaussian(
                    p_wfs.npix, p_centroider.width, p_wfs.npix // 2 + 1,
                    p_wfs.npix // 2 + 1).astype(np.float32)
        elif (p_centroider.type == scons.CentroiderType.CORR):
            p_centroider.weights = util.makegaussian(p_wfs.npix, p_centroider.width,
                                                     p_wfs.npix // 2,
                                                     p_wfs.npix // 2).astype(np.float32)
        else:
            p_centroider.weights = util.makegaussian(
                    p_wfs.npix, p_centroider.width, p_wfs.npix // 2 + 0.5,
                    p_wfs.npix // 2 + 0.5).astype(np.float32)


def init_controller(i: int, p_controller: conf.Param_controller, p_wfss: list,
                    p_geom: conf.Param_geom, p_dms: list, p_atmos: conf.Param_atmos,
                    ittime: float, p_tel: conf.Param_tel, rtc: Rtc, dms: Dms,
                    wfs: Sensors, tel: Telescope, atmos: Atmos, do_refslp=False,
                    dataBase={}, use_DB=False):
    """
        Initialize the controller part of rtc

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
    if (p_controller.type != scons.ControllerType.GEO):
        nwfs = p_controller.nwfs
        if (len(p_wfss) == 1):
            nwfs = p_controller.nwfs
            # TODO fixing a bug ... still not understood
        nvalid = sum([p_wfss[k]._nvalid for k in nwfs])
        p_controller.set_nvalid([p_wfss[k]._nvalid for k in nwfs])
    # parameter for add_controller(_geo)
    ndms = p_controller.ndm.tolist()
    p_controller.set_nactu([p_dms[n]._ntotact for n in ndms])
    nactu = np.sum([p_dms[j]._ntotact for j in ndms])
    alt = np.array([p_dms[j].alt for j in p_controller.ndm], dtype=np.float32)

    list_dmseen = [p_dms[j].type for j in p_controller.ndm]
    if (p_controller.type == scons.ControllerType.GEO):
        Nphi = np.where(p_geom._spupil)[0].size
    else:
        Nphi = -1

    rtc.add_controller(nactu, p_controller.delay, p_controller.type, dms, list_dmseen,
                       alt, p_controller.ndm.size, Nphi)

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

    rtc.init_proj(i, dms, indx_dm, unitpervolt, indx_pup, indx_mpup, roket=roket)


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
    KL2V = None
    if p_controller.kl_imat:
        KL2V = shao.compute_KL2V(p_controller, dms, p_dms, p_geom, p_atmos, p_tel)

        # TODO: Fab et/ou Vincent: quelle normalisation appliquée ? --> à mettre direct dans compute_KL2V
        # En attendant, la version de Seb (retravaillée)
        pushkl = np.ones(KL2V.shape[0])
        a = 0
        for p_dm in p_dms:
            pushkl[a:a + p_dm._ntotact] = p_dm.push4imat
            a += p_dm._ntotact
        for k in range(KL2V.shape[1]):
            klmaxVal = np.abs(KL2V[:, k]).max()
            KL2V[:, k] = KL2V[:, k] / klmaxVal * pushkl

    shao.imat_init(i, rtc, dms, p_dms, wfs, p_wfss, p_tel, p_controller, KL2V,
                   dataBase=dataBase, use_DB=use_DB)

    if p_controller.modopti:
        print("Initializing Modal Optimization : ")
        p_controller.nrec = int(2**np.ceil(np.log2(p_controller.nrec)))
        if p_controller.nmodes is None:
            p_controller.nmodes = sum([p_dms[j]._ntotact for j in range(len(p_dms))])

        KL2V = shao.compute_KL2V(p_controller, dms, p_dms, p_geom, p_atmos, p_tel)

        rtc.init_modalOpti(i, p_controller.nmodes, p_controller.nrec, KL2V,
                           p_controller.gmin, p_controller.gmax, p_controller.ngain,
                           1. / ittime)
        ol_slopes = shao.openLoopSlp(tel, atmos, wfs, rtc, p_controller.nrec, i, p_wfss)
        rtc.load_open_loop_slopes(i, ol_slopes)
        rtc.modal_control_optimization(i)
    else:
        shao.cmat_init(i, rtc, p_controller, p_wfss, p_atmos, p_tel, p_dms, KL2V=KL2V,
                       nmodes=p_controller.nmodes)

        rtc.set_gain(i, p_controller.gain)
        mgain = np.ones(
                sum([p_dms[j]._ntotact for j in range(len(p_dms))]), dtype=np.float32)
        cc = 0
        for ndm in p_dms:
            mgain[cc:cc + ndm._ntotact] = ndm.gain
            cc += ndm._ntotact
        rtc.set_mgain(i, mgain)


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
    rtc.init_cured(i, p_wfss[0].nxsub, p_wfss[0]._isvalid, p_controller.cured_ndivs,
                   tt_flag)
    rtc.set_gain(i, p_controller.gain)


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
    p_controller._imat = shao.imat_geom(wfs, dms, p_wfss, p_dms, p_controller)
    # imat_init(i,rtc,p_rtc,dms,wfs,p_wfss,p_tel,clean=1,simul_name=simul_name)
    rtc.set_imat(i, p_controller._imat)
    rtc.set_gain(i, p_controller.gain)
    size = sum([p_dms[j]._ntotact for j in range(len(p_dms))])
    mgain = np.ones(size, dtype=np.float32)
    rtc.set_mgain(i, mgain)
    shao.do_tomo_matrices(i, rtc, p_wfss, dms, atmos, wfs, p_controller, p_geom, p_dms,
                          p_tel, p_atmos)
    shao.cmat_init(i, rtc, p_controller, p_wfss, p_atmos, p_tel, p_dms)


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
    decayFactor = np.zeros(size, dtype=np.float32)
    mgain = np.zeros(size, dtype=np.float32)
    matE = np.zeros((size, size), dtype=np.float32)
    cmat = np.zeros((size, np.sum(p_controller.nvalid) * 2), dtype=np.float32)

    rtc.set_decayFactor(i, decayFactor)
    rtc.set_mgain(i, mgain)
    rtc.set_cmat(i, cmat)
    rtc.set_matE(i, matE)
