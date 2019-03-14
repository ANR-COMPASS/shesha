import numpy as np
import naga as ng
import os
from shesha.sutra_wrap import Rtc_UFF as Rtc
from shesha.supervisor.compassSupervisor import CompassSupervisor as Supervisor
from scipy.ndimage.measurements import center_of_mass

precision = 1e-2
sup = Supervisor(
        os.getenv("COMPASS_ROOT") + "/shesha/data/par/par4bench/scao_sh_16x16_8pix.py")
sup.initConfig()
sup._sim.wfs.d_wfs[0].set_fakecam(True)
sup._sim.wfs.d_wfs[0].set_maxFluxPerPix(int(sup.config.p_wfs0._nphotons // 2))
sup._sim.wfs.d_wfs[0].set_maxPixValue(2**16 - 1)
sup.singleNext()
sup.openLoop()
sup.closeLoop()
sup._sim.doControl(0)
rtc = Rtc()
rtc.add_centroider(sup._sim.c, sup.config.p_wfs0._nvalid,
                   sup.config.p_wfs0.npix / 2 - 0.5, sup.config.p_wfs0.pixsize, 0, "cog")
rtc.add_controller(sup._sim.c, sup.config.p_wfs0._nvalid, sup.config.p_wfs0._nvalid * 2,
                   sup.config.p_controller0.nactu, sup.config.p_controller0.delay, 0,
                   "generic")
centro = rtc.d_centro[0]
control = rtc.d_control[0]
rtc.d_centro[0].set_npix(sup.config.p_wfs0.npix)
xvalid = np.array(sup._sim.rtc.d_centro[0].d_validx)
yvalid = np.array(sup._sim.rtc.d_centro[0].d_validy)
rtc.d_centro[0].load_validpos(xvalid, yvalid, xvalid.size)
cmat = sup.getCmat(0)
rtc.d_control[0].set_cmat(cmat)
rtc.d_control[0].set_gain(sup.config.p_controller0.gain)
frame = np.array(sup._sim.wfs.d_wfs[0].d_camimg)
rtc.d_centro[0].load_img(frame, frame.shape[0])
rtc.d_centro[0].calibrate_img()

rtc.do_centroids(0)
slp = ng.array(rtc.d_control[0].d_centroids)
rtc.do_control(0)
com = ng.array(rtc.d_control[0].d_com)

dark = np.random.random(frame.shape)
flat = np.random.random(frame.shape)
centro.set_dark(dark, frame.shape[0])
centro.set_flat(flat, frame.shape[0])


def relative_array_error(array1, array2):
    return np.abs((array1 - array2) / array2.max()).max()


def test_initCentro_nvalid():
    assert (centro.nvalid - sup.config.p_wfs0._nvalid < precision)


def test_initCentro_offset():
    assert (centro.offset - (sup.config.p_wfs0.npix / 2 - 0.5) < precision)


def test_initCentro_scale():
    assert (centro.scale - sup.config.p_wfs0.pixsize < precision)


def test_initCentro_type():
    assert (centro.type == "cog")


def test_initControl_nslope():
    assert (control.nslope - sup.config.p_wfs0._nvalid * 2 < precision)


def test_initControl_nactu():
    assert (control.nactu - sup.config.p_controller0.nactu < precision)


def test_initControl_type():
    assert (control.type == "generic")


def test_initControl_delay():
    assert (control.delay - sup.config.p_controller0.delay < precision)


def test_set_npix():
    assert (centro.npix - sup.config.p_wfs0.npix < precision)


def test_load_validposX():
    assert (relative_array_error(np.array(centro.d_validx), xvalid) < precision)


def test_load_validposY():
    assert (relative_array_error(np.array(centro.d_validy), yvalid) < precision)


def test_set_cmat():
    assert (relative_array_error(ng.array(control.d_cmat).toarray(), cmat) < precision)


def test_set_gain():
    assert (control.gain - sup.config.p_controller0.gain < precision)


def test_load_img():
    assert (relative_array_error(np.array(centro.d_img_raw), frame) < precision)


def test_set_dark():
    assert (relative_array_error(ng.array(centro.d_dark).toarray(), dark) < precision)


def test_set_flat():
    assert (relative_array_error(ng.array(centro.d_flat).toarray(), flat) < precision)


def test_calibrate_img():
    centro.calibrate_img()
    imgCal = (frame - dark) * flat
    assert (relative_array_error(ng.array(centro.d_img).toarray(), imgCal) < precision)


def test_doCentroids_cog():
    bincube = np.array(sup._sim.wfs.d_wfs[0].d_bincube)
    slopes = np.zeros(sup.config.p_wfs0._nvalid * 2)
    offset = centro.offset
    scale = centro.scale
    for k in range(sup.config.p_wfs0._nvalid):
        tmp = center_of_mass(bincube[:, :, k])
        slopes[k] = (tmp[0] - offset) * scale
        slopes[k + sup.config.p_wfs0._nvalid] = (tmp[1] - offset) * scale
    assert (relative_array_error(ng.array(control.d_centroids).toarray(), slopes) <
            precision)


def test_doControl_generic():
    slopes = ng.array(control.d_centroids).toarray()
    gain = control.gain
    cmat = ng.array(control.d_cmat).toarray()
    commands = cmat.dot(slopes) * gain * (-1)
    assert (relative_array_error(ng.array(control.d_com).toarray(), commands) <
            precision)


def test_set_comRange():
    control.set_comRange(-1, 1)
    assert (control.comRange == (-1, 1))


def test_clipping():
    control.set_comRange(-1, 1)
    C = (np.random.random(sup.config.p_controller0.nactu) - 0.5) * 4
    control.set_com(C, C.size)
    rtc.do_clipping(0)
    C_clipped = C.copy()
    C_clipped[np.where(C > 1)] = 1
    C_clipped[np.where(C < -1)] = -1
    assert (relative_array_error(ng.array(control.d_comClipped).toarray(), C_clipped) <
            precision)


def test_add_perturb_voltage():
    C = np.random.random(sup.config.p_controller0.nactu)
    control.add_perturb_voltage("test", C, 1)
    assert (relative_array_error(
            ng.array(control.d_perturb_map["test"][0]).toarray(), C) < precision)


def test_remove_perturb_voltage():
    control.remove_perturb_voltage("test")
    assert (control.d_perturb_map == {})


def test_add_perturb():
    C = np.random.random(sup.config.p_controller0.nactu)
    control.add_perturb_voltage("test", C, 1)
    com = ng.array(control.d_comClipped).toarray()
    control.add_perturb()
    assert (relative_array_error(ng.array(control.d_comClipped).toarray(), com + C) <
            precision)


def test_disable_perturb_voltage():
    control.disable_perturb_voltage("test")
    com = ng.array(control.d_com).toarray()
    control.add_perturb()
    assert (relative_array_error(ng.array(control.d_com).toarray(), com) < precision)


def test_enable_perturb_voltage():
    control.enable_perturb_voltage("test")
    com = ng.array(control.d_comClipped).toarray()
    C = ng.array(control.d_perturb_map["test"][0]).toarray()
    control.add_perturb()
    assert (relative_array_error(ng.array(control.d_comClipped).toarray(), com + C) <
            precision)


def test_reset_perturb_voltage():
    control.reset_perturb_voltage()
    assert (control.d_perturb_map == {})


def test_comp_voltage():
    Vmin = -1
    Vmax = 1
    control.set_comRange(Vmin, Vmax)
    control.comp_voltage()
    C = np.random.random(sup.config.p_controller0.nactu)
    control.add_perturb_voltage("test", C, 1)
    control.set_com(C, C.size)
    com1 = ng.array(control.d_com1).toarray()
    control.comp_voltage()
    delay = sup.config.p_controller0.delay
    if control.d_com2 is not None:
        com2 = ng.array(control.d_com2).toarray()
    else:
        com2 = com1.copy() * 0
    floor = int(delay)
    if floor == 0:
        a = 1 - delay
        b = delay
        c = 0
    elif floor == 1:
        a = 0
        c = delay - floor
        b = 1 - c
    else:
        a = 0
        c = 1
        b = 0
    commands = a * C + b * com1 + c * com2
    comPertu = commands + C
    comPertu[np.where(comPertu > Vmax)] = Vmax
    comPertu[np.where(comPertu < Vmin)] = Vmin
    assert (relative_array_error(ng.array(control.d_voltage).toarray(), comPertu) <
            precision)


def test_remove_centroider():
    rtc.remove_centroider(0)
    assert (rtc.d_centro == [])


def test_doCentroids_tcog():
    rtc.add_centroider(sup._sim.c, sup.config.p_wfs0._nvalid,
                       sup.config.p_wfs0.npix / 2 - 0.5, sup.config.p_wfs0.pixsize, 0,
                       "tcog")

    centro = rtc.d_centro[-1]
    threshold = 5000
    centro.set_threshold(threshold)
    centro.set_npix(sup.config.p_wfs0.npix)
    centro.load_validpos(xvalid, yvalid, xvalid.size)
    centro.load_img(frame, frame.shape[0])
    centro.calibrate_img()
    rtc.do_centroids(0)
    slopes = np.zeros(sup.config.p_wfs0._nvalid * 2)
    offset = centro.offset
    scale = centro.scale
    vx = sup.config.p_wfs0._validsubsx
    vy = sup.config.p_wfs0._validsubsy
    npix = sup.config.p_wfs0.npix
    for k in range(sup.config.p_wfs0._nvalid):
        imagette = frame[vx[k]:vx[k] + npix, vy[k]:vy[k] + npix].astype(
                np.float32) - threshold
        imagette[np.where(imagette < 0)] = 0
        tmp = center_of_mass(imagette)
        slopes[k] = (tmp[0] - offset) * scale
        slopes[k + sup.config.p_wfs0._nvalid] = (tmp[1] - offset) * scale
    assert (relative_array_error(ng.array(control.d_centroids).toarray(), slopes) <
            precision)


def test_doCentroids_bpcog():
    rtc.remove_centroider(0)
    rtc.add_centroider(sup._sim.c, sup.config.p_wfs0._nvalid,
                       sup.config.p_wfs0.npix / 2 - 0.5, sup.config.p_wfs0.pixsize, 0,
                       "bpcog")

    centro = rtc.d_centro[-1]
    bpix = 8
    centro.set_nmax(8)
    centro.set_npix(sup.config.p_wfs0.npix)
    centro.load_validpos(xvalid, yvalid, xvalid.size)
    centro.load_img(frame, frame.shape[0])
    centro.calibrate_img()
    rtc.do_centroids(0)
    bincube = np.array(sup._sim.wfs.d_wfs[0].d_bincube)
    bincube /= bincube.max()
    slopes = np.zeros(sup.config.p_wfs0._nvalid * 2)
    offset = centro.offset
    scale = centro.scale
    vx = sup.config.p_wfs0._validsubsx
    vy = sup.config.p_wfs0._validsubsy
    npix = sup.config.p_wfs0.npix
    for k in range(sup.config.p_wfs0._nvalid):
        imagette = frame[vx[k]:vx[k] + npix, vy[k]:vy[k] + npix].astype(np.float32)
        threshold = np.sort(imagette, axis=None)[-(bpix + 1)]
        imagette -= threshold
        imagette[np.where(imagette < 0)] = 0
        tmp = center_of_mass(imagette)
        slopes[k] = (tmp[0] - offset) * scale
        slopes[k + sup.config.p_wfs0._nvalid] = (tmp[1] - offset) * scale
    assert (relative_array_error(ng.array(control.d_centroids).toarray(), slopes) <
            precision)
