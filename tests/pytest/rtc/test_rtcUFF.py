## @package   shesha.tests
## @brief     Tests the RTC module
## @author    COMPASS Team <https://github.com/ANR-COMPASS>
## @version   5.4.2
## @date      2022/01/24
## @copyright GNU Lesser General Public License
#
#  This file is part of COMPASS <https://anr-compass.github.io/compass/>
#
#  Copyright (C) 2011-2023 COMPASS Team <https://github.com/ANR-COMPASS>
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
import os
from shesha.sutra_wrap import Rtc_UFF as Rtc
from shesha.supervisor.compassSupervisor import CompassSupervisor as Supervisor
from scipy.ndimage.measurements import center_of_mass
from shesha.config import ParamConfig

precision = 1e-2

config = ParamConfig(os.getenv("SHESHA_ROOT") + "/tests/pytest/par/test_sh.py")
sup = Supervisor(config)

sup.wfs._wfs.d_wfs[0].set_fakecam(True)
sup.wfs._wfs.d_wfs[0].set_max_flux_per_pix(int(sup.config.p_wfss[0]._nphotons // 2))
sup.wfs._wfs.d_wfs[0].set_max_pix_value(2**16 - 1)
sup.next()
sup.rtc.open_loop(0)
sup.rtc.close_loop(0)
sup.rtc.do_control(0)
rtc = Rtc()
rtc.add_centroider(sup.context, sup.config.p_wfss[0]._nvalid,
                   sup.config.p_wfss[0].npix / 2 - 0.5, sup.config.p_wfss[0].pixsize, False, 0,
                   "cog")
rtc.add_controller(sup.context,"generic", 0, sup.config.p_controllers[0].delay,
                   sup.config.p_wfss[0]._nvalid * 2, sup.config.p_controllers[0].nactu,
                   idx_centro=np.zeros(1),
                   ncentro=1)
centro = rtc.d_centro[0]
control = rtc.d_control[0]
rtc.d_centro[0].set_npix(sup.config.p_wfss[0].npix)
xvalid = np.array(sup.rtc._rtc.d_centro[0].d_validx)
yvalid = np.array(sup.rtc._rtc.d_centro[0].d_validy)
rtc.d_centro[0].load_validpos(xvalid, yvalid, xvalid.size)
cmat = sup.rtc.get_command_matrix(0)
rtc.d_control[0].set_cmat(cmat)
rtc.d_control[0].set_gain(sup.config.p_controllers[0].gain)
frame = np.array(sup.wfs._wfs.d_wfs[0].d_camimg)
rtc.d_centro[0].load_img(frame, frame.shape[0])
rtc.d_centro[0].calibrate_img()

rtc.do_centroids(0)
slp = np.array(rtc.d_control[0].d_centroids)
rtc.do_control(0)
com = np.array(rtc.d_control[0].d_com)

dark = np.random.random(frame.shape)
flat = np.random.random(frame.shape)
centro.set_dark(dark, frame.shape[0])
centro.set_flat(flat, frame.shape[0])


def relative_array_error(array1, array2):
    return np.abs((array1 - array2) / array2.max()).max()


def test_initCentro_nvalid():
    assert (centro.nvalid - sup.config.p_wfss[0]._nvalid < precision)


def test_initCentro_offset():
    assert (centro.offset - (sup.config.p_wfss[0].npix / 2 - 0.5) < precision)


def test_initCentro_scale():
    assert (centro.scale - sup.config.p_wfss[0].pixsize < precision)


def test_initCentro_type():
    assert (centro.type == "cog")


def test_initControl_nslope():
    assert (control.nslope - sup.config.p_wfss[0]._nvalid * 2 < precision)


def test_initControl_nactu():
    assert (control.nactu - sup.config.p_controllers[0].nactu < precision)


def test_initControl_type():
    assert (control.type == "generic")


def test_initControl_delay():
    assert (control.delay - sup.config.p_controllers[0].delay < precision)


def test_set_npix():
    assert (centro.npix - sup.config.p_wfss[0].npix < precision)


def test_load_validposX():
    assert (relative_array_error(np.array(centro.d_validx), xvalid) < precision)


def test_load_validposY():
    assert (relative_array_error(np.array(centro.d_validy), yvalid) < precision)


def test_set_cmat():
    assert (relative_array_error(np.array(control.d_cmat), cmat) < precision)


def test_set_gain():
    assert (control.gain - sup.config.p_controllers[0].gain < precision)


def test_load_img():
    assert (relative_array_error(np.array(centro.d_img_raw), frame) < precision)


def test_set_dark():
    assert (relative_array_error(np.array(centro.d_dark), dark) < precision)


def test_set_flat():
    assert (relative_array_error(np.array(centro.d_flat), flat) < precision)


def test_calibrate_img():
    centro.calibrate_img()
    imgCal = (frame - dark) * flat
    assert (relative_array_error(np.array(centro.d_img), imgCal) < precision)


def test_doCentroids_cog():
    bincube = np.array(sup.wfs._wfs.d_wfs[0].d_bincube)
    slopes = np.zeros(sup.config.p_wfss[0]._nvalid * 2)
    offset = centro.offset
    scale = centro.scale
    for k in range(sup.config.p_wfss[0]._nvalid):
        tmp = center_of_mass(bincube[:, :, k])
        slopes[k] = (tmp[0] - offset) * scale
        slopes[k + sup.config.p_wfss[0]._nvalid] = (tmp[1] - offset) * scale
    assert (relative_array_error(np.array(control.d_centroids), slopes) <
            precision)


def test_do_control_generic():
    slopes = np.array(control.d_centroids)
    gain = control.gain
    cmat = np.array(control.d_cmat)
    commands = cmat.dot(slopes) * gain * (-1)
    assert (relative_array_error(np.array(control.d_com), commands) <
            precision)


def test_set_comRange():
    control.set_comRange(-1, 1)
    assert (control.comRange == (-1, 1))


def test_clipping():
    control.set_comRange(-1, 1)
    C = (np.random.random(sup.config.p_controllers[0].nactu) - 0.5) * 4
    control.set_com(C, C.size)
    rtc.do_clipping(0)
    C_clipped = C.copy()
    C_clipped[np.where(C > 1)] = 1
    C_clipped[np.where(C < -1)] = -1
    assert (relative_array_error(np.array(control.d_com_clipped), C_clipped) <
            precision)


def test_add_perturb_voltage():
    C = np.random.random(sup.config.p_controllers[0].nactu)
    control.add_perturb_voltage("test", C, 1)
    assert (relative_array_error(
            np.array(control.d_perturb_map["test"][0]), C) < precision)


def test_remove_perturb_voltage():
    control.remove_perturb_voltage("test")
    assert (control.d_perturb_map == {})


def test_add_perturb():
    C = np.random.random(sup.config.p_controllers[0].nactu)
    control.add_perturb_voltage("test", C, 1)
    com = np.array(control.d_com_clipped)
    control.add_perturb()
    assert (relative_array_error(np.array(control.d_com_clipped), com + C) <
            precision)


def test_disable_perturb_voltage():
    control.disable_perturb_voltage("test")
    com = np.array(control.d_com)
    control.add_perturb()
    assert (relative_array_error(np.array(control.d_com), com) < precision)


def test_enable_perturb_voltage():
    control.enable_perturb_voltage("test")
    com = np.array(control.d_com_clipped)
    C = np.array(control.d_perturb_map["test"][0])
    control.add_perturb()
    assert (relative_array_error(np.array(control.d_com_clipped), com + C) <
            precision)


def test_reset_perturb_voltage():
    control.reset_perturb_voltage()
    assert (control.d_perturb_map == {})


def test_comp_voltage():
    volt_min = -1
    volt_max = 1
    control.set_comRange(volt_min, volt_max)
    control.comp_voltage()
    C = np.random.random(sup.config.p_controllers[0].nactu)
    control.add_perturb_voltage("test", C, 1)
    control.set_com(C, C.size)
    com0 = np.array(control.d_circularComs0)
    com1 = np.array(control.d_circularComs1)
    control.comp_voltage()
    delay = sup.config.p_controllers[0].delay
    a = delay - int(delay)
    b = 1 - a
    commands = a * com0 + b * com1
    comPertu = commands + C
    comPertu[np.where(comPertu > volt_max)] = volt_max
    comPertu[np.where(comPertu < volt_min)] = volt_min
    assert (relative_array_error(np.array(control.d_voltage), comPertu) <
            precision)


def test_remove_centroider():
    rtc.remove_centroider(0)
    assert (rtc.d_centro == [])


def test_doCentroids_tcog():
    rtc.add_centroider(sup.context, sup.config.p_wfss[0]._nvalid,
                       sup.config.p_wfss[0].npix / 2 - 0.5, sup.config.p_wfss[0].pixsize,
                       False, 0, "tcog")

    centro = rtc.d_centro[-1]
    threshold = 5000
    centro.set_threshold(threshold)
    centro.set_npix(sup.config.p_wfss[0].npix)
    centro.load_validpos(xvalid, yvalid, xvalid.size)
    centro.load_img(frame, frame.shape[0])
    centro.calibrate_img()
    rtc.do_centroids(0)
    slopes = np.zeros(sup.config.p_wfss[0]._nvalid * 2)
    offset = centro.offset
    scale = centro.scale
    vx = sup.config.p_wfss[0]._validsubsx
    vy = sup.config.p_wfss[0]._validsubsy
    npix = sup.config.p_wfss[0].npix
    for k in range(sup.config.p_wfss[0]._nvalid):
        imagette = frame[vx[k]:vx[k] + npix, vy[k]:vy[k] + npix].astype(
                np.float32) - threshold
        imagette[np.where(imagette < 0)] = 0
        tmp = center_of_mass(imagette)
        slopes[k] = (tmp[0] - offset) * scale
        slopes[k + sup.config.p_wfss[0]._nvalid] = (tmp[1] - offset) * scale
    assert (relative_array_error(np.array(control.d_centroids), slopes) <
            precision)


def test_doCentroids_bpcog():
    rtc.remove_centroider(0)
    rtc.add_centroider(sup.context, sup.config.p_wfss[0]._nvalid,
                       sup.config.p_wfss[0].npix / 2 - 0.5, sup.config.p_wfss[0].pixsize,
                       False, 0, "bpcog")

    centro = rtc.d_centro[-1]
    bpix = 8
    centro.set_nmax(8)
    centro.set_npix(sup.config.p_wfss[0].npix)
    centro.load_validpos(xvalid, yvalid, xvalid.size)
    centro.load_img(frame, frame.shape[0])
    centro.calibrate_img()
    rtc.do_centroids(0)
    bincube = np.array(sup.wfs._wfs.d_wfs[0].d_bincube)
    bincube /= bincube.max()
    slopes = np.zeros(sup.config.p_wfss[0]._nvalid * 2)
    offset = centro.offset
    scale = centro.scale
    vx = sup.config.p_wfss[0]._validsubsx
    vy = sup.config.p_wfss[0]._validsubsy
    npix = sup.config.p_wfss[0].npix
    for k in range(sup.config.p_wfss[0]._nvalid):
        imagette = frame[vx[k]:vx[k] + npix, vy[k]:vy[k] + npix].astype(np.float32)
        threshold = np.sort(imagette, axis=None)[-(bpix + 1)]
        imagette -= threshold
        imagette[np.where(imagette < 0)] = 0
        tmp = center_of_mass(imagette)
        slopes[k] = (tmp[0] - offset) * scale
        slopes[k + sup.config.p_wfss[0]._nvalid] = (tmp[1] - offset) * scale
    assert (relative_array_error(np.array(control.d_centroids), slopes) <
            precision)
