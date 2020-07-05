import numpy as np
import naga as ng
import os
from shesha.sutra_wrap import Rtc_FFF as Rtc
from shesha.supervisor.compassSupervisor import CompassSupervisor as Supervisor
from scipy.ndimage.measurements import center_of_mass
from shesha.util.utilities import load_config_from_file

precision = 1e-2

config = load_config_from_file(os.getenv("COMPASS_ROOT") +
        "/shesha/tests/pytest/par/test_pyrhr.py")
sup = Supervisor(config)
sup.next()
sup.rtc.open_loop(0)
sup.rtc.close_loop(0)
sup.rtc.do_control(0)
rtc = Rtc()
rtc.add_centroider(sup.context, sup.config.p_wfs0._nvalid, 0, sup.config.p_wfs0.pixsize,
                   False, 0, "maskedpix")
rtc.add_controller(sup.context, sup.config.p_wfs0._nvalid,
                   sup.config.p_controller0.nslope, sup.config.p_controller0.nactu,
                   sup.config.p_controller0.delay, 0, "generic", idx_centro=np.zeros(1), ncentro=1)
centro = rtc.d_centro[0]
control = rtc.d_control[0]
rtc.d_centro[0].set_npix(sup.config.p_wfs0.npix)
xvalid = np.array(sup.rtc._rtc.d_centro[0].d_validx)
yvalid = np.array(sup.rtc._rtc.d_centro[0].d_validy)
rtc.d_centro[0].load_validpos(xvalid, yvalid, xvalid.size)
cmat = sup.rtc.get_command_matrix(0)
rtc.d_control[0].set_cmat(cmat)
rtc.d_control[0].set_gain(sup.config.p_controller0.gain)
frame = sup.wfs.get_wfs_image(0)
frame /= frame.max()
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


def test_doCentroids_maskedPix():
    binimg = np.array(centro.d_img)
    slopes = np.zeros(xvalid.size)
    psum = binimg[xvalid, yvalid].sum() / slopes.size
    for k in range(slopes.size):
        slopes[k] = binimg[xvalid[k], yvalid[k]] / psum
    assert (relative_array_error(ng.array(control.d_centroids).toarray(), slopes) <
            precision)
