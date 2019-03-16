"""
Computation implementations of command matrix
"""
import numpy as np
import time

import shesha.config as conf
import shesha.constants as scons

from shesha.sutra_wrap import Rtc_FFF as Rtc

from shesha.ao.wfs import noise_cov

import typing
from typing import List


def generic_imat_inversion(
        M2V: np.ndarray,
        modalIMat: np.ndarray,
        modeSelect: np.ndarray = None,
        modeGains: np.ndarray = None,
) -> np.ndarray:
    """ Generic numpy modal interaction matrix inversion function

        :parameters:

            M2V: (nActu x nModes) : modal basis matrix

            modalIMat: (nSlopes x nModes) : modal interaction matrix

            modeSelect: (nModes, dtype=bool): (Optional):
            mode selection, mode at False is filtered

            modeGains: (nModes, dtype=bool): (Optional):
            modal gains to apply. These are gain in the reconstruction sens, ie
            they are applied multiplicatively on the command matrix
        """
    if modeSelect is None:
        modeSelect = np.ones(modalIMat.shape[1], dtype=bool)
    if modeGains is None:
        modeGains = np.ones(modalIMat.shape[1], dtype=np.float32)

    return M2V.dot(modeGains[:, None] * np.linalg.inv(modalIMat[:, modeSelect].T.dot(
            modalIMat[:, modeSelect])).dot(modalIMat[:, modeSelect].T))


def cmat_init(ncontrol: int, rtc: Rtc, p_controller: conf.Param_controller,
              p_wfss: List[conf.Param_wfs], p_atmos: conf.Param_atmos,
              p_tel: conf.Param_tel, p_dms: List[conf.Param_dm],
              nmodes: int = 0) -> None:
    """ Compute the command matrix on the GPU

    :parameters:

        ncontrol: (int) :

        rtc: (Rtc) :

        p_controller: (Param_controller) : controller settings

        p_wfss: (list of Param_wfs) : wfs settings

        p_atmos: (Param_atmos) : atmos settings

        p_tel : (Param_tel) : telescope settings

        p_dms: (list of Param_dm) : dms settings

        M2V : (np.ndarray[ndim=2, dtype=np.float32]): (optional) KL to volts matrix (for KL cmat)

        nmodes: (int) : (optional) number of kl modes
    """
    if (p_controller.type == scons.ControllerType.LS):
        print("Doing imat svd...")
        t0 = time.time()
        rtc.d_control[ncontrol].svdec_imat()
        print("svd done in %f s" % (time.time() - t0))
        eigenv = np.array(rtc.d_control[ncontrol].d_eigenvals)
        imat = np.array(rtc.d_control[ncontrol].d_imat)
        maxcond = p_controller.maxcond
        if (eigenv[0] < eigenv[eigenv.shape[0] - 1]):
            mfilt = np.where((eigenv / eigenv[eigenv.shape[0] - 3]) < 1. / maxcond)[0]
        else:
            mfilt = np.where((1. / (eigenv / eigenv[2])) > maxcond)[0]
        nfilt = mfilt.shape[0]

        print("Building cmat...")
        t0 = time.time()
        if not p_controller.do_kl_imat:
            print("Filtering ", nfilt, " modes")
            rtc.d_control[ncontrol].build_cmat(nfilt)
        else:
            # filter imat
            D_filt = imat.copy()
            # Direct inversion
            Dp_filt = np.linalg.inv(D_filt.T.dot(D_filt)).dot(D_filt.T)
            if (p_controller.klgain is not None):
                Dp_filt *= p_controller.klgain[None, :]
            cmat_filt = p_controller._M2V.dot(Dp_filt)
            rtc.d_control[ncontrol].set_cmat(cmat_filt)

        print("cmat done in %f s" % (time.time() - t0))

    if (p_controller.type == scons.ControllerType.MV):
        Cn = np.zeros(p_controller._imat.shape[0], dtype=np.float32)
        ind = 0
        for k in p_controller.nwfs:
            Cn[ind:ind + 2 * p_wfss[k]._nvalid] = noise_cov(k, p_wfss[k], p_atmos, p_tel)
            ind += 2 * p_wfss[k]._nvalid

        rtc.d_control[ncontrol].load_noisemat(Cn)
        print("Building cmat...")
        rtc.d_control[ncontrol].build_cmat(p_controller.maxcond)

        if (p_controller.TTcond == None):
            p_controller.set_TTcond(p_controller.maxcond)

        if ("tt" in [dm.type for dm in p_dms]):
            rtc.d_control[ncontrol].filter_cmat(p_controller.TTcond)
        print("Done")
    p_controller.set_cmat(np.array(rtc.d_control[ncontrol].d_cmat))
