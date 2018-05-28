"""
Computation implementations of command matrix
"""
import numpy as np
import time

import shesha.config as conf
import shesha.constants as scons

from shesha.sutra_bind.wrap import Rtc

from shesha.ao.wfs import noise_cov

import typing
from typing import List


def cmat_init(ncontrol: int, rtc: Rtc, p_controller: conf.Param_controller,
              p_wfss: List[conf.Param_wfs], p_atmos: conf.Param_atmos,
              p_tel: conf.Param_tel, p_dms: List[conf.Param_dm], KL2V: np.ndarray=None,
              nmodes: int=0) -> None:
    """ Compute the command matrix on the GPU

    :parameters:

        ncontrol: (int) :

        rtc: (Rtc) :

        p_controller: (Param_controller) : controller settings

        p_wfss: (list of Param_wfs) : wfs settings

        p_atmos: (Param_atmos) : atmos settings

        p_tel : (Param_tel) : telescope settings

        p_dms: (list of Param_dm) : dms settings

        KL2V : (np.ndarray[ndim=2, dtype=np.float32]): (optional) KL to volts matrix (for KL cmat)

        nmodes: (int) : (optional) number of kl modes
    """
    if (p_controller.type == scons.ControllerType.LS):
        print("Doing imat svd...")
        t0 = time.time()
        rtc.imat_svd(ncontrol)
        print("svd done in %f s" % (time.time() - t0))
        eigenv = rtc.get_eigenvals(ncontrol)

        imat = rtc.get_imat(ncontrol)
        maxcond = p_controller.maxcond
        if (eigenv[0] < eigenv[eigenv.shape[0] - 1]):
            mfilt = np.where((eigenv / eigenv[eigenv.shape[0] - 3]) < 1. / maxcond)[0]
        else:
            mfilt = np.where((1. / (eigenv / eigenv[2])) > maxcond)[0]
        nfilt = mfilt.shape[0]

        print("Building cmat...")
        t0 = time.time()
        if KL2V is None:
            print("Filtering ", nfilt, " modes")
            rtc.build_cmat(ncontrol, nfilt)
        else:
            ntt = 0
            pii = 0
            for i in range(len(p_dms)):
                ppz = p_dms[i].push4imat
                if ((p_dms[i].type == scons.DmType.PZT) & (ppz == 0)):
                    ppz = 1
                    pii = i
            if ((nmodes == 0) & (ppz != 0)):
                nmodes = p_dms[pii]._ntotact
            if (ppz == 1):
                # filter imat
                D_filt = imat[:, :KL2V.shape[1]]
                # Direct inversion
                Dp_filt = np.linalg.inv(D_filt.T.dot(D_filt)).dot(D_filt.T)
                if (p_controller.klgain is not None):
                    if (p_controller.klgain.shape[0] == KL2V.shape[1]):
                        for i in range(KL2V.shape[1]):
                            Dp_filt[:, i] *= p_controller.klgain[i]
                    else:
                        print("Need size :")
                        print(KL2V.shape[1])
                        raise TypeError("incorect size for klgain vector")
                cmat_filt = KL2V.dot(Dp_filt)
                rtc.set_cmat(ncontrol, cmat_filt)

        print("cmat done in %f s" % (time.time() - t0))

    if (p_controller.type == scons.ControllerType.MV):
        Cn = np.zeros(p_controller._imat.shape[0], dtype=np.float32)
        ind = 0
        for k in p_controller.nwfs:
            Cn[ind:ind + 2 * p_wfss[k]._nvalid] = noise_cov(k, p_wfss[k], p_atmos, p_tel)
            ind += 2 * p_wfss[k]._nvalid

        rtc.load_Cn(ncontrol, Cn)
        print("Building cmat...")
        rtc.build_cmat_mv(ncontrol, p_controller.maxcond)

        if (p_controller.TTcond == 0):
            p_controller.set_TTcond(p_controller.maxcond)

        if ("tt" in [dm.type for dm in p_dms]):
            rtc.filter_cmat(ncontrol, p_controller.TTcond)
        print("Done")
    p_controller.set_cmat(rtc.get_cmat(ncontrol))
