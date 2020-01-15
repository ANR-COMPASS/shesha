## @package   shesha.ao.cmats
## @brief     Computation implementations of command matrix
## @author    COMPASS Team <https://github.com/ANR-COMPASS>
## @version   4.4.0
## @date      2011/01/28
## @copyright GNU Lesser General Public License
#
#  This file is part of COMPASS <https://anr-compass.github.io/compass/>
#
#  Copyright (C) 2011-2019 COMPASS Team <https://github.com/ANR-COMPASS>
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


def svd_for_cmat(D):
    DtD = D.T.dot(D)
    return np.linalg.svd(DtD)


def Btt_for_cmat(rtc, dms, p_dms, p_geom):
    """ Compute a command matrix in Btt modal basis (see error breakdown) and set
    it on the sutra_rtc. It computes by itself the volts to Btt matrix.

    :parameters:

        rtc: (Rtc) : rtc object

        dms: (Dms): dms object

        p_dms: (list of Param_dm): dms settings

        p_geom: (Param_geom): geometry settings

    """

    IFs = basis.compute_IFsparse(dms, p_dms, p_geom).T
    n = IFs.shape[1]
    IFtt = IFs[:, -2:].toarray()
    IFpzt = IFs[:, :n - 2]

    Btt, P = basis.compute_Btt(IFpzt, IFtt)
    return Btt, P


def get_cmat(D, nfilt, Btt=None, rtc=None, svd=None):
    """Compute a command matrix from an interaction matrix 'D'

    usage:
        get_cmat(D,nfilt)
        get_cmat(D,nfilt,Btt=BTT,rtc=RTC)
        get_cmat(D,nfilt,svd=SVD)

    :parameters:
        D: (np.ndarray[ndim=2, dtype=np.float32]): interaction matrix

        nfilt: (int): number of element to filter

        Btt: (np.ndarray[ndim=2, dtype=np.float32]): Btt modal basis

        rtc: (Rtc) :

        svd: (tuple of np.ndarray[ndim=1, dtype=np.float32): svd of D.T*D (obtained from np.linalg.svd)
    """
    nfilt = max(nfilt, 0)  #nfilt is positive
    if (Btt is not None):
        if (svd is not None):
            raise ValueError("Btt and SVD cannt be used together")
        if (rtc is None):
            raise ValueError("Btt cannot be used without rtc")
        n = Btt.shape[1]
        index = np.concatenate((np.arange(n - nfilt - 2), np.array([n - 2,
                                                                    n - 1]))).astype(int)
        Btt_filt = Btt[:, index]
        # Modal interaction basis
        Dm = D.dot(Btt_filt)
        # Direct inversion
        Dmp = np.linalg.inv(Dm.T.dot(Dm)).dot(Dm.T)
        # Command matrix
        cmat = Btt_filt.dot(Dmp)
    else:
        if (svd is not None):
            u = svd[0]
            s = svd[1]
            v = svd[2]
        else:
            u, s, v = svd_for_cmat(D)
        s_filt = 1 / s
        if (nfilt > 0):
            s_filt[-nfilt:] = 0
        DtDx = v.T.dot(np.diag(s_filt)).dot(u.T)
        cmat = DtDx.dot(D.T)

    return cmat.astype(np.float32)
