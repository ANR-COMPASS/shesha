## @package   shesha.ao.basis
## @brief     Functions for modal basis (DM basis, KL, Btt, etc...)
## @author    COMPASS Team <https://github.com/ANR-COMPASS>
## @version   5.0.0
## @date      2020/05/18
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

from shesha.sutra_wrap import Dms, Rtc_FFF as Rtc

import shesha.config as conf
import shesha.constants as scons

from scipy.sparse import csr_matrix

from typing import List
from tqdm import trange


def compute_KL2V(p_controller: conf.Param_controller, dms: Dms, p_dms: list,
                 p_geom: conf.Param_geom, p_atmos: conf.Param_atmos,
                 p_tel: conf.Param_tel):
    """ Compute the Karhunen-Loeve to Volt matrix
    (transfer matrix between the KL space and volt space for a pzt dm)

    :parameters:

        p_controller: (Param_controller) : p_controller settings

        dms : (shesha_dms) : Dms object

        p_dms: (list of Param_dm) : dms settings

        p_geom : (Param_geom) : geometry parameters

        p_atmos : (Param_atmos) : atmos parameters

        p_tel : (Param_tel) : telescope parameters

    :return:

        KL2V : (np.array(np.float32,dim=2)) : KL to Volt matrix
    """
    ntotact = np.array([p_dms[i]._ntotact for i in range(len(p_dms))], dtype=np.int64)
    KL2V = np.zeros((np.sum(ntotact), np.sum(ntotact)), dtype=np.float32)

    indx_act = 0
    nTT = 0

    for i in range(p_controller.ndm.size):
        ndm = p_controller.ndm[i]
        if (p_dms[ndm].type == scons.DmType.PZT):
            tmp = (p_geom._ipupil.shape[0] - (p_dms[ndm]._n2 - p_dms[ndm]._n1 + 1)) // 2
            tmp_e0 = p_geom._ipupil.shape[0] - tmp
            tmp_e1 = p_geom._ipupil.shape[1] - tmp
            pup = p_geom._ipupil[tmp:tmp_e0, tmp:tmp_e1]
            indx_valid = np.where(pup.flatten("F") > 0)[0].astype(np.int32)
            p2m = p_tel.diam / p_geom.pupdiam
            norm = -(p2m * p_tel.diam / (2 * p_atmos.r0))**(5. / 3)

            dms.d_dms[ndm].compute_KLbasis(p_dms[ndm]._xpos, p_dms[ndm]._ypos,
                                           indx_valid, indx_valid.size, norm, 1.0)

            KL2V[indx_act:indx_act + ntotact[ndm], indx_act:indx_act + ntotact[ndm]] = \
                np.fliplr(dms.d_dms[ndm].d_KLbasis)
            indx_act += ntotact[ndm]
        elif (p_dms[ndm].type == scons.DmType.TT):
            nTT += 1
    if (p_controller.nmodes is not None and
                p_controller.nmodes < KL2V.shape[1] - 2 * nTT):
        KL2V = KL2V[:, :p_controller.nmodes]
    else:
        KL2V = KL2V[:, :KL2V.shape[1] - 2 * nTT]
    if (nTT > 1):
        raise ValueError("More than 1 TipTilt found! Stupid")
    if (nTT != 0):
        KL2V[:, :KL2V.shape[1] - 2] = KL2V[:, 2:]
        KL2V[:, KL2V.shape[1] - 2:] = np.zeros((np.sum(ntotact), 2), dtype=np.float32)
        KL2V[np.sum(ntotact) - 2:, KL2V.shape[1] - 2:] = np.identity(2, dtype=np.float32)

    return KL2V


def compute_dm_basis(g_dm, p_dm: conf.Param_dm, p_geom: conf.Param_geom):
    """ Compute a the DM basis as a sparse matrix :
            - push on each actuator
            - get the corresponding dm shape
            - apply pupil mask and store in a column

    :parameters:
        g_dm: (Dm) : Dm object

        p_dm: (Param_dm) : dm settings

        p_geom: (Param_geom) : geom settings

    :return:

        IFbasis = (csr_matrix) : DM IF basis
    """
    tmp = (p_geom._ipupil.shape[0] - (p_dm._n2 - p_dm._n1 + 1)) // 2
    tmp_e0 = p_geom._ipupil.shape[0] - tmp
    tmp_e1 = p_geom._ipupil.shape[1] - tmp
    pup = p_geom._ipupil[tmp:tmp_e0, tmp:tmp_e1]
    indx_valid = np.where(pup.flatten("F") > 0)[0].astype(np.int32)

    #IFbasis = np.ndarray((indx_valid.size, p_dm._ntotact), dtype=np.float32)
    for i in trange(p_dm._ntotact):
        g_dm.reset_shape()
        g_dm.comp_oneactu(i, 1.0)
        shape = np.array(g_dm.d_shape)
        IFvec = csr_matrix(shape.flatten("F")[indx_valid])
        if (i == 0):
            val = IFvec.data
            col = IFvec.indices
            row = np.append(0, IFvec.getnnz())
        else:
            val = np.append(val, IFvec.data)
            col = np.append(col, IFvec.indices)
            row = np.append(row, row[-1] + IFvec.getnnz())
    g_dm.reset_shape()
    IFbasis = csr_matrix((val, col, row))
    return IFbasis


def compute_IFsparse(g_dm: Dms, p_dms: list, p_geom: conf.Param_geom):
    """ Compute the influence functions of all DMs as a sparse matrix :
            - push on each actuator
            - get the corresponding dm shape
            - apply pupil mask and store in a column

    :parameters:

        g_dm: (Dms) : Dms object

        p_dms: (Param_dms) : dms settings

        p_geom: (Param_geom) : geom settings

    :return:

        IFbasis = (csr_matrix) : DM IF basis
    """
    ndm = len(p_dms)
    for i in range(ndm):
        IFi = compute_dm_basis(g_dm.d_dms[i], p_dms[i], p_geom)
        if (i == 0):
            val = IFi.data
            col = IFi.indices
            row = IFi.indptr
        else:
            val = np.append(val, IFi.data)
            col = np.append(col, IFi.indices)
            row = np.append(row, row[-1] + IFi.indptr[1:])
    IFsparse = csr_matrix((val, col, row))
    return IFsparse


def command_on_Btt(rtc: Rtc, dms: Dms, p_dms: list, p_geom: conf.Param_geom, nfilt: int):
    """ Compute a command matrix in Btt modal basis (see error breakdown) and set
    it on the sutra_rtc. It computes by itself the volts to Btt matrix.

    :parameters:

        rtc: (Rtc) : rtc object

        dms: (Dms): dms object

        p_dms: (list of Param_dm): dms settings

        p_geom: (Param_geom): geometry settings

        nfilt: (int): number of modes to filter
    """

    IFs = compute_IFsparse(dms, p_dms, p_geom).T
    n = IFs.shape[1]
    IFtt = IFs[:, -2:].copy().toarray()
    IFpzt = IFs[:, :n - 2]

    Btt, P = compute_btt(IFpzt, IFtt)
    compute_cmat_with_Btt(rtc, Btt, nfilt)


def compute_cmat_with_Btt(rtc: Rtc, Btt: np.ndarray, nfilt: int):
    """ Compute a command matrix on the Btt basis and load it in the GPU

    :parameters:

        rtc: (Rtc): rtc object

        Btt: (np.ndarray[ndim=2, dtype=np.float32]) : volts to Btt matrix

        nfilt: (int): number of modes to filter
    """
    D = np.array(rtc.d_control[0].d_imat)
    #D = ao.imat_geom(wfs,config.p_wfss,config.p_controllers[0],dms,config.p_dms,meth=0)
    # Filtering on Btt modes
    Btt_filt = np.zeros((Btt.shape[0], Btt.shape[1] - nfilt))
    Btt_filt[:, :Btt_filt.shape[1] - 2] = Btt[:, :Btt.shape[1] - (nfilt + 2)]
    Btt_filt[:, Btt_filt.shape[1] - 2:] = Btt[:, Btt.shape[1] - 2:]

    # Modal interaction basis
    Dm = D.dot(Btt_filt)
    # Direct inversion
    Dmp = np.linalg.inv(Dm.T.dot(Dm)).dot(Dm.T)
    # Command matrix
    cmat = Btt_filt.dot(Dmp)
    rtc.d_control[0].set_cmat(cmat.astype(np.float32))

    return cmat.astype(np.float32)


def command_on_KL(rtc: Rtc, dms: Dms, p_controller: conf.Param_controller,
                  p_dms: List[conf.Param_dm], p_geom: conf.Param_geom,
                  p_atmos: conf.Param_atmos, p_tel: conf.Param_tel, nfilt: int):
    """ Compute a command matrix in KL modal basis and set
    it on the sutra_rtc. It computes by itself the volts to KL matrix.

    :parameters:

        rtc: (Rtc) : rtc object

        dms: (Dms): dms object

        p_dms: (list of Param_dm): dms settings

        p_geom: (Param_geom): geometry settings

        p_atmos : (Param_atmos) : atmos parameters

        p_tel : (Param_tel) : telescope parameters

        nfilt: (int): number of modes to filter
    """
    KL2V = compute_KL2V(p_controller, dms, p_dms, p_geom, p_atmos, p_tel)
    return compute_cmat_with_KL(rtc, KL2V, nfilt)


def compute_cmat_with_KL(rtc: Rtc, KL2V: np.ndarray, nfilt: int):
    """ Compute a command matrix on the KL basis and load it in the GPU

    :parameters:

        rtc: (Rtc): rtc object

        KL2V: (np.ndarray[ndim=2, dtype=np.float32]) : volts to KL matrix

        nfilt: (int): number of modes to filter
    """
    D = np.array(rtc.d_control[0].d_imat)
    KL2V_filt = np.zeros((KL2V.shape[0], KL2V.shape[1] - nfilt))
    KL2V_filt[:, :KL2V_filt.shape[1] - 2] = KL2V[:, :KL2V.shape[1] - (nfilt + 2)]
    KL2V_filt[:, KL2V_filt.shape[1] - 2:] = KL2V[:, KL2V.shape[1] - 2:]

    # Modal interaction basis
    Dm = D.dot(KL2V_filt)
    # Direct inversion
    Dmp = np.linalg.inv(Dm.T.dot(Dm)).dot(Dm.T)
    # Command matrix
    cmat = KL2V_filt.dot(Dmp)
    rtc.d_control[0].set_cmat(cmat.astype(np.float32))

    return cmat.astype(np.float32)


def compute_fourier(nActu: int, pitch: float, actu_x_pos: np.ndarray,
                    actu_y_pos: np.ndarray, periodic='n'):
    '''
        Values you are looking for are:
            config.p_dm0.nact
            config.p_dm0._pitch
            config.p_dm0._i1
            config.p_dm0._j1
    '''
    # Offset xpos and ypos to get integer indices.
    # Compute nact x nact x nact x nact Fourier basis # Periodic condition n / n-1 as option
    # Extract values for actuators - flatten

    # Periodic may be 'n' or 'n-1'
    # Will work only for squared pitch mirrors so far
    if periodic == 'n':
        n = nActu
    elif periodic == 'n-1':
        n = nActu - 1
    else:
        raise ValueError('periodic can only be "n" or "n-1" to set boundary condition.')
    xnorm = (np.round((actu_x_pos - np.min(actu_x_pos)) / pitch).astype(np.int32)) % n
    ynorm = (np.round((actu_y_pos - np.min(actu_y_pos)) / pitch).astype(np.int32)) % n
    totActu = len(xnorm)

    data = np.zeros((n, n, n, n), dtype=np.float32)
    for i in range(n):
        for j in range(n):
            data[i, j, i, j] = 1.

    data = np.fft.fftn(data, axes=(2, 3))

    takeSine = np.zeros((n, n), dtype=bool)  # Where to take sine instead of cosine
    takeSine[0, n // 2 + 1:] = 1  # Half of first line
    if n % 2 == 0:
        takeSine[n // 2, n // 2 + 1:] = 1  # Half of waffle line

    takeSine[n // 2 + 1:, :] = 1  # Bottom half

    data[takeSine] *= 1j
    data = data.real

    # Extract actuators
    actuPush = data[:, :, xnorm, ynorm]

    # Add a renorm ?

    return actuPush


def compute_btt(IFpzt, IFtt, influ_petal=None, return_delta=False):
    """ Returns Btt to Volts and Volts to Btt matrices

    :parameters:

        IFpzt : (csr_matrix) : influence function matrix of pzt DM, sparse and arrange as (Npts in pup x nactus)

        IFtt : (np.ndarray(ndim=2,dtype=np.float32)) : Influence function matrix of the TT mirror arrange as (Npts in pup x 2)

        influ_petal : (np.ndarray) : Influence function matrix of M4 petals.
                                     Default is None, if set, the Btt produced is also orthogonal
                                     to petal modes, then only driven by petal DM

        return_delta : (bool, optional) : If True, returns delta instead of P. Default is False

    :returns:

        Btt : (np.ndarray(ndim=2,dtype=np.float32)) : Btt to Volts matrix

        P : (np.ndarray(ndim=2,dtype=np.float32)) : Volts to Btt matrix
    """
    N = IFpzt.shape[0]
    n = IFpzt.shape[1]
    if (n > N):
        raise ValueError("Influence functions must be arrange as (Npts_pup x nactus)")

    delta = IFpzt.T.dot(IFpzt).toarray() / N

    # Tip-tilt + piston
    Tp = np.ones((IFtt.shape[0], IFtt.shape[1] + 1))
    Tp[:, :2] = IFtt.copy(
    )  # THIS IS NOT A SPARSE OBJECT !!!!! STOP PUTTING .toarray() HERE PLEASE !!!
    deltaT = IFpzt.T.dot(Tp) / N
    # Tip tilt projection on the pzt dm
    tau = np.linalg.inv(delta).dot(deltaT)
    nfilt = 3  # Piston + tip + tilt

    if influ_petal is not None:
        # Petal basis generation (orthogonal to global piston)
        nseg = influ_petal.toarray().shape[0]
        petal_modes = -1 / (nseg - 1) * np.ones((nseg, (nseg - 1)))
        petal_modes += nseg / (nseg - 1) * np.eye(nseg)[:, 0:(
                nseg - 1)]  # petal modes within the petal dm space
        tau_petal = IFpzt.T.dot(influ_petal).toarray().dot(petal_modes.T)
        tau = np.concatenate((tau_petal, np.linalg.inv(delta).dot(deltaT)), axis=1)
        nfilt = 8
    # Famille generatrice sans tip tilt
    G = np.identity(n)
    tdt = tau.T.dot(delta).dot(tau)
    subTT = tau.dot(np.linalg.inv(tdt)).dot(tau.T).dot(delta)
    G -= subTT

    # Base orthonormee sans TT
    gdg = G.T.dot(delta).dot(G)
    U, s, V = np.linalg.svd(gdg)
    U = U[:, :U.shape[1] - nfilt]
    s = s[:s.size - nfilt]
    L = np.identity(s.size) / np.sqrt(s)
    B = G.dot(U).dot(L)

    # Rajout du TT
    TT = IFtt.T.dot(IFtt) / N
    Btt = np.zeros((n + 2, n - 1))
    Btt[:B.shape[0], :B.shape[1]] = B
    mini = 1. / np.sqrt(np.abs(TT))
    mini[0, 1] = 0
    mini[1, 0] = 0
    Btt[n:, -2:] = mini
    if influ_petal is not None:
        Btt[:n, -7:-2] = tau_petal

    # Calcul du projecteur actus-->modes
    Delta = np.zeros((n + IFtt.shape[1], n + IFtt.shape[1]))
    #IFpzt = rtc.get_IFpztsparse(1).T
    Delta[:-2, :-2] = delta
    Delta[-2:, -2:] = TT
    if return_delta:
        P = Delta
    else:
        P = Btt.T.dot(Delta)

    return Btt.astype(np.float32), P.astype(np.float32)
