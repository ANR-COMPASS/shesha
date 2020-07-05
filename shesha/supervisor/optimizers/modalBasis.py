## @package   shesha.supervisor.optimizers
## @brief     User layer for optimizing AO supervisor loop
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
from shesha.ao import basis
import shesha.util.utilities as util
import shesha.util.make_pupil as mkP
import shesha.constants as scons
import scipy.ndimage
from scipy.sparse.csr import csr_matrix
import numpy as np

class ModalBasis(object):
    """ This optimizer class handles all the modal basis and DM Influence functions
    related operations.

    Attributes:
        _config : (config) : Configuration parameters module

        _dms : (DmCompass) : DmCompass instance

        _target : (TargetCompass) : TargetCompass instance

        slaved_actus : TODO : docstring

        selected_actus : TODO : docstring

        couples_actus : TODO : docstring

        index_under_spiders : TODO : docstring

        modal_basis : (np.ndarray) : Last modal basis computed

        projection_matrix : (np.ndarray) : Last projection_matrix computed
    """
    def __init__(self, config, dms, target):
        """ Instantiate a ModalBasis object

        Parameters:
            config : (config) : Configuration parameters module

            dms : (DmCompass) : DmCompass instance

            target : (TargetCompass) : TargetCompass instance
        """
        self._config = config
        self._dms = dms
        self._target = target
        self.slaved_actus = None
        self.selected_actus = None
        self.couples_actus = None
        self.index_under_spiders = None
        self.modal_basis = None
        self.projection_matrix = None

    def compute_influ_basis(self, dm_index: int) -> csr_matrix:
        """ Computes and return the influence function phase basis of the specified DM
        as a sparse matrix

        Parameters:
            dm_index : (int) : Index of the DM

        Return:
            influ_sparse : (csr_matrix) : influence function phases
        """
        return basis.compute_dm_basis(self._dms._dms.d_dms[dm_index],
                                              self._config.p_dms[dm_index],
                                              self._config.p_geom)

    def compute_modes_to_volts_basis(self, modal_basis_type: str, *, merged: bool = False,
                                     nbpairs: int = None, return_delta: bool = False) -> np.ndarray:
        """ Computes a given modal basis ("KL2V", "Btt", "Btt_petal") and return the 2 transfer matrices

        Parameters:
            modal_basis_type : (str) : modal basis to compute ("KL2V", "Btt", "Btt_petal")

            merged : (bool, optional) :

            nbpairs : (int, optional) :

        Return:
            modal_basis : (np.ndarray) : modes to volts matrix

            projection_matrix : (np.ndarray) : volts to modes matrix (None if "KL")
        """
        if (modal_basis_type == "KL2V"):
            print("Computing KL2V basis...")
            self.modal_basis = basis.compute_KL2V(
                    self._config.p_controllers[0], self._dms._dms,
                    self._config.p_dms, self._config.p_geom,
                    self._config.p_atmos, self._config.p_tel)
            fnz = util.first_non_zero(self.modal_basis, axis=0)
            # Computing the sign of the first non zero element
            #sig = np.sign(modal_basis[[fnz, np.arange(modal_basis.shape[1])]])
            sig = np.sign(self.modal_basis[tuple([
                    fnz, np.arange(self.modal_basis.shape[1])
            ])])  # pour remove le future warning!
            self.modal_basis *= sig[None, :]
            projection_matrix = None
        elif (modal_basis_type == "Btt"):
            print("Computing Btt basis...")
            self.modal_basis, self.projection_matrix = self.compute_btt_basis(
                                                        merged=merged, nbpairs=nbpairs,
                                                        return_delta=return_delta)
            fnz = util.first_non_zero(self.modal_basis, axis=0)
            # Computing the sign of the first non zero element
            #sig = np.sign(modal_basis[[fnz, np.arange(modal_basis.shape[1])]])
            sig = np.sign(self.modal_basis[tuple([
                    fnz, np.arange(self.modal_basis.shape[1])
            ])])  # pour remove le future warning!
            self.modal_basis *= sig[None, :]
        elif (modal_basis_type == "Btt_petal"):
            print("Computing Btt with a Petal basis...")
            self.modal_basis, self.projection_matrix = self.compute_btt_petal()
        else:
            raise ArgumentError("Unsupported modal basis")

        return self.modal_basis, self.projection_matrix

    def compute_btt_basis(self, *, merged: bool = False, nbpairs: int = None,
                          return_delta: bool = False) -> np.ndarray:
        """ Computes the so-called Btt modal basis. The <merged> flag allows merto merge
        2x2 the actuators influence functions for actuators on each side of the spider (ELT case)

        Parameters:
            merged : (bool, optional) : If True, merge 2x2 the actuators influence functions for
                                        actuators on each side of the spider (ELT case). Default
                                        is False

            nbpairs : (int, optional) : Default is None. TODO : description

            return_delta : (bool, optional) : If False (default), the function returns
                                              Btt (modes to volts matrix),
                                              and P (volts to mode matrix).
                                              If True, returns delta = IF.T.dot(IF) / N
                                              instead of P

        Return:
            Btt : (np.ndarray) : Btt modes to volts matrix

            projection_matrix : (np.ndarray) : volts to Btt modes matrix
        """
        dms_basis = basis.compute_IFsparse(self._dms._dms, self._config.p_dms, self._config.p_geom)
        influ_basis = dms_basis[:-2,:]
        tt_basis = dms_basis[-2:,:].toarray()
        if (merged):
            couples_actus, index_under_spiders = self.compute_merged_influ(0,
                    nbpairs=nbpairs)
            influ_basis2 = influ_basis.copy()
            index_remove = index_under_spiders.copy()
            index_remove += list(couples_actus[:, 1])
            print("Pairing Actuators...")
            for i in range(couples_actus.shape[0]):
                influ_basis2[couples_actus[i, 0], :] += influ_basis2[
                        couples_actus[i, 1], :]
            print("Pairing Done")
            boolarray = np.zeros(influ_basis2.shape[0], dtype=np.bool)
            boolarray[index_remove] = True
            self.slaved_actus = boolarray
            self.selected_actus = ~boolarray
            self.couples_actus = couples_actus
            self.index_under_spiders = index_under_spiders
            influ_basis2 = influ_basis2[~boolarray, :]
            influ_basis = influ_basis2

        self.btt, self.projection_matrix = basis.compute_btt(influ_basis.T, tt_basis.T, return_delta=return_delta)

        if (merged):
            btt2 = np.zeros((len(boolarray) + 2, self.btt.shape[1]))
            btt2[np.r_[~boolarray, True, True], :] = self.btt
            btt2[couples_actus[:, 1], :] = btt2[couples_actus[:, 0], :]

            P2 = np.zeros((self.btt.shape[1], len(boolarray) + 2))
            P2[:, np.r_[~boolarray, True, True]] = self.projection_matrix
            P2[:, couples_actus[:, 1]] = P2[:, couples_actus[:, 0]]
            self.btt = btt2
            self.projection_matrix = P2

        return self.btt, self.projection_matrix

    def compute_merged_influ(self, dm_index : int, *, nbpairs: int = None) -> np.ndarray:
        """ Used to compute merged IF from each side of the spider
        for an ELT case (Petalling Effect)

        Parameters:
            dm_index : (int) : DM index

            nbpairs : (int, optional) : Default is None. TODO : description

        Return:
            pairs : (np.ndarray) : TODO description

            discard : (list) : TODO description
        """
        p_geom = self._config.p_geom


        cent = p_geom.pupdiam / 2. + 0.5
        p_tel = self._config.p_tel
        p_tel.t_spiders = 0.51
        spup = mkP.make_pupil(p_geom.pupdiam, p_geom.pupdiam, p_tel, cent,
                              cent).astype(np.float32).T

        p_tel.t_spiders = 0.
        spup2 = mkP.make_pupil(p_geom.pupdiam, p_geom.pupdiam, p_tel, cent,
                               cent).astype(np.float32).T

        spiders = spup2 - spup

        (spidersID, k) = scipy.ndimage.label(spiders)
        spidersi = util.pad_array(spidersID, p_geom.ssize).astype(np.float32)
        px_list_spider = [np.where(spidersi == i) for i in range(1, k + 1)]

        # DM positions in iPupil:
        dm_posx = self._config.p_dms[dm_index]._xpos - 0.5
        dm_posy = self._config.p_dms[dm_index]._ypos - 0.5
        dm_pos_mat = np.c_[dm_posx, dm_posy].T  # one actu per column

        pitch = self._config.p_dms[dm_index]._pitch
        discard = np.zeros(len(dm_posx), dtype=np.bool)
        pairs = []

        # For each of the k pieces of the spider
        for k, px_list in enumerate(px_list_spider):
            pts = np.c_[px_list[1],
                        px_list[0]]  # x,y coord of pixels of the spider piece
            # line_eq = [a, b]
            # Which minimizes leqst squares of aa*x + bb*y = 1
            line_eq = np.linalg.pinv(pts).dot(np.ones(pts.shape[0]))
            aa, bb = line_eq[0], line_eq[1]

            # Find any point of the fitted line.
            # For simplicity, the intercept with one of the axes x = 0 / y = 0
            if np.abs(bb) < np.abs(aa):  # near vertical
                one_point = np.array([1 / aa, 0.])
            else:  # otherwise
                one_point = np.array([0., 1 / bb])

            # Rotation that aligns the spider piece to the horizontal
            rotation = np.array([[-bb, aa], [-aa, -bb]]) / (aa**2 + bb**2)**.5

            # Rotated the spider mask
            rotated_px = rotation.dot(pts.T - one_point[:, None])
            # Min and max coordinates along the spider length - to filter actuators that are on
            # 'This' side of the pupil and not the other side
            min_u, max_u = rotated_px[0].min() - 5. * pitch, rotated_px[0].max(
            ) + 5. * pitch

            # Rotate the actuators
            rotated_actus = rotation.dot(dm_pos_mat - one_point[:, None])
            sel_good_side = (rotated_actus[0] > min_u) & (rotated_actus[0] < max_u)
            threshold = 0.05
            # Actuators below this piece of spider
            sel_discard = (np.abs(rotated_actus[1]) < threshold * pitch) & sel_good_side
            discard |= sel_discard

            # Actuator 'near' this piece of spider
            sel_pairable = (np.abs(rotated_actus[1]) > threshold  * pitch) & \
                            (np.abs(rotated_actus[1]) < 1. * pitch) & \
                            sel_good_side

            pairable_index = np.where(sel_pairable)[0]  # Indices of these actuators
            u_coord = rotated_actus[
                    0, sel_pairable]  # Their linear coord along the spider major axis

            order = np.sort(u_coord)  # Sort by linear coordinate
            order_index = pairable_index[np.argsort(
                    u_coord)]  # And keep track of original indexes

            # i = 0
            # while i < len(order) - 1:
            if (nbpairs is None):
                i = 0
                ii = len(order) - 1
            else:
                i = len(order) // 2 - nbpairs
                ii = len(order) // 2 + nbpairs
            while (i < ii):
                # Check if next actu in sorted order is very close
                # Some lonely actuators may be hanging in this list
                if np.abs(order[i] - order[i + 1]) < .2 * pitch:
                    pairs += [(order_index[i], order_index[i + 1])]
                    i += 2
                else:
                    i += 1
        print('To discard: %u actu' % np.sum(discard))
        print('%u pairs to slave' % len(pairs))
        if np.sum(discard) == 0:
            discard = []
        else:
            list(np.where(discard)[0])
        return np.asarray(pairs), list(np.where(discard)[0])

    def compute_btt_petal(self) -> np.ndarray:
        """ Computes a Btt modal basis with Pistons filtered

        Return:
            Btt : (np.ndarray) : Btt modes to volts matrix

            P : (np.ndarray) : volts to Btt modes matrix
        """
        pzt_index = np.where([d.type is scons.DmType.PZT for d in self._config.p_dms])[0][0]
        influ_pzt = self.compute_influ_basis(pzt_index)
        petal_dm_index = np.where([
                d.influ_type is scons.InfluType.PETAL for d in self._config.p_dms
        ])[0][0]
        influ_petal = self.compute_influ_basis(petal_dm_index)
        tt_index = np.where([d.type is scons.DmType.TT for d in self._config.p_dms])[0][0]
        influ_tt = self.compute_influ_basis(tt_index).toarray()

        self.modal_basis, self.projection_matrix = basis.compute_btt(influ_pzt.T, influ_tt.T, influ_petal=influ_petal)
        return self.modal_basis, self.projection_matrix

    def compute_phase_to_modes(self, modal_basis: np.ndarray) -> np.ndarray:
        """ Return the phase to modes matrix by using the given modal basis

        Parameters:
            modal_basis : (np.ndarray) : Modal basis matrix

        Return:
            phase_to_modes : (np.ndarray) : phase to modes matrix
        """
        nbmode = modal_basis.shape[1]
        phase = self._target.get_tar_phase(0)
        phase_to_modes = np.zeros((nbmode, phase.shape[0], phase.shape[1]))
        S = np.sum(self._config.p_geom._spupil)
        for i in range(nbmode):
            self._dms.set_command((modal_basis[:, i]).copy())
            # self.next(see_atmos=False)
            self._target.raytrace(0, dms=self._dms, ncpa=False, reset=True)
            phase = self._target.get_tar_phase(0, pupil=True)
            # Normalisation pour les unites rms en microns !!!
            norm = np.sqrt(np.sum((phase)**2) / S)
            phase_to_modes[i] = phase / norm
        return phase_to_modes
