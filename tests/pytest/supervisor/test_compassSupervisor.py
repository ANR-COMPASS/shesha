## @package   shesha.tests
## @brief     Tests the supervisor module
## @author    COMPASS Team <https://github.com/ANR-COMPASS>
## @version   5.2.1
## @date      2022/01/24
## @copyright GNU Lesser General Public License
#
#  This file is part of COMPASS <https://anr-compass.github.io/compass/>
#
#  Copyright (C) 2011-2022 COMPASS Team <https://github.com/ANR-COMPASS>
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

from shesha.supervisor.compassSupervisor import CompassSupervisor
from shesha.config import ParamConfig

import os
import numpy as np
import pytest
import shesha.config as conf

config = ParamConfig(os.getenv("COMPASS_ROOT") + "/shesha/tests/pytest/par/test_pyrhr.py")
config.p_controllers[0].set_type("generic")

sup = CompassSupervisor(config)

def test_next():
    sup.next()
    assert(sup.iter == 1)

def test_loop():
    sup.loop(100)
    assert(sup.iter == 101)

def test_reset():
    sup.reset()
    assert(True)

#                 __ _
#    __ ___ _ _  / _(_)__ _
#   / _/ _ \ ' \|  _| / _` |
#   \__\___/_||_|_| |_\__, |
#                     |___/

def test_get_ipupil():
    ipupil = sup.config.get_pupil("i")
    ipupil = sup.config.get_pupil("ipupil")
    assert(ipupil is sup.config.p_geom._ipupil)

def test_get_mpupil():
    mpupil = sup.config.get_pupil("m")
    mpupil = sup.config.get_pupil("mpupil")
    assert(mpupil is sup.config.p_geom._mpupil)

def test_get_spupil():
    spupil = sup.config.get_pupil("spupil")
    spupil = sup.config.get_pupil("s")
    assert(spupil is sup.config.p_geom._spupil)

def test_export_config():
    aodict, datadict = sup.config.export_config()
    assert(True)

#         _                 ___
#    __ _| |_ _ __  ___ ___/ __|___ _ __  _ __  __ _ ______
#   / _` |  _| '  \/ _ (_-< (__/ _ \ '  \| '_ \/ _` (_-<_-<
#   \__,_|\__|_|_|_\___/__/\___\___/_|_|_| .__/\__,_/__/__/
#                                        |_|
def test_enable_atmos():
    sup.atmos.enable_atmos(False)
    assert(sup.atmos.is_enable == False)

def test_set_r0():
    sup.atmos.set_r0(0.15)
    assert(sup.config.p_atmos.r0 == 0.15)

def test_set_wind():
    sup.atmos.set_wind(0,windspeed=10, winddir=45)
    assert(True)

def test_reset_turbu():
    sup.atmos.reset_turbu()
    assert(True)

def test_get_atmos_layer():
    sup.atmos.get_atmos_layer(0)
    assert(True)

def test_move_atmos():
    sup.atmos.move_atmos()
    assert(True)
#    ___        ___
#   |   \ _ __ / __|___ _ __  _ __  __ _ ______
#   | |) | '  \ (__/ _ \ '  \| '_ \/ _` (_-<_-<
#   |___/|_|_|_\___\___/_|_|_| .__/\__,_/__/__/
#                            |_|
def test_set_command():
    sup.dms.set_command(np.zeros(sup.config.p_controllers[0].nactu))
    sup.dms.set_command(np.zeros(sup.config.p_controllers[0].nactu), shape_dm=False)
    sup.dms.set_command(np.zeros(sup.config.p_dms[0]._ntotact), dm_index=0)
    assert(True)

def test_set_one_actu():
    sup.dms.set_one_actu(0, 0, ampli=1)
    assert(True)

def test_get_influ_function():
    sup.dms.get_influ_function(0)
    assert(True)

def test_get_influ_function_ipupil_coords():
    sup.dms.get_influ_function_ipupil_coords(0)
    assert(True)

def test_reset_dm():
    sup.dms.reset_dm()
    assert(True)

def test_get_dm_shape():
    sup.dms.get_dm_shape(0)
    assert(True)

def test_set_dm_registration():
    sup.dms.set_dm_registration(0, dx=0, dy=0, theta=0, G=1)
    assert(True)
#    _____                  _    ___
#   |_   _|_ _ _ _ __ _ ___| |_ / __|___ _ __  _ __  __ _ ______
#     | |/ _` | '_/ _` / -_)  _| (__/ _ \ '  \| '_ \/ _` (_-<_-<
#     |_|\__,_|_| \__, \___|\__|\___\___/_|_|_| .__/\__,_/__/__/
#                 |___/                       |_|
def test_get_tar_image():
    sup.target.get_tar_image(0)
    sup.target.get_tar_image(0, expo_type="le")
    assert(True)

def test_get_tar_phase():
    sup.target.get_tar_phase(0)
    sup.target.get_tar_phase(0, pupil=True)
    assert(True)

def test_set_tar_phase():
    sup.target.set_tar_phase(0, sup.target.get_tar_phase(0))
    assert(True)

def test_reset_strehl():
    sup.target.reset_strehl(0)
    assert(True)

def test_reset_tar_phase():
    sup.target.reset_tar_phase(0)
    assert(True)

def test_get_ncpa_tar():
    sup.target.get_ncpa_tar(0)
    assert(True)

def test_set_ncpa_tar():
    sup.target.set_ncpa_tar(0, sup.target.get_ncpa_tar(0))
    assert(True)

def test_comp_tar_image():
    sup.target.comp_tar_image(0)
    sup.target.comp_tar_image(0, puponly=1, compLE=False)
    assert(True)

def test_comp_strehl():
    sup.target.comp_strehl(0)
    sup.target.comp_strehl(0, do_fit=False)
    assert(True)

def test_get_strehl():
    sup.target.get_strehl(0)
    sup.target.get_strehl(0, do_fit=False)
    assert(True)
#   __      ____     ___
#   \ \    / / _|___/ __|___ _ __  _ __  __ _ ______
#    \ \/\/ /  _(_-< (__/ _ \ '  \| '_ \/ _` (_-<_-<
#     \_/\_/|_| /__/\___\___/_|_|_| .__/\__,_/__/__/
#                                 |_|
def test_get_wfs_image():
    sup.wfs.get_wfs_image(0)
    assert(True)

def test_set_noise():
    sup.wfs.set_noise(0,-1)
    sup.wfs.set_noise(0,1,seed=1235)
    assert(True)

def test_set_gs_mag():
    sup.wfs.set_gs_mag(0, 5)
    assert(True)

def test_compute_wfs_image():
    sup.wfs.compute_wfs_image(0)
    sup.wfs.compute_wfs_image(0, noise=False)
    assert(True)

def test_reset_noise():
    sup.wfs.reset_noise()
    assert(True)

def test_get_wfs_phase():
    sup.wfs.get_wfs_phase(0)
    assert(True)

def test_set_ncpa_wfs():
    sup.wfs.set_ncpa_wfs(0, sup.wfs.get_wfs_phase(0) * 0.)
    assert(True)

def test_get_ncpa_wfs():
    sup.wfs.get_ncpa_wfs(0)
    assert(True)

def test_set_wfs_phase():
    sup.wfs.set_wfs_phase(0, sup.wfs.get_wfs_phase(0))
    assert(True)

def test_set_wfs_pupil():
    sup.wfs.set_wfs_pupil(0, sup.config.p_geom._mpupil)
    assert(True)

@pytest.mark.skipif(sup.config.p_wfss[0].type != "pyrhr", reason="pyrhr only")
def test_set_pyr_modulation_points():
    sup.wfs.set_pyr_modulation_points(0, sup.config.p_wfss[0]._pyr_cx, sup.config.p_wfss[0]._pyr_cy)
    assert(True)

@pytest.mark.skipif(sup.config.p_wfss[0].type != "pyrhr", reason="pyrhr only")
def test_set_pyr_modulation_ampli():
    _ = sup.wfs.set_pyr_modulation_ampli(0,3)
    assert(True)

@pytest.mark.skip(reason="How does it work ?")
def test_set_pyr_multiple_stars_source():

    assert(True)

@pytest.mark.skipif(sup.config.p_wfss[0].type != "pyrhr", reason="pyrhr only")
def test_set_pyr_disk_source_hexa():
    sup.wfs.set_pyr_disk_source_hexa(0, 1)
    assert(True)

@pytest.mark.skipif(sup.config.p_wfss[0].type != "pyrhr", reason="pyrhr only")
def test_set_pyr_disk_source():
    sup.wfs.set_pyr_disk_source(0,3)
    assert(True)

@pytest.mark.skipif(sup.config.p_wfss[0].type != "pyrhr", reason="pyrhr only")
def test_set_pyr_square_source():
    sup.wfs.set_pyr_square_source(0, 3)
    assert(True)

@pytest.mark.skip(reason="How does it work ?")
def test_set_pyr_pseudo_source():
    assert(True)

@pytest.mark.skipif(sup.config.p_wfss[0].type != "pyrhr", reason="pyrhr only")
def test_set_fourier_mask():
    sup.wfs.set_fourier_mask(0, sup.config.p_wfss[0].get_halfxy())
    assert(True)

@pytest.mark.skipif(sup.config.p_wfss[0].type != "pyrhr", reason="pyrhr only")
def test_get_pyrhr_image():
    sup.wfs.get_pyrhr_image(0)
    assert(True)

@pytest.mark.skipif(sup.config.p_wfss[0].type != "pyrhr", reason="pyrhr only")
def test_get_pyr_focal_plane():
    sup.wfs.get_pyr_focal_plane(0)
    assert(True)
#    ___ _       ___
#   | _ \ |_ __ / __|___ _ __  _ __  __ _ ______
#   |   /  _/ _| (__/ _ \ '  \| '_ \/ _` (_-<_-<
#   |_|_\\__\__|\___\___/_|_|_| .__/\__,_/__/__/
#                             |_|
def test_set_perturbation_voltage():
    sup.rtc.set_perturbation_voltage(0,"test",np.zeros(sup.config.p_controllers[0].nactu))
    assert(True)

def test_get_perturbation_voltage():
    sup.rtc.get_perturbation_voltage(0)
    sup.rtc.get_perturbation_voltage(0, name="test")
    assert(True)

def test_get_slopes():
    sup.rtc.get_slopes(0)
    assert(True)

def test_close_loop():
    sup.rtc.close_loop()
    assert(True)

def test_open_loop():
    sup.rtc.open_loop()
    assert(True)

def test_set_ref_slopes():
    sup.rtc.set_ref_slopes(np.zeros(sup.config.p_controllers[0].nslope))
    assert(True)

def test_get_ref_slopes():
    sup.rtc.get_ref_slopes()
    assert(True)

def test_set_gain():
    sup.rtc.set_gain(0, 0.2)
    assert(True)

def test_get_interaction_matrix():
    sup.rtc.get_interaction_matrix(0)
    assert(True)

def test_get_command_matrix():
    sup.rtc.get_command_matrix(0)
    assert(True)

def test_set_command_matrix():
    sup.rtc.set_command_matrix(0, sup.rtc.get_command_matrix(0))
    assert(True)

@pytest.mark.skip(reason="not implemented")
def test_get_intensities():
    sup.rtc.get_intensities()
    assert(True)

def test_set_flat():
    sup.rtc.set_flat(0, sup.wfs.get_wfs_image(0) * 0 + 1.)
    assert(True)

def test_set_dark():
    sup.rtc.set_dark(0,sup.wfs.get_wfs_image(0) * 0)
    assert(True)

def test_compute_slopes():
    sup.rtc.compute_slopes(0)
    assert(True)

def test_reset_perturbation_voltage():
    sup.rtc.reset_perturbation_voltage(0)
    assert(True)

def test_remove_perturbation_voltage():
    sup.rtc.set_perturbation_voltage(0, "test", np.zeros(sup.config.p_controllers[0].nactu))
    assert(True)

@pytest.mark.skipif(sup.config.p_controllers[0].type != "ls", reason="LS only")
def test_get_err():
    sup.rtc.get_err(0)
    assert(True)

def test_get_voltages():
    sup.rtc.get_voltages(0)
    assert(True)

@pytest.mark.skipif(sup.config.p_controllers[0].type != "generic", reason="Generic only")
def test_set_integrator_law():
    sup.rtc.set_integrator_law(0)
    assert(True)

@pytest.mark.skipif(sup.config.p_controllers[0].type != "generic", reason="Generic only")
def test_set_2matrices_law():
    sup.rtc.set_2matrices_law(0)
    assert(True)

@pytest.mark.skipif(sup.config.p_controllers[0].type != "generic", reason="Generic only")
def test_set_modal_integrator_law():
    sup.rtc.set_modal_integrator_law(0)
    assert(True)

@pytest.mark.skipif(sup.config.p_controllers[0].type != "generic", reason="Generic only")
def test_set_decay_factor():
    sup.rtc.set_decay_factor(0, np.ones(sup.config.p_controllers[0].nactu))
    assert(True)

@pytest.mark.skipif(sup.config.p_controllers[0].type != "generic", reason="Generic only")
def test_set_E_matrix():
    sup.rtc.set_E_matrix(0, np.identity(sup.config.p_controllers[0].nactu))
    assert(True)

def test_reset_ref_slopes():
    sup.rtc.reset_ref_slopes(0)
    assert(True)

@pytest.mark.skipif(sup.config.p_centroiders[0].type != "tcog", reason="tcog only")
def test_set_centroider_threshold():
    sup.rtc.set_centroider_threshold(0, 10)
    assert(True)

@pytest.mark.skipif(sup.config.p_centroiders[0].type != "pyr", reason="pyr only")
def test_get_pyr_method() -> str:
    sup.rtc.get_pyr_method(0)
    assert(True)

@pytest.mark.skipif(sup.config.p_centroiders[0].type != "pyr", reason="pyr only")
def test_set_pyr_method():
    sup.rtc.set_pyr_method(0, 0)
    sup.rtc.set_pyr_method(0, 1)
    sup.rtc.set_pyr_method(0, 2)
    sup.rtc.set_pyr_method(0, 3)
    assert(True)

@pytest.mark.skipif(sup.config.p_controllers[0].type != "generic", reason="Generic only")
def test_set_modal_gains():
    sup.rtc.set_modal_gains(0, np.ones(sup.config.p_controllers[0].nactu))
    assert(True)

@pytest.mark.skipif(sup.config.p_controllers[0].type != "generic", reason="Generic only")
def test_get_modal_gains():
    sup.rtc.get_modal_gains
    assert(True)

@pytest.mark.skipif(sup.config.p_centroiders[0].type != "maskedpix", reason="maskedpix only")
def test_get_masked_pix():
    sup.rtc.get_masked_pix(0)
    assert(True)

def test_get_command():
    sup.rtc.get_command(0)
    assert(True)

def test_set_command():
    sup.rtc.set_command(0, np.zeros(sup.config.p_controllers[0].nactu))
    assert(True)

def test_reset_command():
    sup.rtc.reset_command()
    assert(True)

def test_get_slopes_geom():
    sup.rtc.get_slopes_geom(0)
    assert(True)

def test_get_image_raw():
    sup.rtc.get_image_raw(0)
    assert(True)

def test_get_image_calibrated():
    sup.rtc.get_image_calibrated(0)
    assert(True)

def test_get_image_cal():
    sup.rtc.get_image_raw(0)
    assert(True)

@pytest.mark.skipif(sup.config.p_centroiders[0].type != "maskedpix", reason="maskedpix only")
def test_get_selected_pix():
    sup.rtc.get_selected_pix()
    assert(True)

def test_do_ref_slopes():
    sup.rtc.do_ref_slopes(0)
    assert(True)

def test_do_control():
    sup.rtc.do_control(0)
    assert(True)

def test_do_calibrate_img():
    sup.rtc.do_calibrate_img(0)
    assert(True)

def test_do_centroids():
    sup.rtc.do_centroids(0)
    assert(True)

def test_do_centroids_geom():
    sup.rtc.do_centroids_geom(0)    # default 
    sup.rtc.do_centroids_geom(0, geom_type=0)  # type 0
    sup.rtc.do_centroids_geom(0, geom_type=1)  # type 1
    sup.rtc.do_centroids_geom(0, geom_type=2)  # type 2
    assert(True)

def test_apply_control():
    sup.rtc.apply_control(0)
    assert(True)

def test_do_clipping():
    sup.rtc.do_clipping(0)
    assert(True)

def test_set_scale():
    sup.rtc.set_scale(0, 1)
    assert(True)

@pytest.mark.skipif(sup.cacao is False, reason="cacao only")
def test_publish():
    sup.rtc.publish()
    assert(True)
#    __  __         _      _ ___          _
#   |  \/  |___  __| |__ _| | _ ) __ _ __(_)___
#   | |\/| / _ \/ _` / _` | | _ \/ _` (_-< (_-<
#   |_|  |_\___/\__,_\__,_|_|___/\__,_/__/_/__/
#
def test_compute_influ_basis():
    sup.basis.compute_influ_basis(0)
    assert(True)

def test_compute_influ_delta():
    sup.basis.compute_influ_delta(0)
    assert(True)

def test_compute_modes_to_volts_basis():
    _ = sup.basis.compute_modes_to_volts_basis("KL2V")
    _ = sup.basis.compute_modes_to_volts_basis("Btt")
    assert(True)

def test_compute_btt_basis():
    sup.basis.compute_btt_basis()
    assert(True)

@pytest.mark.skipif(sup.config.p_tel.type_ap == "EELT", reason="EELT pupil only")
def test_compute_merged_influ():
    sup.basis.compute_merged_influ(0)
    assert(True)

@pytest.mark.skipif("petal" not in [dm.influ_type for dm in sup.config.p_dms], reason="Petal dm only")
def test_compute_btt_petal():
    sup.basis.compute_btt_petal()
    assert(True)

def test_compute_phase_to_modes():
    sup.basis.compute_phase_to_modes(sup.basis.projection_matrix)
    assert(True)
#     ___      _ _ _             _   _
#    / __|__ _| (_) |__ _ _ __ _| |_(_)___ _ _
#   | (__/ _` | | | '_ \ '_/ _` |  _| / _ \ ' \
#    \___\__,_|_|_|_.__/_| \__,_|\__|_\___/_||_|
#
def test_apply_volts_and_get_slopes():
    sup.calibration.apply_volts_and_get_slopes(0)
    sup.calibration.apply_volts_and_get_slopes(0, noise=True, turbu=True, reset=False)
    assert(True)

def test_do_imat_modal():
    sup.calibration.do_imat_modal(0, np.ones(sup.basis.modal_basis.shape[0]), sup.basis.modal_basis)
    assert(True)

def test_do_imat_phase():
    sup.calibration.do_imat_phase(0, np.zeros((5, sup.wfs.get_ncpa_wfs(0).shape[0], sup.wfs.get_ncpa_wfs(0).shape[0])))
    assert(True)

def test_compute_modal_residuals():
    sup.calibration.compute_modal_residuals(sup.basis.projection_matrix)
    assert(True)

#
# ModalGains
#

def test_set_modal_basis():
    nactu = sup.config.p_controllers[0].nactu
    sup.modalgains.set_modal_basis(np.ones((nactu, nactu)))
    assert(True)

def test_get_modal_basis():
    sup.modalgains.get_modal_basis()
    assert(True)

def test_set_cmat_modal():
    nslope = sup.config.p_controllers[0].nslope
    nactu = sup.config.p_controllers[0].nactu
    sup.modalgains.set_cmat_modal(np.ones((nactu, nslope)))
    assert(True)

def test_get_modal_gains():
    sup.modalgains.get_modal_gains()
    assert(True)

def test_set_mask():
    sup.modalgains.set_mask(np.zeros(sup.config.p_controllers[0].nactu))
    assert(True)

def test_set_initial_gain():
    sup.modalgains.set_initial_gain(1)
    assert(True)

def test_set_config():
    sup.modalgains.set_config(0.1, 0.05, 0.05, 0.0)
    assert(True)

def test_adapt_modal_gains():
    sup.modalgains.adapt_modal_gains(False)
    assert(True)

def test_reset_mgains():
    sup.modalgains.reset_mgains()
    assert(True)

def test_reset_close():
    sup.modalgains.reset_close()
    assert(True)

def test_update_modal_meas():
    sup.modalgains.update_modal_meas()
    assert(True)

def test_update_mgains():
    sup.modalgains.update_mgains()
    assert(True)

# All of the "generic_linear" tests are run on the p_controller[1] controller:
@pytest.mark.skipif(sup.config.p_controllers[1].type != "generic_linear",
                        reason="Generic Linear only")
def test_get_x_buffer():
    sup.rtc._get_x_buffer(1)
    assert(True)

@pytest.mark.skipif(sup.config.p_controllers[1].type != "generic_linear",
                        reason="Generic Linear only")
def test_get_s_buffer():
    sup.rtc._get_s_buffer(1)
    assert(True)

@pytest.mark.skipif(sup.config.p_controllers[1].type != "generic_linear",
                        reason="Generic Linear only")
def test_get_u_in_buffer():
    sup.rtc._get_u_in_buffer(1)
    assert(True)

@pytest.mark.skipif(sup.config.p_controllers[1].type != "generic_linear",
                        reason="Generic Linear only")
def test_get_u_out_buffer():
    sup.rtc._get_u_out_buffer(1)
    assert(True)

@pytest.mark.skipif(sup.config.p_controllers[1].type != "generic_linear",
                        reason="Generic Linear only")
def test_get_A_matrix():
    sup.rtc._get_A_matrix(1,0)
    assert(True)

@pytest.mark.skipif(sup.config.p_controllers[1].type != "generic_linear",
                        reason="Generic Linear only")
def test_get_L_matrix():
    sup.rtc._get_L_matrix(1,0)
    assert(True)

@pytest.mark.skipif(sup.config.p_controllers[1].type != "generic_linear",
                        reason="Generic Linear only")
def test_get_K_matrix():
    sup.rtc._get_K_matrix(1)
    assert(True)

@pytest.mark.skipif(sup.config.p_controllers[1].type != "generic_linear",
                        reason="Generic Linear only")
def test_get_D_matrix():
    sup.rtc._get_D_matrix(1)
    assert(True)

@pytest.mark.skipif(sup.config.p_controllers[1].type != "generic_linear",
                        reason="Generic Linear only")
def test_get_F_matrix():
    sup.rtc._get_F_matrix(1)
    assert(True)

@pytest.mark.skipif(sup.config.p_controllers[1].type != "generic_linear",
                        reason="Generic Linear only")
def test_get_iir_a_vector():
    sup.rtc._get_iir_a_vector(1,0)
    assert(True)

@pytest.mark.skipif(sup.config.p_controllers[1].type != "generic_linear",
                        reason="Generic Linear only")
def test_get_iir_b_vector():
    sup.rtc._get_iir_b_vector(1,0)
    assert(True)

@pytest.mark.skipif(sup.config.p_controllers[1].type != "generic_linear",
                        reason="Generic Linear only")
def test_set_A_matrix():
    sup.rtc.set_A_matrix(1,0,sup.rtc._get_A_matrix(1,0))
    assert(True)

@pytest.mark.skipif(sup.config.p_controllers[1].type != "generic_linear",
                        reason="Generic Linear only")
def test_set_L_matrix():
    sup.rtc.set_L_matrix(1,0,sup.rtc._get_L_matrix(1,0))
    assert(True)

@pytest.mark.skipif(sup.config.p_controllers[1].type != "generic_linear",
                        reason="Generic Linear only")
def test_set_K_matrix():
    sup.rtc.set_K_matrix(1,sup.rtc._get_K_matrix(1))
    assert(True)

@pytest.mark.skipif(sup.config.p_controllers[1].type != "generic_linear",
                        reason="Generic Linear only")
def test_set_D_matrix():
    sup.rtc.set_D_matrix(1,sup.rtc._get_D_matrix(1))
    assert(True)

@pytest.mark.skipif(sup.config.p_controllers[1].type != "generic_linear",
                        reason="Generic Linear only")
def test_set_F_matrix():
    sup.rtc.set_F_matrix(1,sup.rtc._get_F_matrix(1))
    assert(True)

@pytest.mark.skipif(sup.config.p_controllers[1].type != "generic_linear",
                        reason="Generic Linear only")
def test_set_iir_a_vector():
    sup.rtc.set_iir_a_vector(1,0,sup.rtc._get_iir_a_vector(1,0))
    assert(True)

@pytest.mark.skipif(sup.config.p_controllers[1].type != "generic_linear",
                        reason="Generic Linear only")
def test_set_iir_b_vector():
    sup.rtc.set_iir_b_vector(1,0,sup.rtc._get_iir_b_vector(1,0))
    assert(True)