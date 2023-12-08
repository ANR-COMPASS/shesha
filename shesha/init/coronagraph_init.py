## @package   shesha.init.coronagraph_init
## @brief     Initialization of a Coronagraph object
## @author    COMPASS Team <https://github.com/ANR-COMPASS>
## @version   5.5.0
## @date      2023/03/02
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
import os
import numpy as np
import astropy.io.fits as pfits
import shesha.config as conf
import shesha.constants as scons
import shesha.util.coronagraph_utils as util

# ---------------------------------- #
#      Initialization functions      #
# ---------------------------------- #


# TODO : add some checks (dimension, type, etc...)

def init_coronagraph(p_corono: conf.Param_corono, pupdiam):
    """ Initialize the coronagraph
    """
    wavelength_0 = p_corono._wavelength_0
    delta_wav = p_corono._delta_wav
    nb_wav = p_corono._nb_wav

    # wavelength array init
    if (delta_wav == 0):
        p_corono.set_nb_wav(1)
        p_corono.set_wav_vec(np.array([wavelength_0]))
    elif (nb_wav == 1):
        p_corono.set_delta_wav(0.)
        p_corono.set_wav_vec(np.array([wavelength_0]))
    else:
        wav_vec = np.linspace(wavelength_0 - delta_wav / 2,
                              wavelength_0 + delta_wav / 2,
                              num=nb_wav,
                              endpoint=True)
        p_corono.set_wav_vec(wav_vec)

    # pupils and mask init
    if p_corono._type == scons.CoronoType.SPHERE_APLC:
        init_sphere_aplc(p_corono, pupdiam)
    elif p_corono._type == scons.CoronoType.PERFECT:
        init_perfect_coronagraph(p_corono, pupdiam)
    else:
        init_apodizer(p_corono, pupdiam)
        init_focal_plane_mask(p_corono)
        init_lyot_stop(p_corono, pupdiam)

def init_sphere_aplc(p_corono: conf.Param_corono, pupdiam):
    """ Dedicated function for SPHERE APLC coronagraph init

    References:
        APLC data : SPHERE user manual, appendix A6 'NIR coronagraphs'
            https://www.eso.org/sci/facilities/paranal/instruments/sphere/doc.html
        IRDIS data : ESO instrument description
            https://www.eso.org/sci/facilities/paranal/instruments/sphere/inst.html
    """
    # apodizer init
    p_corono.set_apodizer_name(scons.ApodizerType.SPHERE_APLC_APO1)
    init_apodizer(p_corono, pupdiam)

    # fpm init
    if p_corono._focal_plane_mask_name == None:
        p_corono.set_focal_plane_mask_name(scons.FpmType.SPHERE_APLC_fpm_ALC2)
    init_focal_plane_mask(p_corono)

    # lyot stop init
    p_corono.set_lyot_stop_name(scons.LyotStopType.SPHERE_APLC_LYOT_STOP)
    init_lyot_stop(p_corono, pupdiam)

    # image init
    if p_corono._dim_image == None:
        p_corono.set_dim_image(256)
    if p_corono._image_sampling == None:
        irdis_plate_scale = 12.25  # [mas]
        VLT_pupil_diameter = 8  # [m]
        lambda_over_D = p_corono._wavelength_0 * 1e-6 / VLT_pupil_diameter  # [rad]
        image_sampling = (lambda_over_D * 180 / np.pi * 3600 * 1000) / irdis_plate_scale
        p_corono.set_image_sampling(image_sampling)

def init_perfect_coronagraph(p_corono: conf.Param_corono, pupdiam):
    """ Dedicated function for perfect coronagraph init
    """
    pass

def init_apodizer(p_corono: conf.Param_corono, pupdiam):
    """ Apodizer init
    """
    if p_corono._apodizer_name == scons.ApodizerType.SPHERE_APLC_APO1:
        apodizer = util.make_sphere_apodizer(pupdiam)
    elif p_corono._apodizer_name == None:
        apodizer = np.ones((pupdiam, pupdiam))
    elif isinstance(p_corono._apodizer_name, str):
        if not os.path.exists(p_corono._apodizer_name):
            error_string = "apodizer keyword (or path) '{}'".format(p_corono._apodizer_name) \
                         + " is not a known keyword (or path)"
            raise ValueError(error_string)
        apodizer = pfits.getdata(p_corono._apodizer_name)
    else:
        raise TypeError('apodizer name should be a string')
    p_corono.set_apodizer(apodizer)

def init_focal_plane_mask(p_corono: conf.Param_corono):
    """ Focal plane mask init
    """
    if p_corono._focal_plane_mask_name == scons.FpmType.CLASSICAL_LYOT:
        classical_lyot = True
    elif p_corono._focal_plane_mask_name in (scons.FpmType.SPHERE_APLC_fpm_ALC1,
                                             scons.FpmType.SPHERE_APLC_fpm_ALC2,
                                             scons.FpmType.SPHERE_APLC_fpm_ALC3):
        classical_lyot = True
        if (p_corono._focal_plane_mask_name == scons.FpmType.SPHERE_APLC_fpm_ALC1):
            fpm_radius_in_mas = 145 / 2  # [mas]
        elif (p_corono._focal_plane_mask_name == scons.FpmType.SPHERE_APLC_fpm_ALC2):
            fpm_radius_in_mas = 185 / 2  # [mas]
        elif (p_corono._focal_plane_mask_name == scons.FpmType.SPHERE_APLC_fpm_ALC3):
            fpm_radius_in_mas = 240 / 2  # [mas]
        VLT_pupil_diameter = 8  # [m]
        lambda_over_D = p_corono._wavelength_0 * 1e-6 / VLT_pupil_diameter  # [rad]
        fpm_radius = fpm_radius_in_mas / (lambda_over_D * 180 / np.pi * 3600 * 1000)  # [lambda/D]
        p_corono.set_lyot_fpm_radius(fpm_radius)
    else:
        classical_lyot = False

    if classical_lyot:
        p_corono.set_babinet_trick(True)
        if p_corono._fpm_sampling == None:
            p_corono.set_fpm_sampling(20.)
        lyot_fpm_radius_in_pix = p_corono._fpm_sampling * p_corono._lyot_fpm_radius
        dim_fpm = 2 * int(lyot_fpm_radius_in_pix) + 2
        p_corono.set_dim_fpm(dim_fpm)
        fpm = util.classical_lyot_fpm(p_corono._lyot_fpm_radius,
                                      p_corono._dim_fpm,
                                      p_corono._fpm_sampling,
                                      p_corono._wav_vec)

    elif isinstance(p_corono._focal_plane_mask_name, str):
        if not os.path.exists(p_corono._focal_plane_mask_name):
            error_string = "focal plane mask keyword (or path) '{}'".format(p_corono._focal_plane_mask_name) \
                         + " is not a known keyword (or path)"
            raise ValueError(error_string)
        fpm_array = pfits.getdata(p_corono._focal_plane_mask_name)
        p_corono.set_dim_fpm(fpm_array.shape[0])
        print(p_corono._dim_fpm)
        if p_corono._fpm_sampling == None:
            p_corono.set_fpm_sampling(p_corono._image_sampling)
        if len(fpm_array.shape) == 2:
            fpm = [fpm_array] * p_corono._nb_wav
        elif len(fpm_array.shape) == 3:
            fpm = []
            for i in range(p_corono._nb_wav):
                fpm.append(fpm_array[:, :, i])
    p_corono.set_focal_plane_mask(fpm)

def init_lyot_stop(p_corono: conf.Param_corono, pupdiam):
    """ Lyot stop init
    """
    if p_corono._lyot_stop_name == scons.LyotStopType.SPHERE_APLC_LYOT_STOP:
        lyot_stop = util.make_sphere_lyot_stop(pupdiam)
    elif p_corono._lyot_stop_name == None:
        lyot_stop = np.ones((pupdiam, pupdiam))
    elif isinstance(p_corono._lyot_stop_name, str):
        if not os.path.exists(p_corono._lyot_stop_name):
            error_string = "Lyot stop keyword (or path) '{}'".format(p_corono._lyot_stop_name) \
                         + " is not a known keyword (or path)"
            raise ValueError(error_string)
        lyot_stop = pfits.getdata(p_corono._lyot_stop_name)
    else:
        raise TypeError('Lyot stop name should be a string')
    p_corono.set_lyot_stop(lyot_stop)

def init_mft(p_corono: conf.Param_corono, pupdiam, planes, center_on_pixel=False):
    """ Initialize mft matrices
    """
    dim_fpm = p_corono._dim_fpm
    fpm_sampling = p_corono._fpm_sampling
    dim_image = p_corono._dim_image
    image_sampling = p_corono._image_sampling
    wavelength_0 = p_corono._wavelength_0
    wav_vec = p_corono._wav_vec

    norm0 = np.zeros(len(wav_vec))

    if planes == 'apod_to_fpm':
        AA_apod_to_fpm = np.zeros((dim_fpm, pupdiam, len(wav_vec)), dtype=np.complex64)
        BB_apod_to_fpm = np.zeros((pupdiam, dim_fpm, len(wav_vec)), dtype=np.complex64)

        for w, wavelength in enumerate(wav_vec):
            wav_ratio = wavelength / wavelength_0
            nbres = dim_fpm / fpm_sampling / wav_ratio
            AA_apod_to_fpm[:,:,w], BB_apod_to_fpm[:,:,w], norm0[w] = mft_matrices(pupdiam,
                                                                                  dim_fpm, nbres)

        return AA_apod_to_fpm, BB_apod_to_fpm, norm0

    elif planes == 'fpm_to_lyot':
        AA_fpm_to_lyot = np.zeros((pupdiam, dim_fpm, len(wav_vec)), dtype=np.complex64)
        BB_fpm_to_lyot = np.zeros((dim_fpm, pupdiam, len(wav_vec)), dtype=np.complex64)

        for w, wavelength in enumerate(wav_vec):
            wav_ratio = wavelength / wavelength_0
            nbres = dim_fpm / fpm_sampling / wav_ratio

            AA_fpm_to_lyot[:,:,w], BB_fpm_to_lyot[:,:,w], norm0[w] = mft_matrices(dim_fpm,
                                                                                  pupdiam, nbres,inverse=True)

        return AA_fpm_to_lyot, BB_fpm_to_lyot, norm0

    elif planes == 'lyot_to_image':
        AA_lyot_to_image = np.zeros((dim_image, pupdiam, len(wav_vec)), dtype=np.complex64)
        BB_lyot_to_image = np.zeros((pupdiam, dim_image, len(wav_vec)), dtype=np.complex64)

        for w, wavelength in enumerate(wav_vec):
            wav_ratio = wavelength / wavelength_0
            nbres = dim_image / image_sampling / wav_ratio
            if center_on_pixel:
                AA_lyot_to_image[:,:,w], BB_lyot_to_image[:,:,w], norm0[w] = mft_matrices(pupdiam,
                                                                                dim_image,
                                                                                nbres,
                                                                                X_offset_output=0.5,
                                                                                Y_offset_output=0.5)
            else:
                AA_lyot_to_image[:,:,w], BB_lyot_to_image[:,:,w], norm0[w] = mft_matrices(pupdiam,
                                                                                        dim_image,
                                                                                        nbres)

        return AA_lyot_to_image, BB_lyot_to_image, norm0


# ---------------------------------------- #
#      Matrix Fourier Transform (MFT)      #
#      Generic functions from ASTERIX      #
# ---------------------------------------- #

def mft_matrices(dim_input,
        dim_output,
        nbres,
        real_dim_input=None,
        inverse=False,
        norm='ortho',
        X_offset_input=0,
        Y_offset_input=0,
        X_offset_output=0,
        Y_offset_output=0):
    """
    docstring
    """
    # check dimensions and type of dim_input
    error_string_dim_input = "'dim_input' must be an int (square input) or tuple of ints of dimension 2"
    if np.isscalar(dim_input):
        if isinstance(dim_input, int):
            dim_input_x = dim_input
            dim_input_y = dim_input
        else:
            raise TypeError(dim_input)
    elif isinstance(dim_input, tuple):
        if all(isinstance(dims, int) for dims in dim_input) & (len(dim_input) == 2):
            dim_input_x = dim_input[0]
            dim_input_y = dim_input[1]
        else:
            raise TypeError(error_string_dim_input)
    else:
        raise TypeError(error_string_dim_input)

    # check dimensions and type of real_dim_input
    if real_dim_input == None:
        real_dim_input = dim_input
    error_string_real_dim_input = "'real_dim_input' must be an int (square input pupil) or tuple of ints of dimension 2"
    if np.isscalar(real_dim_input):
        if isinstance(real_dim_input, int):
            real_dim_input_x = real_dim_input
            real_dim_input_y = real_dim_input
        else:
            raise TypeError(error_string_real_dim_input)
    elif isinstance(real_dim_input, tuple):
        if all(isinstance(dims, int) for dims in real_dim_input) & (len(real_dim_input) == 2):
            real_dim_input_x = real_dim_input[0]
            real_dim_input_y = real_dim_input[1]
        else:
            raise TypeError(error_string_real_dim_input)
    else:
        raise TypeError(error_string_real_dim_input)

    # check dimensions and type of dim_output
    error_string_dim_output = "'dim_output' must be an int (square output) or tuple of ints of dimension 2"
    if np.isscalar(dim_output):
        if isinstance(dim_output, int):
            dim_output_x = dim_output
            dim_output_y = dim_output
        else:
            raise TypeError(error_string_dim_output)
    elif isinstance(dim_output, tuple):
        if all(isinstance(dims, int) for dims in dim_output) & (len(dim_output) == 2):
            dim_output_x = dim_output[0]
            dim_output_y = dim_output[1]
        else:
            raise TypeError(error_string_dim_output)
    else:
        raise TypeError(error_string_dim_output)

    # check dimensions and type of nbres
    error_string_nbr = "'nbres' must be an float or int (square output) or tuple of float or int of dimension 2"
    if np.isscalar(nbres):
        if isinstance(nbres, (float, int)):
            nbresx = float(nbres)
            nbresy = float(nbres)
        else:
            raise TypeError(error_string_nbr)
    elif isinstance(nbres, tuple):
        if all(isinstance(nbresi, (float, int)) for nbresi in nbres) & (len(nbres) == 2):
            nbresx = float(nbres[0])
            nbresy = float(nbres[1])
        else:
            raise TypeError(error_string_nbr)
    else:
        raise TypeError(error_string_nbr)

    if real_dim_input is not None:
        nbresx = nbresx * dim_input_x / real_dim_input_x
        nbresy = nbresy * dim_input_y / real_dim_input_y

    X0 = dim_input_x / 2 + X_offset_input
    Y0 = dim_input_y / 2 + Y_offset_input

    X1 = dim_output_x / 2 + X_offset_output
    Y1 = dim_output_y / 2 + Y_offset_output

    xx0 = ((np.arange(dim_input_x) - X0 + 1 / 2) / dim_input_x)  # Entrance image
    xx1 = ((np.arange(dim_input_y) - Y0 + 1 / 2) / dim_input_y)  # Entrance image
    uu0 = ((np.arange(dim_output_x) - X1 + 1 / 2) / dim_output_x) * nbresx  # Fourier plane
    uu1 = ((np.arange(dim_output_y) - Y1 + 1 / 2) / dim_output_y) * nbresy  # Fourier plane

    if not inverse:
        if norm == 'backward':
            norm0 = 1.
        elif norm == 'forward':
            norm0 = nbresx * nbresy / dim_input_x / dim_input_y / dim_output_x / dim_output_y
        elif norm == 'ortho':
            norm0 = np.sqrt(nbresx * nbresy / dim_input_x / dim_input_y / dim_output_x / dim_output_y)
        sign_exponential = -1

    else:
        if norm == 'backward':
            norm0 = nbresx * nbresy / dim_input_x / dim_input_y / dim_output_x / dim_output_y
        elif norm == 'forward':
            norm0 = 1.
        elif norm == 'ortho':
            norm0 = np.sqrt(nbresx * nbresy / dim_input_x / dim_input_y / dim_output_x / dim_output_y)
        sign_exponential = 1

    AA = np.exp(sign_exponential * 1j * 2 * np.pi * np.outer(uu0, xx0)).astype('complex64')
    BB = np.exp(sign_exponential * 1j * 2 * np.pi * np.outer(xx1, uu1)).astype('complex64')

    return AA, BB, norm0

def mft_multiplication(image, AA, BB, norm):
    """ Computes matrix multiplication for MFT
    """
    return norm * ((AA @ image.astype('complex64')) @ BB)
