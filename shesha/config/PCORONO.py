## @package   shesha.config.PCORONO
## @brief     Param_corono class definition
## @author    COMPASS Team <https://github.com/ANR-COMPASS>
## @version   5.4.0
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

import numpy as np
from . import config_setter_utils as csu
import shesha.constants as scons


#################################################
# P-Class (parametres) Param_corono
#################################################
class Param_corono:

    def __init__(self):
        self.__type = None  # 'custom', 'SPHERE_APLC', or 'perfect'
        """ Type of coronograph """

        self.__asterix_parfile = None
        """ ASTERIX parameter file path """
        self.__asterix_datadir = None
        """ ASTERIX data directory path """

        self.__wavelength_0 = None
        """ Central wavelength in the coronagraph """
        self.__delta_wav = 0  # optional
        """ Spectral bandwidth """
        self.__nb_wav = 1  # optional
        """ Number of simulated wavelength in the spectral bandwidth """
        self.__wav_vec = None  # unsettable
        """ Array of simulated wavelengths """

        self.__apodizer = None
        """ Apodizer pupil """
        self.__apodizer_name = None  # optional : 'SPHERE_APLC_apodizer_APO1' or 'user_path'
        """ Apodizer string name or user path """

        self.__focal_plane_mask = None
        """ Focal plane mask complex amplitudes """
        self.__focal_plane_mask_name = None  # 'classical_Lyot', 'user_path' or 'SPHERE_APLC_fpm_ALC1' (or ALC2 or ALC3)
        """ Focal plane mask string name or user path """
        self.__fpm_sampling = None  # optional
        """ Size of lambda / D in the fpm plane, in pixel unit """
        self.__lyot_fpm_radius = None
        """ Focal plane mask radius in lamda / D unit, for a classical Lyot fpm only """
        self.__dim_fpm = None
        """ Size of the focal plane mask in pixel """
        self.__babinet_trick = False
        """ Flag for using Babinet's trick """

        self.__lyot_stop = None
        """ Lyot pupil """
        self.__lyot_stop_name = None  # optional : 'SPHERE_APLC_Lyot_stop' or 'user_path'
        """ Lyot stop string name or user path """

        self.__dim_image = None
        """ Size of the science image in pixel """
        self.__image_sampling = None
        """ Size of lambda / D in the image plane, in pixel unit """


    def get_type(self):
        """ Get the coronograph type

        :return: (str) : coronograph type
        """
        return self.__type

    def set_type(self, t):
        """ Set the coronograph type

        Args:
            t: (str) : coronograph type
        """
        self.__type = scons.check_enum(scons.CoronoType, t)

    _type = property(get_type, set_type)

    def get_asterix_parfile(self):
        """ Get the path of asterix parfile

        :return: (str) : asterix parfile path
        """
        return self.__asterix_parfile

    def set_asterix_parfile(self, f):
        """ set the path of asterix parfile

        :f: (str) : asterix parfile path
        """
        self.__asterix_parfile = f

    _asterix_parfile = property(get_asterix_parfile, set_asterix_parfile)

    def get_asterix_datadir(self):
        """ Get the path of asterix datadir

        :return: (str) : asterix datadir path
        """
        return self.__asterix_datadir

    def set_asterix_datadir(self, f):
        """ set the path of asterix datadir

        :f: (str) : asterix datadir path
        """
        self.__asterix_datadir = f

    _asterix_datadir = property(get_asterix_datadir, set_asterix_datadir)

    def get_wavelength_0(self):
        """ Get the central wavelength in the coronagraph

        :return: (float) : central wavelength
        """
        return self.__wavelength_0

    def set_wavelength_0(self, w):
        """ Set the central wavelength in the coronagraph
        
        :param w: central wavelength
        """
        self.__wavelength_0 = w

    _wavelength_0 = property(get_wavelength_0, set_wavelength_0)

    def get_delta_wav(self):
        """ Get the spectral bandwith

        :return: (float) : bandwidth
        """
        return self.__delta_wav

    def set_delta_wav(self, w):
        """ Set the spectral bandwidth
        
        :param w: (float) : spectral bandwidth
        """
        self.__delta_wav = w

    _delta_wav = property(get_delta_wav, set_delta_wav)

    def get_nb_wav(self):
        """ Get the number of simulated wavelength in the spectral bandwidth

        :return: (int) : number of wavelengths
        """
        return self.__nb_wav

    def set_nb_wav(self, n):
        """ Set the number of simulated wavelength in the spectral bandwidth
        
        :param n: (int) : number of wavelengths
        """
        self.__nb_wav = csu.enforce_int(n)

    _nb_wav = property(get_nb_wav, set_nb_wav)

    def get_wav_vec(self):
        """ Get the wavelengths array

        :return: (np.ndarray) : wavelengths array
        """
        return self.__wav_vec

    def set_wav_vec(self, w):
        """ Set the wavelengths array
        
        :param w: (np.ndarray) : wavelengths array
        """
        self.__wav_vec = w

    _wav_vec = property(get_wav_vec, set_wav_vec)

    def get_apodizer(self):
        """ Get the apodizer pupil

        :return: (np.ndarray) : apodizer
        """
        return self.__apodizer

    def set_apodizer(self, apod):
        """ Set the apodizer pupil
        
        :param apod: (np.ndarray) : apodizer
        """
        self.__apodizer = apod

    _apodizer = property(get_apodizer, set_apodizer)

    def get_apodizer_name(self):
        """ Get the apodizer keyword or user path

        :return: (str) : apodizer keyword or path
        """
        return self.__apodizer_name

    def set_apodizer_name(self, apod):
        """ Set the apodizer keyword or user path
        
        :param apod: (str) : apodizer keyword or path
        """
        self.__apodizer_name = apod

    _apodizer_name = property(get_apodizer_name, set_apodizer_name)

    def get_focal_plane_mask(self):
        """ Get the focal plane mask complex amplitudes

        :return: (list of np.ndarray) : focal plane mask
        """
        return self.__focal_plane_mask

    def set_focal_plane_mask(self, fpm):
        """ Set the focal plane complex amplitudes
        
        :param fpm: (list of np.ndarray) : focal plane mask
        """
        self.__focal_plane_mask = fpm

    _focal_plane_mask = property(get_focal_plane_mask, set_focal_plane_mask)

    def get_focal_plane_mask_name(self):
        """ Get the focal plane mask keyword or user path

        :return: (str) : focal plane mask keyword or path
        """
        return self.__focal_plane_mask_name

    def set_focal_plane_mask_name(self, fpm):
        """ Set the focal plane mask keyword or user path
        
        :param fpm: (str) : focal plane mask keyword or path
        """
        self.__focal_plane_mask_name = fpm

    _focal_plane_mask_name = property(get_focal_plane_mask_name, set_focal_plane_mask_name)

    def get_fpm_sampling(self):
        """ Get the sampling in the focal plane mask
        sampling = size of lambda / D in pixel units

        :return: (float) : focal plane mask sampling
        """
        return self.__fpm_sampling

    def set_fpm_sampling(self, sp):
        """ Set the sampling in the focal plane mask
        sampling = size of lambda / D in pixel units
        
        :param sp: (float) : focal plane mask sampling
        """
        self.__fpm_sampling = sp

    _fpm_sampling = property(get_fpm_sampling, set_fpm_sampling)

    def get_lyot_fpm_radius(self):
        """ Get the radius of the classical Lyot focal plane mask
        in lambda / D units

        :return: (float) : classical Lyot fpm radius
        """
        return self.__lyot_fpm_radius

    def set_lyot_fpm_radius(self, r):
        """ Set the radius of the classical Lyot focal plane mask
        in lambda / D units

        :param r: (float) : classical Lyot fpm radius
        """
        self.__lyot_fpm_radius = r

    _lyot_fpm_radius = property(get_lyot_fpm_radius, set_lyot_fpm_radius)

    def get_dim_fpm(self):
        """ Get the size of the focal plane mask support in pixel units
        
        :return: (int) : fpm support size in pixel
        """
        return self.__dim_fpm

    def set_dim_fpm(self, n):
        """ Set the size of the focal plane mask support in pixel units
        
        :param n: (int) : fpm support size in pixel
        """
        self.__dim_fpm = csu.enforce_int(n)

    _dim_fpm = property(get_dim_fpm, set_dim_fpm)

    def get_babinet_trick(self):
        """ Get the Babinet's trick flag
        
        :return: (bool) : Babinet's trick flag
        """
        return self.__babinet_trick

    def set_babinet_trick(self, b):
        """ Set the Babinet's trick flag
        
        :param b: (bool) : Babinet's trick flag
        """
        self.__babinet_trick = csu.enforce_or_cast_bool(b)

    _babinet_trick = property(get_babinet_trick, set_babinet_trick)

    def get_lyot_stop(self):
        """ Get the Lyot stop pupil

        :return: (np.ndarray) : Lyot stop pupil
        """
        return self.__lyot_stop

    def set_lyot_stop(self, ls):
        """ Set the Lyot stop pupil
        
        :param ls: (np.ndarray) : Lyot stop pupil
        """
        self.__lyot_stop = ls

    _lyot_stop = property(get_lyot_stop, set_lyot_stop)

    def get_lyot_stop_name(self):
        """ Get the Lyot stop keyword or user path

        :return: (str) : Lyot stop keyword or path
        """
        return self.__lyot_stop_name

    def set_lyot_stop_name(self, ls):
        """ Set the Lyot stop keyword or user path
        
        :param ls: (str) : Lyot stop keyword or path
        """
        self.__lyot_stop_name = ls

    _lyot_stop_name = property(get_lyot_stop_name, set_lyot_stop_name)

    def get_dim_image(self):
        """ Get the size of the science image in pixel

        :return: (int) : image size in pixel
        """
        return self.__dim_image

    def set_dim_image(self, n):
        """ Set the size of the science image in pixel
        
        :param n: (int) : image size in pixel
        """
        self.__dim_image = csu.enforce_int(n)

    _dim_image = property(get_dim_image, set_dim_image)

    def get_image_sampling(self):
        """ Get the sampling in the image
        sampling = size of lambda / D in pixel units

        :return: (float) : image sampling
        """
        return self.__image_sampling

    def set_image_sampling(self, sp):
        """ Set the sampling in the image
        sampling = size of lambda / D in pixel units
        
        :param sp: (float) : image sampling
        """
        self.__image_sampling = sp

    _image_sampling = property(get_image_sampling, set_image_sampling)
