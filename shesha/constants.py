## @package   shesha.constants
## @brief     Numerical constants for shesha and config enumerations for safe-typing
## @author    COMPASS Team <https://github.com/ANR-COMPASS>
## @version   5.4.1
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
from aenum import MultiValueEnum


class CONST:
    RAD2ARCSEC = 3600. * 360. / (2 * np.pi)
    ARCSEC2RAD = 2. * np.pi / (360. * 3600.)
    RAD2DEG = 180. / np.pi
    DEG2RAD = np.pi / 180.


def check_enum(cls, name):
    """
        Create a safe-type enum instance from bytes contents
    """

    if not isinstance(name, str) or \
            not name in vars(cls).values():
        raise ValueError("Invalid enumeration value for enum %s, value %s" % (cls, name))
    return name


class DmType:
    """
        Types of deformable mirrors
    """

    PZT = 'pzt'
    TT = 'tt'
    KL = 'kl'


class PatternType:
    """
        Types of Piezo DM patterns
    """

    SQUARE = 'square'
    HEXA = 'hexa'
    HEXAM4 = 'hexaM4'


class KLType:
    """
        Possible KLs for computations
    """

    KOLMO = 'kolmo'
    KARMAN = 'karman'


class InfluType:
    """
        Influence function types
    """

    DEFAULT = 'default'
    RADIALSCHWARTZ = 'radialSchwartz'
    SQUARESCHWARTZ = 'squareSchwartz'
    BLACKNUTT = 'blacknutt'
    GAUSSIAN = 'gaussian'
    BESSEL = 'bessel'
    PETAL = 'petal'


class ControllerType:
    """
        Controller types
    """

    GENERIC = 'generic'
    GENERIC_LINEAR = 'generic_linear'
    LS = 'ls'
    MV = 'mv'
    CURED = 'cured'
    GEO = 'geo'


class CommandLawType:
    """
        Command law types for generic controller only
    """

    INTEGRATOR = 'integrator'
    MODAL_INTEGRATOR = 'modal_integrator'
    TWO_MATRICES = '2matrices'


class CentroiderType:
    """
        Centroider types
    """

    COG = 'cog'
    TCOG = 'tcog'
    WCOG = 'wcog'
    BPCOG = 'bpcog'
    CORR = 'corr'
    PYR = 'pyr'
    MASKEDPIX = 'maskedpix'


class CentroiderFctType:
    MODEL = 'model'
    GAUSS = 'gauss'


class PyrCentroiderMethod:
    """
        Pyramid centroider methods
        Local flux normalization (eq SH quad-cell, ray optics. Ragazzonni 1996)
        Global flux normalization (Verinaud 2004, most > 2010 Pyr applications)
        Resulting (A+/-B-/+C-D)/(A+B+C+D) or sin((A+/-B-/+C-D)/(A+B+C+D))
        ref. code sutra_centroider_pyr.h
    """
    NOSINUSGLOBAL = 0
    SINUSGLOBAL = 1
    NOSINUSLOCAL = 2
    SINUSLOCAL = 3
    OTHER = 4


class WFSType:
    """
        WFS Types
    """
    SH = 'sh'
    PYRHR = 'pyrhr'
    PYRLR = 'pyrlr'


class TargetImageType:
    """
        Target Images
    """

    SE = 'se'
    LE = 'le'


class ApertureType:
    """
        Telescope apertures
    """
    GENERIC = 'Generic'
    EELT_NOMINAL = 'EELT-Nominal'  # Alexis Carlotti method
    EELT = 'EELT'  # E. Gendron method
    EELT_BP1 = 'EELT-BP1'
    EELT_BP3 = 'EELT-BP3'
    EELT_BP5 = 'EELT-BP5'
    EELT_CUSTOM = 'EELT-Custom'
    VLT = 'VLT'
    KECK = 'keck'
    VLT_NOOBS = 'VLT-NoObs'

class SpiderType:
    """
        Spiders
    """
    FOUR = 'four'
    SIX = 'six'


class ProfType:
    """
        Sodium profile for LGS
    """
    GAUSS1 = 'Gauss1'
    GAUSS2 = 'Gauss2'
    GAUSS3 = 'Gauss3'
    EXP = 'Exp'
    MULTIPEAK = 'Multipeak'
    FILES = dict({
            GAUSS1: "allProfileNa_withAltitude_1Gaussian.npy",
            GAUSS2: "allProfileNa_withAltitude_2Gaussian.npy",
            GAUSS3: "allProfileNa_withAltitude_3Gaussian.npy",
            EXP: "allProfileNa_withAltitude.npy",
            MULTIPEAK: "multipeakProfileNa_withAltitude.npy"
    })


class FieldStopType:
    """
        WFS field stop
    """
    SQUARE = 'square'
    ROUND = 'round'
    NONE = 'none'

class PupilType(MultiValueEnum):
    """Compass pupil enumeration
    """
    SPUPIL = "spupil", "s"
    MPUPIL = "mpupil", "m"
    IPUPIL = "ipupil", "i"

class CoronoType:
    """ Coronograph types
    """
    MODULE = "module"
    PERFECT = "perfect"
    CUSTOM = "custom"
    SPHERE_APLC = "SPHERE_APLC"

class ApodizerType:
    """ Apodizer types
    """
    SPHERE_APLC_APO1 = "SPHERE_APLC_apodizer_APO1"

class FpmType:
    """ Focal plane mask types
    """
    CLASSICAL_LYOT = "classical_Lyot"
    SPHERE_APLC_fpm_ALC1 = "SPHERE_APLC_fpm_ALC1"
    SPHERE_APLC_fpm_ALC2 = "SPHERE_APLC_fpm_ALC2"
    SPHERE_APLC_fpm_ALC3 = "SPHERE_APLC_fpm_ALC3"

class LyotStopType:
    """ Lyot stop types
    """
    SPHERE_APLC_LYOT_STOP = "SPHERE_APLC_Lyot_stop"

class MftType:
    """ MFT types
    """
    IMG = "img"
    PSF = "psf"
    FPM = "fpm"
    LYOT= "lyot"
class ExposureType:
    """ Exposure type
    """
    LE = "le"
    SE = "se"