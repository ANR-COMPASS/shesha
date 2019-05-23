""" @package shesha.constants
Numerical constants for shesha
Config enumerations for safe-typing
"""

import numpy as np


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
    LS = 'ls'
    MV = 'mv'
    CURED = 'cured'
    GEO = 'geo'


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
    FILES = dict({
            GAUSS1: "allProfileNa_withAltitude_1Gaussian.npy",
            GAUSS2: "allProfileNa_withAltitude_2Gaussian.npy",
            GAUSS3: "allProfileNa_withAltitude_3Gaussian.npy",
            EXP: "allProfileNa_withAltitude.npy"
    })


class FieldStopType:
    """
        WFS field stop
    """
    SQUARE = 'square'
    ROUND = 'round'
