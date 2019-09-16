## @package   shesha.util.make_pupil
## @brief     Pupil creation functions
## @author    COMPASS Team <https://github.com/ANR-COMPASS>
## @version   4.3.0
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
import os
import scipy.ndimage.interpolation as interp

from . import hdf5_util as h5u
from . import utilities as util

from shesha.constants import ApertureType, SpiderType

EELT_data = os.environ.get('SHESHA_ROOT') + "/data/apertures/"


def make_pupil(dim, pupd, tel, xc=-1, yc=-1, real=0, halfSpider=False):
    """Initialize the system pupil

    :parameters:

        dim: (long) : = p_geom.pupdiam

        pupd: (long) : linear size of total pupil = p_geom.pupdiam

        tel: (Param_tel) : Telescope structure

        xc: (int) = p_geom.pupdiam / 2. - 0.5

        yc: (int) = p_geom.pupdiam / 2. - 0.5

        real: (int)

    TODO: complete
    """

    if tel.type_ap == ApertureType.EELT_NOMINAL:
        N_seg = 798
        return make_EELT(dim, pupd, tel, N_seg)
    elif (tel.type_ap == ApertureType.EELT):
        return generateEeltPupilMask(dim, tel.t_spiders, xc, yc, tel.diam / dim, tel.gap,
                                     tel.pupangle, D=tel.diam, halfSpider=halfSpider,
                                     pitch=1.244683637214, nseg=33, inner_rad=4.1,
                                     outer_rad=15.4, R=95.7853, nominalD=40,
                                     half_seg=0.75, refl=tel.referr)
    elif (tel.type_ap == ApertureType.KECK):
        seg_corner = 1.8
        kpitch = seg_corner / 2 * np.sqrt(3)
        knseg = 7
        kinner_rad = 0.9
        kouter_rad = 3.4
        kR = 85
        knominalD = 10.96
        khalf_seg = 0.9
        return generateEeltPupilMask(dim, tel.t_spiders, xc, yc, tel.diam / dim, tel.gap,
                                     tel.pupangle, D=tel.diam, cobs=tel.cobs,halfSpider=halfSpider,
                                     pitch=kpitch, nseg=knseg, inner_rad=0.9,
                                     outer_rad=3.4, R=kR, nominalD=knominalD,
                                     half_seg=0.9, refl=tel.referr, rotSpiderDegree=-30)
    elif tel.type_ap == ApertureType.EELT_BP1:
        print("ELT_pup_cobs = %5.3f" % 0.339)
        N_seg = 768
        return make_EELT(dim, pupd, tel, N_seg)
    elif tel.type_ap == ApertureType.EELT_BP3:
        print("ELT_pup_cobs = %5.3f" % 0.503)
        N_seg = 672
        return make_EELT(dim, pupd, tel, N_seg)
    elif tel.type_ap == ApertureType.EELT_BP5:
        print("ELT_pup_cobs = %5.3f" % 0.632)
        N_seg = 558
        return make_EELT(dim, pupd, tel, N_seg)
    elif tel.type_ap == ApertureType.EELT_CUSTOM:
        print("EELT_CUSTOM not implemented. Falling back to EELT-Nominal")
        tel.set_type_ap(ApertureType.EELT_NOMINAL)
        return make_EELT(dim, pupd, tel, N_seg)
    elif tel.type_ap == ApertureType.VLT:
        tel.set_cobs(0.14)
        print("force_VLT_pup_cobs = %5.3f" % 0.14)
        return make_VLT(dim, pupd, tel)
    elif tel.type_ap == ApertureType.GENERIC:
        return make_pupil_generic(dim, pupd, tel.t_spiders, tel.spiders_type, xc, yc,
                                  real, tel.cobs)
    else:
        raise NotImplementedError("Missing Pupil type.")


def make_pupil_generic(dim, pupd, t_spiders=0.01, spiders_type=SpiderType.SIX, xc=0,
                       yc=0, real=0, cobs=0):
    """
        Initialize the system pupil

    :parameters:

        dim: (long) : linear size of ???

        pupd: (long) : linear size of total pupil

        t_spiders: (float) : secondary supports ratio.

        spiders_type: (str) :  secondary supports type: "four" or "six".

        xc: (int)

        yc: (int)

        real: (int)

        cobs: (float) : central obstruction ratio.

    TODO: complete
    """

    pup = util.dist(dim, xc, yc)

    if (real == 1):
        pup = np.exp(-(pup / (pupd * 0.5))**60.0)**0.69314
    else:
        pup = (pup < (pupd + 1.) / 2).astype(np.float32)

    if (cobs > 0):
        if (real == 1):
            pup -= np.exp(-(util.dist(dim, xc, yc) / (pupd * cobs * 0.5))**60.)**0.69314
        else:
            pup -= (util.dist(dim, xc, yc) < (pupd * cobs + 1.) * 0.5).astype(np.float32)

            step = 1. / dim
            first = 0.5 * (step - 1)

            X = np.tile(np.arange(dim, dtype=np.float32) * step + first, (dim, 1))

            if (t_spiders < 0):
                t_spiders = 0.01
            t_spiders = t_spiders * pupd / dim

            if (spiders_type == "four"):

                s4_2 = 2 * np.sin(np.pi / 4)
                t4 = np.tan(np.pi / 4)

                spiders_map = ((X.T > (X + t_spiders / s4_2) * t4) +
                               (X.T < (X - t_spiders / s4_2) * t4)).astype(np.float32)
                spiders_map *= ((X.T > (-X + t_spiders / s4_2) * t4) +
                                (X.T < (-X - t_spiders / s4_2) * t4)).astype(np.float32)

                pup = pup * spiders_map

            elif (spiders_type == "six"):

                #angle = np.pi/(180/15.)
                angle = 0
                s2ma_2 = 2 * np.sin(np.pi / 2 - angle)
                s6pa_2 = 2 * np.sin(np.pi / 6 + angle)
                s6ma_2 = 2 * np.sin(np.pi / 6 - angle)
                t2ma = np.tan(np.pi / 2 - angle)
                t6pa = np.tan(np.pi / 6 + angle)
                t6ma = np.tan(np.pi / 6 - angle)

                spiders_map = ((X.T > (-X + t_spiders / s2ma_2) * t2ma) +
                               (X.T < (-X - t_spiders / s2ma_2) * t2ma))
                spiders_map *= ((X.T > (X + t_spiders / s6pa_2) * t6pa) +
                                (X.T < (X - t_spiders / s6pa_2) * t6pa))
                spiders_map *= ((X.T > (-X + t_spiders / s6ma_2) * t6ma) +
                                (X.T < (-X - t_spiders / s6ma_2) * t6ma))
                pup = pup * spiders_map

    print("Generic pupil created")
    return pup


def make_VLT(dim, pupd, tel):
    """
        Initialize the VLT pupil

    :parameters:

        dim: (long) : linear size of ???

        pupd: (long) : linear size of total pupil

        tel: (Param_tel) : Telescope structure
    """

    if (tel.set_t_spiders == -1):
        print("force t_spider =%5.3f" % (0.09 / 18.))
        tel.set_t_spiders(0.09 / 18.)
    angle = 50.5 * np.pi / 180.  # --> 50.5 degre *2 d'angle entre les spiders

    Range = (0.5 * (1) - 0.25 / dim)
    X = np.tile(np.linspace(-Range, Range, dim, dtype=np.float32), (dim, 1))

    R = np.sqrt(X**2 + (X.T)**2)

    pup = ((R < 0.5) & (R > (tel.cobs / 2))).astype(np.float32)

    if (tel.set_t_spiders == -1):
        print('No spider')
    else:
        spiders_map = (
                (X.T >
                 (X - tel.cobs / 2 + tel.t_spiders / np.sin(angle)) * np.tan(angle)) +
                (X.T < (X - tel.cobs / 2) * np.tan(angle))) * (X > 0) * (X.T > 0)
        spiders_map += np.fliplr(spiders_map)
        spiders_map += np.flipud(spiders_map)
        spiders_map = interp.rotate(spiders_map, tel.pupangle, order=0, reshape=False)

        pup = pup * spiders_map

    print("VLT pupil created")
    return pup


def make_EELT(dim, pupd, tel, N_seg=-1):
    """
        Initialize the EELT pupil

    :parameters:

        dim: (long) : linear size of ???

        pupd: (long) : linear size of total pupil

        tel: (Param_tel) : Telescope structure

        N_seg: (int)

    TODO: complete
    TODO : add force rescal pup elt
    """
    if (N_seg == -1):
        EELT_file = EELT_data + "EELT-Custom_N" + str(dim) + "_COBS" + str(
                100 * tel.cobs) + "_CLOCKED" + str(tel.pupangle) + "_TSPIDERS" + str(
                        100 * tel.t_spiders) + "_MS" + str(
                                tel.nbrmissing) + "_REFERR" + str(
                                        100 * tel.referr) + ".h5"
    else:
        EELT_file = EELT_data + tel.type_ap.decode('UTF-8') + "_N" + str(
                dim) + "_COBS" + str(100 * tel.cobs) + "_CLOCKED" + str(
                        tel.pupangle) + "_TSPIDERS" + str(
                                100 * tel.t_spiders) + "_MS" + str(
                                        tel.nbrmissing) + "_REFERR" + str(
                                                100 * tel.referr) + ".h5"
    if (os.path.isfile(EELT_file)):
        print("reading EELT pupil from file ", EELT_file)
        pup = h5u.readHdf5SingleDataset(EELT_file)
    else:
        print("creating EELT pupil...")
        file = EELT_data + "Coord_" + tel.type_ap.decode('UTF-8') + ".dat"
        data = np.fromfile(file, sep="\n")
        data = np.reshape(data, (data.size // 2, 2))
        x_seg = data[:, 0]
        y_seg = data[:, 1]

        file = EELT_data + "EELT_MISSING_" + tel.type_ap.decode('UTF-8') + ".dat"
        k_seg = np.fromfile(file, sep="\n").astype(np.int32)

        W = 1.45 * np.cos(np.pi / 6)

        #tel.set_diam(39.146)
        #tel.set_diam(37.)
        Range = (0.5 * (tel.diam * dim / pupd) - 0.25 / dim)
        X = np.tile(np.linspace(-Range, Range, dim, dtype=np.float32), (dim, 1))

        if (tel.t_spiders == -1):
            print("force t_spider =%5.3f" % (0.014))
            tel.set_t_spiders(0.014)
        #t_spiders=0.06
        #tel.set_t_spiders(t_spiders)

        if (tel.nbrmissing > 0):
            k_seg = np.sort(k_seg[:tel.nbrmissing])

        file = EELT_data + "EELT_REF_ERROR" + ".dat"
        ref_err = np.fromfile(file, sep="\n")

        #mean_ref = np.sum(ref_err)/798.
        #std_ref = np.sqrt(1./798.*np.sum((ref_err-mean_ref)**2))
        #mean_ref=np.mean(ref_err)
        std_ref = np.std(ref_err)

        ref_err = ref_err * tel.referr / std_ref

        if (tel.nbrmissing > 0):
            ref_err[k_seg] = 1.0

        pup = np.zeros((dim, dim))

        t_3 = np.tan(np.pi / 3.)
        if N_seg == -1:

            vect_seg = tel.vect_seg
            for i in vect_seg:
                Xt = X + x_seg[i]
                Yt = X.T + y_seg[i]
                pup+=(1-ref_err[i])*(Yt<0.5*W)*(Yt>=-0.5*W)*(0.5*(Yt+t_3*Xt)<0.5*W) \
                                   *(0.5*(Yt+t_3*Xt)>=-0.5*W)*(0.5*(Yt-t_3*Xt)<0.5*W) \
                                   *(0.5*(Yt-t_3*Xt)>=-0.5*W)

        else:
            for i in range(N_seg):
                Xt = X + x_seg[i]
                Yt = X.T + y_seg[i]
                pup+=(1-ref_err[i])*(Yt<0.5*W)*(Yt>=-0.5*W)*(0.5*(Yt+t_3*Xt)<0.5*W) \
                                   *(0.5*(Yt+t_3*Xt)>=-0.5*W)*(0.5*(Yt-t_3*Xt)<0.5*W) \
                                   *(0.5*(Yt-t_3*Xt)>=-0.5*W)
        if (tel.t_spiders == 0):
            print('No spider')
        else:
            t_spiders = tel.t_spiders * (tel.diam * dim / pupd)

            s2_6 = 2 * np.sin(np.pi / 6)
            t_6 = np.tan(np.pi / 6)

            spiders_map = np.abs(X) > t_spiders / 2
            spiders_map *= (
                    (X.T >
                     (X + t_spiders / s2_6) * t_6) + (X.T <
                                                      (X - t_spiders / s2_6) * t_6))
            spiders_map *= (
                    (X.T >
                     (-X + t_spiders / s2_6) * t_6) + (X.T <
                                                       (-X - t_spiders / s2_6) * t_6))

            pup = pup * spiders_map

        if (tel.pupangle != 0):
            pup = interp.rotate(pup, tel.pupangle, reshape=False, order=0)

        print("writing EELT pupil to file ", EELT_file)
        h5u.writeHdf5SingleDataset(EELT_file, pup)

    print("EELT pupil created")
    return pup


def make_phase_ab(dim, pupd, tel, pup=None, xc=-1, yc=-1, real=0, halfSpider=False):
    """Compute the EELT M1 phase aberration

    :parameters:

        dim: (long) : linear size of ???

        pupd: (long) : linear size of total pupil

        tel: (Param_tel) : Telescope structure

        pup: (?)

    TODO: complete
    """

    if ((tel.type_ap == ApertureType.GENERIC) or (tel.type_ap == ApertureType.VLT)):
        return np.zeros((dim, dim)).astype(np.float32)

    elif tel.type_ap == ApertureType.EELT:

        return generateEeltPupilMask(
                dim, 0, xc, yc, tel.diam / dim, tel.gap, tel.pupangle, D=tel.diam,
                halfSpider=halfSpider, pitch=1.244683637214, nseg=33, inner_rad=4.1,
                outer_rad=15.4, R=95.7853, nominalD=40, half_seg=0.75,
                refl=[tel.std_piston, tel.std_tt, tel.std_tt])
    elif (tel.type_ap == ApertureType.KECK):
        seg_corner = 1.8
        kpitch = seg_corner / 2 * np.sqrt(3)
        knseg = 7
        kinner_rad = 0.9
        kouter_rad = 3.4
        kR = 85
        knominalD = 10.96
        khalf_seg = 0.9
        return generateEeltPupilMask(
                dim, 0, xc, yc, tel.diam / dim, tel.gap, tel.pupangle, D=tel.diam,cobs=tel.cobs,
                halfSpider=halfSpider, pitch=kpitch, nseg=knseg, inner_rad=0.9,
                outer_rad=3.4, R=kR, nominalD=knominalD, half_seg=0.9,
                refl=[tel.std_piston, tel.std_tt, tel.std_tt])
    else:
        ab_file = EELT_data + "aberration_" + tel.type_ap.decode('UTF-8') + \
                "_N" + str(dim) + "_NPUP" + str(np.where(pup)[0].size) + "_CLOCKED" + str(
                tel.pupangle) + "_TSPIDERS" + str(
                        100 * tel.t_spiders) + "_MS" + str(tel.nbrmissing) + "_REFERR" + str(
                                100 * tel.referr) + "_PIS" + str(
                                        tel.std_piston) + "_TT" + str(tel.std_tt) + ".h5"
        if (os.path.isfile(ab_file)):
            print("reading aberration phase from file ", ab_file)
            phase_error = h5u.readHdf5SingleDataset(ab_file)
        else:
            print("computing M1 phase aberration...")

            std_piston = tel.std_piston
            std_tt = tel.std_tt

            W = 1.45 * np.cos(np.pi / 6)

            file = EELT_data + "EELT_Piston_" + tel.type_ap.decode('UTF-8') + ".dat"
            p_seg = np.fromfile(file, sep="\n")
            #mean_pis=np.mean(p_seg)
            std_pis = np.std(p_seg)
            p_seg = p_seg * std_piston / std_pis
            N_seg = p_seg.size

            file = EELT_data + "EELT_TT_" + tel.type_ap.decode('UTF-8') + ".dat"
            tt_seg = np.fromfile(file, sep="\n")

            file = EELT_data + "EELT_TT_DIRECTION_" + tel.type_ap.decode('UTF-8') + ".dat"
            tt_phi_seg = np.fromfile(file, sep="\n")

            phase_error = np.zeros((dim, dim))
            phase_tt = np.zeros((dim, dim))
            phase_defoc = np.zeros((dim, dim))

            file = EELT_data + "Coord_" + tel.type_ap.decode('UTF-8') + ".dat"
            data = np.fromfile(file, sep="\n")
            data = np.reshape(data, (data.size // 2, 2))
            x_seg = data[:, 0]
            y_seg = data[:, 1]

            Range = (0.5 * (tel.diam * dim / pupd) - 0.25 / dim)

            X = np.tile(np.linspace(-Range, Range, dim, dtype=np.float32), (dim, 1))
            t_3 = np.tan(np.pi / 3.)

            for i in range(N_seg):
                Xt = X + x_seg[i]
                Yt = X.T + y_seg[i]
                SEG=(Yt<0.5*W)*(Yt>=-0.5*W)*(0.5*(Yt+t_3*Xt)<0.5*W) \
                                *(0.5*(Yt+t_3*Xt)>=-0.5*W)*(0.5*(Yt-t_3*Xt)<0.5*W) \
                                *(0.5*(Yt-t_3*Xt)>=-0.5*W)

                if (i == 0):
                    N_in_seg = np.sum(SEG)
                    Hex_diam = 2 * np.max(
                            np.sqrt(Xt[np.where(SEG)]**2 + Yt[np.where(SEG)]**2))

                if (tt_seg[i] != 0):
                    TT = tt_seg[i] * (
                            np.cos(tt_phi_seg[i]) * Xt + np.sin(tt_phi_seg[i]) * Yt)
                    mean_tt = np.sum(TT[np.where(SEG == 1)]) / N_in_seg
                    phase_tt += SEG * (TT - mean_tt)

                #TODO defocus

                phase_error += SEG * p_seg[i]

            N_EELT = np.where(pup)[0].size
            if (np.sum(phase_tt) != 0):
                phase_tt *= std_tt / np.sqrt(
                        1. / N_EELT * np.sum(phase_tt[np.where(pup)]**2))

            #TODO defocus

            phase_error += phase_tt + phase_defoc

            if (tel.pupangle != 0):
                phase_error = interp.rotate(phase_error, tel.pupangle, reshape=False,
                                            order=2)

            print("phase aberration created")
            print("writing aberration filel to file ", ab_file)
            h5u.writeHdf5SingleDataset(ab_file, phase_error)

        return phase_error


"""

 _____ _   _____   ____  ___ ____ ___
| ____| | |_   _| |  _ \|_ _/ ___/ _ \
|  _| | |   | |   | |_) || | |  | | | |
| |___| |___| |   |  _ < | | |__| |_| |
|_____|_____|_|   |_| \_\___\____\___/


"""


def generateEeltPupilMask(npt, dspider, i0, j0, pixscale, gap, rotdegree, D=40.0,                                        cobs=0,
                          centerMark=0, halfSpider=False, pitch=1.244683637214, nseg=33,
                          inner_rad=4.1, outer_rad=15.4, R=95.7853, nominalD=40,
                          half_seg=0.75, refl=None, rotSpiderDegree=None):
    """
    Generates a boolean pupil mask of the binary EELT pupil
    on a map of size (npt, npt).


    :returns: pupil image (npt, npt), boolean
    :param int npt: size of the output array
    :param float dspider: width of spiders in meters
    :param float i0, j0: index of pixels where the pupil should be centred = p_geom.pupdiam / 2. - 0.5
                         Can be floating-point indexes.
    :param float pixscale: size of a pixel of the image, in meters = ptel.diam/(p_geom.pupdiam / 2. - 0.5)
    :param float gap: gap between 2 segments in metres
    :param float rotdegree: rotation angle of the pupil, in degrees.
    :param float D: diameter of the pupil. For the nominal EELT, D shall
                    be set to 40.0
    :param int centerMark: when centerMark!=0, a pixel is added at the centre of
        symmetry of the pupil in order to debug things using compass.
        centerMark==1 draws a point
        centerMark==2 draws 2 lines
    :param bool halfSpider: half Spider computation flag
    :param float pitch: segment pitch
    :param int nseg: number of segments across the diameter
    :param float inner_rad: Inner radius [meters]
    :param float outter_rad: outter radius [meters]
    :param float R: M1 curvature radius
    :param float nominalD: diameter needed to get nominal aperture after projection
    :param float half_seg: segment half size
    :param float refl: std of the reflectivity of each segment

    :Example:
    npt = p_geom.pupdiam
    D = p_tel.diam
    i0 = npt / 2. - 0.5
    j0 = npt / 2. - 0.5
    rotdegree = 0.
    pixscale = D/(npt / 2. - 0.5)
    dspider = 0.51
    gap = 0.0
    pup = generateEeltPupilMask(npt, dspider, i0, j0, pixscale, gap, rotdegree)

    """
    rot = rotdegree * np.pi / 180

    if rotSpiderDegree is None:
        rotSpider = rot
    else:
        rotSpider = rotSpiderDegree * np.pi / 180

    # Generation of segments coordinates.
    # hx and hy have a shape [6,798] describing the 6 vertex of the 798
    # hexagonal mirrors
    #hx, hy = generateCoordSegments( D, rot)
    hx, hy = generateCoordSegments(D, rot, pitch=pitch, nseg=nseg, inner_rad=inner_rad,
                                   outer_rad=outer_rad, R=R, nominalD=nominalD)
    # From the data of hex mirrors, we build the pupil image using
    # boolean
    #pup = generateSegmentProperties(True, hx, hy, i0, j0, pixscale, gap, npt, D)
    if (refl == 0):
        refl = True
    elif np.isscalar(refl):
        referr = np.random.random(hx.size)
        referr = referr * refl / np.std(referr)
        refl = np.ones(hx.size) - referr
    elif type(refl) == list:
        if len(refl) == 3:
            refpist = np.random.random(hx.size)
            refpist = refpist * refl[0] / np.std(refpist)
            reftip = np.random.random(hx.size)
            reftip = reftip * refl[1] / np.std(reftip)
            reftilt = np.random.random(hx.size)
            reftilt = reftilt * refl[2] / np.std(reftilt)
            refl = np.array([refpist, reftip, reftilt])
    else:
        raise ValueError(
                "refl param must be None, scalar (reflectivity std error) or list of 3 elements (piston, tip and tilt std errors)"
        )

    pup = generateSegmentProperties(refl, hx, hy, i0, j0, pixscale, gap, npt, D,
                                    nominalD=nominalD, pitch=pitch, half_seg=half_seg)
    # SPIDERS ............................................
    nspider = 3  # for the day where we have more/less spiders ;-)
    if (dspider > 0 and nspider > 0):
        if (halfSpider is True):
            pup = pup * fillHalfSpider(npt, nspider, dspider, i0, j0, pixscale, rotSpider)
        else:
            pup = pup * fillSpider(npt, nspider, dspider, i0, j0, pixscale, rotSpider)

    # Rajout d'un pixel au centre (pour marquer le centre) ou d'une croix,
    # selon la valeur de centerMark
    if centerMark:
        pup = np.logical_xor(pup, centrePourVidal(npt, i0, j0, centerMark))

    if cobs > 0:
        obstru = (util.dist(pup.shape[0], pup.shape[0] // 2 + 0.5, pup.shape[0] // 2 + 0.5) >= (pup.shape[0] * cobs + 1.) * 0.5).astype(np.float32)
        pup *= obstru
    return pup


def fillPolygon(x, y, i0, j0, scale, gap, N, index=0):
    """
    From a list of points defined by their 2 coordinates list
    x and y, creates a filled polygon with sides joining the points.
    The polygon is created in an image of size (N, N).
    The origin (x,y)=(0,0) is mapped at pixel i0, j0 (both can be
    floating-point values).
    Arrays x and y are supposed to be in unit U, and scale is the
    pixel size in U units.

    :returns: filled polygon (N, N), boolean
    :param float x, y: list of points defining the polygon
    :param float i0, j0: index of pixels where the pupil should be centred.
                         Can be floating-point indexes.
    :param float scale: size of a pixel of the image, in same unit as x and y.
    :param float N: size of output image.

    :Example:
    x = np.array([1,-1,-1.5,0,1.1])
    y = np.array([1,1.5,-0.2,-2,0])
    N = 200
    i0 = N/2
    j0 = N/2
    gap = 0.
    scale = 0.03
    pol = fillPolygon(x, y, i0, j0, scale, gap, N, index=2)

    """
    # define coordinates map centred on (i0,j0) with same units as x,y.
    X = (np.arange(N) - i0) * scale
    Y = (np.arange(N) - j0) * scale
    X, Y = np.meshgrid(X, Y, indexing='ij')  # indexage [x,y]

    # define centre x0, y0 of the polygon
    x0 = np.mean(x)
    y0 = np.mean(y)

    # compute angles of all pixels coordinates of the map, and all
    # corners of the polygon
    T = (np.arctan2(Y - y0, X - x0) + 2 * np.pi) % (2 * np.pi)
    t = (np.arctan2(y - y0, x - x0) + 2 * np.pi) % (2 * np.pi)

    # on va voir dans quel sens ca tourne. Je rajoute ca pour que ca marche
    # quel que soit le sens de rotation des points du polygone.
    # En fait, j'aurais peut etre pu classer les points par leur angle, pour
    # etre sur que ca marche meme si les points sont donnes dans ts les cas
    sens = np.median(np.diff(t))
    if sens < 0:
        x = x[::-1]
        y = y[::-1]
        t = t[::-1]

    # re-organise order of polygon points so that it starts from
    # angle = 0, or at least closest to 0.
    imin = t.argmin()  # position of the minimum
    if imin != 0:
        x = np.roll(x, -imin)
        y = np.roll(y, -imin)
        t = np.roll(t, -imin)

    # For each couple of consecutive corners A, B, of the polygon, one fills
    # the triangle AOB with True.
    # Last triangle has a special treatment because it crosses the axis
    # with theta=0=2pi
    n = x.shape[0]  # number of corners of polygon
    indx, indy = (np.array([], dtype=np.int), np.array([], dtype=np.int))
    distedge = np.array([], dtype=np.float)
    for i in range(n):
        j = i + 1  # j=element next i except when i==n : then j=0 (cycling)
        if j == n:
            j = 0
            sub = np.where((T >= t[-1]) | (T <= (t[0])))
        else:
            sub = np.where((T >= t[i]) & (T <= t[j]))
        # compute unitary vector des 2 sommets
        dy = y[j] - y[i]
        dx = x[j] - x[i]
        vnorm = np.sqrt(dx**2 + dy**2)
        dx /= vnorm
        dy /= vnorm
        # calcul du produit vectoriel
        crossprod = dx * (Y[sub] - y[i]) - dy * (X[sub] - x[i])
        tmp = crossprod > gap
        indx = np.append(indx, sub[0][tmp])
        indy = np.append(indy, sub[1][tmp])
        distedge = np.append(distedge, crossprod[tmp])

    # choice of what is returned : either only the indexes, or the
    # boolean map
    if index == 1:
        return (indx, indy, distedge)
    elif index == 2:
        a = np.zeros((N, N))
        a[indx, indy] = distedge
        return a
    else:
        a = np.zeros((N, N), dtype=np.bool)
        a[indx, indy] = True  # convention [x,y]

    return a


def compute6Segments(pupNoSpiders, N, pixscale, dspider, i0, j0, rot=0):
    """
    N = p_geom.pupdiam
    i0 = j0 = N / 2. - 0.5
    D = p_tel.diam
    pixscale = D/N
    dspider = 0.51

    Utilisee dans compass/shesha/shesha/supervisor/canapassSupervisor.py
    pour le slaving des actus.
    """
    npts = 200
    msk = np.zeros((6, pupNoSpiders.shape[0], pupNoSpiders.shape[1]))
    mid = int(pupNoSpiders.shape[0] / 2)
    tmp = np.zeros((pupNoSpiders.shape[0], pupNoSpiders.shape[1]))
    for angle in np.linspace(0, 60, npts):
        tmp += compute1Spider(0, N, dspider, i0, j0, pixscale, angle * 2 * np.pi / 360)
    msk[5, :, :] = tmp < npts * pupNoSpiders
    msk[5, mid:, :] = 0
    msk[2, :, :] = tmp < npts * pupNoSpiders
    msk[2, :mid, :] = 0
    tmp *= 0
    for angle in np.linspace(0, 60, npts):
        tmp += compute1Spider(1, N, dspider, i0, j0, pixscale, angle * 2 * np.pi / 360)
    msk[0, :, :] = tmp < npts * pupNoSpiders
    msk[0, mid:, :] = 0
    msk[3, :, :] = tmp < npts * pupNoSpiders
    msk[3, :mid, :] = 0
    tmp *= 0
    for angle in np.linspace(0, 60, npts):
        tmp += compute1Spider(2, N, dspider, i0, j0, pixscale, angle * 2 * np.pi / 360)
    msk[1, :, :] = tmp < npts * pupNoSpiders
    msk[1, :, :mid] = 0
    msk[4, :, :] = tmp < npts * pupNoSpiders
    msk[4, :, mid:] = 0
    return msk


def compute1Spider(nspider, N, dspider, i0, j0, scale, rot):
    """
    Fonction de fab pour creer le slaving.
    La fonction cree un tableau de booleens avec une seule spider.
    Utilisee par la fonction compute6Segments()
    """
    a = np.ones((N, N), dtype=np.bool)
    X = (np.arange(N) - i0) * scale
    Y = (np.arange(N) - j0) * scale
    X, Y = np.meshgrid(X, Y, indexing='ij')  # convention d'appel [x,y]
    w = 2 * np.pi / 6
    i = nspider
    nn = (abs(X * np.cos(i * w - rot) + Y * np.sin(i * w - rot)) < dspider / 2.)
    a[nn] = False
    return a


def fillSpider(N, nspider, dspider, i0, j0, scale, rot):
    """
    Creates a boolean spider mask on a map of dimensions (N,N)
    The spider is centred at floating-point coords (i0,j0).

    :returns: spider image (boolean)
    :param int N: size of output image
    :param int nspider: number of spiders
    :param float dspider: width of spiders
    :param float i0: coord of spiders symmetry centre
    :param float j0: coord of spiders symmetry centre
    :param float scale: size of a pixel in same unit as dspider
    :param float rot: rotation angle in radians

    """
    a = np.ones((N, N), dtype=np.bool)
    X = (np.arange(N) - i0) * scale
    Y = (np.arange(N) - j0) * scale
    X, Y = np.meshgrid(X, Y, indexing='ij')  # convention d'appel [x,y]
    w = 2 * np.pi / nspider
    for i in range(nspider):
        nn = (abs(X * np.cos(i * w - rot) + Y * np.sin(i * w - rot)) < dspider / 2.)
        a[nn] = False
    return a


def fillHalfSpider(N, nspider, dspider, i0, j0, scale, rot):
    a = np.ones((N, N), dtype=np.bool)
    b = np.ones((N, N), dtype=np.bool)
    X = (np.arange(N) - i0) * scale
    Y = (np.arange(N) - j0) * scale
    X, Y = np.meshgrid(X, Y, indexing='ij')  # convention d'appel [x,y]
    w = 2 * np.pi / nspider
    for i in range(nspider):
        right = (X * np.cos(i * w - rot) + Y * np.sin(i * w - rot) <
                 dspider / 2) * (X * np.cos(i * w - rot) + Y * np.sin(i * w - rot) > 0.)
        left = (X * np.cos(i * w - rot) + Y * np.sin(i * w - rot) >
                -dspider / 2) * (X * np.cos(i * w - rot) + Y * np.sin(i * w - rot) < 0.)
        a[right] = False
        b[left] = False
    return a, b


def createHexaPattern(pitch, supportSize):
    """
    Cree une liste de coordonnees qui decrit un maillage hexagonal.
    Retourne un tuple (x,y).

    Le maillage est centre sur 0, l'un des points est (0,0).
    Une des pointes de l'hexagone est dirigee selon l'axe Y, au sens ou le
    tuple de sortie est (x,y).

    :param float pitch: distance between 2 neighbour points
    :param int supportSize: size of the support that need to be populated

    """
    V3 = np.sqrt(3)
    nx = int(np.ceil((supportSize / 2.0) / pitch) + 1)
    x = pitch * (np.arange(2 * nx + 1) - nx)
    ny = int(np.ceil((supportSize / 2.0) / pitch / V3) + 1)
    y = (V3 * pitch) * (np.arange(2 * ny + 1) - ny)
    x, y = np.meshgrid(x, y, indexing='ij')
    x = x.flatten()
    y = y.flatten()
    peak_axis = np.append(x, x + pitch / 2.)  # axe dirige selon sommet
    flat_axis = np.append(y, y + pitch * V3 / 2.)  # axe dirige selon plat
    return flat_axis, peak_axis


def generateCoordSegments(D, rot, pitch=1.244683637214, nseg=33, inner_rad=4.1,
                          outer_rad=15.4, R=95.7853, nominalD=40):
    """
    Computes the coordinates of the corners of all the hexagonal
    segments of M1.
    Result is a tuple of arrays(6, 798).

    Parameters
    -----------------------------------------
    D: (float) : pupil diameter in meters (it must be set to 40.0 m for the ELT)
    rot: (float) : pupil rotation angle in radians
    pitch: (float): Segment pitch [meters]
    nseg: (int) : number of segments across the diameter
    inner_rad : (float): Inner radius [meters]
    outer_rad : (float): Outer radius [meters]
    R : (float): Curvature radius of the M1
    nominalD: (float): diameter for nominal pupil

    """
    V3 = np.sqrt(3)
    #pitch = 1.227314    # no correction du bol
    #pitch = 1.244683637214  # diametre du cerle INSCRIT
    # diamseg = pitch*2/V3  # diametre du cercle contenant TOUT le segment
    # print("segment diameter : %.6f\n" % diamseg)

    # Creation d'un pattern hexa avec pointes selon la variable <ly>
    lx, ly = createHexaPattern(pitch, (nseg + 2) * pitch)
    ll = np.sqrt(lx**2 + ly**2)
    # Elimination des segments non valides grace a 2 nombres parfaitement
    # empiriques ajustes a-la-mano.
    #inner_rad, outer_rad = 4.1, 15.4   # nominal, 798 segments
    nn = (ll > inner_rad * pitch) & (ll < outer_rad * pitch)
    lx = lx[nn]
    ly = ly[nn]
    lx, ly = reorganizeSegmentsOrderESO(lx, ly)
    ll = np.sqrt(lx**2 + ly**2)

    # n = ll.shape[0]
    # print("Nbre de segments : %d\n" % n)
    # Creation d'un hexagone-segment avec pointe dirigee vers
    # variable <hx> (d'ou le cos() sur hx)
    th = np.linspace(0, 2 * np.pi, 7)[0:6]
    hx = np.cos(th) * pitch / V3
    hy = np.sin(th) * pitch / V3

    # Le maillage qui permet d'empiler des hexagones avec sommets 3h-9h
    # est un maillage hexagonal avec sommets 12h-6h, donc a 90°.
    # C'est pour ca qu'il a fallu croiser les choses avant.
    x = (lx[None, :] + hx[:, None])
    y = (ly[None, :] + hy[:, None])
    r = np.sqrt(x**2 + y**2)
    #R = 95.7853
    rrc = R / r * np.arctan(r / R)  # correction factor
    x *= rrc
    y *= rrc

    #nominalD = 40.0   # size of the OFFICIAL E-ELT
    if D != nominalD:
        x *= D / nominalD
        y *= D / nominalD

    # Rotation matrices
    mrot = np.array([[np.cos(rot), np.sin(rot)], [-np.sin(rot), np.cos(rot)]])

    # rotation of coordinates
    # le tableau [x,y] est de taille (2,6,798). Faut un transpose a la con
    # pour le transformer en (6,2,798) pour pouvoir faire le np.dot
    # correctement. En sortie, xrot est (2,6,798).
    xyrot = np.dot(mrot, np.transpose(np.array([x, y]), (1, 0, 2)))

    return xyrot[0], xyrot[1]


def gendron():
    """
    La fonction est appelee quand l'utilisateur a demande une pupille
    ELT, et renseigne un diametre de telescope different de 40 metres.

    Faut vraiment que je commente ou t'as compris ??

    """
    mymsg = [
            "\n\n\n\n", "__        ___    ____  _   _ ___ _   _  ___ _",
            "\ \      / / \  |  _ \| \ | |_ _| \ | |/ ___|",
            " \ \ /\ / / _ \ | |_) |  \| || ||  \| | |  _ ",
            "  \ V  V / ___ \|  _ <| |\  || || |\  | |_| |",
            "   \_/\_/_/   \_\_| \_\_| \_|___|_| \_|\____|", " \n",
            "Vous utilisez un telescope de type ELT. Ce telescope",
            "est fait pour etre utilise avec un diametre de 40 m.", " ",
            "Or, vous utilisez un diametre different. Cela signifie",
            "que le telescope que vous etes en train de creer a une",
            "taille differente du veritable E-ELT de l'ESO.", "  ",
            "  * Soit vous savez exactement ce que vous faites, auquel",
            "cas bonne route.", " ",
            "  * Soit vous desirez creer LE vrai E-ELT et il faut changer",
            "plusieurs choses:",
            "    1) le diametre telescope de votre fichier de parametres et",
            "       le renseigner a 40 metres.",
            "       p_tel.set_diam(40.0) # Nominal size for the real EELT",
            "    2) le nombre d'actionneurs de M4 a 75",
            "       p_dm0.set_nact(75) # 75 actu in 40m for pitch=54.05cm",
            "    3) option: tourner la pupille de 90 degres pour revenir au",
            "       cas initial de compass",
            "       p_tel.set_pupangle(90.)  # ELT pup rotation in degrees"
            "  ", "\n\n"
    ]
    for ligne in mymsg:
        print(ligne)


def reorganizeSegmentsOrderESO(x, y):
    """
    Reorganisation des segments facon ESO.
    Voir
    ESO-193058 Standard Coordinate System and Basic Conventions

    :param float x: tableau des centres X des segments
    :param float y: idem Y
    :return tuple (x,y): meme tuple que les arguments d'entree, mais tries.

    """
    # pi/2, pi/6, 2.pi, ...
    pi_3 = np.pi / 3
    pi_6 = np.pi / 6
    pix2 = 2 * np.pi
    # calcul des angles
    t = (np.arctan2(y, x) + pi_6 - 1e-3) % (pix2)
    X = np.array([])
    Y = np.array([])
    A = 100.
    for k in range(6):
        sector = (t > k * pi_3) & (t < (k + 1) * pi_3)
        u = k * pi_3
        distance = (A * np.cos(u) - np.sin(u)) * x[sector] + (
                np.cos(u) + A * np.sin(u)) * y[sector]
        indsort = np.argsort(distance)
        X = np.append(X, x[sector][indsort])
        Y = np.append(Y, y[sector][indsort])
    return X, Y


def getdatatype(truc):
    """
    Returns the data type of a numpy variable, either scalar value or array
    """
    if np.isscalar(truc):
        return type(truc)
    else:
        return type(truc.flatten()[0])


def generateSegmentProperties(attribute, hx, hy, i0, j0, scale, gap, N, D, softGap=0,
                              nominalD=40, pitch=1.244683637214, half_seg=0.75):
    """
    Builds a 2D image of the pupil with some attributes for each of the
    segments. Those segments are described from arguments hx and hy, that
    are produced by the function generateCoordSegments(D, rot).

    When attribute is a phase, then it must be a float array of dimension
    [3, 798] with the dimension 3 being piston, tip, and tilt.
    Units of phase is xxx rms, and the output of the procedure will be
    in units of xxx.


    :returns: pupil image (N, N), with the same type of input argument attribute

    :param float/int/bool attribute: scalar value or 1D-array of the reflectivity of
           the segments or 2D array of phase
           If attribute is scalar, the value will be replicated for all segments.
           If attribute is a 1D array, then it shall contain the reflectivities
           of all segments.
           If attribute is a 2D array then it shall contain the piston, tip
           and tilt of the segments. The array shall be of dimension
           [3, 798] that contains [piston, tip, tilt]
           On output, the data type of the pupil map will be the same as attribute.
    :param float hx, hy: arrays [6,:] describing the segment shapes. They are
        generated using generateCoordSegments()
    :param float dspider: width of spiders in meters
    :param float i0, j0: index of pixels where the pupil should be centred.
                         Can be floating-point indexes.
    :param float scale: size of a pixel of the image, in meters.
    :param float gap: half-space between segments in meters
    :param int N: size of the output array (N,N)
    :param float D: diameter of the pupil. For the nominal EELT, D shall
                    be set to 40.0
    :param bool softGap: if False, the gap between segments is binary 0/1
          depending if the pixel is within the gap or not. If True, the gap
          is a smooth region of a fwhm of 2 pixels with a depth related to the
          gap width.
    :param float nominalD: diameter needed to get nominal pupil aperture
    :param float pitch: segment pitch
    :param float half_seg: segment half size



    attribute = np.ones(798)+np.random.randn(798)/20.
    N = 800
    i0 = N/2
    j0 = N/2
    rotdegree = 0.0
    scale = 41./N
    gap = 0.03

    """

    # number of segments
    nseg = hx.shape[-1]
    # If <attribute> is a scalar, then we make a list. It will be required
    # later on to set the attribute to each segment.
    if np.isscalar(attribute):
        attribute = np.array([attribute] * nseg)

    # the pupil map is created with the same data type as <attribute>
    pupil = np.zeros((N, N), dtype=getdatatype(attribute))

    # average coord of segments
    x0 = np.mean(hx, axis=0)
    y0 = np.mean(hy, axis=0)
    # avg coord of segments in pixel indexes
    x0 = x0 / scale + i0
    y0 = y0 / scale + j0
    # size of mini-support
    hexrad = half_seg * D / nominalD / scale
    ix0 = np.floor(x0 - hexrad).astype(int) - 1
    iy0 = np.floor(y0 - hexrad).astype(int) - 1
    segdiam = np.ceil(hexrad * 2 + 1).astype(int) + 1

    n = attribute.shape[0]
    if n != 3:
        # attribute is a signel value : either reflectivity, or boolean,
        # or just piston.
        if softGap != 0:
            # Soft gaps
            # The impact of gaps are modelled using a simple function: Lorentz, 1/(1+x**2)
            # The fwhm is always equal to 2 pixels because the gap is supposed
            # to be "small/invisible/undersampled". The only visible thing is
            # the width of the impulse response, chosen 2-pixel wide to be
            # well sampled.
            # The "depth" is related to the gap width. The integral of a Lorentzian
            # of 2 pix wide is PI. Integral of a gap of width 'gap' in pixels is 'gap'.
            # So the depth equals to gap/scale/np.pi.
            for i in range(nseg):
                indx, indy, distedge = fillPolygon(hx[:, i], hy[:, i], i0 - ix0[i],
                                                   j0 - iy0[i], scale, gap * 0., segdiam,
                                                   index=1)
                pupil[indx + ix0[i], indy + iy0[i]] = attribute[i] * (
                        1. - (gap / scale / np.pi) / (1 + (distedge / scale)**2))
        else:
            # Hard gaps
            for i in range(nseg):
                indx, indy, distedge = fillPolygon(hx[:, i], hy[:, i], i0 - ix0[i],
                                                   j0 - iy0[i], scale, gap, segdiam,
                                                   index=1)
                pupil[indx + ix0[i], indy + iy0[i]] = attribute[i]
    else:
        # attribute is [piston, tip, tilt]
        minimap = np.zeros((segdiam, segdiam))
        xmap = np.arange(segdiam) - segdiam / 2
        xmap, ymap = np.meshgrid(xmap, xmap, indexing='ij')  # [x,y] convention
        #pitch = 1.244683637214        # diameter of inscribed circle
        diamseg = pitch * 2 / np.sqrt(3)  # diameter of circumscribed circle
        diamfrizou = (pitch + diamseg) / 2 * D / nominalD  # average diameter of the 2
        # Calcul du facteur de mise a l'echelle pour l'unite des tilts.
        # xmap et ymap sont calculees avec un increment de +1 pour deux pixels
        # voisins, donc le facteur a appliquer est tel que l'angle se conserve
        # donc factunit*1 / scale = 4*factunit
        factunit = 4 * scale / diamfrizou
        for i in range(nseg):
            indx, indy, _ = fillPolygon(hx[:, i], hy[:, i], i0 - ix0[i], j0 - iy0[i],
                                        scale, 0., segdiam, index=1)
            minimap = attribute[0, i] + (factunit * attribute[1, i]) * xmap + (
                    factunit * attribute[2, i]) * ymap
            pupil[indx + ix0[i], indy + iy0[i]] = minimap[indx, indy]

    return pupil


def centrePourVidal(N, i0, j0, centerMark):
    """
    Renvoie une image de boolens (False) de taille (N,N) avec un point
    ou une croix (True) centree sur (i0, j0).
    :param int N: taille de l'image de sortie
    :param float i0, j0: position du marqueur de sortie
    :param int centerMark: 0 (pour rien), 1 (option point) ou 2 (option croix)
    """
    scale = 1.0
    res = 0
    X = (np.arange(N) - i0) * scale
    Y = (np.arange(N) - j0) * scale
    X, Y = np.meshgrid(X, Y, indexing='ij')  # convention d'appel [x,y]
    if centerMark == 1:
        res = (X**2 + Y**2) < 1
    if centerMark == 2:
        res = (np.abs(X) < 0.9) | (np.abs(Y) < 0.9)
    return res
