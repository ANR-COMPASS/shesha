'''
On the fly modification of the WFS
'''
import shesha.config as conf
import shesha.constants as scons
from shesha.constants import CONST

import shesha.util.utilities as util

import numpy as np
from shesha.sutra_bind.wrap import Sensors, Rtc


def comp_new_pyr_ampl(rtc: Rtc, n: int, p_centroider: conf.Param_centroider,
                      p_wfss: list, p_tel: conf.Param_tel, ampli: float):
    """ Set the pyramid modulation amplitude

    :parameters:

        rtc: (Rtc): rtc object

        n : (int): centroider index

        p_centroider : (Param_centroider) : pyr centroider settings

        ampli : (float) : new amplitude in units of lambda/D

        p_wfss : (list of Param_wfs) : list of wfs parameters

        p_tel : (Param_tel) : Telescope parameters
    """
    nwfs = p_centroider.nwfs
    pwfs = p_wfss[nwfs]
    pwfs.set_pyr_ampl(ampli)

    pixsize = pwfs._qpixsize * CONST.ARCSEC2RAD
    scale_fact = 2 * np.pi / pwfs._Nfft * \
        (pwfs.Lambda * 1e-6 / p_tel.diam) / pixsize * ampli
    cx = scale_fact * \
        np.sin((np.arange(pwfs.pyr_npts, dtype=np.float32))
               * 2. * np.pi / pwfs.pyr_npts)
    cy = scale_fact * \
        np.cos((np.arange(pwfs.pyr_npts, dtype=np.float32))
               * 2. * np.pi / pwfs.pyr_npts)
    pwfs.set_pyr_cx(cx)
    pwfs.set_pyr_cy(cy)

    scale = pwfs.Lambda * 1e-6 / p_tel.diam * ampli * 180. / np.pi * 3600.

    rtc.set_pyr_ampl(nwfs, cx, cy, scale)


def noise_cov(nw: int, p_wfs: conf.Param_wfs, p_atmos: conf.Param_atmos,
              p_tel: conf.Param_tel):
    """ Compute the diagonal of the noise covariance matrix for a SH WFS (arcsec^2)
    Photon noise: (pi^2/2)*(1/Nphotons)*(d/r0)^2 / (2*pi*d/lambda)^2
    Electronic noise: (pi^2/3)*(wfs.noise^2/N^2photons)*wfs.npix^2*(wfs.npix*wfs.pixsize*d/lambda)^2 / (2*pi*d/lambda)^2

    :parameters:

        nw: wfs number

        p_wfs: (Param_wfs) : wfs settings

        p_atmos: (Param_atmos) : atmos settings

        p_tel: (Param_tel) : telescope settings

    :return:

        cov : (np.ndarray(ndim=1,dtype=np.float64)) : noise covariance diagonal
    """
    cov = np.zeros(2 * p_wfs._nvalid)
    if (p_wfs.noise >= 0):
        m = p_wfs._validsubsy
        n = p_wfs._validsubsx
        ind = m * p_wfs.nxsub + n
        flux = np.copy(p_wfs._fluxPerSub)
        flux = flux.reshape(flux.size, order='F')
        flux = flux[ind]
        Nph = flux * p_wfs._nphotons

        r0 = (p_wfs.Lambda / 0.5)**(6.0 / 5.0) * p_atmos.r0

        sig = (np.pi ** 2 / 2) * (1 / Nph) * \
            (1. / r0) ** 2  # Photon noise in m^-2
        # Noise variance in rad^2
        sig /= (2 * np.pi / (p_wfs.Lambda * 1e-6))**2
        sig *= CONST.RAD2ARCSEC**2

        Ns = p_wfs.npix  # Number of pixel
        Nd = (p_wfs.Lambda * 1e-6) * CONST.RAD2ARCSEC / p_wfs.pixsize
        sigphi = (np.pi ** 2 / 3.0) * (1 / Nph ** 2) * (p_wfs.noise) ** 2 * \
            Ns ** 2 * (Ns / Nd) ** 2  # Phase variance in m^-2
        # Noise variance in rad^2
        sigsh = sigphi / (2 * np.pi / (p_wfs.Lambda * 1e-6))**2
        sigsh *= CONST.RAD2ARCSEC**2  # Electronic noise variance in arcsec^2

        cov[:len(sig)] = sig + sigsh
        cov[len(sig):] = sig + sigsh

    return cov


def comp_new_fstop(wfs: Sensors, n: int, p_wfs: conf.Param_wfs, fssize: float,
                   fstop: bytes):
    """ Compute a new field stop for pyrhr WFS

    :parameters:

        n : (int) : WFS index

        wfs : (Param_wfs) : WFS parameters

        fssize : (float) : field stop size [arcsec]

        fstop : (string) : "square" or "round" (field stop shape)
    """
    fsradius_pixels = int(fssize / p_wfs._qpixsize / 2.)
    if (fstop == scons.FieldStopType.ROUND):
        p_wfs.fstop = fstop
        focmask = util.dist(p_wfs._Nfft, xc=p_wfs._Nfft / 2. + 0.5,
                            yc=p_wfs._Nfft / 2. + 0.5) < (fsradius_pixels)
        # fstop_area = np.pi * (p_wfs.fssize/2.)**2. #UNUSED
    elif (p_wfs.fstop == scons.FieldStopType.SQUARE):
        p_wfs.fstop = fstop
        y, x = np.indices((p_wfs._Nfft, p_wfs._Nfft))
        x -= (p_wfs._Nfft - 1.) / 2.
        y -= (p_wfs._Nfft - 1.) / 2.
        focmask = (np.abs(x) <= (fsradius_pixels)) * \
            (np.abs(y) <= (fsradius_pixels))
        # fstop_area = p_wfs.fssize**2. #UNUSED
    else:
        msg = "p_wfs " + str(n) + ". fstop must be round or square"
        raise ValueError(msg)

    # pyr_focmask = np.roll(focmask,focmask.shape[0]/2,axis=0)
    # pyr_focmask = np.roll(pyr_focmask,focmask.shape[1]/2,axis=1)
    pyr_focmask = focmask * 1.0  # np.fft.fftshift(focmask*1.0)
    p_wfs._submask = np.fft.fftshift(pyr_focmask).astype(np.float32)
    p_wfs_fssize = fssize
    wfs.set_submask(n, p_wfs._submask)
