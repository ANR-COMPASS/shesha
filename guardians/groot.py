"""
GROOT (Gpu-based Residual errOr cOvariance maTrix)
Python module for modelization of error covariance matrix
"""
import numpy as np
import h5py
from shesha.sutra_wrap import carmaWrap_context, Groot

import time
from rich.progress import track

from guardians import gamora
from guardians import drax, starlord
import matplotlib.pyplot as plt
plt.ion()

#gpudevices = np.array([0, 1, 2, 3], dtype=np.int32)
gpudevices = np.array([0], dtype=np.int32)

cxt = carmaWrap_context.get_instance_ngpu(gpudevices.size, gpudevices)


def compute_Cerr(filename, modal=True, ctype="float", speed=None, H=None, theta=None,
                 r0=None, L0=None):
    """ Returns the residual error covariance matrix using GROOT from a ROKET file
    :parameter:
        filename : (string) : full path to the ROKET file
        modal : (bool) : if True (default), Cerr is returned in the Btt modal basis,
                         in the actuator basis if False
        ctype : (string) : "float" or "double"
        speed: (np.ndarray(ndim=1, dtype=np.float32)): (optionnal) wind speed for each layer [m/s]
        H: (np.ndarray(ndim=1, dtype=np.float32)): (optionnal) altitude of each layer [m]
        theta: (np.ndarray(ndim=1, dtype=np.float32)): (optionnal) wind direction for each layer [rad]
        r0: (float): (optionnal) Fried parameter @ 0.5 µm [m]
        L0: (np.ndarray(ndim=1, dtype=np.float32)): (optionnal) Outer scale [m]

    :return:
        Cerr : (np.ndarray(dim=2, dtype=np.float32)) : residual error covariance matrix
    """
    f = h5py.File(filename, 'r')
    Lambda_tar = f.attrs["_ParamTarget__Lambda"][0]
    Lambda_wfs = f.attrs["_ParamWfs__Lambda"]
    dt = f.attrs["_ParamLoop__ittime"]
    gain = f.attrs["_ParamController__gain"]
    wxpos = f.attrs["_ParamWfs__xpos"][0]
    wypos = f.attrs["_ParamWfs__ypos"][0]
    if r0 is None:
        r0 = f.attrs["_ParamAtmos__r0"]
    r0 = r0 * (Lambda_tar / Lambda_wfs)**(6. / 5.)
    RASC = 180. / np.pi * 3600.
    xpos = f["dm.xpos"][:]
    ypos = f["dm.ypos"][:]
    p2m = f.attrs["_ParamTel__diam"] / f.attrs["_ParamGeom__pupdiam"]
    pupshape = int(2**np.ceil(np.log2(f.attrs["_ParamGeom__pupdiam"]) + 1))
    xactu = (xpos - pupshape / 2) * p2m
    yactu = (ypos - pupshape / 2) * p2m
    if H is None:
        H = f.attrs["_ParamAtmos__alt"]
    if L0 is None:
        L0 = f.attrs["_ParamAtmos__L0"]
    if speed is None:
        speed = f.attrs["_ParamAtmos__windspeed"]
    if theta is None:
        theta = (f.attrs["_ParamAtmos__winddir"] * np.pi / 180.)
    frac = f.attrs["_ParamAtmos__frac"]

    Htheta = np.linalg.norm([wxpos, wypos]) / RASC * H
    vdt = speed * dt / gain
    angleht = np.arctan2(wypos, wxpos)
    fc = 1 / (2 * (xactu[1] - xactu[0]))
    scale = (1 / r0)**(5 / 3.) * frac * (Lambda_tar / (2 * np.pi))**2
    Nact = f["Nact"][:]
    Nact = np.linalg.inv(Nact)
    P = f["P"][:]
    Btt = f["Btt"][:]
    Tf = Btt[:-2, :-2].dot(P[:-2, :-2])
    IF, T = drax.get_IF(filename)
    IF = IF.T
    T = T.T
    N = IF.shape[0]
    deltaTT = T.T.dot(T) / N
    deltaF = IF.T.dot(T) / N
    pzt2tt = np.linalg.inv(deltaTT).dot(deltaF.T)

    if (ctype == "float"):
        groot = Groot(cxt, cxt.active_device, Nact.shape[0],
                      int(f.attrs["_ParamAtmos__nscreens"]), angleht,
                      vdt.astype(np.float32), Htheta.astype(np.float32), L0, theta,
                      scale.astype(np.float32), pzt2tt.astype(np.float32),
                      Tf.astype(np.float32), Nact.astype(np.float32),
                      xactu.astype(np.float32), yactu.astype(np.float32), fc)
    else:
        raise TypeError("Unknown ctype : must be float")
    tic = time.time()
    groot.compute_Cerr()
    Cerr = np.array(groot.d_Cerr)
    cov_err_groot = np.zeros((Nact.shape[0] + 2, Nact.shape[0] + 2))
    cov_err_groot[:-2, :-2] = Cerr
    cov_err_groot[-2:, -2:] = np.array(groot.d_TT)
    tac = time.time()
    print("Cee computed in : %.2f seconds" % (tac - tic))
    if (modal):
        cov_err_groot = P.dot(cov_err_groot).dot(P.T)

    f.close()
    return cov_err_groot


def compute_Cerr_cpu(filename, modal=True):
    """ Returns the residual error covariance matrix using CPU version of GROOT
    from a ROKET file
    :parameter:
        filename : (string) : full path to the ROKET file
        modal : (bool) : if True (default), Cerr is returned in the Btt modal basis,
                         in the actuator basis if False
    :return:
        Cerr : (np.ndarray(dim=2, dtype=np.float32)) : residual error covariance matrix
    """
    f = h5py.File(filename, 'r')

    tabx, taby = starlord.tabulateIj0()
    Lambda_tar = f.attrs["_ParamTarget__Lambda"][0]
    Lambda_wfs = f.attrs["_ParamWfs__Lambda"]
    dt = f.attrs["_ParamLoop__ittime"]
    gain = f.attrs["_ParamController__gain"]
    wxpos = f.attrs["_ParamWfs__xpos"][0]
    wypos = f.attrs["_ParamWfs__ypos"][0]
    r0 = f.attrs["_ParamAtmos__r0"] * (Lambda_tar / Lambda_wfs)**(6. / 5.)
    RASC = 180. / np.pi * 3600.
    xpos = f["dm.xpos"][:]
    ypos = f["dm.ypos"][:]
    p2m = f.attrs["_ParamTel__diam"] / f.attrs["_ParamGeom__pupdiam"]
    pupshape = int(2**np.ceil(np.log2(f.attrs["_ParamGeom__pupdiam"]) + 1))
    xactu = (xpos - pupshape / 2) * p2m
    yactu = (ypos - pupshape / 2) * p2m
    Ccov = np.zeros((xpos.size, xpos.size))
    Caniso = np.zeros((xpos.size, xpos.size))
    Cbp = np.zeros((xpos.size, xpos.size))
    xx = np.tile(xactu, (xactu.shape[0], 1))
    yy = np.tile(yactu, (yactu.shape[0], 1))
    xij = xx - xx.T
    yij = yy - yy.T

    for atm_layer in range(f.attrs["_ParamAtmos__nscreens"]):
        H = f.attrs["_ParamAtmos__alt"][atm_layer]
        L0 = f.attrs["_ParamAtmos__L0"][atm_layer]
        speed = f.attrs["_ParamAtmos__windspeed"][atm_layer]
        theta = f.attrs["_ParamAtmos__winddir"][atm_layer] * np.pi / 180.
        frac = f.attrs["_ParamAtmos__frac"][atm_layer]

        Htheta = np.linalg.norm([wxpos, wypos]) / RASC * H
        vdt = speed * dt / gain
        # Covariance matrices models on actuators space
        M = np.zeros((xpos.size, xpos.size))
        Mvdt = M.copy()
        Mht = M.copy()
        Mhvdt = M.copy()
        angleht = np.arctan2(wypos, wxpos)
        fc = xactu[1] - xactu[0]

        M = np.linalg.norm([xij, yij], axis=0)
        Mvdt = np.linalg.norm([xij - vdt * np.cos(theta), yij - vdt * np.sin(theta)],
                              axis=0)
        Mht = np.linalg.norm(
                [xij - Htheta * np.cos(angleht), yij - Htheta * np.sin(angleht)], axis=0)
        Mhvdt = np.linalg.norm([
                xij - vdt * np.cos(theta) - Htheta * np.cos(angleht),
                yij - vdt * np.sin(theta) - Htheta * np.sin(angleht)
        ], axis=0)

        Ccov += 0.5 * (starlord.dphi_lowpass(Mhvdt, fc, L0, tabx, taby) -
                       starlord.dphi_lowpass(Mht, fc, L0, tabx, taby) - starlord.
                       dphi_lowpass(Mvdt, fc, L0, tabx, taby) + starlord.dphi_lowpass(
                               M, fc, L0, tabx, taby)) * (1. / r0)**(5. / 3.) * frac

        Caniso += 0.5 * (
                starlord.dphi_lowpass(Mht, fc, L0, tabx, taby) - starlord.dphi_lowpass(
                        M, fc, L0, tabx, taby)) * (1. / r0)**(5. / 3.) * frac
        Cbp += 0.5 * (starlord.dphi_lowpass(Mvdt, fc, L0, tabx, taby) - starlord.
                      dphi_lowpass(M, fc, L0, tabx, taby)) * (1. / r0)**(5. / 3.) * frac

    Sp = (Lambda_tar / (2 * np.pi))**2
    Ctt = (Caniso + Caniso.T) * Sp
    Ctt += ((Cbp + Cbp.T) * Sp)
    Ctt += ((Ccov + Ccov.T) * Sp)

    P = f["P"][:]
    Btt = f["Btt"][:]
    Tf = Btt[:-2, :-2].dot(P[:-2, :-2])

    IF, T = drax.get_IF(filename)
    IF = IF.T
    T = T.T
    N = IF.shape[0]
    deltaTT = T.T.dot(T) / N
    deltaF = IF.T.dot(T) / N
    pzt2tt = np.linalg.inv(deltaTT).dot(deltaF.T)

    Nact = f["Nact"][:]
    N1 = np.linalg.inv(Nact)
    Ctt = N1.dot(Ctt).dot(N1)
    ttcomp = pzt2tt.dot(Ctt).dot(pzt2tt.T)
    Ctt = Tf.dot(Ctt).dot(Tf.T)
    cov_err = np.zeros((Ctt.shape[0] + 2, Ctt.shape[0] + 2))
    cov_err[:-2, :-2] = Ctt
    cov_err[-2:, -2:] = ttcomp
    if (modal):
        cov_err = P.dot(cov_err).dot(P.T)
    f.close()

    return cov_err


def test_Cerr(filename):
    """ Compute PSF of aniso and bandwidth from GROOT model and ROKET to compare

    Args:
        filename:(str):path to the ROKET file
    """
    C = drax.get_covmat_contrib(filename, ["bandwidth", "tomography"])
    Cerr = compute_Cerr(filename)
    _, _, psfr, _ = gamora.psf_rec_Vii(filename, covmodes=C.astype(np.float32),
                                       fitting=False)
    _, _, psf, _ = gamora.psf_rec_Vii(filename, cov=Cerr.astype(np.float32),
                                      fitting=False)
    drax.cutsPSF(filename, psfr, psf)
    print("PSFR SR: ", psfr.max())
    print("PSF SR: ", psf.max())
    psf = drax.ensquare_PSF(filename, psf, 20)
    psfr = drax.ensquare_PSF(filename, psfr, 20)
    plt.matshow(np.log10(np.abs(psfr)))
    plt.colorbar()
    plt.title("PSF_R")
    plt.matshow(
            np.log10(np.abs(psf)), vmax=np.log10(np.abs(psfr)).max(),
            vmin=np.log10(np.abs(psfr)).min())
    plt.colorbar()
    plt.title("PSF")
    plt.matshow(
            np.log10(np.abs(psfr - psf)), vmax=np.log10(np.abs(psfr)).max(),
            vmin=np.log10(np.abs(psfr)).min())
    plt.colorbar()
    plt.title("PSF_R - PSF")

    return psf, psfr


def compare_GPU_vs_CPU(filename):
    """ Compare results of GROOT vs its CPU version in terms of execution time
    and precision on the PSF renconstruction
    :parameter:
        filename : (string) : full path to the ROKET file

    """
    from carmaWrap import timer as carmaWrap_timer
    timer = carmaWrap_timer()

    timer.start()
    timer.stop()
    synctime = timer.total_time
    timer.reset()

    timer.start()
    cov_err_gpu_s = compute_Cerr(filename)
    timer.stop()
    gpu_time_s = timer.total_time - synctime
    timer.reset()

    timer.start()
    cov_err_gpu_d = compute_Cerr(filename, ctype="double")
    timer.stop()
    gpu_time_d = timer.total_time - synctime
    timer.reset()

    tic = time.time()
    cov_err_cpu = compute_Cerr_cpu(filename)
    tac = time.time()
    cpu_time = tac - tic

    otftel, otf2, psf_cpu, gpu = gamora.psf_rec_Vii(filename, fitting=False,
                                                    cov=cov_err_cpu.astype(np.float32))
    otftel, otf2, psf_gpu_s, gpu = gamora.psf_rec_Vii(
            filename, fitting=False, cov=cov_err_gpu_s.astype(np.float32))
    otftel, otf2, psf_gpu_d, gpu = gamora.psf_rec_Vii(
            filename, fitting=False, cov=cov_err_gpu_d.astype(np.float32))

    print("-----------------------------------------")
    print("CPU time : ", cpu_time, " s ")
    print("GPU time simple precision : ", gpu_time_s, " s ")
    print("GPU time double precision : ", gpu_time_d, " s ")
    print("Max absolute difference in PSFs simple precision : ",
          np.abs(psf_cpu - psf_gpu_s).max())
    print("Max absolute difference in PSFs double precision : ",
          np.abs(psf_cpu - psf_gpu_d).max())
    gamora.cutsPSF(filename, psf_cpu, psf_gpu_s)
    gamora.cutsPSF(filename, psf_cpu, psf_gpu_d)


def compute_Ca_cpu(filename, modal=True):
    """ Returns the aliasing error covariance matrix using CPU version of GROOT
    from a ROKET file
    :parameter:
        filename : (string) : full path to the ROKET file
        modal : (bool) : if True (default), Ca is returned in the Btt modal basis,
                         in the actuator basis if False
    :return:
        Ca : (np.ndarray(dim=2, dtype=np.float32)) : aliasing error covariance matrix
    """
    f = h5py.File(filename, 'r')
    nsub = f["R"][:].shape[1] // 2
    nssp = f.attrs["_ParamWfs__nxsub"][0]
    validint = f.attrs["_ParamTel__cobs"]
    x = np.linspace(-1, 1, nssp)
    x, y = np.meshgrid(x, x)
    r = np.sqrt(x * x + y * y)
    rorder = np.sort(r.reshape(nssp * nssp))
    ncentral = nssp * nssp - np.sum(r >= validint)
    validext = rorder[ncentral + nsub]
    valid = (r < validext) & (r >= validint)
    ivalid = np.where(valid)
    xvalid = ivalid[0] + 1
    yvalid = ivalid[1] + 1
    ivalid = (xvalid, yvalid)
    d = f.attrs["_ParamTel__diam"] / (f.attrs["_ParamDm__nact"][0] - 1)
    r0 = f.attrs["_ParamAtmos__r0"] * (f.attrs["_ParamTarget__Lambda"] / 0.5)**(
            6. / 5.)
    RASC = 180 / np.pi * 3600.

    scale = 0.23 * (d / r0)**(5 / 3.) * \
        (f.attrs["_ParamTarget__Lambda"] * 1e-6 / (2 * np.pi * d))**2 * RASC**2

    mask = np.zeros((nssp + 2, nssp + 2))
    Ca = np.identity(nsub * 2)

    for k in range(nsub):
        mask *= 0
        mask[xvalid[k], yvalid[k]] = 1
        mask[xvalid[k], yvalid[k] - 1] = -0.5
        mask[xvalid[k], yvalid[k] + 1] = -0.5
        Ca[k, :nsub] = mask[ivalid].flatten()
        mask *= 0
        mask[xvalid[k], yvalid[k]] = 1
        mask[xvalid[k] - 1, yvalid[k]] = -0.5
        mask[xvalid[k] + 1, yvalid[k]] = -0.5
        Ca[k + nsub, nsub:] = mask[ivalid].flatten()

    R = f["R"][:]
    Ca = R.dot(Ca * scale).dot(R.T)
    if (modal):
        P = f["P"][:]
        Ca = P.dot(Ca).dot(P.T)
    f.close()
    return Ca


def compute_Cn_cpu(filename, model="data", modal=True):
    """ Returns the noise error covariance matrix using CPU version of GROOT
    from a ROKET file
    :parameter:
        filename : (string) : full path to the ROKET file
        modal : (bool) : if True (default), Cn is returned in the Btt modal basis,
                         in the actuator basis if False
    :return:
        Cn : (np.ndarray(dim=2, dtype=np.float32)) : noise error covariance matrix
    """
    f = h5py.File(filename, 'r')
    if (model == "data"):
        N = f["noise"][:]
        Cn = N.dot(N.T) / N.shape[1]
        if modal:
            P = f["P"][:]
            Cn = P.dot(Cn).dot(P.T)
    else:
        nslopes = f["R"][:].shape[1]
        Cn = np.zeros(nslopes)
        noise = f.attrs["_ParamWfs__noise"][0]
        RASC = 180 / np.pi * 3600.
        if (noise >= 0):
            Nph = f.attrs["_ParamWfs__zerop"] * 10 ** (-0.4 * f.attrs["_ParamWfs__gsmag"]) * \
                f.attrs["_ParamWfs__optthroughput"] * \
                (f.attrs["_ParamTel__diam"] / f.attrs["_ParamWfs__nxsub"]
                 ) ** 2. * f.attrs["_ParamLoop__ittime"]

            r0 = (f.attrs["_ParamWfs__Lambda"] / 0.5)**(
                    6.0 / 5.0) * f.attrs["_ParamAtmos__r0"]

            sig = (np.pi ** 2 / 2) * (1 / Nph) * \
                (1. / r0) ** 2  # Photon noise in m^-2
            # Noise variance in arcsec^2
            sig = sig * (
                    (f.attrs["_ParamWfs__Lambda"] * 1e-6) / (2 * np.pi))**2 * RASC**2

            Ns = f.attrs["_ParamWfs__npix"]  # Number of pixel
            Nd = (f.attrs["_ParamWfs__Lambda"] *
                  1e-6) * RASC / f.attrs["_ParamWfs__pixsize"]
            sigphi = (np.pi ** 2 / 3.0) * (1 / Nph ** 2) * (f.attrs["_ParamWfs__noise"]) ** 2 * \
                Ns ** 2 * (Ns / Nd) ** 2  # Phase variance in m^-2
            # Noise variance in arcsec^2
            sigsh = sigphi * \
                ((f.attrs["_ParamWfs__Lambda"] * 1e-6) / (2 * np.pi)) ** 2 * RASC ** 2

            Cn[:len(sig)] = sig + sigsh
            Cn[len(sig):] = sig + sigsh

        Cn = np.diag(Cn)
        R = f["R"][:]
        Cn = R.dot(Cn).dot(R.T)
        if (modal):
            P = f["P"][:]
            Cn = P.dot(Cn).dot(P.T)
    f.close()
    return Cn


def compute_OTF_fitting(filename, otftel):
    """
    Modelize the OTF due to the fitting using dphi_highpass

    Args:
        filename: (str) : ROKET hdf5 file path
        otftel: (np.ndarray) : Telescope OTF
    :return:
        otf_fit: (np.ndarray) : Fitting OTF
        psf_fit (np.ndarray) : Fitting PSF
    """
    f = h5py.File(filename, 'r')
    r0 = f.attrs["_ParamAtmos__r0"] * (f.attrs["_ParamTarget__Lambda"][0] / 0.5)**(
            6. / 5.)
    # ratio_lambda = 2 * np.pi / f.attrs["_ParamTarget__Lambda"][0]
    # Telescope OTF
    spup = drax.get_pup(filename)
    mradix = 2
    fft_size = mradix**int((np.log(2 * spup.shape[0]) / np.log(mradix)) + 1)
    mask = np.ones((fft_size, fft_size))
    mask[np.where(otftel < 1e-5)] = 0

    x = np.arange(fft_size) - fft_size / 2
    pixsize = f.attrs["_ParamTel__diam"] / f.attrs["_ParamGeom__pupdiam"]
    x = x * pixsize
    r = np.sqrt(x[:, None] * x[:, None] + x[None, :] * x[None, :])
    tabx, taby = starlord.tabulateIj0()
    dphi = np.fft.fftshift(
            starlord.dphi_highpass(
                    r, f.attrs["_ParamTel__diam"] / (f.attrs["_ParamDm__nact"][0] - 1),
                    tabx, taby) * (1 / r0)**(5 / 3.))  # * den * ratio_lambda**2 * mask
    otf_fit = np.exp(-0.5 * dphi) * mask
    otf_fit = otf_fit / otf_fit.max()

    psf_fit = np.fft.fftshift(np.real(np.fft.ifft2(otftel * otf_fit)))
    psf_fit *= (fft_size * fft_size / float(np.where(spup)[0].shape[0]))

    f.close()
    return otf_fit, psf_fit


def compute_PSF(filename):
    """
    Modelize the PSF using GROOT model for aniso and bandwidth, Gendron model for aliasing,
    dphi_highpass for fitting, noise extracted from datas. Non linearity not taken into account

    Args:
        filename: (str) : ROKET hdf5 file path
    :return:
        psf: (np.ndarray) : PSF
    """
    tic = time.time()
    spup = drax.get_pup(filename)
    Cab = compute_Cerr(filename)
    Cn = compute_Cn_cpu(filename)
    Ca = compute_Calias(filename)
    Cee = Cab + Cn + Ca
    otftel, otf2, psf, gpu = gamora.psf_rec_Vii(filename, fitting=False,
                                                cov=(Cee).astype(np.float32))
    otf_fit, psf_fit = compute_OTF_fitting(filename, otftel)
    psf = np.fft.fftshift(np.real(np.fft.ifft2(otf_fit * otf2 * otftel)))
    psf *= (psf.shape[0] * psf.shape[0] / float(np.where(spup)[0].shape[0]))
    tac = time.time()
    print("PSF computed in ", tac - tic, " seconds")

    return psf


def compute_Calias_gpu(filename, slopes_space=False, modal=True, npts=3):
    f = h5py.File(filename, 'r')
    nsub = f["R"][:].shape[1] // 2
    nssp = f.attrs["_ParamWfs__nxsub"][0]
    # npix = f.attrs["_ParamWfs__npix"][0]
    validint = f.attrs["_ParamTel__cobs"]
    x = np.linspace(-1, 1, nssp)
    x, y = np.meshgrid(x, x)
    r = np.sqrt(x * x + y * y)
    rorder = np.sort(r.reshape(nssp * nssp))
    ncentral = nssp * nssp - np.sum(r >= validint, dtype=np.int32)
    validext = rorder[ncentral + nsub]
    valid = (r < validext) & (r >= validint)
    ivalid = np.where(valid)
    r0 = f.attrs["_ParamAtmos__r0"]
    Lambda_wfs = f.attrs["_ParamWfs__Lambda"][0]
    d = f.attrs["_ParamTel__diam"] / nssp
    RASC = 180 / np.pi * 3600
    scale = 0.5 * (1 / r0)**(5 / 3)
    c = (RASC * Lambda_wfs * 1e-6 / 2 / np.pi) / d**2
    h = d / (npts - 1)
    x = (np.arange(nssp) - nssp / 2) * d
    x, y = np.meshgrid(x, x)
    x = x[ivalid].astype(np.float32)
    y = y[ivalid].astype(np.float32)
    fc = 1 / (2 * d)  #/ npix
    scale = scale * c**2 * (h / 3)**2
    coeff = simpson_coeff(npts)
    weights = np.zeros(npts)
    for k in range(npts):
        weights[k] = (coeff[k:] * coeff[:npts - k]).sum()
    groot = Groot(cxt, cxt.active_device, nsub, weights.astype(np.float32), scale, x, y,
                  fc, d, npts)
    groot.compute_Calias()
    CaXX = np.array(groot.d_CaXX)
    Ca = np.zeros((2 * CaXX.shape[0], 2 * CaXX.shape[0]))
    Ca[:CaXX.shape[0], :CaXX.shape[0]] = CaXX
    Ca[CaXX.shape[0]:, CaXX.shape[0]:] = np.array(groot.d_CaYY)
    if not slopes_space:
        R = f["R"][:]
        Ca = R.dot(Ca).dot(R.T)
        if modal:
            P = f["P"][:]
            Ca = P.dot(Ca).dot(P.T)
    f.close()

    return Ca


def compute_Calias(filename, slopes_space=False, modal=True, npts=3):
    """ Returns the aliasing slopes covariance matrix using CPU version of GROOT
    from a ROKET file and a model based on structure function
    :parameter:
        filename : (string) : full path to the ROKET file
        slopes_space: (bool): (optionnal) if True, return the covariance matrix in the slopes space
        modal: (bool): (optionnal) if True, return the covariance matrix in the modal space
    :return:
        Ca : (np.ndarray(dim=2, dtype=np.float32)) : aliasing error covariance matrix
    """

    f = h5py.File(filename, 'r')
    tabx, taby = starlord.tabulateIj0()
    nsub = f["R"][:].shape[1] // 2
    nssp = f.attrs["_ParamWfs__nxsub"][0]
    # npix = f.attrs["_ParamWfs__npix"][0]
    validint = f.attrs["_ParamTel__cobs"]
    x = np.linspace(-1, 1, nssp)
    x, y = np.meshgrid(x, x)
    r = np.sqrt(x * x + y * y)
    rorder = np.sort(r.reshape(nssp * nssp))
    ncentral = nssp * nssp - np.sum(r >= validint, dtype=np.int32)
    validext = rorder[ncentral + nsub]
    valid = (r < validext) & (r >= validint)
    ivalid = np.where(valid)
    r0 = f.attrs["_ParamAtmos__r0"]
    Lambda_wfs = f.attrs["_ParamWfs__Lambda"][0]
    d = f.attrs["_ParamTel__diam"] / nssp
    RASC = 180 / np.pi * 3600
    scale = 0.5 * (1 / r0)**(5 / 3)
    c = (RASC * Lambda_wfs * 1e-6 / 2 / np.pi) / d**2
    x = (np.arange(nssp) - nssp / 2) * d
    x, y = np.meshgrid(x, x)
    x = x[ivalid]
    y = y[ivalid]
    fc = d  #/ npix
    xx = np.tile(x, (nsub, 1))
    yy = np.tile(y, (nsub, 1))
    # Ca = compute_Calias_element(xx, yy, fc, d, nsub, tabx, taby)
    # Ca += compute_Calias_element(xx, yy, fc, d, nsub, tabx, taby, xoff=0.5)
    # Ca += compute_Calias_element(xx, yy, fc, d, nsub, tabx, taby, xoff=-0.5)
    # Ca += compute_Calias_element(xx, yy, fc, d, nsub, tabx, taby, yoff=0.5)
    # Ca += compute_Calias_element(xx, yy, fc, d, nsub, tabx, taby, yoff=-0.5)
    # Ca = Ca * scale / 5
    Ca = np.zeros((2 * nsub, 2 * nsub))
    coeff = simpson_coeff(npts)
    # for k in track(range(npts)):
    #     weight = (coeff[k:] * coeff[:npts - k]).sum()
    #     Ca += compute_Calias_element_XX(xx, yy, fc, d, nsub, tabx, taby, yoff=k /
    #                                     (npts - 1)) * weight
    #     Ca += compute_Calias_element_YY(xx, yy, fc, d, nsub, tabx, taby, xoff=k /
    #                                     (npts - 1)) * weight
    #     if k > 0:
    #         Ca += compute_Calias_element_XX(xx, yy, fc, d, nsub, tabx, taby, yoff=-k /
    #                                         (npts - 1)) * weight
    #         Ca += compute_Calias_element_YY(xx, yy, fc, d, nsub, tabx, taby, xoff=-k /
    #                                         (npts - 1)) * weight
    if (npts > 1):
        h = d / (npts - 1)
    else:
        h = 1
    for k in track(range(npts)):
        for p in track(range(npts)):
            Ca += (compute_Calias_element_XX(xx, yy, fc, d, nsub, tabx, taby,
                                             yoff=(k - p) * h) * coeff[k] * coeff[p])
            Ca += (compute_Calias_element_YY(xx, yy, fc, d, nsub, tabx, taby,
                                             xoff=(k - p) * h) * coeff[k] * coeff[p])

    if not slopes_space:
        R = f["R"][:]
        Ca = R.dot(Ca).dot(R.T)
        if modal:
            P = f["P"][:]
            Ca = P.dot(Ca).dot(P.T)
    f.close()

    return Ca * scale * c**2 * (h / 3)**2


def simpson_coeff(n):
    """
    Returns the n weights to apply for a Simpson integration on n elements
    Args:
        n: (int): number of elements, must be odd
    :return:
        coeff: (np.array[ndims=1,dtype=np.int64]): simpson coefficients
    """
    if (n == 1):
        coeff = np.ones(n)
    else:
        if (n % 2):
            coeff = np.ones(n)
            coeff[1::2] = 4
            coeff[2:-1:2] = 2
        else:
            raise ValueError("n must be odd")

    return coeff


def compute_Calias_element_XX(xx, yy, fc, d, nsub, tabx, taby, xoff=0, yoff=0):
    """
        Compute the element of the aliasing covariance matrix

    Args:
        Ca: (np.ndarray(ndim=2, dtype=np.float32)): aliasing covariance matrix to fill
        xx: (np.ndarray(ndim=2, dtype=np.float32)): X positions of the WFS subap
        yy: (np.ndarray(ndim=2, dtype=np.float32)): Y positions of the WFS subap
        fc: (float): cut-off frequency for structure function
        d: (float): subap diameter
        nsub: (int): number of subap
        tabx: (np.ndarray(ndim=1, dtype=np.float32)): X tabulation for dphi
        taby: (np.ndarray(ndim=1, dtype=np.float32)): Y tabulation for dphi
        xoff: (float) : (optionnal) offset to apply on the WFS xpos (units of d)
        yoff: (float) : (optionnal) offset to apply on the WFS ypos (units of d)
    """
    xx = xx - xx.T  #+ xoff * d
    yy = yy - yy.T  #+ yoff * d
    #xx = np.triu(xx) - np.triu(xx, -1).T
    #yy = np.triu(yy) - np.triu(yy, -1).T
    Ca = np.zeros((2 * nsub, 2 * nsub))

    # XX covariance
    AB = np.linalg.norm([xx, yy + yoff], axis=0)
    Ab = np.linalg.norm([xx - d, yy + yoff], axis=0)
    aB = np.linalg.norm([xx + d, yy + yoff], axis=0)
    # ab = AB

    Ca[:nsub, :nsub] += starlord.dphi_highpass(
            Ab, fc, tabx, taby) + starlord.dphi_highpass(
                    aB, fc, tabx, taby) - 2 * starlord.dphi_highpass(AB, fc, tabx, taby)

    return Ca


def compute_Calias_element_YY(xx, yy, fc, d, nsub, tabx, taby, xoff=0, yoff=0):
    """
        Compute the element of the aliasing covariance matrix

    Args:
        Ca: (np.ndarray(ndim=2, dtype=np.float32)): aliasing covariance matrix to fill
        xx: (np.ndarray(ndim=2, dtype=np.float32)): X positions of the WFS subap
        yy: (np.ndarray(ndim=2, dtype=np.float32)): Y positions of the WFS subap
        fc: (float): cut-off frequency for structure function
        d: (float): subap diameter
        nsub: (int): number of subap
        tabx: (np.ndarray(ndim=1, dtype=np.float32)): X tabulation for dphi
        taby: (np.ndarray(ndim=1, dtype=np.float32)): Y tabulation for dphi
        xoff: (float) : (optionnal) offset to apply on the WFS xpos (units of d)
        yoff: (float) : (optionnal) offset to apply on the WFS ypos (units of d)
    """
    xx = xx - xx.T  #+ xoff * d
    yy = yy - yy.T  #+ yoff * d
    #xx = np.triu(xx) - np.triu(xx, -1).T
    #yy = np.triu(yy) - np.triu(yy, -1).T
    Ca = np.zeros((2 * nsub, 2 * nsub))

    # YY covariance
    CD = np.linalg.norm([xx + xoff, yy], axis=0)
    Cd = np.linalg.norm([xx + xoff, yy - d], axis=0)
    cD = np.linalg.norm([xx + xoff, yy + d], axis=0)
    # cd = CD

    Ca[nsub:, nsub:] += starlord.dphi_highpass(
            Cd, fc, tabx, taby) + starlord.dphi_highpass(
                    cD, fc, tabx, taby) - 2 * starlord.dphi_highpass(CD, fc, tabx, taby)

    return Ca


def compute_Calias_element_XY(xx, yy, fc, d, nsub, tabx, taby, xoff=0, yoff=0):
    """
        Compute the element of the aliasing covariance matrix

    Args:
        Ca: (np.ndarray(ndim=2, dtype=np.float32)): aliasing covariance matrix to fill
        xx: (np.ndarray(ndim=2, dtype=np.float32)): X positions of the WFS subap
        yy: (np.ndarray(ndim=2, dtype=np.float32)): Y positions of the WFS subap
        fc: (float): cut-off frequency for struture function
        d: (float): subap diameter
        nsub: (int): number of subap
        tabx: (np.ndarray(ndim=1, dtype=np.float32)): X tabulation for dphi
        taby: (np.ndarray(ndim=1, dtype=np.float32)): Y tabulation for dphi
        xoff: (float) : (optionnal) offset to apply on the WFS xpos (units of d)
        yoff: (float) : (optionnal) offset to apply on the WFS ypos (units of d)
    """
    xx = xx - xx.T + xoff * d
    yy = yy - yy.T + yoff * d
    Ca = np.zeros((2 * nsub, 2 * nsub))

    # YY covariance
    aD = np.linalg.norm([xx + d / 2, yy + d / 2], axis=0)
    ad = np.linalg.norm([xx + d / 2, yy - d / 2], axis=0)
    Ad = np.linalg.norm([xx - d / 2, yy - d / 2], axis=0)
    AD = np.linalg.norm([xx - d / 2, yy + d / 2], axis=0)

    Ca[nsub:, :nsub] = 0.25 * (
            starlord.dphi_highpass(Ad, d, tabx, taby) + starlord.dphi_highpass(
                    aD, d, tabx, taby) - starlord.dphi_highpass(AD, d, tabx, taby) -
            starlord.dphi_highpass(ad, d, tabx, taby))
    Ca[:nsub, nsub:] = Ca[nsub:, :nsub].copy()
    return Ca


def compute_Calias_element(xx, yy, fc, d, nsub, tabx, taby, xoff=0, yoff=0):
    """
        Compute the element of the aliasing covariance matrix

    Args:
        Ca: (np.ndarray(ndim=2, dtype=np.float32)): aliasing covariance matrix to fill
        xx: (np.ndarray(ndim=2, dtype=np.float32)): X positions of the WFS subap
        yy: (np.ndarray(ndim=2, dtype=np.float32)): Y positions of the WFS subap
        fc: (float): cut-off frequency for structure function
        d: (float): subap diameter
        nsub: (int): number of subap
        tabx: (np.ndarray(ndim=1, dtype=np.float32)): X tabulation for dphi
        taby: (np.ndarray(ndim=1, dtype=np.float32)): Y tabulation for dphi
        xoff: (float) : (optionnal) offset to apply on the WFS xpos (units of d)
        yoff: (float) : (optionnal) offset to apply on the WFS ypos (units of d)
    """
    xx = xx - xx.T + xoff * d
    yy = yy - yy.T + yoff * d
    Ca = np.zeros((2 * nsub, 2 * nsub))

    # XX covariance
    AB = np.linalg.norm([xx, yy], axis=0)
    Ab = np.linalg.norm([xx - d, yy], axis=0)
    aB = np.linalg.norm([xx + d, yy], axis=0)
    # ab = AB

    Ca[:nsub, :nsub] += starlord.dphi_highpass(
            Ab, fc, tabx, taby) + starlord.dphi_highpass(
                    aB, fc, tabx, taby) - 2 * starlord.dphi_highpass(AB, fc, tabx, taby)

    # YY covariance
    CD = AB
    Cd = np.linalg.norm([xx, yy - d], axis=0)
    cD = np.linalg.norm([xx, yy + d], axis=0)
    # cd = CD

    Ca[nsub:, nsub:] += starlord.dphi_highpass(
            Cd, fc, tabx, taby) + starlord.dphi_highpass(
                    cD, fc, tabx, taby) - 2 * starlord.dphi_highpass(CD, fc, tabx, taby)

    # XY covariance

    # aD = np.linalg.norm([xx + d/2, yy + d/2], axis=0)
    # ad = np.linalg.norm([xx + d/2, yy - d/2], axis=0)
    # Ad = np.linalg.norm([xx - d/2, yy - d/2], axis=0)
    # AD = np.linalg.norm([xx - d/2, yy + d/2], axis=0)
    #
    # Ca[nsub:,:nsub] = 0.25 * (starlord.dphi_highpass(Ad, d, tabx, taby)
    #                 + starlord.dphi_highpass(aD, d, tabx, taby)
    #                 - starlord.dphi_highpass(AD, d, tabx, taby)
    #                 - starlord.dphi_highpass(ad, d, tabx, taby)) * (1 / r0)**(5. / 3.)
    # Ca[:nsub,nsub:] = Ca[nsub:,:nsub].copy()
    return Ca


def compute_dCmm(filename, ws=None, wd=None, dk=1):
    """ Returns the derivative slopes covariance matrix using CPU version of GROOT
    from a ROKET file and a model based on structure function
    :parameter:
        filename : (string) : full path to the ROKET file
        ws: (np.array[ndim=1, dtype=np.float32]): wind speed per layer [m/s]
        wd: (np.array[ndim=1, dtype=np.float32]): wind direction per layer [deg]
        dk: (int): slopes shift [iterations]
    :return:
        dCmm : (np.ndarray(dim=2, dtype=np.float32)) : d/dt(slopes)*slopes
    """

    f = h5py.File(filename, 'r')
    if ws is None:
        ws = f.attrs["_ParamAtmos__windspeed"]
    if wd is None:
        wd = f.attrs["_ParamAtmos__winddir"]
    dt = f.attrs["_ParamLoop__ittime"] * dk
    L0 = f.attrs["_ParamAtmos__L0"]
    frac = f.attrs["_ParamAtmos__frac"]
    nsub = f["R"][:].shape[1] // 2
    nssp = f.attrs["_ParamWfs__nxsub"][0]
    validint = f.attrs["_ParamTel__cobs"]
    x = np.linspace(-1, 1, nssp)
    x, y = np.meshgrid(x, x)
    r = np.sqrt(x * x + y * y)
    rorder = np.sort(r.reshape(nssp * nssp))
    ncentral = nssp * nssp - np.sum(r >= validint, dtype=np.int32)
    validext = rorder[ncentral + nsub]
    valid = (r < validext) & (r >= validint)
    ivalid = np.where(valid)
    r0 = f.attrs["_ParamAtmos__r0"]
    Lambda_wfs = f.attrs["_ParamWfs__Lambda"][0]
    d = f.attrs["_ParamTel__diam"] / nssp
    RASC = 180 / np.pi * 3600
    scale = 0.5 * (1 / r0)**(5 / 3) * (RASC * Lambda_wfs * 1e-6 / 2 / np.pi)**2 / d**2
    x = (np.arange(nssp) - nssp / 2) * d
    x, y = np.meshgrid(x, x)
    x = x[ivalid]
    y = y[ivalid]
    xx = np.tile(x, (nsub, 1))
    yy = np.tile(y, (nsub, 1))
    f.close()
    dCmm = np.zeros((2 * nsub, 2 * nsub))
    for k in range(ws.size):
        dCmm += frac[k] * compute_dCmm_element(xx, yy, d, nsub, ws[k], wd[k], dt, L0[k])

    return dCmm * scale


def compute_dCmm_element(xx, yy, d, nsub, ws, wd, dt, L0):
    """
        Compute the element of the derivative slopes covariance matrix

    Args:
        xx: (np.ndarray(ndim=2, dtype=np.float32)): X positions of the WFS subap
        yy: (np.ndarray(ndim=2, dtype=np.float32)): Y positions of the WFS subap
        d: (float): subap diameter
        nsub: (int): number of subap
        ws: (float): wind speed per layer [m/s]
        wd: (float): wind direction per layer [deg]
        dt: (float): iteration time [s]
        L0: (float): outer scale [m]
    """
    xij = xx - xx.T
    yij = yy - yy.T
    dCmm = np.zeros((2 * nsub, 2 * nsub))
    vdt = ws * dt
    wd = wd / 180 * np.pi

    # XX covariance
    AB = np.linalg.norm([-xij + vdt * np.cos(wd), -yij + vdt * np.sin(wd)], axis=0)
    Ab = np.linalg.norm([-xij - d + vdt * np.cos(wd), -yij + vdt * np.sin(wd)], axis=0)
    aB = np.linalg.norm([-xij + d + vdt * np.cos(wd), -yij + vdt * np.sin(wd)], axis=0)

    dCmm[:nsub, :nsub] += starlord.rodconan(Ab, L0) + starlord.rodconan(
            aB, L0) - 2 * starlord.rodconan(AB, L0)

    AB = np.linalg.norm([xij + vdt * np.cos(wd), yij + vdt * np.sin(wd)], axis=0)
    Ab = np.linalg.norm([xij - d + vdt * np.cos(wd), yij + vdt * np.sin(wd)], axis=0)
    aB = np.linalg.norm([xij + d + vdt * np.cos(wd), yij + vdt * np.sin(wd)], axis=0)

    dCmm[:nsub, :nsub] -= (starlord.rodconan(Ab, L0) + starlord.rodconan(aB, L0) -
                           2 * starlord.rodconan(AB, L0))

    # YY covariance
    CD = np.linalg.norm([-xij + vdt * np.cos(wd), -yij + vdt * np.sin(wd)], axis=0)
    Cd = np.linalg.norm([-xij + vdt * np.cos(wd), -yij - d + vdt * np.sin(wd)], axis=0)
    cD = np.linalg.norm([-xij + vdt * np.cos(wd), -yij + d + vdt * np.sin(wd)], axis=0)

    dCmm[nsub:, nsub:] += starlord.rodconan(Cd, L0) + starlord.rodconan(
            cD, L0) - 2 * starlord.rodconan(CD, L0)

    CD = np.linalg.norm([xij + vdt * np.cos(wd), yij + vdt * np.sin(wd)], axis=0)
    Cd = np.linalg.norm([xij + vdt * np.cos(wd), yij - d + vdt * np.sin(wd)], axis=0)
    cD = np.linalg.norm([xij + vdt * np.cos(wd), yij + d + vdt * np.sin(wd)], axis=0)

    dCmm[nsub:, nsub:] -= (starlord.rodconan(Cd, L0) + starlord.rodconan(cD, L0) -
                           2 * starlord.rodconan(CD, L0))
    # XY covariance

    # aD = np.linalg.norm([xx + d/2, yy + d/2], axis=0)
    # ad = np.linalg.norm([xx + d/2, yy - d/2], axis=0)
    # Ad = np.linalg.norm([xx - d/2, yy - d/2], axis=0)
    # AD = np.linalg.norm([xx - d/2, yy + d/2], axis=0)
    #
    # dCmm[nsub:,:nsub] = 0.25 * (starlord.rodconan(Ad, d, tabx, taby)
    #                 + starlord.dphi_highpass(aD, d, tabx, taby)
    #                 - starlord.dphi_highpass(AD, d, tabx, taby)
    #                 - starlord.dphi_highpass(ad, d, tabx, taby)) * (1 / r0)**(5. / 3.)
    # dCmm[:nsub,nsub:] = dCmm[nsub:,:nsub].copy()
    return 0.25 * dCmm
