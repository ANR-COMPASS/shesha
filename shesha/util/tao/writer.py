import numpy as np
import astropy.io.fits as fits
import shutil

#filtering
#for:
#mavis_ltao keep 1320 values
#mavis_mcao keep 5400 values
#mavis_moao keep 1320 values
#


def used_actu(xpos, ypos,*, Np=-1):
    """return the indices of the used actuators

    Args:
        xpos: (np.ndarray[ndim=1, dtype=np.int32]): (optional) actuator position in x

        ypos: (np.ndarray[ndim=1, dtype=np.int32]): (optional) actuator position in y

    Kwargs:
        Np: (int): (optional) number of actuators along the diameter
    """

    u = np.unique(xpos)
    pMin = u.min()
    u -= pMin
    if (Np > 0 and Np != u.size):
        raise ValueError("Number of actuator along the diameter unconsistent")
    else:
        Np = u.size
    #if(not np.testing.assert_array_almost_equal(np.arange(Np)*u[1],u,1)):
    #    print((np.arange(Np)*u[1]-u).max())
    #    raise ValueError("non uniform positions")
    X = (xpos - pMin) / u[1]
    Y = (ypos - pMin) / u[1]
    return (Y * Np + X).astype(np.int32)


def get_idx(p_dm, *, xpos=None, ypos=None):
    """return a correspondance between the covariance matrix indices and the covariance map indices

    Args:
        p_dm: (ParamDm): dm settings

    Kwargs:
        xpos: (np.ndarray[ndim=1, dtype=np.int32]): (optional) actuator position in x

        ypos: (np.ndarray[ndim=1, dtype=np.int32]): (optional) actuator position in y

    Returns:
        index_map :  (np.ndarray[ndim=1, dtype=np.int32]) : correspondance between the covariance matrix indices and the covariance map indices
    """

    if (xpos is not None and ypos is not None):
        csI = used_actu(xpos, ypos)
    else:
        csI = p_dm.csI

    Np = p_dm.nact
    # dm.csI: valid actuators
    xx = np.tile(np.arange(Np), (Np, 1)).flatten('C')[csI]
    xx = -np.tile(xx, (xx.size, 1))
    dx = xx - xx.T

    yy = np.tile(np.arange(Np), (Np, 1)).flatten('F')[csI]
    yy = -np.tile(yy, (yy.size, 1))
    dy = yy - yy.T
    #  // transformation des decalages en indice de tableau
    dx += Np
    dy += Np

    return dx.flatten("F") + (p_dm.nact * 2 - 1) * (dy.flatten("F") - 1)


def get_abs2fi(sup, *,dm=0):
    size = sup.config.p_geom.pupdiam
    N = 2**(int(np.log(2 * size) / np.log(2) + 1))  #SutraTarget:138,

    supportFi = np.zeros((N, N), dtype=np.float32)
    fi = sup.config.p_dms[0]._influ[:, :, 0] * 1e-6
    supportFi[:fi.shape[0], :fi.shape[1]] = fi

    abs2fi = np.abs(np.fft.fft2(supportFi.T))**2

    return abs2fi.T


def OTF_telescope(sup):
    """otf = OTF_telescope(fourier)

   Computes the OTF of the telescope, so that
   > fft(OTF_telescope()).re
   produces a PSF normalized with max(psf)=SR=1.0

    """
    # size = sup.config.p_geom.pupdiam
    # N = 2**(int(np.log(2 * size) / np.log(2) + 1))  #SutraTarget:138,
    # ud = sup.config.p_tel.diam / sup.config.p_geom.pupdiam
    # computation of pupil
    # x = ud / (sup.config.p_tel.diam / 2.) * (np.arange(N) + 1 -
    #                                          (N / 2 + 1))  # x exprime en rayon pupille
    # x2 = np.tile(x * x, (x.size, 1))
    # r = np.sqrt(x2 + x2.T)

    #pup=(r<=1.0)*1 * (r>sup.config.p_tel.cobs)*1
    pup = sup.config.p_geom._ipupil
    #  factor that will normalize the psf
    #  with PSF(0)=1.00 when diffraction-limited
    #  surface_pup_m2 = tomo.tel.diam^2*(1-tomo.tel.obs^2)*pi/4;
    # surface_pup_m2 = sup.config.p_tel.diam**2 * (
    #         1 - sup.config.p_tel.cobs**2) * np.pi / 4.
    # surface_pup_pix = surface_pup_m2 / ud**2
    # factnorm = surface_pup_pix**2
    #  compute FTO of telescope. Computing the psf using
    #  just fft(FTO).re produces a psf with max(psf)=SR
    #  In fact, FTOtel is normalized so that sum(FTOtel)=1.
    #  FTOtel = autocorrelation(pup) / factnorm;
    #FTOtel=autocorrelation(pup)/factnorm
    FTOtel = autocorrelation(pup) / np.sum(pup)**2
    return FTOtel


def get_subaps(sup, *, wfs="all"):
    """ Return the number of valid subaps (per wfs) as well as their position

    Args:
        sup : (CompassSupervisor) : current supervisor

    Kwargs:
        wfs : (str) : (optional), default "all" wfs used by tao ( among "all", "lgs", "ngs")

    """
    #X=(sup.config.p_wfss[0]._validpuppixx-mpup.shape[0]/2-1)*(sup.config.p_tel.diam/sup.config.p_wfss[0].nxsub/sup.config.p_wfss[0]._pdiam )
    nsubap = []
    X = []
    Y = []
    if (wfs == "ngs"):
        p_wfss = sup.config.p_wfs_ngs
    elif (wfs == "lgs"):
        p_wfss = sup.config.p_wfs_lgs + [sup.config.p_wfs_ngs[-1]]
    else:  # case all
        p_wfss = sup.config.p_wfs_lgs + sup.config.p_wfs_ngs
    for wfs in p_wfss:
        validX = wfs._validpuppixx
        validY = wfs._validpuppixy
        toMeter = (sup.config.p_tel.diam / wfs.nxsub / wfs._pdiam)
        validX = (validX - validX.max() / 2) * toMeter
        validY = (validY - validY.max() / 2) * toMeter
        X += list(validX)
        Y += list(validY)
        nsubap.append(len(validX))
    return nsubap, X, Y


def autocorrelation(a):
    """ computes the autocorrelation so that

    max(aa) == sum(a^2)

    Args:
        a: (np.ndarray[ndim=2, dtype=np.float32]): matrix to compute the autocorrelation on

    """
    if (a.ndim == 2):
        b = np.abs(np.fft.fft2(a))
        b = np.fft.ifft2(b * b).real * a.size
    elif (a.ndim == 1):
        b = np.abs(np.fft.fft(a))
        b = np.fft.ifft(b * b).real
    else:
        print("error: autocorrelation: expect dim 1 or 2")
        return
    n2 = a.size  # N*N
    b /= n2
    return b


def func_influ(x, y, x0):
    #/* DOCUMENT opd_metres = func_influ(x,y,x0)
    #
    #   The arguments <x>, <y> and <x0> must all be in the same units.
    #
    #   In the particular case where <x0> is set to x0=dm.x0, <x> and <y>
    #   must be expressed with no unit, they are variables defined over the
    #   unitary circle of R=1.
    #
    #   In any case, the function returns the influence function (=OPD,
    #   optical path difference) of the actuators expressed in METERS.
    #
    #
    #  // allez on va dire que ce sont des metres !
    return 1.e-6 * np.exp(-(x * x + y * y) / (2 * x0 * x0))


def generate_files(sup, *,path=".", single_file=False, dm_use_tt=False, wfs="all",
    lgs_filter_cst=0.1, tar=-1):
    """write inputs parameters

    sys-params.txt: contains the system parameters
     idx.fits     :
     otftel.fits  :
     abs2fi.fits  :
     subaps.fits  : number and position of the subapertures

    Args:
        sup : (CompassSupervisor) : current supervisor

    Kwargs:
        path : (str): (optional), default './' path where the files are written

        single_file : (bool): (optional), default=False write a single fits File

        wfs : (str) : (optional), default "all" wfs used by tao ( among "all", "lgs", "ngs")

        lgs_filter_cst : (float) : (optional) add constant to filter out lgs tt

        tar : (list) : (optional), default -1 index of the target
    """
    p_dm = sup.config.p_dms[0]
    if (p_dm.type == 'tt'):
        print("ERROR: first dm must not be a 'tip-tilt")
        return
    nact = p_dm.nact
    ntotact = p_dm._ntotact

    if (dm_use_tt):
        p_dm_tt = sup.config.p_dms[-1]
        if (p_dm_tt.type != 'tt'):
            print("ERROR: tip-tilt dm must be the last one")
            return
        ntotact += 2

    write_sys_param(sup, path=path, wfs=wfs, lgs_filter_cst=lgs_filter_cst, tar=tar)
    write_atm_param(sup, path=path)
    idx = get_idx(p_dm, xpos=p_dm._xpos, ypos=p_dm._ypos)
    otf = OTF_telescope(sup)
    abs2fi = get_abs2fi(sup)
    nsubaps, X, Y = get_subaps(sup, wfs=wfs)
    if (not single_file):
        hdu_idx = fits.PrimaryHDU(idx)
        hdu_idx.header["NACT"] = nact
        hdu_idx.header["NTOTACT"] = ntotact
        hdul = fits.HDUList([hdu_idx])
        hdul.writeto(path + "/idx.fits", overwrite=1)
        fits.writeto(path + "/otftel.fits", otf, overwrite=1)
        fits.writeto(path + "/abs2fi.fits", abs2fi, overwrite=1)

        hdu_prime = fits.PrimaryHDU(np.zeros(0))
        hdu_nsubap = fits.ImageHDU(nsubaps, name="NSUBAP")
        hdu_Xpos = fits.ImageHDU(X, name="XPOS")
        hdu_Ypos = fits.ImageHDU(Y, name="YPOS")
        hdul = fits.HDUList([hdu_prime, hdu_nsubap, hdu_Xpos, hdu_Ypos])
        hdul.writeto(path + "/subaps.fits", overwrite=1)
    else:
        hdu_prime = fits.PrimaryHDU(np.zeros(0))
        hdu_nsubap = fits.ImageHDU(nsubaps, name="NSUBAP")
        hdu_Xpos = fits.ImageHDU(X, name="XPOS")
        hdu_Ypos = fits.ImageHDU(Y, name="YPOS")
        hdu_idx = fits.ImageHDU(idx, name="IDX")
        hdu_idx.header["NACT"] = nact
        hdu_idx.header["NTOTACT"] = ntotact
        hdu_abs2fi = fits.ImageHDU(abs2fi, name="ABS2FI")
        hdu_otf = fits.ImageHDU(otf, name="OTF")
        hdul = fits.HDUList([
                hdu_prime, hdu_nsubap, hdu_Xpos, hdu_Ypos, hdu_idx, hdu_abs2fi, hdu_otf
        ])
        hdul.writeto(path + "/sys-inputs.fits", overwrite=1)


def to_str(a=""):
    """  transform a np.array into a string

    Kwargs:
        a : (np.ndarray[ndim=1, dtype=np.int32]) : input array
    """
    string = ""
    if (type(a) is np.ndarray):
        for i in range(a.size):
            string += str(a[i]) + " "
    if isinstance(a, list):
        for i in range(len(a)):
            string += str(a[i]) + " "
    else:
        string = str(a)

    return string

def write_sys_param(sup, path=".", wfs="all", lgs_filter_cst=0.1, tar=-1):
    """ Write a sysParam file for tao based on the compass configuration

    Args:
        sup : (CompassSupervisor) : current supervisor

    Kwargs:
        path : (str) : (optional), "./" path to the sysParam file

        wfs : (str) : (optional), default "all" wfs used by tao ( among "all", "lgs", "ngs")

        lgs_filter_cst : (float) : (optional) add constant to filter out lgs tt

        tar : (list) : (optional), default -1 index of the target
    """
    bdw = 3.3e-7
    lgs_depth = 5000.
    through_atm = 1.
    p_wfs_ngs = sup.config.p_wfs_ngs
    p_wfs_lgs = sup.config.p_wfs_lgs
    if (wfs == "ngs"):
        p_wfss = p_wfs_ngs
    elif (wfs == "lgs"):
        p_wfss = p_wfs_lgs + [p_wfs_ngs[-1]]
    else:  # case all
        p_wfss = p_wfs_lgs + p_wfs_ngs
    p_wfs_ts = sup.config.p_wfs_ts
    p_targets = sup.config.p_targets
    p_tel = sup.config.p_tel
    p_loop = sup.config.p_loop

    if (len(p_wfs_lgs) > 0):
        lgs_flux = p_wfs_lgs[0].lgsreturnperwatt * p_wfs_lgs[0].laserpower * p_wfs_lgs[
                0].optthroughput * 10**4
        lgs_pix_size = p_wfs_lgs[0].pixsize
        lambda_lgs = p_wfs_lgs[0].Lambda * 1e-6
        through_lgs = p_wfs_lgs[0].optthroughput
        spot_width = p_wfs_lgs[0].beamsize
        lgs_alt = p_wfs_lgs[0].gsalt
    else:
        lgs_flux = 7.e6
        lgs_pix_size = 0.7
        lambda_lgs = 5.89e-07
        through_lgs = 0.382
        spot_width = 0.8
        lgs_alt = 90000

    if (len(p_wfs_ts) > 0):
        ts_xpos = [w.xpos for w in p_wfs_ts]
        ts_ypos = [w.ypos for w in p_wfs_ts]
    else:
        ts_xpos = []
        ts_ypos = []

    f = open(path + "/sys-params.txt", "w")
    f.write("diam       : meter     : Telescope diameter\n")
    f.write(to_str(p_tel.diam))
    f.write("\nobs        : percent   : Central obscuration\n")
    f.write(to_str(p_tel.cobs))
    f.write("\ntFrame     : second    : frame rate\n")
    f.write(to_str(p_loop.ittime))
    f.write("\nnW         :           : number of WFS\n")
    f.write(to_str(len(p_wfss)))
    f.write("\nnLgs       :           : number of LGS\n")
    f.write(to_str(len(p_wfs_lgs)))
    f.write("\nnTS        :           : number of Truth Sensor\n")
    f.write(to_str(len(p_wfs_ts)))
    f.write("\nnTarget    :           : number of Target\n")
    if(tar==-1):
        f.write(to_str(len(p_targets)))
    else:
        f.write("1")
    f.write("\nNssp       :           : number of subaperture per wfs along the diameter\n")
    f.write(to_str([wfs.nxsub for wfs in p_wfss]))
    f.write("\nfracsub    : %         : Minimal illumination fraction for valid subap\n")
    f.write("-1")  #to_str(p_wfss[0].fracsub))
    f.write("\ngsAlt      : meter^-1  : inverse of lazer altitude\n")
    f.write(to_str([1 / w.gsalt for w in p_wfs_lgs] + [0 for w in p_wfs_ngs]))
    f.write("\ntype       :           : guide star type (1:NGS, 2:LGS)\n")
    f.write(to_str([2 for w in p_wfs_lgs] + [1 for w in p_wfs_ngs]))
    f.write("\nalphaX_as  : arcsec    : pointing direction of the wfs on x axis\n")
    f.write(to_str([w.xpos for w in p_wfss]))
    f.write("\nalphaY_as  : arcsec    : pointing direction of the wfs on y axis\n")
    f.write(to_str([w.ypos for w in p_wfss]))
    f.write("\nXPup       : meter     : pupil shift of the WFS\n")
    f.write(to_str([0 for i in range(len(p_wfss))]))
    f.write("\nYPup       : meter     : pupil shift of the WFS\n")
    f.write(to_str([0 for i in range(len(p_wfss))]))
    f.write("\nthetaML    :           : rotation of the microlenses\n")
    f.write(to_str([0 for i in range(len(p_wfss))]))
    f.write("\nthetaCam   :           : rotation of the camera\n")
    f.write(to_str([0 for i in range(len(p_wfss))]))
    f.write("\nsensibility:           : sensitivity coeff of this WFS\n")
    f.write(to_str([1 for i in range(len(p_wfss))]))
    f.write("\ntracking   : arcsec^2  : telescope tracking error parameters (x^2, y^2 and xy)\n"
            )
    f.write(to_str("1 1 1"))
    f.write("\npasDPHI    :           : Precision of DPHI precomputation. //deprecated\n"
            )
    f.write(to_str(0.0001))
    f.write("\nncpu       :           : Number of CPU used (only with openMP)\n")
    f.write(to_str(1))
    f.write("\nmrNGS      :           : magnitude of NGS\n")
    if (len(p_wfs_ngs) > 0):
        f.write(to_str([w.gsmag for w in p_wfs_ngs]))
    else:
        f.write(to_str([0.0]))
    f.write("\nlgsFlux    : (ph/m2/s) : LGS photon return at M1\n")
    f.write(to_str(lgs_flux))
    f.write("\nngsPixSize : arcsec    : NGS pixel size\n")
    if (len(p_wfs_ngs) > 0):
        f.write(to_str(p_wfs_ngs[0].pixsize))
    else:
        f.write(to_str(0.0))
    f.write("\nlgsPixSize : arcsec    : LGS pixel size\n")
    f.write(to_str(lgs_pix_size))
    f.write("\nlambdaNGS  : meter     : wave length for NGS\n")
    if (len(p_wfs_ngs) > 0):
        f.write(to_str(p_wfs_ngs[0].Lambda * 1e-6))
    else:
        f.write(to_str(0.0))
    f.write("\nlambdaLGS  : meter     : wave length for LGS\n")
    f.write(to_str(lambda_lgs))
    f.write("\nbdw_m      : meter     : bandwidth\n")
    f.write(to_str(bdw))
    f.write("\nthroughNGS : percent   : transmission for NGS\n")
    if (len(p_wfs_ngs) > 0):
        f.write(to_str(p_wfs_ngs[0].optthroughput))
    else:
        f.write(to_str(0.0))
    f.write("\nthroughLGS : percent   : transmission for LGS\n")
    f.write(to_str(through_lgs))
    f.write("\nthroughAtm : percent   : atmosphere transmission\n")
    f.write(to_str(through_atm))
    f.write("\nRON        : nb of e-  : Read Out Noise \n")
    f.write(to_str(int(np.ceil(p_wfss[0].noise))))
    f.write("\nlgsCst     :           : constant on lgs (simulate that LGS cannot measure tip-tilt and focus)\n")
    f.write(to_str(lgs_filter_cst))
    f.write("\nspotWidth  : arcsec    : lazer width\n")
    f.write(to_str(spot_width))
    f.write("\nlgsAlt     : meter     : sodium layer altitude\n")
    f.write(to_str(lgs_alt))
    f.write("\nlgsDepth   : meter     : depth of the sodium layer\n")
    f.write(to_str(lgs_depth))
    f.write("\ntargetX_as : arcsec    :  taget direction on x axis\n")
    if(tar==-1):
        f.write(to_str(ts_xpos + [t.xpos for t in p_targets]))
    elif(isinstance(tar,(list,np.ndarray))):
        f.write(to_str(ts_xpos + [tar[0]]))
    else:
        f.write(to_str(ts_xpos + [p_targets[tar].xpos]))
    f.write("\ntargetY_as : arcsec    :  taget direction on y axis\n")
    if(tar==-1):
        f.write(to_str(ts_ypos + [t.ypos for t in p_targets]))
    elif(isinstance(tar,(list,np.ndarray))):
        f.write(to_str(ts_ypos + [tar[1]]))
    else:
        f.write(to_str(ts_ypos + [p_targets[tar].ypos]))


def write_atm_param(sup, *,path="."):
    """ Write a atmParam file for tao based on the compass configuration

    Args:
        sup : (CompassSupervisor) : current supervisor

    Kwargs:
        path : (str) : (optional), default "./" path to the atmParam file
    """
    f = open(path + "/prof-1-atmos-night0.txt", "w")
    f.write("Nlayer\n")
    f.write(to_str(sup.config.p_atmos.nscreens))
    f.write("\nr0 @ wfs lambda\n")
    f.write(to_str(sup.config.p_atmos.r0))
    f.write("\ncn2 ESO units\n")
    f.write(to_str(sup.config.p_atmos.get_frac().tolist()))
    f.write("\nh in meters\n")
    f.write(to_str(sup.config.p_atmos.get_alt().tolist()))
    f.write("\nl0 in meters\n")
    f.write(to_str(sup.config.p_atmos.get_L0().tolist()))
    f.write("\nwind direction \n")
    f.write(to_str(sup.config.p_atmos.get_winddir().tolist()))
    f.write("\nwind speed meter/s^-1\n")
    f.write(to_str(sup.config.p_atmos.get_windspeed().tolist()))
    f.close()
    shutil.copyfile(path + "/prof-1-atmos-night0.txt", path + "/prof0-atmos-night0.txt")


def write_meta_Dx(meta_Dx, *,nTS=0, nmeas=None, trans=True, path="."):
    """Write command matrices

    split the meta command matrix

    Args:
        meta_Dx: (np.ndarray[ndim=2, dtype=np.float32]): "meta" command matrix

    Kwargs:
        nTS: (int): (optional), default=0. Number of truth sensors, command matrices are written as Di.fits where 'i' belongs to [0,nTS[ , if nTS<1 write the whole matrix as Dx.fits

        nmeas: (np.ndarray[ndim=1, dtype=np.int32]): (optional) if set, must contains the number of measurements for each TS, the matrix is split according to theses numbers. By default, the matrix is split evenly between the nTS truth sensors

        trans: (bool): (optional), default=True. Transpose the matrix if true

        path: (str): (optional), default './' path where the files are written
        """
    if (nTS < 1):
        if (trans):
            fits.writeto(path + "/Dx.fits", meta_Dx.T, overwrite=True)
        else:
            fits.writeto(path + "/Dx.fits", meta_Dx, overwrite=True)
        return

    if (nmeas is None):
        n = meta_Dx.shape[1] // nTS
        nmeas = np.arange(0, meta_Dx.shape[1] + n, n)
    else:
        nmeas = np.append(0, nmeas.cumsum())

    for i in range(nTS):
        print(i + 1, "out of", nTS, end='\r')
        Dx = meta_Dx[:, nmeas[i]:nmeas[i + 1]]
        if (trans):
            fits.writeto(path + "/Dx" + str(i) + ".fits", Dx.T, overwrite=True)
        else:
            fits.writeto(path + "/Dx" + str(i) + ".fits", Dx, overwrite=True)
