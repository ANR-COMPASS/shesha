#
"""
Test Line

ipython -i /home/fvidal/compass/shesha/test/scripts/script_PYR39m_optimGain.py /home/fvidal/compass/shesha/data/par/MICADO/micado_39m_PYR_ELTPupil.py 500 0.1 5 0.5 17 100 py3Test 92 5
"""


def sendMail(message, title):
    import smtplib
    from email.mime.text import MIMEText
    smtp = smtplib.SMTP('smtp.obspm.fr')
    msg = MIMEText(title)
    msg['From'] = 'micmac'
    msg['To'] = 'Script micmac'
    msg['Subject'] = message
    smtp.sendmail('micmac@obspm.fr', ["fabrice.vidal@obspm.fr"], msg.as_string())


#try:
import cProfile
import pstats as ps
import sys, os
sys.path.insert(0, os.environ["SHESHA_ROOT"] + "/widgets/")
sys.path.insert(0, os.environ["SHESHA_ROOT"] + "/lib/")
sys.path.insert(0, os.environ["SHESHA_ROOT"] + "/AOlib/")
sys.path.insert(0, os.environ["SHESHA_ROOT"] + "/src/shesha_util/")

#from adoptLib import computeKLModesImat, computeCmatModal
from shesha.util import tools
import numpy as np
import carmaWrap as ch
import shesha.config as ao
import shesha.sim
import shesha.constants as scons
from shesha.constants import CONST

import time
import matplotlib.pyplot as plt
import hdf5_util as h5u
import resDataBase as db
import astropy.io.fits as pf
import glob
import pandas as pd
import compassConfigToFile as cf
import make_pupil as mkP

if (len(sys.argv) == 1):
    error = 'command line should be:"python -i test.py parameters_filename"\n with "parameters_filename" the path to the parameters file'
    raise Exception(error)
elif (len(sys.argv) == 2):
    print("Using Internal parameters...")
    """
    -----------------
            INPUTS
    -----------------
    """
    freq = 500
    gain = 1
    magnitude = 11
    nKL_Filt = 450
    MODU = 5
    RON = 0.1
    NSSP = 92
    simulName = "PYR_39m_RoundPupil_FromHippo6"
else:
    print("-------------------------------------")
    print("DETECTED BASH SCRIPT with parameters:")
    print((sys.argv))
    print("-------------------------------------")

    freq = float(sys.argv[2])  # AO Loop frequency
    RON = float(sys.argv[3])  # noise on the WFS measurement in electrons
    MODU = float(sys.argv[4])  # modulation radius
    gain = float(sys.argv[5])  # global loop gain
    magnitude = float(sys.argv[6])  # gs magnitude
    nKL_Filt = int(float(sys.argv[7]))  # Nb KL filtered
    simulName = sys.argv[8]  # Nb KL filtered
    NSSP = int(sys.argv[9])  # Number of ssp (pixels of pyramid)
    GPU = int(sys.argv[10])  # GPU number
    comment = "SRVsGSVsNControlledModes"

print(("Freq=", freq))
print(("RON=", RON))
print(("MODU=", MODU))
print(("gain=", gain))
print(("magnitude=", magnitude))
print(("nKL_Filt=", nKL_Filt))
print(("GPU=", GPU))
print(("simulName=", simulName))

pathResults = "/volumes/hra/micado/" + simulName

dBResult = pathResults + "/" + simulName + ".h5"
savePSFs = True
PYR = True

imatFromFile = False
#iMatName = "iMat39mPYR_MODU_"+str(int(MODU))+".fits"
#KL2VName = "KL2VNorm39mPYR_MODU_"+str(int(MODU))+".fits"
#gainModalName = "gains4K_MODU_"+str(int(MODU))+".fits"

#iMatName = "iMat_MODU_5_ELTPUPIL.fits"
#KL2VName = "KL2VNorm_MODU_5_ELTPUPIL.fits"
#gainModalName = "gains4K_MODU_5_ELTPUPIL.fits"

imat0_PATH = "/home/fvidal/dataSimus"
iMatName = "imatDiffraction_ELTPYR_35Layers.fits"
gainModalName = "gains4K_ELTPYR_35Layers.fits"
KL2VName = "KL2VNorm_ELTPYR_35Layers.fits"
"""
iMatName = "iMat_MODU_2_ELTPUPIL.fits"
KL2VName = "KL2VNorm_MODU_2_ELTPUPIL.fits"
gainModalName = "gains4K_MODU_2_ELTPUPIL.fits"
"""

ModalBasisType = "Btt"

PSFWithOtherPupil = True

niter = 8096
saveCBData = True
nbLoopData = 512
"""
simulName = "PYR_39m"
pathResults="/home/fvidal/dataSimus/PYR_39m_RoundPupil_RUN1/"
dBResult = "/home/fvidal/dataSimus/PYR_39m_RoundPupil_RUN1.h5"
imat0_PATH = "/home/fvidal/compass/shesha/test/scripts"
savePSFs = False
imatFromFile = False
"""

if (GPU == 1):
    GPUs = np.array([4, 5, 6, 7], dtype=np.int32)
else:
    GPUs = np.array([GPU], dtype=np.int32)
print(("Using GPUs: ", GPUs))

GPUs = np.array([4, 5, 6, 7], dtype=np.int32)
#GPUs = np.array([0,1,2,3], dtype=np.int32)

if (not glob.glob(pathResults)):
    print("Results folder not found. Creating it now:")
    tools.system("mkdir " + pathResults)
if (not glob.glob(pathResults + "/PSFs/")):
    print("PSFs folder not found. Creating it now:")
    tools.system("mkdir " + pathResults + "/PSFs/")
if (not glob.glob(pathResults + "/AODATA/")):
    print("AODATA folder not found. Creating it now:")
    tools.system("mkdir " + pathResults + "/AODATA/")
if (not glob.glob(pathResults + "/CircularBuffers/")):
    print("CircularBuffers folder not found. Creating it now:")
    tools.system("mkdir " + pathResults + "/CircularBuffers/")

#get parameters from file
param_file = sys.argv[1]  # par filename
if (param_file.split('.')[-1] == "py"):
    filename = param_file.split('/')[-1]
    param_path = param_file.split(filename)[0]
    sys.path.insert(0, param_path)
    exec("import %s as config" % filename.split(".py")[0])
    #sys.path.remove(param_path)
elif (param_file.split('.')[-1] == "h5"):
    sys.path.insert(0, os.environ["SHESHA_ROOT"] + "/data/par/par4bench/")
    import scao_sh_16x16_8pix as config
    #sys.path.remove(os.environ["SHESHA_ROOT"]+"/data/par/par4bench/")
    h5u.configFromH5(param_file, config)
else:
    raise ValueError("Parameter file extension must be .py or .h5")

print(("param_file is", param_file))

if (hasattr(config, "simul_name")):
    if (config.simul_name is None):
        simul_name = ""
    else:
        simul_name = config.simul_name
else:
    simul_name = ""
print(("simul name is", simul_name))

matricesToLoad = {}
if (simul_name == ""):
    clean = 1
else:
    clean = 0
    param_dict = h5u.params_dictionary(config)
    matricesToLoad = h5u.checkMatricesDataBase(os.environ["SHESHA_ROOT"] + "/data/",
                                               config, param_dict)
c = ch.carmaWrap_context(devices=GPUs)


class wao_class():

    def __init__(self, config, wfs, tel, atm, dms, tar, rtc):
        self.config = config
        self.wfs = wfs
        self.tel = tel
        self.atm = atm
        self.dms = dms
        self.tar = tar
        self.rtc = rtc


def makeFITSHeader(filepath, df):
    hdulist = pf.open(filepath)  # read file
    header = hdulist[0].header
    names = np.sort(list(set(df))).tolist()

    for name in names:
        if (name != "centroider.thresh"):
            val = df[name][0]
            if (type(val) is list):
                value = ""
                for v in val:
                    value += (str(v) + " ")
                value = value.replace("\n", "")
            elif (type(val) is np.ndarray):
                value = ""
                for v in val:
                    value += (str(v) + " ")
                value = value.replace("\n", "")

            else:
                value = val

        if ((type(value) is str)):
            if (len(value) > 50):
                print(("warning", name, "keyword has been cut to 100 characters"))
                #header.set(name, value[:50],'')
            else:
                header.set(name, value, '')

    hdulist.writeto(filepath, clobber=True)  # Save changes to file


def initSimu(config, c):
    param_dict = h5u.params_dictionary(config)
    matricesToLoad = h5u.checkMatricesDataBase(os.environ["SHESHA_ROOT"] + "/data/",
                                               config, param_dict)
    print("->wfs")
    wfs, tel = ao.wfs_init(config.p_wfss, config.p_atmos, config.p_tel, config.p_geom,
                           config.p_target, config.p_loop, config.p_dms)
    print("->atmos")
    atm = ao.atmos_init(c, config.p_atmos, config.p_tel, config.p_geom, config.p_loop,
                        config.p_wfss, wfs, config.p_target, rank=0)
    print("->dm")
    dms = ao.dm_init(config.p_dms, config.p_wfss, wfs, config.p_geom, config.p_tel)
    print("->target")
    tar = ao.target_init(c, tel, config.p_target, config.p_atmos, config.p_geom,
                         config.p_tel, config.p_dms)
    print("->rtc")
    rtc = ao.rtc_init(tel, wfs, config.p_wfss, dms, config.p_dms, config.p_geom,
                      config.p_rtc, config.p_atmos, atm, config.p_tel, config.p_loop,
                      do_refslp=False, clean=clean, simul_name=simul_name,
                      load=matricesToLoad, doimat=0)

    h5u.validDataBase(os.environ["SHESHA_ROOT"] + "/data/", matricesToLoad)

    print("====================")
    print("init done")
    print("====================")
    print("objects initialzed on GPU:")
    print("--------------------------------------------------------")
    print(atm)
    print(wfs)
    print(dms)
    print(tar)
    print(rtc)
    return wfs, tel, atm, dms, tar, rtc


def compute_modal_residuals(P, rtc, dms, tar):
    rtc.do_control_geo(1, dms, tar, 0)
    #self.rtc.do_control_geo_on(1, self.dms,self.tar, 0)
    v = rtc.get_command(1)
    ai = P.dot(v) * 1000.  # np rms units
    return ai


def loop(n, wfs, tel, atm, dms, tar, rtc, move_atmos=True, noise=True, loopData=0,
         P=None):
    t0 = time.time()
    print("----------------------------------------------------")
    print("iter# | S.E. SR | L.E. SR | Est. Rem. | framerate")
    print("----------------------------------------------------")
    """
    ph = tar.get_image(0, "se")
    pupBig = ph*0.

    phsize = pup.shape[0]
    npup = pupBig.shape[0] # wao.wfs.get_pyrimghr(0).shape

    pupBig[(npup-phsize)/2:(npup+phsize)/2, (npup-phsize)/2:(npup+phsize)/2] = pup
    PSFLEArray = np.zeros((config.p_target.ntargets, ph.shape[0],ph.shape[1]))
    PSFSEArray = np.zeros((config.p_target.ntargets, ph.shape[0],ph.shape[1]))

    """
    RmsErrorTot = []
    ph = tar.get_image(0, "se")
    PSFtarget = np.zeros((config.p_target.ntargets, ph.shape[0], ph.shape[1]))
    sr_se = []
    numiter = []
    if (loopData):
        if (loopData > n):
            loopData = n
        slopes = np.zeros((loopData, rtc.get_centroids(0).shape[0]))
        volts = np.zeros((loopData, rtc.get_voltage(0).shape[0]))
    else:
        slopes = volts = None
    ii = 0
    jj = 0
    sr_se = np.zeros((n / 10, config.p_target.ntargets))
    sr_le = np.zeros((n / 10, config.p_target.ntargets))

    for i in range(n):
        if (move_atmos):
            atm.move_atmos()

        for t in range(config.p_target.ntargets):
            tar.atmos_trace(t, atm, tel)
            tar.dmtrace(t, dms)
        for w in range(len(config.p_wfss)):
            wfs.raytrace(w, "all", tel, atm, dms)
            wfs.sensors_compimg(w, noise=noise)

        rtc.do_centroids(0)
        if (loopData):
            if (i >= (n - loopData)):
                #print("Recording loop Data")
                s = rtc.get_centroids(0)
                v = rtc.get_voltage(0)
                volts[ii, :] = v.copy()
                slopes[ii, :] = s.copy()
                ii += 1
        rtc.docontrol(0)
        rtc.doclipping(0, -1e5, 1e5)
        rtc.apply_control(0)

        signal_le = ""
        signal_se = ""
        if (P is not None):
            ai = compute_modal_residuals(P, rtc, dms, tar)
            tarPhaseError = np.sqrt(np.sum(ai**2))
        else:
            tarPhaseError = 0.
        RmsErrorTot.append(tarPhaseError)
        print("tarPhaseError =", tarPhaseError, "nm rms")
        if ((i + 1) % 10 == 0):
            print(("Iter#:", i + 1, "/", n))
            t = 0
            SRTmp = np.zeros(config.p_target.ntargets)
            SRTmp2 = np.zeros(config.p_target.ntargets)

            for t in range(config.p_target.ntargets):
                if (PSFWithOtherPupil):
                    SR = list([0, 0])
                    SR[0] = PSFSEArray[t, :, :].max()  # SE SR
                    SR[1] = (PSFLEArray[t, :, :] / (i + 1)).max()  # LE SR
                else:
                    SR = tar.get_strehl(t)
                #print("Tar %d at %3.2fMicrons:" % (t+1, tar.Lambda[t]))
                signal_se += "SR S.E %3.2fMicrons:: %1.2f   " % (tar.Lambda[t], SR[0])
                signal_le += "SR L.E %3.2fMicrons:: %1.2f   " % (tar.Lambda[t], SR[1])
                SRTmp[t] = SR[0] * 100
                SRTmp2[t] = SR[1] * 100
            print((signal_se + signal_le))
            sr_se[jj, :] = SRTmp.copy()
            sr_le[jj, :] = SRTmp2.copy()

            #sr_se.append()
            #sr_se.append(SR[0])
            numiter.append(i + 1)
            jj += 1

    t1 = time.time()
    print((" loop execution time:", t1 - t0, "  (", n, "iterations), ", (t1 - t0) / n,
           "(mean)  ", n / (t1 - t0), "Hz"))
    SRList = []
    for t in range(config.p_target.ntargets):
        SR = tar.get_strehl(t)
        PSFtarget[t, :, :] = tar.get_image(t, "le")
        SRList.append(SR[1])  # Saving Last Long Exp SR

    return SRList, tar.Lambda.tolist(), sr_se.astype(int), sr_le.astype(
            int), numiter, slopes, volts, PSFtarget, RmsErrorTot


SR = []
colnames = h5u.params_dictionary(config)  # config values internal to compass
simunames = {
        "PSFFilenames": None,
        "rmsError": None,
        "rmsErrorList": None,
        "comment": None,
        "NCPA": None,
        "NCPAList": None,
        "ModalType": None,
        "srir": None,
        "gainModal": None,
        "lambdaTarget": None,
        "nbBrightest": None,
        "sr_le": None,
        "sr_se": None,
        "numiter": None,
        "NklFilt": None,
        "NklTot": None,
        "Nkl": None,
        "eigenvals": None,
        "Nphotons": None,
        "Nactu": None,
        "RON": None,
        "Nslopes": None
}  # Added values computed by the simu..

resAll = db.readDataBase(fullpath=dBResult)  # Reads all the database if exists
if (not (type(resAll) == pd.core.frame.DataFrame)):
    print("Creating compass database")
    resAll = db.createDf(list(colnames.keys()) + list(
            simunames.keys()))  # Creates the global compass Db

# -----------------------------------------------------------------------------
# ----------- Replacing values from user defined variables-------------------
# -----------------------------------------------------------------------------
config.p_wfs0.set_nxsub(NSSP)
if (PYR):
    decalage = int((240 - 4 - (NSSP * 2)) / 2. + NSSP / 2)
    config.p_wfs0.set_pyr_pup_sep(decalage)
    rMod = MODU
    config.p_wfs0.set_pyr_npts(int(np.ceil(int(rMod * 2 * 3.141592653589793) / 4.) * 4))
    config.p_wfs0.set_pyr_ampl(rMod)

config.p_loop.set_ittime(1 / freq)
config.p_wfs0.set_noise(RON)
config.p_loop.set_niter(niter)

config.p_wfs0.set_gsmag(magnitude)

res = pd.DataFrame(columns=list(colnames.keys()) + list(simunames.keys()))  # Create Db
wfs, tel, atm, dms, tar, rtc = initSimu(config, c)  # Init COMPASS Simu!

if (PSFWithOtherPupil):
    #pp = config.p_geom.get_spupil().shape[0]
    cent = config.p_geom.pupdiam / 2. + 0.5
    oldsetting = int(config.p_tel.t_spiders)
    config.p_tel.set_t_spiders(-1)  # Enabling spiders for pupil computation
    pupELTSpiders = mkP.make_pupil(config.p_geom.pupdiam, config.p_geom.pupdiam,
                                   config.p_tel, cent, cent).astype(np.float32)
    config.p_tel.set_t_spiders(
            oldsetting)  # Disabling spiders in case of it is used else where in GPU...
    #PUPILPATH = "/home/fvidal/dataSimus/pupELTwithSpiders_368.fits"
    #PUPILPATH = "/home/fvidal/dataSimus/pupELTwithSpiders.fits"
    #PUPILPATH = "/home/fvidal/dataSimus/pupELTwithSpiders_1472.fits"
    for target in range(config.p_target.ntargets):  # Apply E-ELT pupil for each target
        tar.set_pupil(target, pupELTSpiders.astype(np.float32))

# ------------ ADOPT ----------------
ADOPTPATH = os.getenv("ADOPTPATH")
sys.path.insert(0, ADOPTPATH)
import adoptCompass as adoptComm
import adoptVariables as adoptVar
import aoCalib as cal
config_fileName = ADOPTPATH + "/config/ADOPT.conf"
wao = wao_class(config, wfs, tel, atm, dms, tar, rtc)
cf.returnConfigfromWao(wao, filepath=config_fileName)
com = adoptComm.command_class(wao, ao)
aoAd = adoptVar.ao_class(adoptVar.ao_attributes, adoptVar.wfs_attributes,
                         adoptVar.dm_attributes, config_fileName)
com.initComm(aoAd)
com.do_ref_slopes()

#KL2V = com.getKL2V()
#
nfilt = nKL_Filt

# Computing imat on diffraction limited source.
if (imatFromFile):
    print("Reloading imat KL2V and gains4K from files...")
    #print(imat0_PATH+"/"+iMatName)
    #print(imat0_PATH+"/gains4K_MODU_"+str(int(MODU))+".fits")
    imat = pf.get_data(imat0_PATH + "/" + iMatName)
    modal_basis = pf.get_data(imat0_PATH + "/" + KL2VName)
    gains4KRAW = pf.get_data(imat0_PATH + "/" + gainModalName)
    gains4K = np.zeros(imat.shape[0] - nfilt)
    gains4K[:-2] = gains4KRAW[:imat.shape[0] - nfilt - 2]
    gains4K[-2:] = gains4KRAW[-2:]
    gainopt = gains4K.copy()
else:
    KL2V = com.getKL2V()
    KL2VNorm = cal.normalizeKL2V(KL2V)
    print("Computing Imat Diffraction Limited")
    imat = cal.computeImatModal(com, KL2VNorm, aoAd.dm0.push4iMat, aoAd.dm1.push4iMat,
                                withTurbu=False, noise=False)
    gains = np.linspace(1., 1., aoAd.Nactu - 2 - nfilt)
    gains[-2:] = 1.0
    cmat0, cmatKL0 = cal.computeCmatModal(imat, KL2VNorm, nfilt, gains)
    com.set_command_matrix(cmat0)
    com.close_loop()
    print("Closing Loop with Imat Diffraction Limited")

    # Closing loop until we reach the fitting error for the given ao config + turbulence conditions (seeing ect...) but without noise and bandwidth (screen is frozen)
    SR, lambdaTargetList, sr_se, numiter, _, _, _ = loop(200, wfs, tel, atm, dms, tar,
                                                         rtc, move_atmos=True,
                                                         noise=True)
    print("SR After 200 iterations of closed loop:")

    if (PYR):
        cmat0, cmatModal0 = cal.computeCmatModal(imat, modal_basis, nfilt, gains)
        com.set_command_matrix(cmat0)
        com.close_loop()
        print("Closing Loop with Imat Diffraction Limited")

        # Closing loop until we reach the fitting error for the given ao config + turbulence conditions (seeing ect...) but without noise and bandwidth (screen is frozen)
        SR, lambdaTargetList, sr_se, sr_le, numiter, _, _, _, _ = loop(
                100, wfs, tel, atm, dms, tar, rtc, move_atmos=True, noise=True, P=P)
        print("SR After 100 iterations of closed loop:")

        # Computing 2nd imat  with this best conditions (no noise + limited by fitting)
        imatTurbu = cal.computeImatModal(com, modal_basis, aoAd.dm0.push4iMat,
                                         aoAd.dm1.push4iMat, withTurbu=True, noise=False)
        gains4K = cal.computeOptimGainK(imat, imatTurbu, nfilt)

        date = time.strftime("_%d-%m-%Y_%H:%M:%S_")
        gainModalName = "gains4K_" + date + ".fits"
        iMatName = "imat_" + date + ".fits"
        KL2VName = "modal_basis_" + date + ".fits"
        # saving imat, modal basis, and gains...
        pf.writeto(pathResults + "/AODATA/" + iMatName, imat)
        pf.writeto(pathResults + "/AODATA/" + KL2VName, modal_basis)
        pf.writeto(pathResults + "/AODATA/" + gainModalName, gains4K)
        gainopt = gains4K.copy()
    else:
        gainopt = np.linspace(1., 1., aoAd.Nactu - 2 - nfilt)
        gainopt[-2:] = 1.0
cmatT, cmatKLT = cal.computeCmatModal(imat, modal_basis, nfilt, gainopt * gain)
cmat = cmatT
com.set_command_matrix(cmatT)
com.close_loop()
com.resetSR()

# ------------------------------------------------------------------------------
# --------------------- Modal Optim. ----------------------------------------
# ------------------------------------------------------------------------------
# Taking 2048 loop data for Optim Modal gain optim ("a la" Gendron & Lena)
# closing loop by adding noise + bandwidth and wait a bit that loop converge...
"""
SR, lambdaTargetList, sr_le, sr_se, numiter, _, _ = loop(200,wfs,tel,atm,dms,tar,rtc, noise=True)
com.resetSR()
SR, lambdaTargetList, sr_le, sr_se, numiter, slopes, volts = loop(2048,wfs,tel,atm,dms,tar,rtc, noise=True, loopData=True)
V2KL  =np.linalg.pinv(KL2VNorm)
sol = cal.recPseudoopen_loop(slopes, volts, imat, V2KL, gains4K, nfilt, 1/aoAd.Fe, aoAd.Fe)
gainoptCorr = cal.modalControlOptimizationopen_loopData(sol.T, cmatKL0, KL2VNorm, gmax = 1.0, Fs = aoAd.Fe, latency = 1/aoAd.Fe, BP = 1e12,ngain=200)
gainopt = gainopt*gainoptCorr
cmatOptim,_ = cal.computeCmatModal(imat, KL2VNorm, nfilt, gainopt);
com.set_command_matrix(cmatOptim)

com.close_loop()
"""
# ------------------------------------------------------------------------------

#cmat = pf.get_data(os.environ["SHESHA_ROOT"]+"/test/scripts/cmatKLGood.fits").byteswap().newbyteorder()
#rtc.set_cmat(0, cmat.copy().astype(np.float32))

# -----------------------------------------------------------------------------
# ----------- !!!!!! Starting real loop !!!!!!-------------------
# -----------------------------------------------------------------------------

print("Starting Real Loop")
com.resetSR()
# com.resetDm()
SR, lambdaTargetList, sr_se, sr_le, numiter, slopesCB, voltsCB, PSFtarget, rmsErrorList = loop(
        config.p_loop.niter, wfs, tel, atm, dms, tar, rtc, loopData=nbLoopData, P=P)

if (saveCBData):
    PYRImage = wfs.get_pyrimg(0)
    date = time.strftime("_%d-%m-%Y_%H:%M:%S_")
    slopesCBName = "slopesCB_" + date + ".fits"
    voltsCBName = "voltsCB_" + date + ".fits"
    PYRIMAGEName = "pyrImageCB_" + date + ".fits"
    SRHistorySEName = "SRHistorySE_" + date + ".fits"
    SRHistoryLEName = "SRHistoryLE_" + date + ".fits"
    pf.writeto(pathResults + "/CircularBuffers/" + slopesCBName, slopesCB.copy())
    pf.writeto(pathResults + "/CircularBuffers/" + voltsCBName, voltsCB.copy())
    pf.writeto(pathResults + "/CircularBuffers/" + PYRIMAGEName, PYRImage.copy())
    pf.writeto(pathResults + "/CircularBuffers/" + SRHistorySEName, sr_se.copy())
    pf.writeto(pathResults + "/CircularBuffers/" + SRHistoryLEName, sr_le.copy())

else:
    slopesCBName = ""
    voltsCBName = ""
    PYRIMAGEName = ""
    SRHistorySEName = ""
    SRHistoryLEName = ""

# ------------- Saving config and results in data frame -----
dfparams = h5u.params_dictionary(config)  # get the current compass config
for i in range(len(dfparams)):
    if (type(dfparams[list(dfparams.keys())[i]]) is list):  # If list
        if (len(dfparams[list(dfparams.keys())[i]]) == 1):  # If list has only 1 element:
            dfparams[list(dfparams.keys())[i]] = dfparams[list(dfparams.keys())[i]][0]

dfparams.update(simunames)  # Add the simunames params

res = db.fillDf(res, dfparams)  # Saving dictionnary config
res.loc[0, "iMatName"] = iMatName
res.loc[0, "ModalName"] = KL2VName
res.loc[0, "gainModalName"] = gainModalName
res.loc[0, "slopesCBName"] = slopesCBName
res.loc[0, "voltsCBName"] = voltsCBName
res.loc[0, "SRHistoryLEName"] = SRHistorySEName
res.loc[0, "SRHistorySEName"] = SRHistoryLEName
res.loc[0, "PYRIMAGEName"] = PYRIMAGEName
res.loc[0, "comment"] = comment
res.loc[0, "NklFilt"] = nKL_Filt
res.loc[0, "Nkl"] = imat.shape[0] - nfilt - 2
res.loc[0, "NklTot"] = cmat.shape[1] - 2
res.loc[0, "Nactu"] = cmat.shape[1]
res.loc[0, "Nslopes"] = cmat.shape[0]
res.loc[0, "Nphotons"] = config.p_wfs0._nphotons
res.loc[0, "RON"] = RON
#res.eigenvals.values[0] = rtc.getEigenvals(0)
res.srir.values[0] = SR  # Saving computed values
res.lambdaTarget.values[0] = lambdaTargetList
res.loc[0, "gsmag"] = config.p_wfs0.gsmag
res.loc[0, "gain"] = gain
#res.loc[0, "type_ap"] = str(res.loc[0, "type_ap"][0])
#res.loc[0, "type_wfs"] = str(res.loc[0, "type_wfs"][0])
res.loc[0, "type_dm"] = "pzt, tt"
#res.loc[0, "npix"] = res.loc[0, "npix"][0]
#res.loc[0, "nbBrightest"] = res.loc[0, "nbBrightest"][0]
res.loc[0, "pixsizeInMeters"] = (config.p_tel.diam / config.p_geom.get_spupil().shape[0])
# PSF pixsize =
#res.sr_le.values[0] = sr_le
#res.sr_se.values[0] = sr_se
#res.numiter.values[0] = numiter
res.loc[0, "simulname"] = simulName
res.rmsErrorList.values[0] = rmsErrorList
res.rmsError.values[0] = np.average(np.array(rmsErrorList))

# --------------- PSF Stuff ----------------------
print("Saving PSFs...")
PSFNameList = []
for t in range(config.p_target.ntargets):

    date = time.strftime("_%d-%m-%Y_%H:%M:%S_")
    lam = "%3.2f" % tar.Lambda.tolist()[t]
    lam = lam.replace(".", "_")
    PSFName = "PYR_TARGET_" + str(t + 1) + "_Lambda_" + lam + "_" + date + ".fits"
    PSFNameList.append(PSFName)
    #PSFNameList.append("NOT SAVED")
    if (savePSFs):
        pf.writeto(pathResults + "/PSFs/" + PSFName, PSFtarget[t, :, :].copy(),
                   clobber=True)
    lam2 = "%3.2f" % tar.Lambda.tolist()[t]
    res.loc[0, "SR_%s" % lam2] = SR[t]
    PSFPixsize = (tar.Lambda.tolist()[t] * 1e-6) / (
            wao.config.p_tel.diam / wao.config.p_geom.get_spupil().shape[0] *
            wao.config.p_geom.get_ipupil().shape[0]) * 206265.
    res.loc[0, "pixsizeArcSec_%s" % lam2] = PSFPixsize
    filepath = pathResults + "/PSFs/" + PSFName
    if (savePSFs):
        #"Add the SR and wavelegth value at the top of the PSF header file"
        hdulist = pf.open(filepath)  # read file
        header = hdulist[0].header
        header["SR"] = SR[t]
        header["wavelengthMic"] = tar.Lambda.tolist()[t]
        header["pixsizeArcSec"] = PSFPixsize
        hdulist.writeto(filepath, clobber=True)  # Save changes to file
        # Adding all the parameters to the header
        makeFITSHeader(filepath, res)
    else:
        res.PSFFilenames.values[0] = ["PSF NOT SAVED"]
print("Done")
res.PSFFilenames.values[0] = PSFNameList

resAll = db.fillDf(resAll, res)  # Saving in global DB
#resAll.to_hdf("/home/fvidal/compass/trunk/shesha/test/scripts/resultatsScripts/SH39m.h5", "resAll", complevel=9,complib='blosc')
resAll.to_hdf(dBResult, "resAll", complevel=9, complib='blosc')
#db.saveDataBase(resAll)

print("Simulation Done...")
"""
except Exception as e:
    if(e):
        print "Simu failed"
        print e
        sendMail("Script failed!" + str(sys.argv),  str(e)+ str("\nline: ") +str(sys.exc_info()[-1].tb_lineno))
"""
"""Convert resAll not well formatted:



for i in range(len(dfparams)):
  if(type(dfparams[dfparams.keys()[i]]) is list): # If list
    if(len(dfparams[dfparams.keys()[i]]) == 1): # If list has only 1 element:
      dfparams[dfparams.keys()[i]] = dfparams[dfparams.keys()[i]][0]




for i in range(len(list(set(resAll)))):
  if(type(resAll[list(set(resAll))[i]][0]) is list): # If list
    if(len(resAll[list(set(resAll))[i]][0]) == 1): # If list has only 1 element:
      key = list(set(resAll))[i]
      for j in range(len(resAll)):
        try:
          resAll.loc[j, key] = resAll.loc[j, key][0]
        except:
          print "Warning! Could not convert list "+key+" with 1 element to 1 element"
          pass




"""
