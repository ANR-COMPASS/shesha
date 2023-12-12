import os
from astropy.io import fits

from shesha.ao import imats
from shesha.ao import cmats

from shesha.util.tao import writer


def init(tao_settings ,sup,*,n_filt=0,wfs="all",dm_use_tt=False):
    """Initialize the MOAO mode

    compute meta matrix of interaction / command and write parameter files

    Args:
        tao_settings : (dict) : tao settings variables

        sup : (CompassSupervisor) : compass supervisor

    Kwargs
        n_filt : (int) : number of Imat eigenvalues to filter out

        wfs : (str) : (optional), default "all" wfs used by tao ( among "all", "lgs", "ngs")

        dm_use_tt : (bool) :(optional), default False DM compensating TT
    """


    #compute meta imat
    meta_D = imats.get_metaD(sup)
    #get svd of (D.T*D)
    SVD = cmats.svd_for_cmat(meta_D)
    #plt.plot(SVD[1])
    meta_Dx = cmats.get_cmat(meta_D, nfilt=n_filt, svd=SVD)

    #write MOAO pipeline inputs
    data_path = tao_settings["INPUT_PATH"]

    lgs_filter_cst = 0.1
    if(dm_use_tt):
        lgs_filter_cst = 0.
    writer.generate_files(sup, path=data_path, single_file=True,
        dm_use_tt=dm_use_tt,wfs=wfs, lgs_filter_cst=lgs_filter_cst)
    writer.write_meta_Dx(meta_Dx, nTS=sup.config.NTS, path=data_path)


def reconstructor(tao_settings, *,apply_log="./log"):
    """Initialize the LTAO mode

    compute meta matrix of interaction / command and write parameter files

    Args:
        tao_settings : (dict)  : tao settings variables

    Kwargs:
        apply_log    : (str)   : (optional), default "./log" tao log file name
    """

    flags = tao_settings["STARPU_FLAGS"]
    tao_path = tao_settings["TAO_PATH"]
    data_path = tao_settings["INPUT_PATH"]
    gpus = tao_settings["GPU_IDS"]
    tile_size = str(tao_settings["TILE_SIZE"])

    apply_cmd=flags+" "+tao_path+"/mcao_reconstructor --sys_path="+data_path+" --atm_path="+data_path+" --ncores=1 --gpuIds="+gpus+" --ts="+tile_size+" --sync=1 --warmup=0  >"+apply_log+" 2>&1"

    os.system(apply_cmd)
    return fits.open("./M_mcao.fits")[0].data.T
