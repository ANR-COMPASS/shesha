import os
from astropy.io import fits

from shesha.ao import imats
from shesha.ao import cmats

from shesha.util.tao import writer

def init(tao_settings,sup,*,n_filt=10, wfs="all", dm_use_tt=False):
    """Initialize the LTAO mode

    compute meta matrix of interaction / command and write parameter files

    Args:
        tao_settings : (dict) : tao settings variables

        sup : CompassSupervisor : compass supervisor

    Kwargs:
        wfs : (str) : (optional), default "all" wfs used by tao ( among "all", "lgs", "ngs")

        n_filt : (int) : number of Imat eigenvalues to filter out

        dm_use_tt : (bool) : (optional), default False using a TT DM
    """

    #compute meta imat
    meta_D = imats.get_metaD(sup,0,0)
    #get svd of (D.T*D)
    SVD = cmats.svd_for_cmat(meta_D)
    #plt.plot(SVD[1])
    meta_Dx = cmats.get_cmat(meta_D,nfilt=n_filt,svd=SVD)

    #write MOAO pipeline inputs
    data_path = tao_settings["INPUT_PATH"]
    lgs_filter_cst=0.1
    if(dm_use_tt):
        lgs_filter_cst=0.
    writer.generate_files(sup, path=data_path, single_file=True,
        dm_use_tt=dm_use_tt, wfs=wfs, lgs_filter_cst=lgs_filter_cst)
    writer.write_meta_Dx(meta_Dx,nTS=sup.config.NTS,path=data_path)


def reconstructor(tao_settings,*,apply_log="./log"):
    """Initialize the LTAO mode

    compute meta matrix of interaction / command and write parameter files

    Args:
        tao_settings : (dict)  : tao settings variables

    Kwargs:
        apply_log : (str)   : tao log file name

    Returns:
        tor : () : tomographic reconstructor
    """

    flags = tao_settings["STARPU_FLAGS"]
    tao_path = tao_settings["TAO_PATH"]
    data_path = tao_settings["INPUT_PATH"]
    gpus = tao_settings["GPU_IDS"]
    tile_size = str( tao_settings["TILE_SIZE"])
    apply_cmd = flags + " " + tao_path + "/ltao_reconstructor --sys_path=" \
        + data_path + " --atm_path=" + data_path + " --ncores=1 --gpuIds=" \
        + gpus + " --ts=" + tile_size + " --sync=1 --warmup=0  >" + apply_log \
        +" 2>&1"
    os.system(apply_cmd)
    return fits.open("M_ltao_0.fits")[0].data.T
