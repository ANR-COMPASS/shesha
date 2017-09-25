import numpy as np  # type: ignore
import time

import shesha_config as conf
import shesha_constants as scons
import shesha_init.lgs_init as lgs
import shesha_util.hdf5_utils as h5u

from Sensors import Sensors
from Dms import Dms
from Rtc import Rtc

from typing import List  # Mypy checker

from tqdm import tqdm


def imat_geom(wfs: Sensors, dms: Dms, p_wfss: List[conf.Param_wfs],
              p_dms: List[conf.Param_dm], p_controller: conf.Param_controller,
              meth: int=0) -> np.ndarray:
    """Compute the interaction matrix with a geometric method

    :parameters:
        wfs: (Sensors) : Sensors object
        dms: (Dms) : Dms object
        p_wfss: (list of Param_wfs) : wfs settings
        p_dms: (list of Param_dm) : dms settings
        p_controller: (Param_controller) : controller settings
        meth: (int) : (optional) method type (0 or 1)
    """

    nwfs = p_controller.nwfs.size
    ndm = p_controller.ndm.size
    imat_size1 = 0
    imat_size2 = 0

    for nw in range(nwfs):
        nm = p_controller.nwfs[nw]
        imat_size1 += p_wfss[nw]._nvalid * 2

    for nmc in range(ndm):
        imat_size2 += p_dms[nmc]._ntotact

    imat_cpu = np.zeros((imat_size1, imat_size2), dtype=np.float32)
    ind = 0
    cc = 0
    print("Doing imat geom...")
    for nmc in range(ndm):
        nm = p_controller.ndm[nmc]
        dms.resetdm(p_dms[nm].type_dm, p_dms[nm].alt)
        for i in tqdm(range(p_dms[nm]._ntotact), desc="DM%d"%nmc):
            dms.oneactu(p_dms[nm].type_dm, p_dms[nm].alt, i, p_dms[nm].push4imat)
            nslps = 0
            for nw in range(nwfs):
                n = p_controller.nwfs[nw]
                wfs.raytrace(n, b"dm", tel=None, atmos=None, dms=dms, rst=1)
                wfs.slopes_geom(n, meth)
                imat_cpu[nslps:nslps + p_wfss[n]._nvalid * 2, ind] = wfs.get_slopes(n)
                nslps += p_wfss[n]._nvalid * 2
            imat_cpu[:, ind] = imat_cpu[:, ind] / p_dms[nm].push4imat
            ind = ind + 1
            cc = cc + 1
            dms.resetdm(p_dms[nm].type_dm, p_dms[nm].alt)

    return imat_cpu


def imat_init(ncontrol: int, rtc: Rtc, dms: Dms, p_dms: list, wfs: Sensors, p_wfss: list,
              p_tel: conf.Param_tel, p_controller: conf.Param_controller, kl=None,
              dataBase: dict={}, use_DB: bool=False) -> None:
    """Initialize and compute the interaction matrix on the GPU

    :parameters:
        ncontrol: (int) : controller's index
        rtc: (Rtc) : Rtc object
        dms: (Dms) : Dms object
        p_dms: (Param_dms) : dms settings
        wfs: (Sensors) : Sensors object
        p_wfss: (list of Param_wfs) : wfs settings
        p_tel: (Param_tel) : telescope settings
        p_controller: (Param_controller) : controller settings
        kl:(np.array) :  KL_matrix
        dataBase:(dict): (optional) dict containing paths to files to load
        use_DB:(bool) : (optional) use dataBase flag
    """
    # first check if wfs is using lgs
    # if so, load new lgs spot, just for imat
    for i in range(len(p_wfss)):
        if (p_wfss[i].gsalt > 0):
            # TODO: check that
            save_profile = p_wfss[i].proftype
            p_wfss[i].proftype = scons.ProfType.GAUSS1
            lgs.prep_lgs_prof(p_wfss[i], i, p_tel, wfs, imat=1)

    if "imat" in dataBase:
        imat = h5u.load_imat_from_dataBase(dataBase)
        rtc.set_imat(ncontrol, imat)
    else:
        t0 = time.time()
        if kl is not None:
            ntt = scons.DmType.TT in [d.type_dm for d in p_dms]
            rtc.do_imat_kl(ncontrol, p_controller, dms, p_dms, kl, ntt)
        else:
            rtc.do_imat(ncontrol, dms)
        print("done in %f s" % (time.time() - t0))
        if use_DB:
            h5u.save_imat_in_dataBase(rtc.get_imat(ncontrol))
    p_controller.set_imat(rtc.get_imat(ncontrol))

    # Restore original profile in lgs spots
    for i in range(len(p_wfss)):
        if (p_wfss[i].gsalt > 0):
            p_wfss[i].proftype = save_profile
            lgs.prep_lgs_prof(p_wfss[i], i, p_tel, wfs)
