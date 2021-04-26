
from shesha.util.writers.yao.general import *
from shesha.util.writers.yao.wfs     import *
from shesha.util.writers.yao.dm      import *
from shesha.util.writers.yao.targets import *
from shesha.util.writers.yao.atmos   import *
from shesha.util.writers.yao.loop    import *
from shesha.util.writers.yao.gs      import *
from shesha.util.writers import common

def write_parfiles(sup, *,  param_file_name="./yao.par",
                            fits_file_name="./yao.fits",
                            screen_dir="\"./yao_screen\"",
                            n_wfs=None,
                            controller_id=-1,
                            influ_index=0,
                            imat_type="controller"):
    """Write parameter files for YAO simulations

    Args:
        sup : (CompassSupervisor) : supervisor

    Kwargs:
        param_file_name : (str) : (optional), default "./yao.par" name of the yao parameter file

        fits_file_name : (str) : (optional), default "./yao.fits" name of fits file containing sub-apertures and actuator position etc

        screen_dir : (str) : (optional), default "./yao_screen" path to the yao turbulent screen files

        n_wfs : (int) : (optional), number of WFS (default: all wfs)

        controller_id : (int) : index of te controller (default : all)

        influ_index : (int) : actuator index to get the influence function from

        imat_type : (str) : (optional), default "controller" use of regular controller or split tomography (among "controller", "splitTomo")
    """
    conf = sup.config
    if(n_wfs is None):
        n_wfs = len(conf.p_wfss)
    zerop = conf.p_wfss[0].zerop
    lgs_return_per_watt = max([w.lgsreturnperwatt for w in conf.p_wfss])

    print("writing parameter file to " + param_file_name)
    write_general(param_file_name, conf.p_geom, conf.p_controllers,
                  conf.p_tel, conf.simul_name)
    wfs_offset = 0
    dm_offset = 0
    ndm = init_dm(param_file_name)
    for sub_system, c in enumerate(conf.p_controllers):
        dms = [ conf.p_dms[i]  for i in c.get_ndm() ]
        ndm += write_dms (param_file_name, dms ,sub_system=sub_system + 1,
                        offset=dm_offset)
        dm_offset = dm_offset+len(dms)
    finish_dm(param_file_name, ndm)
    gs = init_wfs(param_file_name)
    for sub_system, c in enumerate(conf.p_controllers):
        wfss = [ conf.p_wfss[i] for i in c.get_nwfs()]
        n_ngs, n_lgs = write_wfss(param_file_name, wfss, sub_system=sub_system + 1,
                                n_wfs=n_wfs, offset=wfs_offset)
        gs = (gs[0] + n_ngs, gs[1] + n_lgs)
        wfs_offset = wfs_offset + len(wfss)
    finish_wfs(param_file_name, gs[0], gs[1])
    write_targets(param_file_name, conf.p_targets)
    write_gs(param_file_name, zerop, lgs_return_per_watt,
             conf.p_geom.zenithangle)
    write_atm(param_file_name, conf.p_atmos, screen_dir,conf.p_geom.zenithangle)
    write_loop(param_file_name, conf.p_loop, conf.p_controllers[0])
    common.write_data(fits_file_name, sup, wfss_indices=np.arange(n_wfs),
       controller_id=controller_id, influ=influ_index, compose_type=imat_type)
