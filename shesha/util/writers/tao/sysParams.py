import json
import numpy as np
from shesha.util.writers import common

def write_json_sys_param(sup, *, wfss_indices=None, ts=False, dms_indices=None, file_name="./sys-params.json"):
    """Return a json representation of the AO system

    Args:
        sup : (CompassSupervisor) : supervisor to get the json representation from

    Kargs:
        wfss_indices : (list(int)) : list of wfs indices added into the json

        dms_indices : (list(int)) : list of dm indices added into the json

        file_name : (str) : output file name
    """

    if(wfss_indices is None):
        wfss_indices = list(range(len(sup.config.p_wfss)))
    elif(isinstance(wfss_indices,int)):
         wfss_indices = list(range(wfss_indices))

    if(dms_indices is None):
        dms_indices = list(range(len(sup.config.p_dms)))
    elif(isinstance(dms_indices,int)):
         dms_indices = list(range(dms_indices))

    # general
    sys_json={
        "diam" : {
            "comment": " meter      : telescope diameter",
            "value" : sup.config.p_tel.get_diam()
        },
        "cobs" : {
            "comment": " percent    : central obscuration",
            "value" :  sup.config.p_tel.get_cobs()
        },
        "tFrame": {
            "comment": " second     : frame rate",
            "value": sup.config.p_loop.ittime
        },
        "fracsub": {
            "comment": "Minimal illumination fraction for valid subap",
            "value": sup.config.p_wfss[0].get_fracsub()
        },
        "throughAtm": {
            "comment": "percent    : atmosphere transmission",
            "value": 1.0
        },
        "tracking": {
            "comment": "arcsec^2  : telescope tracking error parameters (x^2, y^2 and xy)",
            "value": [
                1.0,
                1.0,
                1.0
            ]
        }
    }

    # diam = sup.config.p_tel.get_diam()
    geom = sup.config.p_geom
    #WFSs
    lgs_json = []
    ngs_json = []
    target_json = None
    ts_json = None
    for i in wfss_indices:
        w = sup.config.p_wfss[i]
        if w in sup.config.p_wfs_lgs:
            lgs_json.append(common.wfs_to_json(w,geom,"lgs"))
        elif w in sup.config.p_wfs_ngs:
            if( i == (len(sup.config.p_wfss) - 1) ):
                target_json = common.wfs_to_json(w,geom,"target",
                    x_pos = [t.xpos for t in sup.config.p_targets],
                y_pos = [t.ypos for t in sup.config.p_targets] )
            else:
                ngs_json.append(common.wfs_to_json(w,geom,"ngs"))
    if ts :
        w = sup.config.p_wfs_ts
        if(w[0].nxsub == 0):
            argmax = np.array([sup.config.p_wfss[i].nxsub for i in wfss_indices]).argmax()
            w[0].set_nxsub(sup.config.p_wfss[argmax].nxsub)
            w[0].set_pdiam(sup.config.p_wfss[argmax]._pdiam)
        ts_json = common.wfs_to_json(w,geom,"ts")

    wfs_json = {
        "notice_lgs" : common.wfs_json_notice("lgs"),
        "notice_ngs" : common.wfs_json_notice("ngs"),
        "lgs" : lgs_json,
        "ngs" : ngs_json,
        "ts" : ts_json,
        "target":target_json}

    sys_json["wfs"] = wfs_json

    #DMs
    dm_json = []
    for i in dms_indices:
        d = sup.config.p_dms[i]
        dm_json.append(common.dm_to_json(d, geom))
    sys_json["dm"] = dm_json

    f = open(file_name, "w")
    f.write(json.dumps({"instrument":sys_json},indent=4))
    f.close()
