import numpy as np
import json

def get_subap_pos_pixel(wfs):
    """Return the coordinates of the valid subapertures of a given WFS

    these coordinates are given in pixels

    Args:
        wfs : Param_wfs : wfs to get the subapertures position from

    Return:
        valid_X : (np.ndarray[ndim=1, dtype=np.float64]) : subapertures positions along axis x

        valid_Y : (np.ndarray[ndim=1, dtype=np.float64]) : subapertures positions along axis y
    """

    return wfs._validpuppixx-2 , wfs._validpuppixy-2


def get_subap_pos_meter(sup, wfs_id):
    """Return the coordinates of the valid subapertures of a given WFS

    these coordinates are given in meters and centered

    Args:
        sup : (compassSupervisor) : supervisor

        wfs_id : (int) : index of the WFS

    Return :
        valid_X : (np.ndarray[ndim=1, dtype=np.float64]) : subapertures positions along axis x

        valid_Y : (np.ndarray[ndim=1, dtype=np.float64]) : subapertures positions along axis y
    """

    config = sup.config
    wfs = config.p_wfss[wfs_id]
    geom = config.p_geom
    valid_X, valid_Y = get_subap_pos_pixel(wfs)
    total = geom.pupdiam/wfs.nxsub*(wfs.nxsub-1)
    valid_X = (valid_X-total/2)*geom.get_pixsize()
    valid_Y = (valid_Y-total/2)*geom.get_pixsize()
    return valid_X, valid_Y


def wfs_to_json(wfs, geom, type, *, x_pos=None, y_pos=None):
    """return a json description of a wfs

    Args:
        wfs : (Param_wfs) : wfs to represent as json

        geom : (Param_geom) : geom settings

        type : (string) : wfs type ("lgs", "ngs" "target" or "ts")

    Kargs:
        x_pos : (list(float)) : x coordinates of the targets ()

        y_pos : (list(float)) : y coordinates of the targets ()
    """
    types = ["lgs", "ngs", "target", "ts"]
    if(type not in types):
        ValueError("type must be one of "+str(types))

    wfs_json={}

    if(type == "ts"):
        wfs_json = {
            "nssp" : wfs[0].get_nxsub(),
            "alphaX_as" : [w.get_xpos() for w in wfs],
            "alphaY_as" : [w.get_ypos() for w in wfs]
        }

    elif(type == "target"):
        if(x_pos is None or len(x_pos) != len(y_pos)):
            ValueError("pointing direction of WFS target must be provided (x_pos, y_pos)")
        wfs_json = {
            "nssp" : wfs.get_nxsub(),
            "alphaX_as" : x_pos,
            "alphaY_as" : y_pos
        }

    else :
        bdw = 3.3e-7
        lgs_depth = 5000.
        lgs_cst = 0.1
        wfs_json = {
            "nssp" : wfs.get_nxsub(),
            "alphaX_as" : wfs.get_xpos(),
            "alphaY_as" : wfs.get_ypos(),
            "XPup" : wfs.get_dx() * geom.get_pixsize(),
            "YPup" : wfs.get_dy() * geom.get_pixsize(),
            "thetaML" : wfs.get_thetaML() ,
            "thetaCam" : 0 ,
            "sensitivity" : 0 ,
            "pixSize" :  wfs.get_pixsize(),
            "lambdaWFS" : wfs.get_Lambda() ,
            "bandwidth" : bdw ,
            "throughput" : wfs.get_optthroughput() ,
            "RON" : wfs.get_noise()
        }

        if(wfs.get_gsalt()>0):
            if(type == "ngs"):
                ValueError("wfs is not a NGS (gsalt > 0)")

            wfs_json["lgsAlt"] = wfs.get_gsalt()
            wfs_json["lgsDepth"] = lgs_depth
            wfs_json["lgsFlux"] = wfs.lgsreturnperwatt * wfs.laserpower * \
                wfs.optthroughput * 10**4
            wfs_json["spotWidth"] = wfs.get_beamsize()
            wfs_json["lgsCst"] = lgs_cst

        else:
            if(type == "lgs"):
                ValueError("wfs is not a LGS (gsalt == 0) ")
            wfs_json["magnitude"] = wfs.get_gsmag()

    return wfs_json

def wfs_json_notice(type):
    """Return the notice of the wfs json representation

    Args:
        type : (string) : wfs type ("lgs", "ngs" or "target")
    """
    if(type != "lgs" and type != "ngs" and type != "target"):
        ValueError("type must be either \"lgs\",  \"ngs\" or \"target\"")
    if(type == "target"):
        notice = {
            "nssp" : "            : number of subapertures along the diameter",
            "alphaX_as" : " arcsec     : list of pointing direction of the wfs (on x axis)",
            "alphaY_as" : " arcsec     : list of pointing direction of the wfs (on y axis)",
        }
    else :
        notice = {
            "nssp" : "            : number of subapertures along the diameter",
            "alphaX_as" : " arcsec     : pointing direction of the wfs (on x axis)",
            "alphaY_as" : " arcsec     : pointing direction of the wfs (on y axis)",
            "XPup" : " meter      : pupil shift of the WFS (on axis x)",
            "YPup" : " meter      : pupil shift of the WFS (on axis y)",
            "thetaML" : " radian     : rotation of the camera",
            "thetaCam" : " radian     : rotation of the microlenses",
            "sensitivity" : "            : sensitivity coeff of this WFS",
            "pixSize" :  " arcsec     : WFS pixel size",
            "lambdaWFS" : " meter      : WFS wavelength",
            "bandwidth" : " meter      : WFS bandwidth",
            "throughput" : " percent    : transmission for the GS",
            "RON" :  " nb of e-   : Read Out Noise",
        }
        if(type == "lgs"):
            notice["lgsAlt"] = " meter      : laser guide star altitude"
            notice["lgsDepth"] = " meter      : laser guide star depth"
            notice["lgsFlux"] = " (ph/m2/s)  : LGS photon return at M1"
            notice["spotWidth"] = " arcsec     : lazer width"
            notice["lgsCst"] = "            : constant on lgs (simulate that LGS cannot measure tip-tilt and focus, for Linear Algebra purpose)"
        if(type == "ngs"):
            notice["magnitude"] = "            : guide stars magnitude"

    return notice
