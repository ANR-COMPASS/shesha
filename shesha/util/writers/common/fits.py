import numpy as np
from shesha.util.writers.common import dm
from shesha.util.writers.common import wfs
from astropy.io import fits

def wfs_to_fits_hdu(sup, wfs_id):
    """Return a fits Header Data Unit (HDU) representation of a single WFS

    Args:
        sup : (compasSSupervisor) : supervisor

        wfs_id : (int) : index of the WFS in the supervisor

    Returns:
        hdu : (ImageHDU) : fits representation of the WFS
    """
    hdu_name = "WFS" + str(wfs_id)
    X,Y = wfs.get_subap_pos_meter(sup, wfs_id)
    valid_subap = np.array([X,Y],dtype=np.float64)
    hdu = fits.ImageHDU( valid_subap, name=hdu_name)
    hdu.header["NSSP"] = sup.config.p_wfss[wfs_id].get_nxsub()
    hdu.header["SSPSIZE"] = sup.config.p_wfss[wfs_id].get_subapd()
    return hdu

def dm_to_fits_hdu(sup, dm_id):
    """Return a fits Header Data Unit (HDU) representation of a single DM

    Args:
        sup : (compasSSupervisor) : supervisor

        wfs_id : (int) : index of the DM in the supervisor

    Returns:
        hdu : (ImageHDU) : fits representation of the DM
    """
    hdu_name = "DM" + str(dm_id)
    X,Y = dm.get_actu_pos_meter(sup, dm_id)
    valid_subap = np.array([X,Y],dtype=np.float64)
    hdu = fits.ImageHDU( valid_subap, name=hdu_name)
    hdu.header["NACTU"] = sup.config.p_dms[dm_id].get_nact()
    hdu.header["PITCH"] = sup.config.p_dms[dm_id].get_pitch()
    hdu.header["COUPLING"] = sup.config.p_dms[dm_id].get_coupling()
    hdu.header["ALT"] = sup.config.p_dms[dm_id].get_alt()
    return hdu

def dm_influ_to_fits_hdu(sup, dm_id, *, influ_index=-1):
    """Return a fits Header Data Unit (HDU) holding the influence functions of a specific DM

    Args:
        sup : (compasSSupervisor) : supervisor

        wfs_id : (int) : index of the DM in the supervisor

    Kwargs:
        influ_index : (int) : (optional) default -1, index of the actuator to get the influence function from. -1 : get all influence functions

    Returns:
        hdu : (ImageHDU) : hdu holding the DM influence functions
    """
    hdu_name = "INFLU_DM" + str(dm_id)
    if influ_index < 0 :
        influ_fct = sup.config.p_dms[dm_id].get_influ().astype(np.float64)
    else :
        influ_fct = sup.config.p_dms[dm_id].get_influ()[:,:,influ_index].astype(np.float64)
    hdu = fits.ImageHDU( influ_fct, name=hdu_name)
    return hdu

def write_data(file_name, sup, *, wfss_indices=None, dms_indices=None,
               controller_id=0, influ=0, compose_type="controller"):
    """ Write data for yao compatibility

    write into a single fits:
        * number of valide subapertures
        * number of actuators
        * subapertures position (2-dim array x,y) in meters centered
        * actuator position (2-dim array x,y) in pixels starting from 0
        * interaction matrix (2*nSubap , nactu)
        * command matrix (nacy , 2*nSubap)

    Args:
        file_name : (str) : data file name

        sup : (compasSSupervisor) : supervisor

    Kargs:
        wfss_indices : (list[int]) : optional, default all, list of the wfs indices to include

        dms_indices : (list[int]) : optional, default all, list of the DM indices to include

        controller_id : (int) : optional, index of the controller passed to yao

        influ : (int) : optional, actuator index for the influence function

        compose_type : (str) : optional, possibility to specify split tomography case ("controller" or "splitTomo")
    """
    print("writing data to" + file_name)
    hdul=fits.HDUList([])

    # setting list of wfs and dm
    conf = sup.config
    if(wfss_indices is None):
        wfss_indices = np.arange(len(conf.p_wfss))
    if(dms_indices is None):
        dms_indices = []
        for i in range(len(conf.p_dms)):
            if( conf.p_dms[i].type != "tt"):
                dms_indices.append(i)

    #cout the number of lgs
    n_lgs = 0
    for i in wfss_indices :
        if(conf.p_wfss[i].get_gsalt() > 0):
            n_lgs += 1

    #primary hdu contains only keywords for sanity check
    hdu = fits.PrimaryHDU(np.zeros(1,dtype=np.int32))
    hdu.header["DIAM"] = conf.p_tel.get_diam()
    hdu.header["COBS"] = conf.p_tel.get_cobs()
    hdu.header["NLGS"] = n_lgs
    hdu.header["NNGS"] = len(wfss_indices) - n_lgs
    hdu.header["NDM" ] = len(dms_indices)
    hdu.header["PIXSIZE"] = conf.p_geom.get_pixsize()

    #add primary hdu to list
    hdul.append(hdu)

    # add wfss
    for i in wfss_indices:
        hdul.append( wfs_to_fits_hdu(sup, i))

    # add dm
    for i in dms_indices:
        hdul.append(dm_to_fits_hdu(sup, i))
        hdul.append(dm_influ_to_fits_hdu(sup, i, influ_index = influ))

    # if(controller_id > -1):
        # IMAT
        # interaction_mat=imat.compose_imat(sup, compose_type=compose_type,
        #                   controller_id=controller_id)
        # hdu_imat=fits.ImageHDU(interaction_mat,name="IMAT")

        # CMAT
        # hdu_cmat=fits.ImageHDU(sup.rtc.get_command_matrix(controller_id),
        #                        name="CMAT")

    print("\t* number of subaperture per WFS")
    print("\t* subapertures position")
    print("\t* number of actuator per DM")
    print("\t* actuators position")
    print("\t* Imat")
    print("\t* Cmat")

    hdul.writeto(file_name, overwrite=1)