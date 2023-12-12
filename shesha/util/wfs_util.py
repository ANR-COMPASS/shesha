import numpy as np
from astropy.io import fits


wfs_fits_content="""
    The primary header contains the keywords:
    * TYPE : the WFS type (only supported for now sh:shack-hartmann).

    * XCENTER, YCENTER are the coordinates of the centre of the pupil, expressed in pixels,
        in a reference frame conformable to (i,j) coords. The translation from pixels to meters
        can be done using:
        meters = (pixels - XCENTER) * PIXSIZE

    * PIXSIZE : the size of the pixels on the maps in meters.

    * PUPM is the diameter of pupil stop (meters).

    * SUBAPD : the subaperture diameter (meters) i.e. the side of the subaperture square.

    This FITS file contains a single extension:
    * Extension 'XPOS_YPOS' are the coordinates (xpos, ypos) of the bottom left corner of the
        subapertures in the pupil (in meters).
"""

def add_doc_content(*content):
    """adds content to a docstring (to be used as decorator)"""
    def dec(obj):
        obj.__doc__ = obj.__doc__.format(content)
        return obj
    return dec


def write_wfs_custom_fits(file_name:str, WFS_type:str, xpos : np.ndarray, ypos :np.ndarray,
    xcenter, ycenter,pixsize, pupm,subap_diam) -> fits.HDUList:
    """Write a custom_wfs fits file based on user provided data

    Args:
        file_name : (str)       : name of the wfs fits file

        WFS_type  : (str)       : the wfs type (sh:shack-hartmann)

        xpos      : (np.ndarray): x coordinate of the bottom left corner of the subapertures
                                  in the pupil (meters)

        ypos      : (np.ndarray): y coordinate of the bottom left corner of the subapertures
                                  in the pupil (meters)

        xcenter   : (float)     : x coordinate of the centre of the pupil, expressed in pixels

        ycenter   : (float)     : y coordinate of the centre of the pupil, expressed in pixels

        pixsize   : (float)     : size of the pixels on the maps in meters

        pupm      : (float)     : diameter of pupil stop (meters).

        subap_diam: (float)     : subaperture diameter (meters) i.e. side of the subaperture square.

    Returns:
        (HDUList) : wfs data

    """
    if(WFS_type != 'sh'):
        raise RuntimeError("Only Shack-Hartmann ('sh') supported at the moment")
    fits_version=1.2
    primary_hdu = fits.PrimaryHDU()
    primary_hdu.header['VERSION'] = (fits_version, 'file format version')
    primary_hdu.header['TYPE']    = (WFS_type    , 'WFS type')
    primary_hdu.header['XCENTER'] = (xcenter     , 'WFS centre along X in pixels')
    primary_hdu.header['YCENTER'] = (ycenter     , 'WFS centre along Y in pixels')
    primary_hdu.header['PIXSIZE'] = (pixsize     , 'pixel size (meters)')
    primary_hdu.header['PUPM']    = (pupm        , 'nominal pupil diameter (meters)')
    primary_hdu.header['SUBAPD']  = (subap_diam  , 'subaperture diameter (pix):side of the subap')

    xpos_ypos = np.c_[xpos, ypos].T.astype(np.float64)
    image_hdu = fits.ImageHDU(xpos_ypos, name="XPOS_YPOS")
    custom_wfs = fits.HDUList([primary_hdu, image_hdu])
    custom_wfs.writeto(file_name, overwrite=True)
    return custom_wfs

@add_doc_content(wfs_fits_content)
def export_custom_wfs(file_name:str, p_wfs, p_geom, *, p_tel=None):
    """Return an HDUList (FITS) with the data required for a WFS

    and write the associated fits file

    {}

    Args :
        file_name : (str)       : name of the wfs fits file

        p_wfs     : (ParamWfs) : wfs settings

    """
    pixsize = p_geom.get_pixsize()
    diam = p_geom.pupdiam * p_geom._pixsize
    if(p_tel is not None):
        diam = p_tel.diam
    diam_pix = p_wfs.get_validsubsx().max()+ p_wfs.npix - p_wfs.get_validsubsx().min()
    scale =  diam / diam_pix
    if( p_wfs.get_validsubsx().max()*scale+p_wfs.get_subapd()- diam != 0):
        print("WARNING : provided diameter does not match exactly the wfs settings")
        print("          the wfs settings span a diameter of {}m (procided : {})".format(
            p_wfs.get_validsubsx().max()*scale+p_wfs.get_subapd(), diam
        ))

    xpos = p_wfs.get_validpuppixx()
    ypos = p_wfs.get_validpuppixy()
    centerx = (xpos.max()+ p_wfs.npix + xpos.min())/2
    centery = (ypos.max()+ p_wfs.npix + xpos.min())/2
    wfs_custom = write_wfs_custom_fits(file_name, p_wfs.type, xpos, ypos, centerx, centery, pixsize,
        diam, p_wfs.npix)
    return wfs_custom