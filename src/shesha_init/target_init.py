"""
Initialization of a Target object
"""

import shesha_config as conf

import shesha_constants as scons
from shesha_constants import CONST

import numpy as np
from sutra_bind.wrap import naga_context, Target, Target_brahma, Telescope


def target_init(ctxt: naga_context, telescope: Telescope, p_target: conf.Param_target,
                p_atmos: conf.Param_atmos, p_tel: conf.Param_tel,
                p_geom: conf.Param_geom, dm=None, brahma=False):
    """Create a cython target from parametres structures

    :parameters:
        ctxt: (naga_context) :
        telescope: (Telescope): Telescope object
        target: (Param_target) : target_settings
        p_atmos: (Param_atmos) : atmos settings
        p_tel: (Param_tel) : telescope settings
        p_geom: (Param_geom) : geom settings
        dm: (Param_dm) : (optional) dm settings
        brahma: (bool): (optional) BRAHMA flag
    :return:
        tar: (Target): Target object
    """
    type_target = b"atmos"

    if (p_target.ntargets > 0):
        if (p_target.dms_seen is None and dm is not None):
            for i in range(p_target.ntargets):
                p_target.dms_seen = np.arange(len(dm))

    sizes = np.ones(p_target.ntargets, dtype=np.int64) * p_geom.pupdiam

    ceiled_pupil = np.ceil(p_geom._spupil)

    ceiled_pupil[np.where(ceiled_pupil > 1)] = 1

    if (p_target.apod == 1):
        Npts = 0
        # TODO apodizer, Npts=nb element of apodizer>0
        ceiled_apodizer = np.ceil(p_geom._apodizer * p_geom._spupil)
        ceiled_apodizer[np.where(ceiled_apodizer > 1)] = 1
        if (brahma):
            target = Target_brahma(ctxt, telescope, p_target.ntargets, p_target.xpos,
                                   p_target.ypos, p_target.Lambda, p_target.mag,
                                   p_target.zerop, sizes, Npts)
        else:
            target = Target(ctxt, telescope, p_target.ntargets, p_target.xpos,
                            p_target.ypos, p_target.Lambda, p_target.mag, p_target.zerop,
                            sizes, Npts)

    else:
        Npts = np.sum(ceiled_pupil)
        if (brahma):
            target = Target_brahma(ctxt, telescope, p_target.ntargets, p_target.xpos,
                                   p_target.ypos, p_target.Lambda, p_target.mag,
                                   p_target.zerop, sizes, Npts)
        else:
            target = Target(ctxt, telescope, p_target.ntargets, p_target.xpos,
                            p_target.ypos, p_target.Lambda, p_target.mag, p_target.zerop,
                            sizes, Npts)

    # cc=i
    for i in range(p_target.ntargets):
        if (p_atmos.nscreens > 0):
            for j in range(p_atmos.nscreens):
                xoff = p_target.xpos[i] * CONST.ARCSEC2RAD * \
                    p_atmos.alt[j] / p_atmos.pupixsize
                yoff = p_target.ypos[i] * CONST.ARCSEC2RAD * \
                    p_atmos.alt[j] / p_atmos.pupixsize
                xoff += float((p_atmos.dim_screens[j] - p_geom._n) / 2)
                yoff += float((p_atmos.dim_screens[j] - p_geom._n) / 2)
                pupdiff = (p_geom._n - p_geom.pupdiam) / 2
                xoff += pupdiff
                yoff += pupdiff
                target.add_layer(i, type_target, p_atmos.alt[j], xoff, yoff)

        # if (y_dm != []) {
        if (dm is not None):
            # j=ddd
            # for (ddd=1;ddd<=numberof(*y_target(cc).dms_seen);ddd++) {
            for j in range(p_target.dms_seen.size):
                # k=dd
                # dd = (*y_target(cc).dms_seen)(ddd)
                k = p_target.dms_seen[j]
                dims = dm[k]._n2 - dm[k]._n1 + 1
                dim = p_geom._mpupil[2].size
                dim_dm = max(dim, dims)
                xoff = p_target.xpos[i] * CONST.ARCSEC2RAD * \
                    dm[k].alt / p_tel.diam * p_geom.pupdiam
                yoff = p_target.ypos[i] * CONST.ARCSEC2RAD * \
                    dm[k].alt / p_tel.diam * p_geom.pupdiam

                xoff += float((dim_dm - p_geom._n) / 2)
                yoff += float((dim_dm - p_geom._n) / 2)

                pupdiff = (p_geom._n - p_geom.pupdiam) / 2
                xoff += pupdiff
                yoff += pupdiff

                if (dm[k].type == scons.DmType.KL):
                    xoff += 2
                    yoff += 2
                target.add_layer(i, dm[k].type, dm[k].alt, xoff, yoff)

        target.init_strehlmeter(i)

    return target
