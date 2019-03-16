"""
Functions used for modal optimization control
"""
import numpy as np
from shesha.sutra_wrap import Sensors, Telescope, Rtc_FFF as Rtc, Atmos


def openLoopSlp(tel: Telescope, atmos: Atmos, wfs: Sensors, rtc: Rtc, nrec: int,
                ncontrol: int, p_wfss: list):
    """ Return a set of recorded open-loop slopes, usefull for initialize modal control optimization

    :parameters:

        tel: (Telescope) : Telescope object

        atmos: (Atmos) : Atmos object

        wfs: (Sensors) : Sensors object

        rtc: (Rtc) : Rtc object

        nrec: (int) : number of samples to record

        ncontrol: (int) : controller's index

        p_wfss: (list of Param_wfs) : wfs settings
    """
    # TEST IT
    ol_slopes = np.zeros((sum([2 * p_wfss[i]._nvalid
                               for i in range(len(p_wfss))]), nrec), dtype=np.float32)

    print("Recording " + str(nrec) + " open-loop slopes...")
    for i in range(nrec):
        atmos.move_atmos()

        if (p_wfss is not None and wfs is not None):
            for j in range(len(p_wfss)):
                wfs.d_wfs[j].d_gs.raytrace(atmos)
                wfs.d_wfs[j].d_gs.raytrace(tel)
                wfs.d_wfs[j].comp_image()
                rtc.d_centro[j].get_cog()
                ol_slopes[j * p_wfss[j]._nvalid * 2:(j + 1) * p_wfss[j]._nvalid *
                          2, i] = np.array(wfs.d_wfs[j].d_slopes)
    print("Done")
    return ol_slopes
