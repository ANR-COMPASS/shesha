from importlib import reload
import numpy as np
from astropy.io import fits

from shesha.ao import imats
from shesha.ao import cmats

from shesha.util.tao import writer
from shesha.util.tao import ltao
from shesha.util.tao import mcao
reload(ltao)
reload(mcao)

TILE_SIZE="1000"

STARPU_FLAGS=""

#variable necessary to run TAO
TAO_SETTINGS={"SCHED":"dmdas",
      "STARPU_FLAGS":"",
      "GPU_IDS":0,
      "TILE_SIZE":TILE_SIZE,
      "INPUT_PATH":0,
      "TAO_PATH":0
      }


def check():
    """Checks that variable are initialized
    """
    stop=0
    try :
        if (not isinstance(TAO_SETTINGS["SCHED"], str)):
            print("you must select a scheduler (dmda,dmdas,dmdar...)\n\tex: TAO_SETTINGS[\"SCHED\"]=\"dmdas\"")
            stop=1
    except:
        print("you must select a scheduler (dmda,dmdas,dmdar...)\n\tex: TAO_SETTINGS[\"SCHED\"]=\"dmdas\"")
        stop=1
    try :
        if( not isinstance(TAO_SETTINGS["GPU_IDS"], str)):
            print("you must define the GPUs to use as a string \n\tex:TAO_SETTINGS[\"GPU_IDS\"]=\"1,2\"")
            stop=1
    except:
        print("you must define the GPUs to use as a string \n\tex:TAO_SETTINGS[\"GPU_IDS\"]=\"1,2\"")
        stop=1
    try :
        if( not isinstance(TAO_SETTINGS["INPUT_PATH"], str)):
            print("you must define the location of the system parameters \n\tex: TAO_SETTINGS[\"INPUT_PATH\"]=\"~/workspace/compass/params\"")
            stop=1
    except:
        print("you must define the location of the system parameters \n\tex: TAO_SETTINGS[\"INPUTPATH\"]=\"~/workspace/compass/params\"")
        stop=1
    try :
        if( not isinstance(TAO_SETTINGS["TAO_PATH"], str)):
            print("you must define the location of the tao executables \n\tex: TAO_SETTINGS[\"TAO_PATH\"]=\"~/workspace/tao/install/bin\"")
            stop=1
    except:
        print("you must define the location of the tao executables \n\tex: TAO_SETTINGS[\"TAOPATH\"]=\"~/workspace/tao/install/bin\"")
        stop=1
    try :
        TAO_SETTINGS["STARPU_FLAGS"]
    except:
        TAO_SETTINGS["STARPU_FLAGS"]=""

    return stop


def init(sup, mod, *,wfs="all", dm_use_tt=False, n_filt=None):
    """ Set up the compass loop

    set the interaction matrix, loop gain and write parameter files for TAO

    Args:

        sup : (CompassSupervisor) : current supervisor

        mod : (module) : AO mode requested (among: ltao , mcao)

    Kwargs:
        wfs : (str) : (optional), default "all" wfs used by tao ( among "all", "lgs", "ngs")

        dm_use_tt : (bool) :(optional), default False using a TT DM

        n_filt : (int) : (optional), default None number of meta interaction matrix singular values filtered out
    """

    #setting open loop
    sup.rtc._rtc.d_control[0].set_polc(True)

    if n_filt is None:
        mod.init(TAO_SETTINGS, sup, dm_use_tt=dm_use_tt, wfs=wfs)
    else:
        mod.init(TAO_SETTINGS, sup, dm_use_tt=dm_use_tt, wfs=wfs, n_filt=n_filt)

def reconstructor(mod):
    """ Compute the TAO reconstructor for a given AO mode

    Args:
        mod : (module)  : AO mode requested (among: ltao , mcao)
    """
    return mod.reconstructor(TAO_SETTINGS)


def run(sup, mod, *, n_iter=1000, initialisation=True, reset=True, wfs="all",
    dm_use_tt=False, n_filt=None):
    """ Computes a tao reconstructor and run a compass loop with it

    Args:
        sup : (CompassSupervisor) : current supervisor

        mod : (module) : AO mode requested (among: ltao , mcao)

    Kwargs
        n_iter : (int) : (optional), default 1000 number of iteration of the ao loop

        initialisation : (bool) : (optional), default True initialise tao (include comptation of meta matrices of interaction/command)

        reset : (bool) : (optional), default True reset the supervisor before the loop

        wfs : (str) : (optional), default "all" wfs used by tao ( among "all", "lgs", "ngs")

        dm_use_tt : (bool) :(optional), default False using a TT DM

        n_filt : (int) : (optional), default None number of meta interaction matrix singular values filtered out
    """
    check()

    #setting open loop
    sup.rtc._rtc.d_control[0].set_polc(True)

    #if generic: need to update imat in controller
    if(np.abs(np.array(sup.rtc._rtc.d_control[0].d_imat)).max()==0):
        #imat not set yet for controller
        sup.rtc._rtc.d_control[0].set_imat(sup.config.p_controllers[0]._imat)
    #update gain
    sup.rtc._rtc.set_gain(0,sup.config.p_controllers[0].gain)

    if(initialisation):
        init(sup, mod, wfs=wfs, dm_use_tt=dm_use_tt, n_filt=n_filt)
    M=reconstructor(mod)
    if(reset):
        sup.reset()
    cmat_shape=sup.rtc.get_command_matrix(0).shape
    if(M.shape[0] != cmat_shape[0] or M.shape[1] != cmat_shape[1]):
        print("ToR shape is not valid:\n\twaiting for:",cmat_shape,"\n\tgot        :",M.shape)
    else:
        sup.rtc.set_command_matrix(0,M)
        if(n_iter>0):
            sup.loop(n_iter)
