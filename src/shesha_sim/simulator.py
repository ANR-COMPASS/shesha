"""
Simulator class definition
Must be instantiated for running a COMPASS simulation script easily
"""
import sys
import os

try:
    from naga import naga_context
except:
    class naga_context:
        def __init__(devices=0):
            pass


import shesha_init as init
import shesha_constants as scons
import shesha_util.hdf5_utils as h5u

import Atmos, Telescope, Target, Rtc, Dms, Sensors
import time

from typing import Iterable, Any, Dict


class Simulator:
    """
    The Simulator class is self sufficient for running a COMPASS simulation
    Initializes and run a COMPASS simulation
    """

    def __init__(self, filepath: str=None, use_DB: bool=False) -> None:
        """
        Initializes a Simulator instance

        :parameters:
            filepath: (str): (optional) path to the parameters file
            use_DB: (bool): (optional) flag to use dataBase system
        """
        self.is_init = False  # type: bool
        self.loaded = False  # type: bool
        self.config = None  # type: Any # types.ModuleType ?
        self.iter = 0  # type: int

        self.c = None  # type: naga_context
        self.atm = None  # type: Atmos.Atmos
        self.tel = None  # type: Telescope.Telescope
        self.tar = None  # type: Target.Target
        self.rtc = None  # type: Rtc.Rtc
        self.wfs = None  # type: Sensors.Sensors
        self.dms = None  # type: Dms.Dms

        self.matricesToLoad = {}  # type: Dict[str,str]
        self.use_DB = use_DB  # type: bool

        if filepath is not None:
            self.load_from_file(filepath)

    def __str__(self) -> str:
        """
        Print the objects created in the Simulator instance
        """
        s = ""
        if self.is_init:
            s += "====================\n"
            s += "Objects initialzed on GPU:\n"
            s += "--------------------------------------------------------\n"

            if self.atm is not None:
                s += self.atm.__str__() + '\n'
            if self.wfs is not None:
                s += self.wfs.__str__() + '\n'
            if self.dms is not None:
                s += self.dms.__str__() + '\n'
            if self.tar is not None:
                s += self.tar.__str__() + '\n'
            if self.rtc is not None:
                s += self.rtc.__str__() + '\n'
        else:
            s += "Simulator is not initialized."

        return s

    def force_context(self) -> None:
        """
        Active all the GPU devices specified in the parameters file
        """
        if self.loaded and self.c is not None:
            current_Id = self.c.get_activeDevice()
            for devIdx in range(len(self.config.p_loop.devices)):
                self.c.set_activeDeviceForce(devIdx)
            self.c.set_activeDevice(current_Id)

    def load_from_file(self, filepath: str) -> None:
        """
        Load the parameters from the parameters file

        :parameters:
            filepath: (str): path to the parameters file

        """
        self.loaded = False
        self.is_init = False
        filename = filepath.split('/')[-1]
        if (len(filepath.split('.')) > 1 and filepath.split('.')[-1] != "py"):
            raise ValueError("Config file must be .py")

        pathfile = filepath.split(filename)[0]
        if (pathfile not in sys.path):
            sys.path.insert(0, pathfile)

        if self.config is not None:
            name = self.config.__name__
            print("Removing previous config")
            self.config = None
            try:
                del sys.modules[name]
            except:
                pass

        print("loading: %s" % filename.split(".py")[0])
        self.config = __import__(filename.split(".py")[0])
        # exec("import %s as wao_config" % filename.split(".py")[0])
        sys.path.remove(pathfile)

        # Set missing config attributes to None
        if not hasattr(self.config, 'p_loop'):
            self.config.p_loop = None
        if not hasattr(self.config, 'p_geom'):
            self.config.p_geom = None
        if not hasattr(self.config, 'p_tel'):
            self.config.p_tel = None
        if not hasattr(self.config, 'p_atmos'):
            self.config.p_atmos = None
        if not hasattr(self.config, 'p_dms'):
            self.config.p_dms = None
        if not hasattr(self.config, 'p_target'):
            self.config.p_target = None
        if not hasattr(self.config, 'p_wfss'):
            self.config.p_wfss = None
        if not hasattr(self.config, 'p_centroiders'):
            self.config.p_tel = None
        if not hasattr(self.config, 'p_controllers'):
            self.config.p_tel = None

        if not hasattr(self.config, 'simul_name'):
            self.config.simul_name = None

        self.loaded = True

    def clear_init(self) -> None:
        """
        Delete objects initialized in a previous simulation
        """
        if self.loaded and self.is_init:
            self.iter = 0

            del self.atm
            self.atm = None
            del self.tel
            self.tel = None
            del self.tar
            self.tar = None
            del self.rtc
            self.rtc = None
            del self.wfs
            self.wfs = None
            del self.dms
            self.dms = None

            # del self.c  # What is this supposed to do ... ?
            # self.c = None

        self.is_init = False

    def init_sim(self) -> None:
        """
        Initializes the simulation by creating all the sutra objects that will be used
        """
        if not self.loaded:
            raise ValueError("Config must be loaded before call to init_sim")
        if (self.config.simul_name is not None and self.use_DB):
            param_dict = h5u.params_dictionary(self.config)
            self.matricesToLoad = h5u.checkMatricesDataBase(
                    os.environ["SHESHA_ROOT"] + "/data/dataBase/", self.config,
                    param_dict)
        self.c = naga_context(devices=self.config.p_loop.devices)
        self.force_context()

        if self.config.p_tel is None or self.config.p_geom is None:
            raise ValueError("Telescope geometry must be defined (p_geom and p_tel)")

        if self.config.p_atmos is not None:
            r0 = self.config.p_atmos.r0
        else:
            raise ValueError('A r0 value through a Param_atmos is required.')

        if self.config.p_loop is not None:
            ittime = self.config.p_loop.ittime
        else:
            raise ValueError(
                    'An ittime (iteration time in seconds) value through a Param_loop is required.'
            )

        print("->tel")
        self.tel = init.tel_init(self.c, self.config.p_geom, self.config.p_tel, r0,
                                 ittime, self.config.p_wfss)

        if self.config.p_atmos is not None:
            #   atmos
            print("->atmos")
            self.atm = init.atmos_init(self.c, self.config.p_atmos, self.config.p_tel,
                                       self.config.p_geom, ittime,
                                       dataBase=self.matricesToLoad, use_DB=self.use_DB)
        else:
            self.atm = None

        if self.config.p_dms is not None:
            #   dm
            print("->dm")
            self.dms = init.dm_init(self.c, self.config.p_dms, self.config.p_tel,
                                    self.config.p_geom, self.config.p_wfss)
        else:
            self.dms = None

        self._tar_init()

        if self.config.p_wfss is not None:
            print("->wfs")
            self.wfs = init.wfs_init(self.c, self.tel, self.config.p_wfss,
                                     self.config.p_tel, self.config.p_geom,
                                     self.config.p_dms, self.config.p_atmos)
        else:
            self.wfs = None

        self._rtc_init(ittime)

        self.is_init = True
        if self.use_DB:
            h5u.validDataBase(os.environ["SHESHA_ROOT"] + "/data/dataBase/",
                              self.matricesToLoad)

    def _tar_init(self) -> None:
        """
        Initializes the Target object in the simulator
        """
        if self.config.p_target is not None:
            print("->target")
            self.tar = init.target_init(self.c, self.tel, self.config.p_target,
                                        self.config.p_atmos, self.config.p_tel,
                                        self.config.p_geom, self.config.p_dms,
                                        brama=False)
        else:
            self.tar = None

    def _rtc_init(self, ittime: float) -> None:
        """
        Initializes the Rtc object in the simulator
        """
        if self.config.p_controllers is not None or self.config.p_centroiders is not None:
            print("->rtc")
            #   rtc
            self.rtc = init.rtc_init(
                    self.c, self.tel, self.wfs, self.dms, self.atm, self.config.p_wfss,
                    self.config.p_tel, self.config.p_geom, self.config.p_atmos, ittime,
                    self.config.p_centroiders, self.config.p_controllers,
                    self.config.p_dms, brama=False, dataBase=self.matricesToLoad,
                    use_DB=self.use_DB)
        else:
            self.rtc = None

    def next(self, *, move_atmos: bool=True, see_atmos: bool=True, nControl: int=0,
             tar_trace: Iterable[int]=None, wfs_trace: Iterable[int]=None,
             apply_control: bool=True) -> None:
        '''
        Iterates the AO loop, with optional parameters

        :parameters:
             move_atmos: (bool): move the atmosphere for this iteration, default: True

             nControl: (int): Controller number to use, default 0 (single control configurations)

             tar_trace: (None or list[int]): list of targets to trace. None equivalent to all.

             wfs_trace: (None or list[int]): list of WFS to trace. None equivalent to all.

             apply_control: (bool): (optional) if True (default), apply control on DMs
        '''
        if tar_trace is None:
            tar_trace = range(self.config.p_target.ntargets)
        if wfs_trace is None:
            wfs_trace = range(len(self.config.p_wfss))

        if move_atmos:
            self.atm.move_atmos()

        if (self.config.p_controllers[nControl].type == scons.ControllerType.GEO):
            for t in tar_trace:
                if see_atmos:
                    self.tar.raytrace(t, b"atmos", atmos=self.atm)
                else:
                    self.tar.reset_phase(t)
                self.tar.raytrace(t, b"telncpa", tel=self.tel, ncpa=1)
                self.tar.raytrace(t, b"dm", dms=self.dms)
                self.rtc.do_control_geo(nControl, self.dms, self.tar, t)
                self.rtc.apply_control(nControl, self.dms)
        else:
            for t in tar_trace:
                if see_atmos:
                    self.tar.raytrace(t, b"atmos", atmos=self.atm)
                else:
                    self.tar.reset_phase(t)
                self.tar.raytrace(t, b"dm", tel=self.tel, dms=self.dms, ncpa=1)
            for w in wfs_trace:

                if see_atmos:
                    self.wfs.raytrace(w, b"atmos", tel=self.tel, atmos=self.atm, ncpa=1)
                else:
                    self.wfs.raytrace(w, b"telncpa", tel=self.tel, rst=1, ncpa=1)

                if not self.config.p_wfss[w].openloop:
                    self.wfs.raytrace(w, b"dm", dms=self.dms)
                self.wfs.comp_img(w)
            self.rtc.do_centroids(nControl)
            self.rtc.do_control(nControl)
            self.rtc.do_clipping(0, -1e5, 1e5)
            if apply_control:
                self.rtc.apply_control(nControl, self.dms)
        self.iter += 1

    def loop(self, n=1, monitoring_freq=100, **kwargs):
        """
        Perform the AO loop for n iterations

        :parameters:
            n: (int): (optional) Number of iteration that will be done
            monitoring_freq: (int): (optional) Monitoring frequency [frames]
        """
        print("----------------------------------------------------")
        print("iter# | S.E. SR | L.E. SR | ETR (s) | Framerate (Hz)")
        print("----------------------------------------------------")
        t0 = time.time()
        for i in range(n):
            self.next(**kwargs)
            if ((i + 1) % monitoring_freq == 0):
                framerate = (i + 1) / (time.time() - t0)
                self.tar.comp_image(0)
                strehltmp = self.tar.get_strehl(0)
                etr = (n - i) / framerate
                print("%d \t %.3f \t  %.3f\t     %.1f \t %.1f" %
                      (i + 1, strehltmp[0], strehltmp[1], etr, framerate))
        t1 = time.time()
        print(" loop execution time:", t1 - t0, "  (", n, "iterations), ", (t1 - t0) / n,
              "(mean)  ", n / (t1 - t0), "Hz")
