"""
Simulator class definition
Must be instantiated for running a COMPASS simulation script easily
"""
import sys
import os
'''
    Binding struct for all initializers - good for subclassing modules
'''


class init:
    from shesha.init.geom_init import tel_init
    from shesha.init.atmos_init import atmos_init
    from shesha.init.rtc_init import rtc_init
    from shesha.init.dm_init import dm_init
    from shesha.init.target_init import target_init
    from shesha.init.wfs_init import wfs_init


import shesha.constants as scons
import shesha.util.hdf5_utils as h5u

import time

from typing import Iterable, Any, Dict
from shesha.sutra_bind.wrap import naga_context, Sensors, Dms, Rtc, Atmos, Telescope, Target


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
        self.atm = None  # type: Atmos
        self.tel = None  # type: Telescope
        self.tar = None  # type: Target
        self.rtc = None  # type: Rtc
        self.wfs = None  # type: Sensors
        self.dms = None  # type: Dms

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
        load_config_from_file(self, filepath)

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

        self._tel_init(r0, ittime)

        self._atm_init(ittime)

        self._dms_init()

        self._tar_init()

        self._wfs_init()

        self._rtc_init(ittime)

        self.is_init = True
        if self.use_DB:
            h5u.validDataBase(os.environ["SHESHA_ROOT"] + "/data/dataBase/",
                              self.matricesToLoad)

    def _tel_init(self, r0: float, ittime: float) -> None:
        """
        Initializes the Telescope object in the simulator
        """
        print("->tel")
        self.tel = init.tel_init(self.c, self.config.p_geom, self.config.p_tel, r0,
                                 ittime, self.config.p_wfss)

    def _atm_init(self, ittime: float) -> None:
        """
        Initializes the Atmos object in the simulator
        """
        if self.config.p_atmos is not None:
            #   atmos
            print("->atmos")
            self.atm = init.atmos_init(
                    self.c, self.config.p_atmos, self.config.p_tel, self.config.p_geom,
                    ittime, p_wfss=self.config.p_wfss, p_target=self.config.p_target,
                    dataBase=self.matricesToLoad, use_DB=self.use_DB)
        else:
            self.atm = None

    def _dms_init(self) -> None:
        """
        Initializes the DMs object in the simulator
        """
        if self.config.p_dms is not None:
            #   dm
            print("->dm")
            self.dms = init.dm_init(self.c, self.config.p_dms, self.config.p_tel,
                                    self.config.p_geom, self.config.p_wfss)
        else:
            self.dms = None

    def _tar_init(self) -> None:
        """
        Initializes the Target object in the simulator
        """
        if self.config.p_target is not None:
            print("->target")
            self.tar = init.target_init(self.c, self.tel, self.config.p_target,
                                        self.config.p_atmos, self.config.p_tel,
                                        self.config.p_geom, self.config.p_dms,
                                        brahma=False)
        else:
            self.tar = None

    def _wfs_init(self) -> None:
        """
        Initializes the WFS object in the simulator
        """
        if self.config.p_wfss is not None:
            print("->wfs")
            self.wfs = init.wfs_init(self.c, self.tel, self.config.p_wfss,
                                     self.config.p_tel, self.config.p_geom,
                                     self.config.p_dms, self.config.p_atmos)
        else:
            self.wfs = None

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
                    self.config.p_dms, brahma=False, dataBase=self.matricesToLoad,
                    use_DB=self.use_DB)
        else:
            self.rtc = None

    def next(self, *, move_atmos: bool=True, see_atmos: bool=True, nControl: int=0,
             tar_trace: Iterable[int]=None, wfs_trace: Iterable[int]=None,
             do_control: bool=True, apply_control: bool=True) -> None:
        '''
        Iterates the AO loop, with optional parameters

        :parameters:
             move_atmos: (bool): move the atmosphere for this iteration, default: True

             nControl: (int): Controller number to use, default 0 (single control configurations)

             tar_trace: (None or list[int]): list of targets to trace. None equivalent to all.

             wfs_trace: (None or list[int]): list of WFS to trace. None equivalent to all.

             apply_control: (bool): (optional) if True (default), apply control on DMs
        '''
        if tar_trace is None and self.tar is not None:
            tar_trace = range(self.config.p_target.ntargets)
        if wfs_trace is None and self.wfs is not None:
            wfs_trace = range(len(self.config.p_wfss))

        if move_atmos and self.atm is not None:
            self.atm.move_atmos()

        if (
                self.config.p_controllers is not None and
                self.config.p_controllers[nControl].type == scons.ControllerType.GEO):
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

                if not self.config.p_wfss[w].openloop and self.dms is not None:
                    self.wfs.raytrace(w, b"dm", dms=self.dms)
                self.wfs.comp_img(w)
        if do_control:
            self.rtc.do_centroids(nControl)
            self.rtc.do_control(nControl)
            self.rtc.do_clipping(0, -1e5, 1e5)
            if apply_control:
                self.rtc.apply_control(nControl, self.dms)
        self.iter += 1

    def print_strehl(self, monitoring_freq: int, t1: float, nCur: int=0, nTot: int=0,
                     nTar: int=0):
        framerate = monitoring_freq / t1
        self.tar.comp_image(nTar)
        strehltmp = self.tar.get_strehl(nTar)
        etr = (nTot - nCur) / framerate
        print("%d \t %.3f \t  %.3f\t     %.1f \t %.1f" % (nCur + 1, strehltmp[0],
                                                          strehltmp[1], etr, framerate))

    def loop(self, n: int=1, monitoring_freq: int=100, **kwargs):
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
        t1 = time.time()
        if n == -1:
            i = 0
            while (True):
                self.next(**kwargs)
                if ((i + 1) % monitoring_freq == 0):
                    self.print_strehl(monitoring_freq, time.time() - t1, i, i)
                    t1 = time.time()
                i += 1

        for i in range(n):
            self.next(**kwargs)
            if ((i + 1) % monitoring_freq == 0):
                self.print_strehl(monitoring_freq, time.time() - t1, i, n)
                t1 = time.time()
        t1 = time.time()
        print(" loop execution time:", t1 - t0, "  (", n, "iterations), ", (t1 - t0) / n,
              "(mean)  ", n / (t1 - t0), "Hz")


def load_config_from_file(sim_class, filepath: str) -> None:
    """
    Load the parameters from the parameters file

    :parameters:
        filepath: (str): path to the parameters file

    """
    sim_class.loaded = False
    sim_class.is_init = False
    filename = filepath.split('/')[-1]
    if (len(filepath.split('.')) > 1 and filepath.split('.')[-1] != "py"):
        raise ValueError("Config file must be .py")

    pathfile = filepath.split(filename)[0]
    if (pathfile not in sys.path):
        sys.path.insert(0, pathfile)

    print("loading: %s" % filename.split(".py")[0])
    sim_class.config = __import__(filename.split(".py")[0])
    del sys.modules[sim_class.config.__name__]  # Forced reload
    sim_class.config = __import__(filename.split(".py")[0])

    # exec("import %s as wao_config" % filename.split(".py")[0])
    sys.path.remove(pathfile)

    # Set missing config attributes to None
    if not hasattr(sim_class.config, 'p_loop'):
        sim_class.config.p_loop = None
    if not hasattr(sim_class.config, 'p_geom'):
        sim_class.config.p_geom = None
    if not hasattr(sim_class.config, 'p_tel'):
        sim_class.config.p_tel = None
    if not hasattr(sim_class.config, 'p_atmos'):
        sim_class.config.p_atmos = None
    if not hasattr(sim_class.config, 'p_dms'):
        sim_class.config.p_dms = None
    if not hasattr(sim_class.config, 'p_target'):
        sim_class.config.p_target = None
    if not hasattr(sim_class.config, 'p_wfss'):
        sim_class.config.p_wfss = None
    if not hasattr(sim_class.config, 'p_centroiders'):
        sim_class.config.p_centroiders = None
    if not hasattr(sim_class.config, 'p_controllers'):
        sim_class.config.p_controllers = None

    if not hasattr(sim_class.config, 'simul_name'):
        sim_class.config.simul_name = None

    sim_class.loaded = True
