## @package   shesha.supervisor
## @brief     User layer for initialization and execution of a COMPASS simulation
## @author    COMPASS Team <https://github.com/ANR-COMPASS>
## @version   5.0.0
## @date      2020/05/18
## @copyright GNU Lesser General Public License
#
#  This file is part of COMPASS <https://anr-compass.github.io/compass/>
#
#  Copyright (C) 2011-2019 COMPASS Team <https://github.com/ANR-COMPASS>
#  All rights reserved.
#  Distributed under GNU - LGPL
#
#  COMPASS is free software: you can redistribute it and/or modify it under the terms of the GNU Lesser
#  General Public License as published by the Free Software Foundation, either version 3 of the License,
#  or any later version.
#
#  COMPASS: End-to-end AO simulation tool using GPU acceleration
#  The COMPASS platform was designed to meet the need of high-performance for the simulation of AO systems.
#
#  The final product includes a software package for simulating all the critical subcomponents of AO,
#  particularly in the context of the ELT and a real-time core based on several control approaches,
#  with performances consistent with its integration into an instrument. Taking advantage of the specific
#  hardware architecture of the GPU, the COMPASS tool allows to achieve adequate execution speeds to
#  conduct large simulation campaigns called to the ELT.
#
#  The COMPASS platform can be used to carry a wide variety of simulations to both testspecific components
#  of AO of the E-ELT (such as wavefront analysis device with a pyramid or elongated Laser star), and
#  various systems configurations such as multi-conjugate AO.
#
#  COMPASS is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the
#  implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
#  See the GNU Lesser General Public License for more details.
#
#  You should have received a copy of the GNU Lesser General Public License along with COMPASS.
#  If not, see <https://www.gnu.org/licenses/lgpl-3.0.txt>.
from shesha.init.rtc_init import rtc_init
from shesha.supervisor.components.sourceCompass import SourceCompass
import shesha.constants as scons
import numpy as np
from typing import Union

class RtcCompass(object):
    """ RTC handler for compass simulation

    Attributes:
        _rtc : (sutraWrap.Rtc) : Sutra rtc instance

        _context : (carmaContext) : CarmaContext instance

        _config : (config module) : Parameters configuration structure module    

        cacao : (bool) : CACAO features enabled in the RTC  
    """
    def __init__(self, context, config, tel, wfs, dms, atm, *, cacao=False):
        """ Initialize a RtcCompass component for rtc related supervision

        Args:
            context : (carmaContext) : CarmaContext instance

            config : (config module) : Parameters configuration structure module

            tel : (TelescopeCompass) : A TelescopeCompass instance

            wfs : (WfsCompass) : A WfsCompass instance

            dms : (DmCompass) : A DmCompass instance

            atm : (AtmosCompass) : An AtmosCompass instance

        Kwargs:
            cacao : (bool) : If True, enables CACAO features in RTC (Default is False)
                                      /!\ Requires OCTOPUS to be installed
        """
        self.cacao = cacao
        self._context = context
        self._config = config # Parameters configuration coming from supervisor init
        print("->rtc init")
        self._rtc = rtc_init(self._context, tel._tel, wfs._wfs, dms._dms, atm._atmos,
                                self._config.p_wfss, self._config.p_tel,
                                self._config.p_geom, self._config.p_atmos, self._config.p_loop.ittime,
                                self._config.p_centroiders, self._config.p_controllers,
                                self._config.p_dms, cacao=cacao)

    def set_perturbation_voltage(self, controller_index: int, name: str,
                                 command: np.ndarray) -> None:
        """ Add circular buffer of offset values to integrator (will be applied at the end of next iteration)

        Args:
            controller_index : (int) : Controller index

            name : (str) : Buffer name

            command : (np.ndarray) : perturbation voltage circular buffer
        """
        if len(command.shape) == 1:
            self._rtc.d_control[controller_index].set_perturb_voltage(name, command, 1)
        elif len(command.shape) == 2:
            self._rtc.d_control[controller_index].set_perturb_voltage(
                    name, command, command.shape[0])
        else:
            raise AttributeError("command should be a 1D or 2D array")

    def get_slopes(self, controller_index: int) -> np.ndarray:
        """ Return the current slopes vector of the controller_index controller

        Args:
            controller_index : (int) : controller index handling the slopes

        Return:
            slopes : (np.ndarray) : Current slopes vector containing slopes of all
                                    the WFS handled by the specified controller
        """
        return np.array(self._rtc.d_control[controller_index].d_centroids)

    def close_loop(self, controller_index: int=None) -> None:
        """ DM receives controller output + pertuVoltage

        Kwargs:
            controller_index: (int): controller index. 
                                               If None (default), apply on all controllers
        """
        if controller_index is None:
            for controller in self._rtc.d_control:
                controller.set_open_loop(0)
        else:
            self._rtc.d_control[controller_index].set_open_loop(0)  # close_loop

    def open_loop(self, controller_index: int=None, reset=True) -> None:
        """ Integrator computation goes to /dev/null but pertuVoltage still applied

        Kwargs:
            controller_index: (int): controller index.
                                     If None (default), apply on all controllers

            reset : (bool) : If True (default), integrator is reset
        """
        if controller_index is None:
            for controller in self._rtc.d_control:
                controller.set_open_loop(1, reset)
        else:
            self._rtc.d_control[controller_index].set_open_loop(1, reset)  # open_loop

    def set_ref_slopes(self, ref_slopes: np.ndarray, *, centro_index : int=None) -> None:
        """ Set given ref slopes in centroider

        Args:
            ref_slopes : (ndarray) : Reference slopes vectoronly set the reference slop

        Kwargs:
            centro_index : (int) : If given, only set the reference slopes vector
                                             used by the specified centroider. If None, the reference
                                             slopes vector must be a concatenation of all the reference
                                             slopes to use for each centroiders handled by the controller
        """
        if (centro_index is None):
            self._rtc.set_centroids_ref(ref_slopes)
        else:
            self._rtc.d_centro[centro_index].set_centroids_ref(ref_slopes)

    def get_ref_slopes(self, centro_index=None) -> np.ndarray:
        """ Get the currently used reference slopes

        Kwargs:
            centro_index : (int) : If given, only get the reference slopes vector
                                             used by the specified centroider. If None, the reference
                                             slopes vector returned is a concatenation of all the reference
                                             slopes used for by centroiders in the RTC

        Return:
            ref_slopes : (np.ndarray) : Reference slopes vector
        """
        ref_slopes = np.empty(0)
        if (centro_index is None):
            for centro in self._rtc.d_centro:
                ref_slopes = np.append(ref_slopes, np.array(centro.d_centroids_ref))
            return ref_slopes
        else:
            return np.array(self._rtc.d_centro[centro_index].d_centroids_ref)

    def set_gain(self, controller_index: int, gain: float) -> None:
        """ Set the scalar gain

        Args:
            controller_index : (int) : Index of the controller to modify

            gain : (float) : scalar gain of modal gain to set
        """
        self._rtc.d_control[controller_index].set_gain(gain)

    def get_interaction_matrix(self, controller_index: int):
        """ Return the interaction matrix of the controller

        Args:
            controller_index: (int): controller index

        Return:
            imat : (np.ndarray) : Interaction matrix currently set in the controller
        """
        return np.array(self._rtc.d_control[controller_index].d_imat)

    def get_command_matrix(self, controller_index: int):
        """ Return the command matrix of the controller

        Args:
            controller_index: (int): controller index

        Return:
            cmat : (np.ndarray) : Command matrix currently used by the controller
        """
        return np.array(self._rtc.d_control[controller_index].d_cmat)

    def set_command_matrix(self, controller_index: int, cmat: np.ndarray) -> None:
        """ Set the command matrix for the controller to use

        Args:
            controller_index : (int) : Controller index to modify

            cmat : (np.ndarray) : command matrix to set
        """
        self._rtc.d_control[controller_index].set_cmat(cmat)

    def get_intensities(self) -> np.ndarray:
        """ Return sum of intensities in subaps. Size nSubaps, same order as slopes
        """
        raise NotImplementedError("Not implemented")

    def set_flat(self, centro_index: int, flat: np.ndarray,):
        """ Load flat field for the given wfs

        Args:
            centro_index : (int) : index of the centroider handling the WFS

            flat : (np.ndarray) : New WFS flat to use
        """
        self._rtc.d_centro[centro_index].set_flat(flat, flat.shape[0])

    def set_dark(self, centro_index: int, dark: np.ndarray):
        """ Load dark for the given wfs

        Args:
            centro_index : (int) : index of the centroider handling the WFS

            dark : (np.ndarray) : New WFS dark to use
        """
        self._rtc.d_centro[centro_index].set_dark(dark, dark.shape[0])

    def compute_slopes(self, controller_index: int):
        """ Compute the slopes handled by a controller, and returns it

        Args:
            controller_index : (int) : Controller index that will compute its slopes

        Return:
            slopes : (np.ndarray) : Slopes vector
        """
        self._rtc.do_centroids(controller_index)
        return self.get_slopes(controller_index)

    def reset_perturbation_voltage(self, controller_index: int) -> None:
        """ Reset the perturbation voltage of the controller_index controller
        (i.e. will remove ALL perturbation voltages.)
        If you want to reset just one, see the function remove_perturbation_voltage()

        Args:
            controller_index : (int) : controller index from where to remove the buffer
        """
        self._rtc.d_control[controller_index].reset_perturb_voltage()

    def remove_perturbation_voltage(self, controller_index: int, name: str) -> None:
        """ Remove the perturbation voltage called <name>, from the controller number <controller_index>.
        If you want to remove all of them, see function reset_perturbation_voltage()

        Args:
            controller_index : (int) : controller index from where to remove the buffer

            name : (str) : Name of the buffer to remove
        """
        self._rtc.d_control[controller_index].remove_perturb_voltage(name)

    def get_perturbation_voltage(self, controller_index: int, *, name: str=None) -> Union[dict, tuple]:
        """ Get a perturbation voltage buffer

        Args:
            controller_index : (int) : controller index from where to get the buffer

        Kwargs:
            name : (str) : Name of the buffer to get. If None, returns all the buffers

        Returns:
            pertu : (dict or tuple) : If name is None, returns a dictionnary with the buffers names as keys
                                      and a tuple (buffer, circular_counter, is_enabled)
        """ 
        pertu_map = self._rtc.d_control[controller_index].d_perturb_map
        if name is None:
            for key in pertu_map.keys():
                pertu_map[key] = (np.array(pertu_map[key][0]), pertu_map[key][1], pertu_map[key][2])
            return pertu_map
        else:
            pertu = pertu_map[name]
            pertu = (np.array(pertu[0]), pertu[1], pertu[2])
            return pertu

    def get_err(self, controller_index: int) -> np.ndarray:
        """ Get integrator increment from controller_index controller

        Args:
            controller_index : (int) : controller index
        """
        return np.array(self._rtc.d_control[controller_index].d_err)

    def get_voltages(self, controller_index: int) -> np.ndarray:
        """ Get voltages vector (i.e. vector sent to the DM) from controller_index controller

        Args:
            controller_index : (int) : controller index

        Return:
            voltages : (np.ndarray) : current voltages vector

        """
        return np.array(self._rtc.d_control[controller_index].d_voltage)

    def set_integrator_law(self, controller_index: int) -> None:
        """ Set the control law to integrator (controller generic only)
            v[k] = v[k-1] + g.R.s[k]

        Args:
            controller_index: (int): controller index
        """
        self._rtc.d_control[controller_index].set_commandlaw("integrator")

    def set_2matrices_law(self, controller_index: int) -> None:
        """ Set the control law to 2matrices (controller generic only)
        v[k] = decayFactor.E.v[k-1] + g.R.s[k]

        Args:
            controller_index: (int): controller index
        """
        self._rtc.d_control[controller_index].set_commandlaw("2matrices")

    def set_modal_integrator_law(self, controller_index: int) -> None:
        """ Set the control law to 2matrices (controller generic only)
        v[k] = v[k-1] + E.g.R.s[k]

        Args:
            controller_index: (int): controller index
        """
        self._rtc.d_control[controller_index].set_commandlaw("modal_integrator")

    def set_decay_factor(self, controller_index: int, decay : np.ndarray) -> None:
        """ Set the decay factor used in 2matrices command law (controller generic only)

        Args:
            controller_index: (int): controller index

            decay : (np.ndarray) : decay factor vector
        """
        self._rtc.d_control[controller_index].set_decayFactor(decay)

    def set_E_matrix(self, controller_index: int, e_matrix : np.ndarray) -> None:
        """ Set the E matrix used in 2matrices or modal command law (controller generic only)

        Args:
            e_matrix : (np.ndarray) : E matrix to set

            controller_index: (int): controller index
        """
        self._rtc.d_control[controller_index].set_matE(e_matrix)

    def reset_ref_slopes(self, controller_index: int) -> None:
        """ Reset the reference slopes of each WFS handled by the specified controller

        Args:
            controller_index: (int): controller index
        """
        for centro in self._rtc.d_centro:
            centro.d_centroids_ref.reset()

    def set_centroider_threshold(self, centro_index: int, thresh: float) -> None:
        """ Set the threshold value of a thresholded COG

        Args:
            centro_index: (int): centroider index

            thresh: (float): new threshold value
        """
        self._rtc.d_centro[centro_index].set_threshold(thresh)

    def get_pyr_method(self, centro_index : int) -> str:
        """ Get pyramid compute method currently used

        Args:
            centro_index: (int): centroider index

        Return:
            method : (str) : Pyramid compute method currently used
        """
        return self._rtc.d_centro[centro_index].pyr_method

    def set_pyr_method(self, centro_index: int, pyr_method : int) -> None:
        """ Set the pyramid method for slopes computation

        Args:
            centro_index : (int) : centroider index

            pyr_method : (int) : new centroiding method (0: nosinus global
                                                1: sinus global
                                                2: nosinus local
                                                3: sinus local)
        """
        self._rtc.d_centro[centro_index].set_pyr_method(pyr_method)  # Sets the pyr method
        self._rtc.do_centroids(0)  # To be ready for the next get_slopess
        print("PYR method set to " + self._rtc.d_centro[centro_index].pyr_method)


    def set_modal_gains(self, controller_index: int, mgain: np.ndarray) -> None:
        """ Sets the modal gain (when using modal integrator control law)

        Args:
            controller_index : (int) : Controller index to modify
            
            mgain : (np.ndarray) : Modal gains to set
        """
        self._rtc.d_control[controller_index].set_modal_gains(mgain)

    def get_modal_gains(self, controller_index: int) -> np.ndarray:
        """ Returns the modal gains (when using modal integrator control law)

        Args:
            controller_index : (int) : Controller index to modify

        Return:
            mgain : (np.ndarray) : Modal gains vector currently used
        """
        return np.array(self._rtc.d_control[controller_index].d_gain)

    def get_masked_pix(self, centro_index: int) -> np.ndarray:
        """ Return the mask of valid pixels used by a maskedpix centroider

        Args:
            centro_index : (int): Centroider index. Must be a maskedpix centroider

        Return:
            mask : (np.ndarray) : Mask used
        """
        if (self._rtc.d_centro[centro_index].type != scons.CentroiderType.MASKEDPIX):
            raise TypeError("Centroider must be a maskedpix one")
        self._rtc.d_centro[centro_index].fill_mask()
        return np.array(self._rtc.d_centro[centro_index].d_mask)

    def get_command(self, controller_index: int) -> np.ndarray:
        """ Returns the last computed command before conversion to voltages

        Args:
            controller_index : (int) : Controller index

        Return:
            com : (np.ndarray) : Command vector
        """
        return np.array(self._rtc.d_control[controller_index].d_com)

    def set_command(self, controller_index: int, com : np.ndarray) -> np.ndarray:
        """ Returns the last computed command before conversion to voltages

        Args:
            controller_index : (int) : Controller index

            com : (np.ndarray) : Command vector to set
        """
        if(com.size != self._config.p_controllers[controller_index].nactu):
            raise ValueError("Dimension mismatch")
        self._rtc.d_control[controller_index].set_com(com, com.size)

    def reset_command(self, controller_index: int = None) -> None:
        """ Reset the controller_index Controller command buffer, reset all controllers if controller_index is None

        Kwargs:
            controller_index : (int) : Controller index
                                      Default is None, i.e. all controllers are reset
        """
        if (controller_index is None):  # All Dms reset
            for control in self._rtc.d_control:
                control.d_com.reset()
        else:
            self._rtc.d_control[controller_index].d_com.reset()

    def get_slopes_geom(self, controller_index: int) -> np.ndarray:
        """ Computes and return the slopes geom from the specified controller

        Args:
            controller_index : (int) : controller index

        Return:
            slopes_geom : (np.ndarray) : geometrically computed slopes
        """
        self._rtc.do_centroids_geom(controller_index)
        slopes_geom = np.array(self._rtc.d_control[controller_index].d_centroids)
        self._rtc.do_centroids(controller_index)  # To return in non-geo state
        return slopes_geom

    def get_selected_pix(self) -> np.ndarray:
        """ Return the pyramid image with only the selected pixels used by the full pixels centroider

        Return:
            selected_pix : (np.ndarray) : PWFS image with only selected pixels
        """
        if (self._config.p_centroiders[0].type != scons.CentroiderType.MASKEDPIX):
            raise TypeError("Centroider must be maskedPix")

        carma_centroids = self._rtc.d_control[0].d_centroids
        self._rtc.d_centro[0].fill_selected_pix(carma_centroids)

        return np.array(self._rtc.d_centro[0].d_selected_pix)

    def do_ref_slopes(self, controller_index: int) -> None:
        """ Computes and set a new reference slopes for each WFS handled by
        the specified controller

        Args:
            controller_index: (int): controller index
        """
        print("Doing reference slopes...")
        self._rtc.do_centroids_ref(controller_index)
        print("Reference slopes done")

    def do_control(self, controller_index: int, *, sources : SourceCompass = None, source_index : int = 0, is_wfs_phase : bool = False) -> None:
        """Computes the command from the Wfs slopes

        Args:
            controller_index: (int): controller index

        Kwargs:
            sources : (SourceCompass) : List of phase screens of a wfs or target sutra object
                                                  If the controller is a GEO one, specify a SourceCompass instance
                                                  from WfsCompass or TargetCompass to project the corresponding phase
            
            source_index : (int) : Index of the phase screen to consider inside <sources>. Default is 0

            is_wfs_phase : (bool) : If True, sources[source_index] is a WFS phase screen.
                                              Else, it is a Target phase screen (Default)
        """
        if (self._rtc.d_control[controller_index].type == scons.ControllerType.GEO):
            if (sources is not None):
                self._rtc.d_control[controller_index].comp_dphi(sources[source_index],
                                                       is_wfs_phase)
        self._rtc.do_control(controller_index)

    def do_calibrate_img(self, controller_index: int) -> None:
        """ Computes the calibrated image from the Wfs image

        Args:
            controller_index: (int): controller index
        """
        self._rtc.do_calibrate_img(controller_index)

    def do_centroids(self, controller_index: int) -> None:
        """ Computes the centroids from the Wfs image

        Args:
            controller_index: (int): controller index
        """
        self._rtc.do_centroids(controller_index)

    def do_centroids_geom(self, controller_index: int) -> None:
        """ Computes the centroids geom from the Wfs image

        Args:
            controller_index: (int): controller index
        """
        self._rtc.do_centroids_geom(controller_index)

    def apply_control(self, controller_index: int, *, comp_voltage: bool = True) -> None:
        """ Computes the final voltage vector to apply on the DM by taking into account delay and perturbation voltages, and shape the DMs

        Args:
            controller_index: (int): controller index

        Kwargs:
            comp_voltage: (bool): If True (default), computes the voltage vector from the command one (delay + perturb). Else, directly applies the current voltage vector
        """
        self._rtc.apply_control(controller_index, comp_voltage)

    def do_clipping(self, controller_index: int) -> None:
        """ Clip the commands between vmin and vmax values set in the RTC

        Args:
            controller_index: (int): controller index
        """
        self._rtc.do_clipping(controller_index)

    def set_scale(self, centroider_index : int, scale : float) -> None:
        """ Update the scale factor of the centroider

        Args:
            centroider_index : (int) : Index of the centroider to update

            scale : (float) : scale factor to apply on slopes
        """
        self._rtc.d_centro[centroider_index].set_scale(scale)

    def publish(self) -> None:
        """ Publish loop data on DDS topics

        /!\ only with cacao enabled, requires OCTOPUS
        """
        if self.cacao:
            self._rtc.publish()
        else:
            raise AttributeError("CACAO must be enabled")
    
    def get_image_raw(self, centroider_index : int) -> np.ndarray:
        """ Return the raw image currently loaded on the specified centroider

        Args:
            centroider_index : (int) : Index of the centroider
        
        Return:
            image_raw : (np.ndarray) : Raw WFS image loaded in the centroider
        """
        return np.array(self._rtc.d_centro[centroider_index].d_img_raw)

    def get_image_calibrated(self, centroider_index : int) -> np.ndarray:
        """ Return the last image calibrated by the specified centroider

        Args:
            centroider_index : (int) : Index of the centroider
        
        Return:
            image_cal : (np.ndarray) : Calibrated WFS image loaded in the centroider
        """
        return np.array(self._rtc.d_centro[centroider_index].d_img)