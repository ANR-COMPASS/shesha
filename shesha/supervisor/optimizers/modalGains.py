import numpy as np

class ModalGains(object):
    """ This optimizer class handles the modal gain optimization related operations
    using the CLOSE algorithm. Should be used with a modal integrator command law.

    Attributes:
        _config : (config) : Configuration parameters module

        _rtc : (RtcCompass) : RtcCompass instance

        _ntotact : (int) : total number of actuators used in the simulation

        modal_basis : (np.ndarray) : KL2V modal basis

        cmat_modal : (np.ndarray) : modal command matrix

        _mask : (np.ndarray) : mask array (containig 0 or 1) filtering the modes

        _ac_idx : (int) : autocorrelation index

        _up_idx : (int) : update index (averaging length of AC estimator before modal gains update)

        _lf : (float) : learning factor of the autocorrelation computation

        _lfdownup : (float) : learning factors for modal gain update

        _trgt : (float) : target value for the autocorrelation optimization

        _initial_gain : (float) : initial value for the modal gains (same for all modes)

        _modal_meas : (list) : list containing previous modal measurements to
                                be used for CLOSE optimization

        _ac_est_0 : (np.ndarray) : autocorrelation estimation for no frames delay

        _ac_est_dt : (np.ndarray) : autocorrelation estimation for _ac_idx delay

        mgains : (np.ndarray) : modal gains that will be updated

        close_iter : (int) : number of iteration of CLOSE optimizations
    """

    def __init__(self, config, rtc):
        """ Instantiate a ModalGains optimizer object.

        Args:
            config : (config module) : Parameters configuration structure module

            rtc : (sutraWrap.Rtc) : Sutra rtc instance
        """
        self._config = config
        self._rtc = rtc
        self._ntotact = config.p_controllers[0].nactu
        # parameters of the CLOSE optimization
        self.modal_basis = None # carrée !!
        self.cmat_modal = None
        self._mask = np.ones(self._ntotact)
        self._ac_idx = int(config.p_controllers[0].delay * 2 + 1)
        self._up_idx = config.p_controllers[0].get_close_update_index()
        self._lf = config.p_controllers[0].get_close_learning_factor()
        self._lfdownup = np.array(config.p_controllers[0].get_lfdownup())
        self._trgt = config.p_controllers[0].get_close_target()
        self._initial_gain = config.p_controllers[0].get_mgain_init()
        # computation intermediaries
        self._modal_meas = []
        self._buffer = []
        self._ac_est_0 = np.zeros((self._ntotact), dtype=np.float32)
        self._ac_est_dt = np.zeros((self._ntotact), dtype=np.float32)
        # out variables
        self.mgains = np.ones(self._ntotact) * self._initial_gain
        self.close_iter = 0
        if (self._config.p_controllers[0].close_opti):
            self._rtc.set_modal_gains(0, self.mgains)
        # print(f"total number of actuators {self._ntotact}")
        # print(f"Autocorrelation index for CLOSE optimization is {self._ac_idx}")

    def update_modal_meas(self):
        """Save the modal measurement of the current iter"""
        if self.cmat_modal is None or self.modal_basis is None :
            raise Exception("Modal basis and cmat modal should be not None")
        slp = self._rtc.get_slopes(0)
        self._modal_meas.append(self.cmat_modal.dot(slp))
        if len(self._modal_meas) == self._ac_idx + 1:
            self._modal_meas.pop(0)

    def update_mgains(self):
        """Compute a new modal gains
        This function computes and updates the modal gains according to the
        CLOSE algorithm.
        """
        #ctrl_modes = self._mask != 0    # where modes are controlled
        ctrl_modes = np.where(self._mask)[0]    # where modes are controlled
        if self.cmat_modal is None or self.modal_basis is None :
            raise Exception("Modal basis and cmat modal should be not None")
        # get new measurement
        slp = self._rtc.get_slopes(0)
        temp_modal_meas = self.cmat_modal.dot(slp)
        self._modal_meas.append(temp_modal_meas)
        # estimate autocorrelation
        if np.all(self._ac_est_0 == 0):

            self._ac_est_0[ctrl_modes] = self._modal_meas[-1][ctrl_modes] ** 2
        else:
            self._ac_est_0[ctrl_modes] = self._ac_est_0[ctrl_modes] * (1 - self._lf) + self._modal_meas[-1][ctrl_modes] ** 2 * self._lf
        if len(self._modal_meas) == self._ac_idx + 1:
            if np.all(self._ac_est_dt == 0):
                self._ac_est_dt[ctrl_modes] = self._modal_meas[0][ctrl_modes] * self._modal_meas[-1][ctrl_modes]
            else:
                self._ac_est_dt[ctrl_modes] = self._ac_est_dt[ctrl_modes] * (1 - self._lf) \
                    + self._modal_meas[0][ctrl_modes] * self._modal_meas[-1][ctrl_modes] * self._lf
            # compute new modal gains
            x = self._ac_est_dt[ctrl_modes] / self._ac_est_0[ctrl_modes] - self._trgt
            self._buffer.append(x)
            if len(self._buffer) >= self._up_idx:
                mean = np.mean(self._buffer, axis=0)
                sign_ac = (mean > 0).astype(np.int8)
                self.mgains[ctrl_modes] = self.mgains[ctrl_modes] * (1 + self._lfdownup[sign_ac] * x)
                self._rtc.set_modal_gains(0, self.mgains)
                self._buffer = []
            self._modal_meas.pop(0)
        self.close_iter +=1

    def reset_close(self):
        """Reset modal gain and computation variables"""
        self.mgains = np.ones(self._ntotact) * self._mask * self._initial_gain
        self._ac_est_0 = np.zeros((self._ntotact), dtype=np.float32)
        self._ac_est_dt = np.zeros((self._ntotact), dtype=np.float32)
        self._modal_meas = []
        self.close_iter = 0
        self._rtc.set_modal_gains(0, self.mgains)
        self.adapt_modal_gains(False)


    def reset_mgains(self):
        """Reset the modal gains only"""
        self.mgains = np.ones(self._ntotact) * self._mask * self._initial_gain

    def adapt_modal_gains(self, flag):
        """Set the flag indicating to use CLOSE optimization.

        Args:
            flag : (bool) : If true, update the modal gains value according to CLOSE algo
        """
        self._config.p_controllers[0].set_close_opti(flag)

    def set_modal_basis(self, modal_basis):
        """Set the modal basis to be used in CLOSE calculation.

        Args:
            modal_basis : (np.ndarray) : modal basis (KL2V) to be used (square)
        """
        if (modal_basis.shape[0] != modal_basis.shape[1]):
            raise Exception("Modal basis should be square matrix")
        self.modal_basis = modal_basis
        self._rtc.set_E_matrix(0, modal_basis)
        print("Modal basis is set")

    def get_modal_basis(self):
        """Get the modal basis

        Returns:
            self.modal_basis : (np.ndarray) : modal basis (KL2V) used in the optimizer
        """
        return self.modal_basis

    def set_cmat_modal(self, cmat_modal):
        """Set cmat modal

        Args:
            cmat_modal : (np.ndarray) : modal command matrix
        """
        self.cmat_modal = cmat_modal
        self._rtc.set_command_matrix(0, cmat_modal)
        print("cmat_modal is set")

    def get_modal_gains(self):
        """Get the modal gains

        Returns:
            self.mgains : (np.ndarray) : modal gains
        """
        return self._rtc.get_modal_gains(0)

    def set_modal_gains(self, mgains):
        """Sets manually the modal gains

        Args:
            mgains : (np.ndarray) : the modal gains array
        """
        self.mgains = mgains
        self._rtc.set_modal_gains(0, mgains)


    def set_mask(self, mask):
        """Set the mode mask

        Args:
            mask : (np.ndarray) : mask array (containig 0 or 1) filtering the modes
        """
        self._mask = mask
        self.mgains[mask == 0] = 0
        self._rtc.set_modal_gains(0, self.mgains)

    def set_initial_gain(self, gain):
        """Set the initial value for modal gains. This function reinitializes the modal gains.

        Args:
            gain: (float) : initial value for modal gains
        """
        self._initial_gain = gain
        self.mgains = np.ones(self._ntotact) * self._initial_gain
        self.mgains[self._mask == 0] = 0
        self._rtc.set_modal_gains(0, self.mgains)
        self._config.p_controllers[0].set_mgain_init(gain)

    def set_config(self, p, qminus, qplus, target, up_idx):
        """Set the 4 parameters for the CLOSE optimization loop

        Args:
            p: (float) : learning factor for autocorrelation

            qminus: (float) : learning factor for mgain optimization when lower than target

            qplus: (float) : learning factor for mgain optimization when higher than target

            target: (float) : autocorrelation target for optimization
        
            up_idx: (int) : modal gains update rate [frame] 
        """
        self._lf = p
        self._config.p_controllers[0].set_close_learning_factor(p)
        self._lfdownup = np.array([qminus, qplus])
        self._config.p_controllers[0].set_lfdownup(qminus, qplus)
        self._trgt = target
        self._config.p_controllers[0].set_close_target(target)
        self._up_idx = up_idx
        self._config.p_controllers[0].set_close_update_index(up_idx)