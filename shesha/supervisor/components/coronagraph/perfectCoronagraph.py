import numpy as np
import shesha.config as conf
import shesha.constants as scons
from shesha.supervisor.components.coronagraph.genericCoronagraph import GenericCoronagraph
from shesha.init.coronagraph_init import init_coronagraph, init_mft, mft_multiplication
from shesha.supervisor.components.targetCompass import TargetCompass
from sutraWrap import PerfectCoronagraph
from carmaWrap import context


class PerfectCoronagraphCompass(GenericCoronagraph):
    """ Class supervising perfect coronagraph component

    Attributes:
        _spupil: (np.ndarray[ndim=2, dtype=np.float32]): Telescope pupil mask

        _pupdiam : (int): Number of pixels along the pupil diameter

        _dim_image :(int): Coronagraphic image dimension

        _p_corono: (Param_corono): Coronagraph parameters

        _target: (TargetCompass): Compass Target used as input for the coronagraph

        _norm_img : (float): Normalization factor for coronagraphic image

        _norm_psf : (float): Normalization factor for PSF

        _coronagraph: (SutraCoronagraph): Sutra coronagraph instance

        _wav_vec: (np.ndarray[ndim=1, dtype=np.float32]): Vector of wavelength

        _AA: (np.ndarray[ndim=3, dtype=np.complex64]): MFT matrix for image computation

        _BB: (np.ndarray[ndim=3, dtype=np.complex64]): MFT matrix for image computation

        _norm0: (np.ndarray[ndim=3, dtype=np.complex64]): MFT matrix for image computation

        _AA_c: (np.ndarray[ndim=3, dtype=np.complex64]): MFT matrix for image computation

        _BB_c: (np.ndarray[ndim=3, dtype=np.complex64]): MFT matrix for psf computation
        
        _norm0_c: (np.ndarray[ndim=3, dtype=np.complex64]): MFT matrix for psf computation

        _indices_pup: (tuple): Tuple of ndarray containing X and Y indices of illuminated 
                                pixels in the pupil
    """
    def __init__(self, context: context, targetCompass: TargetCompass, 
                 p_corono: conf.Param_corono, p_geom: conf.Param_geom):
        """ Initialize a perfect coronagraph instance

        Args:
            context: (CarmaWrap.context): GPU context

            targetCompass: (TargetCompass): Compass Target used as input for the coronagraph

            p_corono: (Param_corono): Coronagraph parameters

            p_geom: (Param_geom): Compass geometry parameters
        """
        init_coronagraph(p_corono, p_geom.pupdiam)
        GenericCoronagraph.__init__(self, p_corono, p_geom, targetCompass)
        self._wav_vec = p_corono._wav_vec

        self._AA, self._BB, self._norm0 = init_mft(p_corono,
                                                   self._pupdiam,
                                                   planes='lyot_to_image')
        self._AA_c, self._BB_c, self._norm0_c = init_mft(p_corono,
                                                         self._pupdiam,
                                                         planes='lyot_to_image',
                                                         center_on_pixel=True)
        self._indices_pup = np.where(self._spupil > 0.)

        self._coronagraph = PerfectCoronagraph(context, self._target.sources[0], 
                                               self._dim_image, self._dim_image, 
                                               self._wav_vec, self._wav_vec.size, 0)
        
        AA = np.rollaxis(np.array(self._AA), 0, self._wav_vec.size)
        BB = np.rollaxis(np.array(self._BB), 0, self._wav_vec.size)
        AA_c = np.rollaxis(np.array(self._AA_c), 0, self._wav_vec.size)
        BB_c = np.rollaxis(np.array(self._BB_c), 0, self._wav_vec.size)
        self._coronagraph.set_mft(AA, BB, self._norm0, scons.MftType.IMG)
        self._coronagraph.set_mft(AA_c, BB_c, self._norm0_c, scons.MftType.PSF)
        self._compute_normalization()

    def _compute_normalization(self):
        """ Computes the normalization factor of coronagraphic images (CPU based)
        """
        self._target.reset_tar_phase(0)
        self.compute_psf(accumulate=False)
        self._norm_img = np.max(self.get_psf(expo_type=scons.ExposureType.SE))
        self._norm_psf = self._norm_img
