import numpy as np
import shesha.config as conf
import shesha.constants as scons
from shesha.supervisor.components.coronagraph.genericCoronagraph import GenericCoronagraph
from shesha.init.coronagraph_init import init_coronagraph, init_mft, mft_multiplication
from shesha.supervisor.components.targetCompass import TargetCompass
from sutraWrap import StellarCoronagraph
from carmaWrap import context
class StellarCoronagraphCompass(GenericCoronagraph):
    """ Class supervising stellar coronagraph component

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

        _AA_apod_to_fpm: (np.ndarray[ndim=3, dtype=np.complex64]): MFT matrix for focal plane

        _BB_apod_to_fpm: (np.ndarray[ndim=2, dtype=np.complex64]): MFT matrix for focal plane

        _norm0_apod_to_fpm: (np.ndarray[ndim=2, dtype=np.complex64]): MFT matrix for focal plane

        _AA_fpm_to_lyot: (np.ndarray[ndim=3, dtype=np.complex64]): MFT matrix for lyot plane

        _BB_fpm_to_lyot: (np.ndarray[ndim=2, dtype=np.complex64]): MFT matrix for lyot plane

        _norm0_fpm_to_lyot: (np.ndarray[ndim=2, dtype=np.complex64]): MFT matrix for lyot plane

        _AA_lyot_to_image_c: (np.ndarray[ndim=2, dtype=np.complex64]): MFT matrix for image computation (centered on pixel)

        _BB_lyot_to_image_c: (np.ndarray[ndim=2, dtype=np.complex64]): MFT matrix for psf computation (centered on pixel)
        
        _norm0_lyot_to_image_c: (np.ndarray[ndim=2, dtype=np.complex64]): MFT matrix for psf computation (centered on pixel)

        _AA_lyot_to_image: (np.ndarray[ndim=2, dtype=np.complex64]): MFT matrix for image computation

        _BB_lyot_to_image: (np.ndarray[ndim=2, dtype=np.complex64]): MFT matrix for psf computation
        
        _norm0_lyot_to_image: (np.ndarray[ndim=2, dtype=np.complex64]): MFT matrix for psf computation

        _indices_pup: (tuple): Tuple of ndarray containing X and Y indices of illuminated 
                                pixels in the pupil
    """
    def __init__(self, context: context, targetCompass: TargetCompass, p_corono: conf.Param_corono, 
                 p_geom: conf.Param_geom):
        """ Initialize a stellar coronagraph instance

        Args:
            context: (CarmaWrap.context): GPU context

            targetCompass: (TargetCompass): Compass Target used as input for the coronagraph

            p_corono: (Param_corono): Coronagraph parameters

            p_geom: (Param_geom): Compass geometry parameters
        """

        init_coronagraph(p_corono, p_geom.pupdiam)
        GenericCoronagraph.__init__(self, p_corono, p_geom, targetCompass)
        self._wav_vec = p_corono._wav_vec

        self._AA_apod_to_fpm, self._BB_apod_to_fpm, self._norm0_apod_to_fpm = init_mft(self._p_corono,
                                                                                       self._pupdiam,
                                                                                       planes='apod_to_fpm')
        self._AA_fpm_to_lyot, self._BB_fpm_to_lyot, self._norm0_fpm_to_lyot = init_mft(self._p_corono,
                                                                                       self._pupdiam,
                                                                                       planes='fpm_to_lyot')
        self._AA_lyot_to_image, self._BB_lyot_to_image, self._norm0_lyot_to_image = init_mft(self._p_corono,
                                                                                             self._pupdiam,
                                                                                             planes='lyot_to_image')
        self._AA_lyot_to_image_c, self._BB_lyot_to_image_c, self._norm0_lyot_to_image_c = init_mft(self._p_corono,
                                                                                                   self._pupdiam,
                                                                                                   planes='lyot_to_image',
                                                                                                   center_on_pixel=True)
        self._coronagraph = StellarCoronagraph(context, self._target.sources[0], 
                                               self._dim_image, self._dim_image, 
                                               self._p_corono._dim_fpm, self._p_corono._dim_fpm, 
                                               self._wav_vec, self._wav_vec.size, 
                                               self._p_corono._babinet_trick, 0)
        
        AA = np.rollaxis(np.array(self._AA_lyot_to_image), 0, self._wav_vec.size)
        BB = np.rollaxis(np.array(self._BB_lyot_to_image), 0, self._wav_vec.size)
        AA_c = np.rollaxis(np.array(self._AA_lyot_to_image_c), 0, self._wav_vec.size)
        BB_c = np.rollaxis(np.array(self._BB_lyot_to_image_c), 0, self._wav_vec.size)
        AA_fpm = np.rollaxis(np.array(self._AA_apod_to_fpm), 0, self._wav_vec.size)
        BB_fpm = np.rollaxis(np.array(self._BB_apod_to_fpm), 0, self._wav_vec.size)
        AA_lyot = np.rollaxis(np.array(self._AA_fpm_to_lyot), 0, self._wav_vec.size)
        BB_lyot = np.rollaxis(np.array(self._BB_fpm_to_lyot), 0, self._wav_vec.size)
        
        self._coronagraph.set_mft(AA, BB, self._norm0_lyot_to_image, scons.MftType.IMG)
        self._coronagraph.set_mft(AA_c, BB_c, self._norm0_lyot_to_image_c, scons.MftType.PSF)
        self._coronagraph.set_mft(AA_fpm, BB_fpm, self._norm0_apod_to_fpm, scons.MftType.FPM)
        self._coronagraph.set_mft(AA_lyot, BB_lyot, self._norm0_fpm_to_lyot, scons.MftType.LYOT)

        self._coronagraph.set_apodizer(self._p_corono._apodizer)
        self._coronagraph.set_lyot_stop(self._p_corono._lyot_stop)
        fpm = np.rollaxis(np.array(self._p_corono._focal_plane_mask), 0, self._wav_vec.size)
        if self._p_corono._babinet_trick:
            fpm = 1. - fpm
        self._coronagraph.set_focal_plane_mask(fpm)
        self._compute_normalization()

    def _compute_normalization(self):
        """ Compute the normalization factor of coronagraphic images
        """
        self._target.reset_tar_phase(0)
        self._coronagraph.compute_image_normalization()
        self._norm_img = np.max(self.get_image(expo_type=scons.ExposureType.SE))
        self.compute_psf(accumulate=False)
        self._norm_psf = np.max(self.get_psf(expo_type=scons.ExposureType.SE))
