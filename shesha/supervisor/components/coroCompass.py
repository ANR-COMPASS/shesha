import numpy as np
import shesha.util.coro_pupil as pup
import astropy.io.fits as pfits

# coronagraph parameters
param_coro = {
    'wavelength_0' : 1667e-9,  # [nm] central wavelength
    'delta_wav' : 54e-9,  # [nm] bandwidth
    'nb_wav' : 3,  # number of simulated wavelengths

    'dim_science' : 256,  # [pixel] size of the science image
    'science_sampling' : 3.382,  # [pixel] lambda / D in pixel
                                 # lambda = wavelength_0, D = tel_diam
    'perfect_coro' : False,  # flag for using perfect coronagraph

    'apodizer' : 'sphere_ALC_apodizer_APO1',  # 'sphere_ALC_apodizer_APO1' or 'user_path'

    'focal_plane_mask' : 'classical_lyot',  # 'classical_lyot' or 'user_path' (what about 'sphere_ALC_fpm_185mas' ?)
    'fpm_radius' : 100,  # [mas] for 'classical_lyot' fpm only

    'lyot_stop' : 'sphere_ALC_lyot_stop',  # 'sphere_ALC_lyot_stop' or 'user_path'

    'propagation' : 'mft', # 'mft', 'mft-babinet' (for classical_lyot fpm only) or 'fft'
}

# init functions
def init_mft(image,
             real_dim_input,
             dim_output,
             nbres,
             inverse=False,
             norm='backward',
             X_offset_input=0,
             Y_offset_input=0,
             X_offset_output=0,
             Y_offset_output=0):
    """
    Initialize matrices for Matrix Fourier Transform computation
    """
    # check dimensions and type of real_dim_input
    error_string_real_dim_input = "'real_dim_input' must be an int (square input pupil) or tuple of ints of dimension 2"
    if np.isscalar(real_dim_input):
        if isinstance(real_dim_input, int):
            real_dim_input_x = real_dim_input
            real_dim_input_y = real_dim_input
        else:
            raise TypeError(error_string_real_dim_input)
    elif isinstance(real_dim_input, tuple):
        if all(isinstance(dims, int) for dims in real_dim_input) & (len(real_dim_input) == 2):
            real_dim_input_x = real_dim_input[0]
            real_dim_input_y = real_dim_input[1]
        else:
            raise TypeError(error_string_real_dim_input)
    else:
        raise TypeError(error_string_real_dim_input)

    # check dimensions and type of dim_output
    error_string_dim_output = "'dim_output' must be an int (square output) or tuple of ints of dimension 2"
    if np.isscalar(dim_output):
        if isinstance(dim_output, int):
            dim_output_x = dim_output
            dim_output_y = dim_output
        else:
            raise TypeError(error_string_dim_output)
    elif isinstance(dim_output, tuple):
        if all(isinstance(dims, int) for dims in dim_output) & (len(dim_output) == 2):
            dim_output_x = dim_output[0]
            dim_output_y = dim_output[1]
        else:
            raise TypeError(error_string_dim_output)
    else:
        raise TypeError(error_string_dim_output)

    # check dimensions and type of nbres
    error_string_nbr = "'nbres' must be an float or int (square output) or tuple of float or int of dimension 2"
    if np.isscalar(nbres):
        if isinstance(nbres, (float, int)):
            nbresx = float(nbres)
            nbresy = float(nbres)
        else:
            raise TypeError(error_string_nbr)
    elif isinstance(nbres, tuple):
        if all(isinstance(nbresi, (float, int)) for nbresi in nbres) & (len(nbres) == 2):
            nbresx = float(nbres[0])
            nbresy = float(nbres[1])
        else:
            raise TypeError(error_string_nbr)
    else:
        raise TypeError(error_string_nbr)

    dim_input_x = image.shape[0]
    dim_input_y = image.shape[1]

    nbresx = nbresx * dim_input_x / real_dim_input_x
    nbresy = nbresy * dim_input_y / real_dim_input_y

    X0 = dim_input_x / 2 + X_offset_input
    Y0 = dim_input_y / 2 + Y_offset_input

    X1 = dim_output_x / 2 + X_offset_output
    Y1 = dim_output_y / 2 + Y_offset_output

    xx0 = ((np.arange(dim_input_x) - X0 + 1 / 2) / dim_input_x)  # Entrance image
    xx1 = ((np.arange(dim_input_y) - Y0 + 1 / 2) / dim_input_y)  # Entrance image
    uu0 = ((np.arange(dim_output_x) - X1 + 1 / 2) / dim_output_x) * nbresx  # Fourier plane
    uu1 = ((np.arange(dim_output_y) - Y1 + 1 / 2) / dim_output_y) * nbresy  # Fourier plane

    if not inverse:
        if norm == 'backward':
            norm0 = 1.
        elif norm == 'forward':
            norm0 = nbresx * nbresy / dim_input_x / dim_input_y / dim_output_x / dim_output_y
        elif norm == 'ortho':
            norm0 = np.sqrt(nbresx * nbresy / dim_input_x / dim_input_y / dim_output_x / dim_output_y)
        sign_exponential = -1

    else:
        if norm == 'backward':
            norm0 = nbresx * nbresy / dim_input_x / dim_input_y / dim_output_x / dim_output_y
        elif norm == 'forward':
            norm0 = 1.
        elif norm == 'ortho':
            norm0 = np.sqrt(nbresx * nbresy / dim_input_x / dim_input_y / dim_output_x / dim_output_y)
        sign_exponential = 1

    AA = np.exp(sign_exponential * 1j * 2 * np.pi * np.outer(uu0, xx0)).astype('complex64')
    BB = np.exp(sign_exponential * 1j * 2 * np.pi * np.outer(xx1, uu1)).astype('complex64')

    return AA, BB, norm0

def init_apodizer(param_coro, pupdiam):

    if param_coro['apodizer'] == 'sphere_ALC_apodizer_APO1':
        apodizer = pup.make_sphere_apodizer(pupdiam)
        return apodizer
    else:
        apodizer_path = param_coro['apodizer']
        apodizer = pfits.getdata(apodizer_path)
        return apodizer

def init_focal_plane_mask(param_coro, pupdiam):

    if param_coro['focal_plane_mask'] == 'classical_lyot':
        fpm = None
    else:
        fpm_path = param_coro['focal_plane_mask']
        fpm = pfits.getdata(fpm_path)
        return fpm

def init_lyot_stop(param_coro, pupdiam):

    if param_coro['lyot_stop'] == 'sphere_ALC_lyot_stop':
        lyot_stop = pup.make_sphere_lyot_stop(pupdiam)
        return lyot_stop
    else:
        lyot_stop_path = param_coro['lyot_stop']
        lyot_stop = pfits.getdata(lyot_stop_path)
        return lyot_stop

# propagation function
def fft_choosecenter():
    pass

def mft(image, AA, BB, norm):
    return norm * (AA @ image) @ BB

def crop_or_pad_image(image, dimout):
    """Crop or padd with zero to a 2D image depending on:

        - if dimout < dim : cropped image around pixel (dim/2,dim/2)
        - if dimout > dim : image around pixel (dim/2,dim/2) surrounded by 0

    AUTHOR: Raphael Galicher

    Parameters
    ----------
    image : 2D array (float, double or complex)
        dim x dim array to crop or pad
    dimout : int
        dimension of the output array
    Returns
    -------
    im_out : 2D array (float)
        resized image
    """
    if float(dimout) < image.shape[0]:
        im_out = np.zeros((image.shape[0], image.shape[1]), dtype=image.dtype)
        im_out = image[int((image.shape[0] - dimout) / 2):int((image.shape[0] + dimout) / 2),
                       int((image.shape[1] - dimout) / 2):int((image.shape[1] + dimout) / 2)]
    elif dimout > image.shape[0]:
        im_out = np.zeros((dimout, dimout), dtype=image.dtype)
        im_out[int((dimout - image.shape[0]) / 2):int((dimout + image.shape[0]) / 2),
               int((dimout - image.shape[1]) / 2):int((dimout + image.shape[1]) / 2)] = image
    else:
        im_out = image
    return im_out

class CoroCompass:
    """
    docstring
    """
    def __init__(self, param_coro, pupdiam):

        self.pupdiam = pupdiam

        self.wavelength_0 = param_coro['wavelength_0']
        self.wav_vec = np.linspace(self.wavelength_0 - param_coro['delta_wav'] / 2,
                                   self.wavelength_0 + param_coro['delta_wav'] / 2,
                                   num=param_coro['nb_wav'],
                                   endpoint=True)

        self.perfect_coro = param_coro['perfect_coro']

        self.apodizer = init_apodizer(param_coro, pupdiam)
        self.fpm = init_focal_plane_mask(param_coro, pupdiam)
        self.lyot_stop = init_lyot_stop(param_coro, pupdiam)

        self.propagation = param_coro['propagation']


    def EF_through_coro_to_detector(self,
                                    entrance_EF,
                                    wavelength,
                                    no_fpm=False,
                                    center_on_pixel=False):
        """
        Compute the complex electric field in the science focal plane, for a given wavelength
        """




