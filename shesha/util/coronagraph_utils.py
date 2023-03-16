import numpy as np


# ------------------------ #
#      Generic pupils      #
# ------------------------ #

def roundpupil(dim_pp, prad, center_pos='b'):
    """Create a circular pupil.

    AUTHORS : Axel Pottier, Johan Mazoyer
    7/9/22 Modified by J Mazoyer to remove the pixel crenellation with rebin and add a better center option

    Parameters
    ----------
    dim_pp : int
        Size of the image (in pixels)
    prad : float
        Size of the pupil radius (in pixels)
    center_pos : string (optional, default 'b')
        Option for the center pixel.
        If 'p', center on the pixel dim_pp//2.
        If 'b', center in between pixels dim_pp//2 -1 and dim_pp//2, for 'dim_pp' odd or even.

    Returns
    -------
    pupilnormal : 2D array
        Output circular pupil
    """

    xx, yy = np.meshgrid(np.arange(dim_pp) - dim_pp // 2, np.arange(dim_pp) - dim_pp // 2)

    if center_pos.lower() == 'b':
        offset = 1 / 2
    elif center_pos.lower() == 'p':
        offset = 0
    else:
        raise ValueError("center_pos must be 'p' (centered on pixel) or 'b' (centered in between 4 pixels)")

    rr = np.hypot(yy + offset, xx + offset)
    pupilnormal = np.zeros((dim_pp, dim_pp))
    pupilnormal[rr <= prad] = 1.0

    return pupilnormal

def classical_lyot_fpm(rad_lyot_fpm, dim_fpm, Lyot_fpm_sampling, wav_vec):
    """Create a classical Lyot coronagraph of radius rad_LyotFP.

        AUTHOR : Johan Mazoyer

        Returns
        -------
        ClassicalLyotFPM_allwl : list of 2D numpy arrays
            The FP masks at all wavelengths.
        """

    rad_LyotFP_pix = rad_lyot_fpm * Lyot_fpm_sampling
    ClassicalLyotFPM = 1. - roundpupil(dim_fpm, rad_LyotFP_pix)

    ClassicalLyotFPM_allwl = []
    for wav in wav_vec:
        ClassicalLyotFPM_allwl.append(ClassicalLyotFPM)

    return ClassicalLyotFPM_allwl


# ------------------------------------ #
#      SPHERE/VLT specific pupils      #
# ------------------------------------ #

def make_VLT_pupil(pupdiam, centralobs = 0.14, spiders = 0.00625, spiders_bool = True, centralobs_bool = True):
    """
    Return a VLT pupil
    based on make_VLT function from shesha/shesha/util/make_pupil.py

    Args :
        pupdiam (int) [pixel] : pupil diameter

        centralobs (float, optional) [fraction of diameter] : central obtruction diameter, default = 0.14

        spiders (float, optional) [fraction of diameter] : spider diameter, default = 0.00625

        spiders_bool (bool, optional) : if False, return the VLT pupil without spiders; default = True

        centralobs_bool (bool, optional) : if False, return the VLT pupil without central obstruction; default = True

    Returns :
        VLT_pupil (2D array) : VLT transmission pupil of shape (pupdiam, pupdiam), filled with 0 and 1
    """
    range = (0.5 * (1) - 0.25 / pupdiam)
    X = np.tile(np.linspace(-range, range, pupdiam, dtype=np.float32), (pupdiam, 1))
    R = np.sqrt(X**2 + (X.T)**2)

    if centralobs_bool :
        VLT_pupil = ((R < 0.5) & (R > (centralobs / 2))).astype(np.float32)
    elif centralobs_bool == False :
        VLT_pupil = (R < 0.5).astype(np.float32)

    if spiders_bool:
        angle = 50.5 * np.pi / 180.  # 50.5 degrees = angle between spiders

        if pupdiam % 2 == 0 :
            spiders_map = (
                (X.T >
                (X - centralobs / 2 + spiders / np.sin(angle)) * np.tan(angle)) +
                (X.T < (X - centralobs / 2) * np.tan(angle))) * (X > 0) * (X.T > 0)
        elif pupdiam % 2 == 1 :
            spiders_map = (
                (X.T >
                (X - centralobs / 2 + spiders / np.sin(angle)) * np.tan(angle)) +
                (X.T < (X - centralobs / 2) * np.tan(angle))) * (X >= 0) * (X.T >= 0)
        spiders_map += np.fliplr(spiders_map)
        spiders_map += np.flipud(spiders_map)
        VLT_pupil = VLT_pupil * spiders_map

    return VLT_pupil

def sphere_apodizer_radial_profile(x):
    """
    Compute the transmission radial profile of the SPHERE APO1 apodizer.
    x is the radial coordinate inside the pupil
    x = 0 at the center of the pupil and x = 1 on the outer edge
    This profile has been estimated with a five order polynomial fit.
    Don't go inside the central obstruction, namely x < 0.14,
    as the fit is no longer reliable.

    Parameters
    ----------
    x : float or array
        distance to the pupil center, in fraction of the pupil radius

    Returns
    -------
    profile : float or array
        corresponding transmission
    """
    a = 0.16544446129778326
    b = 4.840243632913415
    c = -12.02291052479871
    d = 7.549499000031292
    e = -1.031115714037546
    f = 0.8227341447351052
    profile = a * x**5 + b * x**4 + c * x**3 + d * x**2 + e * x + f
    return profile

def make_sphere_apodizer(pupdiam, centralobs = 0.14, radial_profile = sphere_apodizer_radial_profile):
    """
    Return the SPHERE APLC apodizer APO1, based on its radial transmission profile

    can be used with any profile, assuming that the radial coordinates is 0 at center
    and 1 at the outer edge pixel

    Args :
        pupdiam (int) [pixel] : pupil diameter
        
        centralobs (float, optional) [fraction of diameter] : central obtruction diameter

        radialProfile (function, optional) : apodizer radial transmission; default is SPHERE APO1 apodizer

    Returns :
        apodizer (2D array) : apodizer transmission pupil
    """
    # creating VLT pup without spiders
    pup = make_VLT_pupil(pupdiam, centralobs = centralobs, spiders = 0, spiders_bool = False, centralobs_bool = True)

    # applying apodizer radial profile
    X = np.tile(np.linspace(-1, 1, pupdiam, dtype=np.float32), (pupdiam, 1))  # !
    R = np.sqrt(X**2 + (X.T)**2)
    apodizer = pup*radial_profile(R)
    return apodizer

def make_sphere_lyot_stop(pupdiam,
                          centralobs = 0.14,
                          spiders = 0.00625,
                          add_centralobs = 7.3/100,
                          add_spiders = 3.12/100,
                          add_outer_edge_obs = 1.8/100):
    """
    Return the SPHERE Lyot stop

    default values of additional central obstruction, spiders size and
    outer edge obstruction have been estimated by eye on the real lyot stop
    WARNING : this lyot stop does not feature the dead actuators patches

    Args :
        pupdiam (int) [pixel] : pupil diameter
        
        centralobs (float, optional) [fraction of diameter] : central obstruction diameter

        spiders (float, optional) [fraction of diameter] : spider diameter

        add_centralobs (float, optional) [fraction of diameter] : additional diameter of central obstruction
        
        add_spiders (float, optional) [fraction of diameter] : additional diameter of spiders obstruction
        
        add_outer_edge_obs (float, optional) [fraction of diameter] : outer edge obstruction size
    
    Returns :
        lyotStop (2D array) : Sphere lyot Stop transmission pupil of shape (pupdiam, pupdiam), filled with 0 and 1
    """
    lyotCentralObs = centralobs + add_centralobs

    range = 0.5  # !
    X = np.tile(np.linspace(-range, range, pupdiam, dtype=np.float32), (pupdiam, 1))
    R = np.sqrt(X**2 + (X.T)**2)
    lyotCentralMap = ((R < 0.5) & (R > (lyotCentralObs / 2))).astype(np.float32)

    angle = 50.5 * np.pi / 180.  # 50.5 degrees = angle between spiders
    if pupdiam % 2 == 0 :
        lyotSpidersMap = (
            (X.T > (X - centralobs / 2 + (spiders + add_spiders / 2) / np.sin(angle)) * np.tan(angle)) +
            (X.T < (X - centralobs / 2 - add_spiders / 2 / np.sin(angle)) * np.tan(angle))
            ) * (X > 0) * (X.T > 0)
    elif pupdiam % 2 == 1 :
        lyotSpidersMap = (
            (X.T > (X - centralobs / 2 + (spiders + add_spiders / 2) / np.sin(angle)) * np.tan(angle)) +
            (X.T < (X - centralobs / 2 - add_spiders / 2 / np.sin(angle)) * np.tan(angle))
            ) * (X >= 0) * (X.T >= 0)
    lyotSpidersMap += np.fliplr(lyotSpidersMap)
    lyotSpidersMap += np.flipud(lyotSpidersMap)

    X = np.tile(np.linspace(-range, range, pupdiam, dtype=np.float32), (pupdiam, 1))
    R = np.sqrt(X**2 + (X.T)**2)
    lyotOuterEdge = (R < 0.5 - add_outer_edge_obs)
    
    lyotStop = lyotCentralMap*lyotSpidersMap*lyotOuterEdge
    return lyotStop


# --------------------------------------------------- #
#      Useful functions for contrast computation      #
# --------------------------------------------------- #

def ring(center, radius_min, radius_max, shape):
    """ Returns a ring mask

    Args:
        center: (float or (float, float)): center of the ring in pixel

        radius_min: (float): internal radius in pixel

        radius_max: (float): external radius in pixel

        shape: (int or (int, int)): shape of the image in pixel

    Return:
        mask (boolean 2D array) : image of boolean, filled
            with False outside the ring and True inside
    """
    if np.isscalar(shape):
        shape = (shape, shape)
    if np.isscalar(center):
        center = (center, center)
    mask = np.full(shape, False)
    xx, yy = np.meshgrid(np.arange(shape[0]) - center[0], np.arange(shape[1]) - center[1])
    rr = np.hypot(xx, yy)
    mask = np.logical_and((radius_min <= rr), (rr <= radius_max))
    return mask

def compute_contrast(image, center, r_min, r_max, width):
    """ Computes average, standard deviation, minimum and maximum
    over rings at several distances from the center of the image.

    A ring includes the pixels between the following radiuses :
    r_min + k * width - width / 2 and r_min + k * width + width / 2,
    with k = 0, 1, 2... until r_min + k * width > r_max (excluded).

    Args:
        image: (2D array): the coronagraphic image

        center: (float or (float, float)): center of the coronagraphic image in pixel

        r_min: (float): radius of the first ring in pixel

        r_max: (float): maximum radius in pixel

        width: (float): width of one ring in pixel

    Returns:
        distances: (1D array): distances in pixel

        mean: (1D array): corresponding averages

        std: (1D array): corresponding standard deviations

        mini: (1D array): corresponding minimums

        maxi: (1D array): corresponding maximums
    """

    distances = np.arange(r_min, r_max, width)
    mean = []
    std = []
    mini = []
    maxi = []
    for radius in distances:
        mask = ring(center, radius - width / 2, radius + width / 2, image.shape)
        values = image[mask]
        mean.append(np.mean(values))
        std.append(np.std(values))
        mini.append(np.min(values))
        maxi.append(np.max(values))
    mean = np.array(mean)
    std = np.array(std)
    mini = np.array(mini)
    maxi = np.array(maxi)
    return distances, mean, std, mini, maxi
