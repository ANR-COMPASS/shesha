import numpy as np

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
    Return the SPHERE APLC apodizer, based on its radial transmission profile

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
    X = np.tile(np.linspace(-1, 1, pupdiam, dtype=np.float32), (pupdiam, 1))
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

    range = 0.5
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