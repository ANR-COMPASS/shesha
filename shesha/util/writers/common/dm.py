
def get_actu_pos_pixel(dm):
    """return the coordinates in pixel of a given DM actuators

    Args:
        dm : (ParamDm) : Dm to get the actuators position from

    Returns:
        xpos : (np.ndarray[ndim=1, dtype=np.float32]) : actuators positions along axis x

        ypos : (np.ndarray[ndim=1, dtype=np.float32]) : actuators positions along axis y
    """

    return dm._xpos+1, dm._ypos+1


def get_actu_pos_meter(sup, dm_id):
    """return the coordinates in meters of a given DM actuators

    Args:
        sup : (compasSSupervisor) : supervisor

        dm_id : (int) : index of the DM

    Returns:
        xpos : (np.ndarray[ndim=1, dtype=np.float32]) : actuators positions along axis x

        ypos : (np.ndarray[ndim=1, dtype=np.float32]) : actuators positions along axis y
    """

    config = sup.config
    dm=config.p_dms[dm_id]
    geom = config.p_geom
    valid_X = ( dm._xpos - geom.get_cent() ) * geom.get_pixsize()
    valid_Y = ( dm._ypos - geom.get_cent() ) * geom.get_pixsize()
    return valid_X, valid_Y


def dm_to_json(dm, geom):
    """return a json description of a dm

    Args:
        dm : (ParamDm) : dm to represent as json
    """
    dm_json = {
        "n_actu" : dm.get_nact(),
        "h" : dm.get_alt(),
        "coupling" : dm.get_coupling(),
        "shift_x" : dm.get_dx() * geom.get_pixsize(),
        "shift_y" : dm.get_dy() * geom.get_pixsize(),
        "theta" : dm.get_theta()
    }
    return dm_json