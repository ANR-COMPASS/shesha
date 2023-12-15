
def atmos_to_json(atmos, name=""):
    """return a json description of a the atmos

    Args:
        atmos : (ParamAtmos) : compass atmospheric parameters
    """
    json_atm = {
        "nLayer" : atmos.get_nscreens(),
        "r0" : atmos.get_r0(),
        "h" : atmos.get_alt().tolist(),
        "fracCn2" : atmos.get_frac().tolist(),
        "L0" : atmos.get_L0().tolist(),
        "windDir" : atmos.get_winddir().tolist(),
        "windSpeed" : atmos.get_windspeed().tolist()
    }
    if(name != ""):
        json_atm["name"] = name
    return json_atm


def atmos_json_notice():
    notice = {
        "name": "          : profile name",
        "nLayer": "          : number of layers in the turbulent profile",
        "r0": "r0 at 500 nm (fried parameter)",
        "h": " meter    : (list) altitude of each layer",
        "fracCn2": " percent  : (list) cn2 fraction of each layer",
        "L0":  "meter    : (list) outer scale of each layer",
        "windDir": " degree   : (list) wind sirection of each layer",
        "windSpeed": " meter/s  : (list) wind speed of each layer"
    }
    return notice