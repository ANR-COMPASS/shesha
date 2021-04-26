import json
import numpy as np
from shesha.util.writers import common

def write_json_atm_param(sup, *, file_name="./atm-params.json"):

    """Return a json representation of the atmospheric parameters

    Args:
        sup : (CompassSupervisor) : supervisor to get the json representation from
    """
    atm_json={
        "notice" : common.atmos_json_notice(),
        "profiles" : [ common.atmos_to_json(sup.config.p_atmos)]
    }
    f = open(file_name,"w")
    f.write(json.dumps(atm_json,indent=4))
    f.close()