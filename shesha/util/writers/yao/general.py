import numpy as np

def write_general(file_name, geom, controllers, tel, simul_name):
    """Write (append) general simulation parameter to file for YAO use

    Args:
        file_name : (str) : name of the file to append the parameter to

        geom : (ParamGeom) : compass AO geometry parameters

        controllers : ([ParamController]) : list of compass controller parameters

        tel : (ParamTel) : compass telescope parameters

        simul_name : (str) : simulation name
    """
    f = open(file_name,"w")
    f.write("\n\n//------------------------------")
    f.write("\n//general parameters")
    f.write("\n//------------------------------")
    f.write("\nsim.name        = \"" + simul_name + "\";")
    f.write("\nsim.pupildiam   = " + str(geom.pupdiam) + ";")
    f.write("\nsim.debug       = 0;")
    f.write("\nsim.verbose     = 1;")

    f.write("\nmat.file            = \"\";")
    f.write("\nmat.condition = &(" + np.array2string( \
            np.array([np.sqrt(c.maxcond) for c in controllers]), \
            separator=',',max_line_width=300) + ");")

    f.write("\nmat.method = \"none\";")
    #f.write("\nhfield = 15")
    f.write("\nYAO_SAVEPATH = \"\"; // where to save the output to the simulations")

    f.write("\ntel.diam = " + str(tel.diam) + ";")
    f.write("\ntel.cobs = " + str(tel.cobs) + ";")
    f.write("\ndm       = [];")
    f.write("\nwfs      = [];")
