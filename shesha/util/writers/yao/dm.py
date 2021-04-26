YAO_DMTYPE={"pzt":"\"stackarray\"",
            "tt" :"\"tiptilt\""}

def init_dm(file_name):
    """ Initialise dm entry in yao parameter file

    Args:
        file_name : (str) : yao parameter file name
    """
    f = open(file_name, "a+")
    f.write("\n\n//------------------------------")
    f.write("\n//DM  parameters")
    f.write("\n//------------------------------")
    f.close()
    return 0

def write_dm(file_name, dm, index, *, sub_system=1):
    """Write (append) dm parameter to file for YAO use for a single dm

    Args:
        file_name : (str) : name of the file to append the parameter to

        dm : (Param_dm) : compass dm  parameters

        index : (int) : YAO index for dm

        sub_system : (int) : (optional), default 1 index of yao sub-system
    """
    obj = "dm(" + str(index) + ")"
    f = open(file_name,"a+")
    f.write("\ngrow,dm,dms;")
    f.write("\n" + obj + ".type          = " + YAO_DMTYPE[dm.type] + ";")
    f.write("\n" + obj + ".subsystem     = " + str(sub_system) + ";")
    f.write("\n" + obj + ".iffile        = \"\"; // not set by compass")
    f.write("\n" + obj + ".alt           = " + str(dm.alt) + ";")
    f.write("\n" + obj + ".unitpervolt   = " + str(dm.unitpervolt) + ";")
    f.write("\n" + obj + ".push4imat     = " + str(dm.push4imat) + ";")

    if(dm.type != "tt"):
        f.write("\n" + obj + ".nxact         = " + str(dm.nact) + ";")
        f.write("\n" + obj + ".pitch         = " + str(dm._pitch) + ";")
        f.write("\n" + obj + ".thresholdresp = " + str(dm.thresh) + ";")
        f.write("\n" + obj + ".pitchMargin   = " + str(2.2) + "; // not set by compass")
        f.write("\n" + obj + ".elt           = " + str(1) + "; // not set by compass")
        f.write("\n" + obj + ".coupling    = " + str(dm.coupling) + ";")
    f.close()

def write_dms(file_name, dms, *, sub_system=1, offset=0):
    """Write (append) dm parameter to file for YAO

    Args:
        file_name : str       : name of the file to append the parameter to

        dms       : list[Param_dm] : compass dm  parameters list

    Kwargs:
        sub_system : (int) : (optional), default 1 index of yao sub-system

        offset : (int) : (optional), default 0 yao dm index offset

    Returns:
        n_dm : (int) : number of dm passed to yao
    """
    f = open(file_name,"a+")

    i = 1
    for d in dms:
        f.write("\n\n//DM " + str(i + offset))
        f.flush()
        write_dm(file_name, d, i + offset, sub_system=sub_system)
        i += 1

    f.close()
    return len(dms)

def finish_dm(file_name, n_dm):
    """ Finalize wfs section in yao parameter file

    Args:
        file_name : (str) : yao parameter file name

        n_dm : (int) : number of ngs written to yao parameter file
    """
    f=open(file_name, "a+")
    f.write("\n\nndm = " + str(n_dm) + ";")
    f.close()
