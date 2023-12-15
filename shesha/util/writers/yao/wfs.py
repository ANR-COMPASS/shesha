
YAO_WFSTYPE={"sh":"\"hartmann\"", "pyrhr":"\"pyramid\""}

def init_wfs(file_name):
    """ Initialise wfs entry in yao parameter file

    Args:
        file_name : (str) : yao parameter file name
    """
    f = open(file_name,"a+")
    f.write("\n\n//------------------------------")
    f.write("\n//WFS parameters")
    f.write("\n//------------------------------")
    return (0,0)

def write_wfs(file_name, wfs, index, *, sub_system=1):
    """Write (append) wfs parameter to file for YAO use for a single wfs

    Args:
        file_name : (str) : name of the file to append the parameter to

        wfs : (ParamWfs) : compass wfs parameters

        index :(int) : wfs index in ayo parameter file

    Kwargs:
        sub_system : (int) : (optional), default 1 sub_system in yao
    """
    obj = "wfs(" + str(index) + ")"
    f = open(file_name, "a+")
    f.write("\ngrow,wfs,wfss;")
    f.write("\n" + obj + ".type           = " + YAO_WFSTYPE[wfs.type] + ";")
    f.write("\n" + obj + ".subsystem     = " + str(sub_system) + ";")
    f.write("\n" + obj + ".shmethod      = 2" + ";")
    f.write("\n" + obj + ".shnxsub       = " + str(wfs.nxsub) + ";")
    f.write("\n" + obj + ".lambda        = " + str(wfs.Lambda) + ";")
    f.write("\n" + obj + ".pixsize       = " + str(wfs.pixsize) + ";")
    f.write("\n" + obj + ".npixels       = " + str(wfs.npix) + ";")
    f.write("\n" + obj + ".shthreshold   = 0;   // not set by compass")
    f.write("\n" + obj + ".dispzoom      = 1.0; // not set by compass")
    f.write("\n" + obj + ".fracIllum     = " + str(wfs.fracsub) + ";")
    f.write("\n" + obj + ".rotation      = " + str(wfs.thetaML) + ";")
    f.write("\n" + obj + ".shift         = [ " + str(wfs.dx)   + " , " + \
            str(wfs.dy) + " ];")
    f.write("\n" + obj + ".LLTxy         = [ " + str(wfs.lltx) + " , " + \
            str(wfs.llty) + " ];")
    f.write("\n" + obj + ".gspos         = [ " + str(wfs.xpos) + " , " + \
            str(wfs.ypos) + " ];")
    if(wfs.noise<0):
        f.write("\n" +obj + ".noise         = 1;")
        f.write("\n" +obj + ".ron           = 0;")
    else:
        f.write("\n" + obj + ".noise         = 1;")
        f.write("\n" + obj + ".ron           = " + str(wfs.noise) + ";")
    f.write("\n" + obj + ".darkcurrent   = 0 ; // not set by compass ")
    if(wfs.gsalt > 0):
        f.write("\n" + obj + ".gsalt         = " + str(wfs.gsalt) + ";")
        f.write("\n" + obj + ".gsdepth       = " + str(1) + ";")
        f.write("\n" + obj + ".optthroughput = " + str(wfs.optthroughput) +\
                ";")
        f.write("\n" + obj + ".laserpower    = " + str(wfs.laserpower) + ";")
        f.write("\n" + obj + ".filtertilt    = " + str(1) + ";")
        f.write("\n" + obj + ".correctUpTT   = " + str(1) + ";")
        f.write("\n" + obj + ".uplinkgain    = " + str(0.2) + ";")
    f.close()


def write_wfss(file_name, wfss, *, n_wfs=-1, sub_system=1, offset=0):
    """Write (append) wfs parameter to file for YAO use for a wfs list

    Args:
        file_name : (str) : name of the file to append the parameter to

        wfss : (list[ ParamWfs]) : compass wfs parameters list

    Kwargs:
        n_wfs : (int) : (optional), default -1 number of wfs passed to yao (-1 : all wfs)

        sub_system : (int) : (optional), default 1 yao sub system index

        offset : (int) : (optional), default 0 yao wfs index offset

    Returns:
        n_ngs : (int) : number of ngs passed to yao
        n_lgs : (int) : number of lgs passed to yao
    """
    #counting nb of lgs and ngs
    n_ngs=0
    n_lgs=0
    if(n_wfs<0):
        n_wfs = len(wfss)
    for w in wfss[:n_wfs]:
        if(w.gsalt>0):
            n_lgs += 1
        else:
            n_ngs += 1
    n_wfs = n_ngs + n_lgs
    f=open(file_name, "a+")

    i = 1
    for w in wfss[:n_wfs] :
        f.write("\n\n//WFS" + str(i + offset))
        f.flush()
        write_wfs(file_name, w, i + offset, sub_system=sub_system)
        i += 1

    f.close()
    return (n_ngs , n_lgs)

################################

def finish_wfs(file_name, n_ngs, n_lgs):
    """ Finalize wfs section in yao parameter file

    Args:
        file_name : (str) : yao parameter file name

        n_ngs : (int) : number of ngs written to yao parameter file

        n_lgs : (int) : number of lgs written to yao parameter file
    """
    f=open(file_name,"a+")
    f.write("\n\nnngs = "+str(n_ngs)+";")
    f.write("\nnlgs = "+str(n_lgs)+";")
    f.write("\nnwfs = "+str(n_ngs+n_lgs)+";")
    f.close()
