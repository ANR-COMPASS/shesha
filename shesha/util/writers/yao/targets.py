import numpy as np
def write_targets(file_name, tars, *, sub_system=1):
    """Write (append) target parameter to file for YAO use for a single dm

    Args:
        file_name : (str) : name of the file to append the parameter to

        tars : (list[ParamTarget]) : compass target parameters list

    Kwargs:
        sub_system : (int) : (optional), default 1 yao sub system index
    """
    f=open(file_name,"a+")
    f.write("\n\n//------------------------------")
    f.write("\n//TAR  parameters")
    f.write("\n//------------------------------")

    f.write("\ntarget.lambda       = &(" + np.array2string(np.array( \
            [t.Lambda for t in tars]), separator=',', max_line_width=300) + \
            ");") #&([0.55]);
    f.write("\ntarget.xposition    = &(" + np.array2string(np.array(\
            [t.xpos for t in tars]), separator=',', max_line_width=300) + \
            ");") # &mmsepos_asec1;
    f.write("\ntarget.yposition    = &(" + np.array2string(np.array( \
            [t.ypos for t in tars]), separator=',', max_line_width=300) + \
            ");") # &mmsepos_asec2;
    dispzoom = np.ones((len(tars)))
    f.write("\ntarget.dispzoom     = &(" + np.array2string(dispzoom, \
            separator=',',max_line_width=300) + ") ; // not set by compass")
            #+ np.array2string(np.array([t.mag for t in tars]),separator=',',max_line_width=300)+";)") #  &array(5.0,numberof(mmsepos_asec1));
