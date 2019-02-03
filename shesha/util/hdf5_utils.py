"""
Functions for handling the database system
Wrapping of some h5py function for quick HDF5 save
"""
import h5py
import pandas
import os
import numpy as np
from subprocess import check_output


def updateParamDict(pdict, pClass, prefix):
    """
    Update parameters dictionnary pdict with all the parameters of pClass.
    Prefix must be set to define the key value of the new dict entries
    """
    if (isinstance(pClass, list)):
        params = [
                i for i in dir(pClass[0])
                if (not i.startswith('_') and not i.startswith('set_') and
                    not i.startswith('get_'))
        ]
        for k in params:
            pdict.update({
                    prefix + k: [
                            p.__dict__[prefix + k].encode("utf8")
                            if isinstance(p.__dict__[prefix + k], str) else
                            p.__dict__[prefix + k] for p in pClass
                    ]
            })

    else:
        params = [
                i for i in dir(pClass)
                if (not i.startswith('_') and not i.startswith('set_') and
                    not i.startswith('get_'))
        ]

        for k in params:
            if isinstance(pClass.__dict__[prefix + k], str):
                pdict.update({prefix + k: pClass.__dict__[prefix + k].encode("utf8")})
            else:
                pdict.update({prefix + k: pClass.__dict__[prefix + k]})


def params_dictionary(config):
    """ Create and returns a dictionary of all the config parameters with the
    corresponding keys for further creation of database and save files

    :param config: (module) : simulation parameters

    :return param_dict: (dictionary) : dictionary of parameters
    """

    commit = check_output(["git", "rev-parse", "--short", "HEAD"]).strip()

    param_dict = {"simul_name": config.simul_name.encode('utf8'), "commit": commit}

    updateParamDict(param_dict, config.p_loop, "_Param_loop__")
    updateParamDict(param_dict, config.p_geom, "_Param_geom__")
    updateParamDict(param_dict, config.p_tel, "_Param_tel__")
    if config.p_atmos is not None:
        updateParamDict(param_dict, config.p_atmos, "_Param_atmos__")
    if config.p_target is not None:
        updateParamDict(param_dict, config.p_targets, "_Param_target__")
        param_dict.update({"ntargets": len(config.p_targets)})
    if config.p_wfss is not None:
        updateParamDict(param_dict, config.p_wfss, "_Param_wfs__")
        param_dict.update({"nwfs": len(config.p_wfss)})
    if config.p_dms is not None:
        updateParamDict(param_dict, config.p_dms, "_Param_dm__")
        param_dict.update({"ndms": len(config.p_dms)})
    if config.p_controllers is not None:
        updateParamDict(param_dict, config.p_controllers, "_Param_controller__")
        param_dict.update({"ncontrollers": len(config.p_controllers)})
    if config.p_centroiders is not None:
        updateParamDict(param_dict, config.p_centroiders, "_Param_centroider__")
        param_dict.update({"ncentroiders": len(config.p_centroiders)})

    for k in param_dict.keys():
        if type(param_dict[k]) is list:
            param_dict[k] = [d if d is not None else -10 for d in param_dict[k]]
        elif param_dict[k] is None:
            param_dict[k] = -10
    return param_dict


def create_file_attributes(filename, param_dict):
    """ create_file_attributes(filename,config)
    Create an hdf5 file wtih attributes corresponding to all simulation parameters

    :param:

        filename : (str) : full path + filename to create

        config : () : simulation parameters
    """
    f = h5py.File(filename, "w")

    for i in list(param_dict.keys()):
        if (isinstance(param_dict[i], str)):
            attr = param_dict[i].encode("utf-8")
        elif (isinstance(param_dict[i], list)):
            attr = [
                    s.encode("utf-8") if isinstance(s, str) else s for s in param_dict[i]
            ]
        else:
            attr = param_dict[i]
        f.attrs.create(i, attr)
    f.attrs.create("validity", False)
    print(filename, "initialized")
    f.close()


def init_hdf5_files(savepath, param_dict, matricesToLoad):
    """ TODO: docstring
    """
    commit = check_output(["git", "rev-parse", "--short", "HEAD"]).decode('utf8').strip()
    # if not(matricesToLoad.has_key("A")):
    if "A" not in matricesToLoad:
        df = pandas.read_hdf(savepath + "matricesDataBase.h5", "A")
        ind = len(df.index)
        filename = savepath + "turbu/A_" + commit + "_" + str(ind) + ".h5"
        create_file_attributes(filename, param_dict)
        updateDataBase(filename, savepath, "A")

    if not ("dm" in matricesToLoad):
        df = pandas.read_hdf(savepath + "matricesDataBase.h5", "dm")
        ind = len(df.index)
        filename = savepath + "mat/dm_" + commit + "_" + str(ind) + ".h5"
        create_file_attributes(filename, param_dict)
        updateDataBase(filename, savepath, "dm")

    if not ("imat" in matricesToLoad):
        df = pandas.read_hdf(savepath + "matricesDataBase.h5", "imat")
        ind = len(df.index)
        filename = savepath + "mat/imat_" + commit + "_" + str(ind) + ".h5"
        create_file_attributes(filename, param_dict)
        updateDataBase(filename, savepath, "imat")


def initDataBase(savepath, param_dict):
    """ Initialize and create the database for all the saved matrices. This database
    will be placed on the top of the savepath and be named matricesDataBase.h5.

    :parameters:

        savepath : (str) : path to the data repertory

        param_dict : (dictionary) : parameters dictionary
    """
    keys = list(param_dict.keys())
    keys.append("path2file")
    keys.append("validity")
    df = pandas.DataFrame(columns=keys)
    store = pandas.HDFStore(savepath + "matricesDataBase.h5")
    store.put("A", df)
    store.put("imat", df)
    store.put("dm", df)
    store.close()
    print("Matrices database created")


def updateDataBase(h5file, savepath, matrix_type):
    """ Update the database adding a new row to the matrix_type database.

    :parameters:

        h5file : (str) : path to the new h5 file to add

        savepath : (str) : path to the data directory

        matrix_type : (str) : type of matrix to store ("A","B","istx","isty"
                                                         "istx","eigenv","imat","U"
                                                         "pztok" or "pztnok")
    """
    if (matrix_type == "A" or matrix_type == "imat" or matrix_type == "dm"):
        f = h5py.File(h5file, "r")
        store = pandas.HDFStore(savepath + "matricesDataBase.h5")
        df = store[matrix_type]
        ind = len(df.index)
        for i in list(f.attrs.keys()):
            df.loc[ind, i] = f.attrs[i]
        df.loc[ind, "path2file"] = h5file
        df.loc[ind, "validity"] = False
        store.put(matrix_type, df)
        store.close()
        f.close()
    else:
        raise ValueError("Wrong matrix_type specified. See documentation")


def save_hdf5(filename, dataname, data):
    """ save_hdf5(filename, dataname, data)
    Create a dataset in an existing hdf5 file filename and store data in it

    :param:

        filename: (str) : full path to the file

        dataname : (str) : name of the data (imat, cmat...)

        data : np.array : data to save
    """
    f = h5py.File(filename, "r+")
    f.create_dataset(dataname, data=data)
    f.close()


def save_h5(filename, dataname, config, data):
    """ save_hdf5(filename, dataname, config, data)
    Create a hdf5 file and store data in it with full header from config parameters
    Usefull to backtrace data origins

    :param:

        filename: (str) : full path to the file

        dataname : (str) : name of the data (imat, cmat...)

        config : (module) : config parameters

        data : np.array : data to save
    """
    p_dict = params_dictionary(config)
    create_file_attributes(filename, p_dict)
    save_hdf5(filename, dataname, data)
    print(filename, "has been written")


def checkMatricesDataBase(savepath, config, param_dict):
    """ Check in the database if the current config have been already run. If so,
    return a dictionary containing the matrices to load and their path. Matrices
    which don't appear in the dictionary will be computed, stored and added
    to the database during the simulation.
    If the database doesn't exist, this function creates it.

    :parameters:

        savepath : (str) : path to the data repertory

        config : (module) : simulation parameters

        param_dict : (dictionary) : parameters dictionary

    :return:

        matricesToLoad : (dictionary) : matrices that will be load and their path
    """

    matricesToLoad = {}
    if (os.path.exists(savepath + "matricesDataBase.h5")):
        checkTurbuParams(savepath, config, param_dict, matricesToLoad)
        checkDmsParams(savepath, config, param_dict, matricesToLoad)
        #        if(matricesToLoad.has_key("pztok")):
        if "dm" in matricesToLoad:
            checkControlParams(savepath, config, param_dict, matricesToLoad)

    else:
        initDataBase(savepath, param_dict)
    init_hdf5_files(savepath, param_dict, matricesToLoad)
    return matricesToLoad


def checkTurbuParams(savepath, config, pdict, matricesToLoad):
    """ Compare the current turbulence parameters to the database. If similar parameters
    are found, the matricesToLoad dictionary is completed.
    Since all the turbulence matrices are computed together, we only check the parameters
    for the A matrix : if we load A, we load B, istx and isty too.

    :parameters:

        config : (module) : simulation parameters

        matricesToLoad : (dictionary) :  matrices that will be load and their path
    """
    dataBase = pandas.read_hdf(savepath + "matricesDataBase.h5", "A")
    param2test = [
            "_Param_atmos__r0", "_Param_atmos__seeds", "_Param_atmos__L0",
            "_Param_atmos__alt", "_Param_tel__diam", "_Param_tel__cobs",
            "_Param_geom__pupdiam", "_Param_geom__zenithangle", "_Param_target__xpos",
            "_Param_target__ypos", "_Param_wfs__xpos", "_Param_wfs__ypos"
    ]

    for i in dataBase.index:
        cc = 0
        commit = check_output(["git", "rev-parse", "--short", "HEAD"]).strip()
        if (dataBase.loc[i, "validity"] and (dataBase.loc[i, "commit"] == commit)):
            cond = True
            while (cond):
                if (cc >= len(param2test)):
                    break
                else:
                    cond = dataBase.loc[i, param2test[cc]] == pdict[param2test[cc]]
                    if type(cond) is np.ndarray:
                        cond = cond.all()
                    cc += 1
            # For debug
            #############################
            if not cond:
                cc -= 1
                print(param2test[cc] + " has changed from ",
                      dataBase.loc[i, param2test[cc]], " to ", pdict[param2test[cc]])
            ###############################
        else:
            cond = False

        if (cond):
            matricesToLoad["index_turbu"] = i
            matricesToLoad["A"] = dataBase.loc[i, "path2file"]
            # dataBase = pandas.read_hdf(savepath + "matricesDataBase.h5", "B")
            # matricesToLoad["B"] = dataBase.loc[i, "path2file"]
            # dataBase = pandas.read_hdf(savepath + "matricesDataBase.h5",
            #                            "istx")
            # matricesToLoad["istx"] = dataBase.loc[i, "path2file"]
            # dataBase = pandas.read_hdf(savepath + "matricesDataBase.h5",
            #                            "isty")
            # matricesToLoad["isty"] = dataBase.loc[i, "path2file"]
            return


def checkControlParams(savepath, config, pdict, matricesToLoad):
    """ Compare the current controller parameters to the database. If similar parameters
    are found, matricesToLoad dictionary is completed.
    Since all the controller matrices are computed together, we only check the parameters
    for the imat matrix : if we load imat, we load eigenv and U too.

    :parameters:

        config : (module) : simulation parameters

        matricesToLoad : (dictionary) :  matrices that will be load and their path
    """
    dataBase = pandas.read_hdf(savepath + "matricesDataBase.h5", "imat")

    param2test = [
            "_Param_tel__diam", "_Param_tel__t_spiders", "_Param_tel__spiders_type",
            "_Param_tel__pupangle", "_Param_tel__referr", "_Param_tel__std_piston",
            "_Param_tel__std_tt", "_Param_tel__type_ap", "_Param_tel__nbrmissing",
            "_Param_tel__cobs", "_Param_geom__pupdiam", "nwfs", "_Param_wfs__type",
            "_Param_wfs__nxsub", "_Param_wfs__npix", "_Param_wfs__pixsize",
            "_Param_wfs__fracsub", "_Param_wfs__xpos", "_Param_wfs__ypos",
            "_Param_wfs__Lambda", "_Param_wfs__dms_seen", "_Param_wfs__fssize",
            "_Param_wfs__fstop", "_Param_wfs__pyr_ampl", "_Param_wfs__pyr_loc",
            "_Param_wfs__pyr_npts", "_Param_wfs__pyr_pup_sep", "_Param_wfs__pyrtype",
            "ndms", "_Param_dm__type", "_Param_dm__alt", "_Param_dm__coupling",
            "_Param_dm__margin_in", "_Param_dm__margin_out", "_Param_dm__nact",
            "_Param_dm__nkl", "_Param_dm__type_kl", "_Param_dm__push4imat",
            "_Param_dm__thresh", "_Param_dm__unitpervolt", "ncentroiders",
            "_Param_centroider__type", "_Param_centroider__nmax",
            "_Param_centroider__nwfs", "_Param_centroider__sizex",
            "_Param_centroider__sizey", "_Param_centroider__thresh",
            "_Param_centroider__type_fct", "_Param_centroider__weights",
            "_Param_centroider__width"
    ]

    for i in dataBase.index:
        cc = 0
        commit = check_output(["git", "rev-parse", "--short", "HEAD"]).strip()
        if (dataBase.loc[i, "validity"] and (dataBase.loc[i, "commit"] == commit)):
            cond = True
            while (cond):
                if (cc >= len(param2test)):
                    break
                else:
                    cond = dataBase.loc[i, param2test[cc]] == pdict[param2test[cc]]
                    if type(cond) is np.ndarray:
                        cond = cond.all()
                    cc += 1
            # For debug
            #############################
            if not cond:
                cc -= 1
                print(param2test[cc] + " has changed from ",
                      dataBase.loc[i, param2test[cc]], " to ", pdict[param2test[cc]])
            ###############################
        else:
            cond = False

        if (cond):
            matricesToLoad["index_control"] = i
            matricesToLoad["imat"] = dataBase.loc[i, "path2file"]
            return


def checkDmsParams(savepath, config, pdict, matricesToLoad):
    """ Compare the current controller parameters to the database. If similar parameters
    are found, matricesToLoad dictionary is completed.
    Since all the dms matrices are computed together, we only check the parameters
    for the pztok matrix : if we load pztok, we load pztnok too.

    :parameters:

        config : (module) : simulation parameters

        matricesToLoad : (dictionary) :  matrices that will be load and their path
    """
    dataBase = pandas.read_hdf(savepath + "matricesDataBase.h5", "dm")

    param2test = [
            "_Param_tel__diam", "_Param_tel__t_spiders", "_Param_tel__spiders_type",
            "_Param_tel__pupangle", "_Param_tel__referr", "_Param_tel__std_piston",
            "_Param_tel__std_tt", "_Param_tel__type_ap", "_Param_tel__nbrmissing",
            "_Param_tel__cobs", "_Param_geom__pupdiam", "nwfs", "_Param_wfs__type",
            "_Param_wfs__nxsub", "_Param_wfs__npix", "_Param_wfs__pixsize",
            "_Param_wfs__fracsub", "_Param_wfs__xpos", "_Param_wfs__ypos",
            "_Param_wfs__Lambda", "_Param_wfs__dms_seen", "_Param_wfs__fssize",
            "_Param_wfs__fstop", "_Param_wfs__pyr_ampl", "_Param_wfs__pyr_loc",
            "_Param_wfs__pyr_npts", "_Param_wfs__pyrtype", "_Param_wfs__pyr_pup_sep",
            "ndms", "_Param_dm__type", "_Param_dm__alt", "_Param_dm__coupling",
            "_Param_dm__margin_in", "_Param_dm__margin_out", "_Param_dm__nkl",
            "_Param_dm__nact", "_Param_dm__type_kl", "_Param_dm__push4imat",
            "_Param_dm__thresh", "_Param_dm__unitpervolt"
    ]

    for i in dataBase.index:
        cc = 0
        commit = check_output(["git", "rev-parse", "--short", "HEAD"]).strip()
        if (dataBase.loc[i, "validity"] and (dataBase.loc[i, "commit"] == commit)):
            cond = True
            while (cond):
                if (cc >= len(param2test)):
                    break
                else:
                    cond = dataBase.loc[i, param2test[cc]] == pdict[param2test[cc]]
                    if type(cond) is np.ndarray:
                        cond = cond.all()
                    cc += 1
            # For debug
            #############################
            if not cond:
                cc -= 1
                print((param2test[cc] + " has changed from ",
                       dataBase.loc[i, param2test[cc]], " to ", pdict[param2test[cc]]))
            ###############################
        else:
            cond = False

        if (cond):
            matricesToLoad["index_dms"] = i
            matricesToLoad["dm"] = dataBase.loc[i, "path2file"]
            return


def validDataBase(savepath, matricesToLoad):
    """ TODO: docstring
    """
    store = pandas.HDFStore(savepath + "matricesDataBase.h5")
    if not ("A" in matricesToLoad):
        validInStore(store, savepath, "A")
    if not ("dm" in matricesToLoad):
        validInStore(store, savepath, "dm")
    if not ("imat" in matricesToLoad):
        validInStore(store, savepath, "imat")
    store.close()


def validFile(filename):
    """ TODO: docstring
    """
    f = h5py.File(filename, "r+")
    f.attrs["validity"] = True
    f.close()


def validInStore(store, savepath, matricetype):
    """ TODO: docstring
    """
    df = store[matricetype]
    ind = len(df.index) - 1
    df.loc[ind, "validity"] = True
    store[matricetype] = df
    validFile(df.loc[ind, "path2file"])


def configFromH5(filename, config):
    """ TODO: docstring
    """
    #import shesha as ao

    f = h5py.File(filename, "r")

    config.simul_name = str(f.attrs.get("simulname"))
    # Loop
    config.p_loop.set_niter(f.attrs.get("niter"))
    config.p_loop.set_ittime(f.attrs.get("ittime"))

    # geom
    config.p_geom.set_zenithangle(f.attrs.get("zenithangle"))
    config.p_geom.set_pupdiam(f.attrs.get("pupdiam"))

    # Tel
    config.p_tel.set_diam(f.attrs.get("tel_diam"))
    config.p_tel.set_cobs(f.attrs.get("cobs"))
    config.p_tel.set_nbrmissing(f.attrs.get("nbrmissing"))
    config.p_tel.set_t_spiders(f.attrs.get("t_spiders"))
    config.p_tel.set_type_ap(str(f.attrs.get("type_ap")))
    config.p_tel.set_spiders_type(str(f.attrs.get("spiders_type")))
    config.p_tel.set_pupangle(f.attrs.get("pupangle"))
    config.p_tel.set_referr(f.attrs.get("referr"))
    config.p_tel.set_std_piston(f.attrs.get("std_piston"))
    config.p_tel.set_std_tt(f.attrs.get("std_tt"))

    # Atmos
    config.p_atmos.set_r0(f.attrs.get("r0"))
    config.p_atmos.set_nscreens(f.attrs.get("nscreens"))
    config.p_atmos.set_frac(f.attrs.get("frac"))
    config.p_atmos.set_alt(f.attrs.get("atm.alt"))
    config.p_atmos.set_windspeed(f.attrs.get("windspeed"))
    config.p_atmos.set_winddir(f.attrs.get("winddir"))
    config.p_atmos.set_L0(f.attrs.get("L0"))
    config.p_atmos.set_seeds(f.attrs.get("seeds"))

    # Target
    config.p_target.set_nTargets(f.attrs.get("ntargets"))
    config.p_target.set_xpos(f.attrs.get("target.xpos"))
    config.p_target.set_ypos(f.attrs.get("target.ypos"))
    config.p_target.set_Lambda(f.attrs.get("target.Lambda"))
    config.p_target.set_mag(f.attrs.get("target.mag"))
    if (f.attrs.get("target.dms_seen") > -1):
        config.p_target.set_dms_seen(f.attrs.get("target.dms_seen"))

    # WFS
    config.p_wfss = []
    for i in range(f.attrs.get("nwfs")):
        config.p_wfss.append(ao.Param_wfs())
        config.p_wfss[i].set_type(str(f.attrs.get("type")[i]))
        config.p_wfss[i].set_nxsub(f.attrs.get("nxsub")[i])
        config.p_wfss[i].set_npix(f.attrs.get("npix")[i])
        config.p_wfss[i].set_pixsize(f.attrs.get("pixsize")[i])
        config.p_wfss[i].set_fracsub(f.attrs.get("fracsub")[i])
        config.p_wfss[i].set_xpos(f.attrs.get("wfs.xpos")[i])
        config.p_wfss[i].set_ypos(f.attrs.get("wfs.ypos")[i])
        config.p_wfss[i].set_Lambda(f.attrs.get("wfs.Lambda")[i])
        config.p_wfss[i].set_gsmag(f.attrs.get("gsmag")[i])
        config.p_wfss[i].set_optthroughput(f.attrs.get("optthroughput")[i])
        config.p_wfss[i].set_zerop(f.attrs.get("zerop")[i])
        config.p_wfss[i].set_noise(f.attrs.get("noise")[i])
        config.p_wfss[i].set_atmos_seen(f.attrs.get("atmos_seen")[i])
        config.p_wfss[i].set_fstop(str(f.attrs.get("fstop")[i]))
        config.p_wfss[i].set_pyr_npts(f.attrs.get("pyr_npts")[i])
        config.p_wfss[i].set_pyr_ampl(f.attrs.get("pyr_ampl")[i])
        config.p_wfss[i].set_pyrtype(str(f.attrs.get("pyrtype")[i]))
        config.p_wfss[i].set_pyr_loc(str(f.attrs.get("pyr_loc")[i]))
        config.p_wfss[i].set_fssize(f.attrs.get("fssize")[i])
        if ((f.attrs.get("dms_seen")[i] > -1).all()):
            config.p_wfss[i].set_dms_seen(f.attrs.get("dms_seen")[i])

        # LGS
        config.p_wfss[i].set_gsalt(f.attrs.get("gsalt")[i])
        config.p_wfss[i].set_lltx(f.attrs.get("lltx")[i])
        config.p_wfss[i].set_llty(f.attrs.get("llty")[i])
        config.p_wfss[i].set_laserpower(f.attrs.get("laserpower")[i])
        config.p_wfss[i].set_lgsreturnperwatt(f.attrs.get("lgsreturnperwatt")[i])
        config.p_wfss[i].set_proftype(str(f.attrs.get("proftype")[i]))
        config.p_wfss[i].set_beamsize(f.attrs.get("beamsize")[i])

    # DMs
    config.p_dms = []
    if (f.attrs.get("ndms")):
        for i in range(f.attrs.get("ndms")):
            config.p_dms.append(ao.Param_dm())
            config.p_dms[i].set_type(str(f.attrs.get("type")[i]))
            config.p_dms[i].set_nact(f.attrs.get("nact")[i])
            config.p_dms[i].set_alt(f.attrs.get("dm.alt")[i])
            config.p_dms[i].set_thresh(f.attrs.get("dm.thresh")[i])
            config.p_dms[i].set_coupling(f.attrs.get("coupling")[i])
            config.p_dms[i].set_unitpervolt(f.attrs.get("unitpervolt")[i])
            config.p_dms[i].set_push4imat(f.attrs.get("push4imat")[i])

    # Centroiders
    config.p_centroiders = []
    if (f.attrs.get("ncentroiders")):
        for i in range(f.attrs.get("ncentroiders")):
            config.p_centroiders.append(ao.Param_centroider())
            config.p_centroiders[i].set_nwfs(f.attrs.get("centro.nwfs")[i])
            config.p_centroiders[i].set_type(str(f.attrs.get("type")[i]))
            config.p_centroiders[i].set_type_fct(str(f.attrs.get("type_fct")[i]))
            config.p_centroiders[i].set_nmax(f.attrs.get("nmax")[i])
            config.p_centroiders[i].set_thresh(f.attrs.get("centroider.thresh")[i])
            if (f.attrs.get("weights")[i]):
                config.p_centroiders[i].set_weights(f.attrs.get("weights")[i])
            config.p_centroiders[i].set_width(f.attrs.get("width")[i])
        config.p_rtc.set_centroiders(config.p_centroiders)

    # Controllers
    config.p_controllers = []
    if (f.attrs.get("ncontrollers")):
        for i in range(f.attrs.get("ncontrollers")):
            config.p_controllers.append(ao.Param_controller())
            config.p_controllers[i].set_type(str(f.attrs.get("type")[i]))
            config.p_controllers[i].set_nwfs(f.attrs.get("control.nwfs")[i])
            config.p_controllers[i].set_ndm(f.attrs.get("ndm")[i])
            config.p_controllers[i].set_maxcond(f.attrs.get("maxcond")[i])
            config.p_controllers[i].set_delay(f.attrs.get("delay")[i])
            config.p_controllers[i].set_gain(f.attrs.get("gain")[i])
            config.p_controllers[i].set_modopti(f.attrs.get("modopti")[i])
            config.p_controllers[i].set_nrec(f.attrs.get("nrec")[i])
            config.p_controllers[i].set_nmodes(f.attrs.get("nmodes")[i])
            config.p_controllers[i].set_gmin(f.attrs.get("gmin")[i])
            config.p_controllers[i].set_gmax(f.attrs.get("gmax")[i])
            config.p_controllers[i].set_ngain(f.attrs.get("ngain")[i])
            config.p_controllers[i].set_TTcond(f.attrs.get("TTcond")[i])
            config.p_controllers[i].set_cured_ndivs(f.attrs.get("cured_ndivs")[i])
        config.p_rtc.set_controllers(config.p_controllers)

    config.p_rtc.set_nwfs(f.attrs.get("nwfs"))

    print("Parameters have been read from ", filename, "header")


def writeHdf5SingleDataset(filename, data, datasetName="dataset"):
    """ Write a hdf5 file containig a single field

    If the file already exists, it will be overwritten

    :parametres:

        filename: (str) : name of the file to write

        data: (np.ndarray) : content of the file

        datasetName: (str) : name of the dataset to write (default="dataset")
    """

    f = h5py.File(filename, "w")
    f.create_dataset(datasetName, data=data)
    f.close()


def readHdf5SingleDataset(filename, datasetName="dataset"):
    """ Read a single dataset from an hdf5 file

    :parameters:

        filename: (str) : name of the file to read from

        datasetName: (str) : name of the dataset to read (default="dataset")
    """

    f = h5py.File(filename, "r")
    data = f[datasetName][:]
    f.close()
    return data


def load_AB_from_dataBase(database, ind):
    """ Read and return A, B, istx and isty from the database

    :parameters:

        database: (dict): dictionary containing paths to matrices to load

        ind: (int): layer index
    """
    print("loading", database["A"])
    f = h5py.File(database["A"], 'r')
    A = f["A_" + str(ind)][:]
    B = f["B_" + str(ind)][:]
    istx = f["istx_" + str(ind)][:]
    isty = f["isty_" + str(ind)][:]
    f.close()

    return A, B, istx, isty


def save_AB_in_database(k, A, B, istx, isty):
    """ Save A, B, istx and isty in the database

    :parameters:

        ind:

        A:

        B:

        istx:

        isty:
    """
    commit = check_output(["git", "rev-parse", "--short", "HEAD"]).decode('utf8').strip()
    print("writing files and updating database")
    df = pandas.read_hdf(
            os.getenv('SHESHA_ROOT') + "/data/dataBase/matricesDataBase.h5", "A")
    ind = len(df.index) - 1
    savename = os.getenv('SHESHA_ROOT') + "/data/dataBase/turbu/A_" + \
        commit + "_" + str(ind) + ".h5"
    save_hdf5(savename, "A_" + str(k), A)
    save_hdf5(savename, "B_" + str(k), B)
    save_hdf5(savename, "istx_" + str(k), istx)
    save_hdf5(savename, "isty_" + str(k), isty)


def load_dm_geom_from_dataBase(database, ndm):
    """ Read and return the DM geometry

    :parameters:

        database: (dict): dictionary containing paths to matrices to load

        ndm: (int): dm index
    """
    print("loading", database["dm"])
    f = h5py.File(database["dm"], 'r')
    influpos = f["influpos_" + str(ndm)][:]
    ninflu = f["ninflu_" + str(ndm)][:]
    influstart = f["influstart_" + str(ndm)][:]
    i1 = f["i1_" + str(ndm)][:]
    j1 = f["j1_" + str(ndm)][:]
    ok = f["ok_" + str(ndm)][:]
    f.close()

    return influpos, ninflu, influstart, i1, j1, ok


def save_dm_geom_in_dataBase(ndm, influpos, ninflu, influstart, i1, j1, ok):
    """ Save the DM geometry in the database

    :parameters:

        ndm:

        influpos:

        ninflu:

        influstart:

        i1:

        j1:
    """
    commit = check_output(["git", "rev-parse", "--short", "HEAD"]).decode('utf8').strip()
    print("writing files and updating database")
    df = pandas.read_hdf(
            os.getenv('SHESHA_ROOT') + "/data/dataBase/matricesDataBase.h5", "dm")
    ind = len(df.index) - 1
    savename = os.getenv('SHESHA_ROOT') + "/data/dataBase/mat/dm_" + \
        commit + "_" + str(ind) + ".h5"
    save_hdf5(savename, "influpos_" + str(ndm), influpos)
    save_hdf5(savename, "ninflu_" + str(ndm), ninflu)
    save_hdf5(savename, "influstart_" + str(ndm), influstart)
    save_hdf5(savename, "i1_" + str(ndm), i1)
    save_hdf5(savename, "j1_" + str(ndm), j1)
    save_hdf5(savename, "ok_" + str(ndm), ok)


def load_imat_from_dataBase(database):
    """ Read and return the imat

    :parameters:

        database: (dict): dictionary containing paths to matrices to load
    """
    print("loading", database["imat"])
    f = h5py.File(database["imat"], 'r')
    imat = f["imat"][:]
    f.close()

    return imat


def save_imat_in_dataBase(imat):
    """ Save the DM geometry in the database

    :parameters:

        imat: (np.ndarray): imat to save
    """
    commit = check_output(["git", "rev-parse", "--short", "HEAD"]).decode('utf8').strip()
    print("writing files and updating database")
    df = pandas.read_hdf(
            os.getenv('SHESHA_ROOT') + "/data/dataBase/matricesDataBase.h5", "imat")
    ind = len(df.index) - 1
    savename = os.getenv('SHESHA_ROOT') + "/data/dataBase/mat/imat_" + \
        commit + "_" + str(ind) + ".h5"
    save_hdf5(savename, "imat", imat)
