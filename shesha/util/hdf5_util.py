## @package   shesha.util.hdf5_util
## @brief     Functions for handling the database system
## @author    COMPASS Team <https://github.com/ANR-COMPASS>
## @version   5.5.0
## @date      2022/01/24
## @copyright GNU Lesser General Public License
#
#  This file is part of COMPASS <https://anr-compass.github.io/compass/>
#
#  Copyright (C) 2011-2023 COMPASS Team <https://github.com/ANR-COMPASS>
#  All rights reserved.
#  Distributed under GNU - LGPL
#
#  COMPASS is free software: you can redistribute it and/or modify it under the terms of the GNU Lesser
#  General Public License as published by the Free Software Foundation, either version 3 of the License,
#  or any later version.
#
#  COMPASS: End-to-end AO simulation tool using GPU acceleration
#  The COMPASS platform was designed to meet the need of high-performance for the simulation of AO systems.
#
#  The final product includes a software package for simulating all the critical subcomponents of AO,
#  particularly in the context of the ELT and a real-time core based on several control approaches,
#  with performances consistent with its integration into an instrument. Taking advantage of the specific
#  hardware architecture of the GPU, the COMPASS tool allows to achieve adequate execution speeds to
#  conduct large simulation campaigns called to the ELT.
#
#  The COMPASS platform can be used to carry a wide variety of simulations to both testspecific components
#  of AO of the E-ELT (such as wavefront analysis device with a pyramid or elongated Laser star), and
#  various systems configurations such as multi-conjugate AO.
#
#  COMPASS is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the
#  implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
#  See the GNU Lesser General Public License for more details.
#
#  You should have received a copy of the GNU Lesser General Public License along with COMPASS.
#  If not, see <https://www.gnu.org/licenses/lgpl-3.0.txt>.

import h5py
import pandas
import os
import numpy as np
from subprocess import check_output

shesha_db = None
try:
    shesha_db = os.environ['SHESHA_DB_ROOT']
except KeyError:
    # if SHESHA_DB_ROOT is not defined, test if SHESHA_ROOT is defined
    if 'SHESHA_ROOT' in os.environ:
        shesha_db = os.environ['SHESHA_ROOT'] + "/data"
    else: # if SHESHA_ROOT is not defined, search for the data directory in the default package location
        if os.path.isdir(os.path.dirname(__file__) + "/../../data"):
            shesha_db = os.path.dirname(__file__) + "/../../data"

if not shesha_db:
    raise RuntimeError("neither SHESHA_DB_ROOT nor SHESHA_ROOT are defined, and the default data directory is not found. Please define SHESHA_DB_ROOT or SHESHA_ROOT to point to the data directory (see documentation).")

shesha_savepath = shesha_db


def updateParamDict(pdict, pClass, prefix):
    """
    Update parameters dictionnary pdict with all the parameters of pClass.
    Prefix must be set to define the key value of the new dict entries
    """
    if (isinstance(pClass, list)):
        params = pClass[0].__dict__.keys()
        for k in params:
            pdict.update({
                    k: [
                            p.__dict__[k].encode("utf8") if isinstance(
                                    p.__dict__[k], str) else
                            p.__dict__[k] for p in pClass
                    ]
            })

    else:
        params = pClass.__dict__.keys()

        for k in params:
            if isinstance(pClass.__dict__[k], str):
                pdict.update({k: pClass.__dict__[k].encode("utf8")})
            else:
                pdict.update({k: pClass.__dict__[k]})


def params_dictionary(config):
    """ Create and returns a dictionary of all the config parameters with the
    corresponding keys for further creation of database and save files

    :param config: (module) : simulation parameters

    :return param_dict: (dictionary) : dictionary of parameters
    """

    commit = check_output(["git", "rev-parse", "--short", "HEAD"]).strip()

    param_dict = {"simul_name": config.simul_name.encode('utf8'), "commit": commit}

    updateParamDict(param_dict, config.p_loop, "_ParamLoop__")
    updateParamDict(param_dict, config.p_geom, "_ParamGeom__")
    updateParamDict(param_dict, config.p_tel, "_ParamTel__")
    if config.p_atmos is not None:
        updateParamDict(param_dict, config.p_atmos, "_ParamAtmos__")
    if config.p_targets is not None:
        updateParamDict(param_dict, config.p_targets, "_ParamTarget__")
        param_dict.update({"ntargets": len(config.p_targets)})
    if config.p_wfss is not None:
        updateParamDict(param_dict, config.p_wfss, "_ParamWfs__")
        param_dict.update({"nwfs": len(config.p_wfss)})
    if config.p_dms is not None:
        updateParamDict(param_dict, config.p_dms, "_ParamDm__")
        param_dict.update({"ndms": len(config.p_dms)})
    if config.p_controllers is not None:
        updateParamDict(param_dict, config.p_controllers, "_ParamController__")
        param_dict.update({"ncontrollers": len(config.p_controllers)})
    if config.p_centroiders is not None:
        updateParamDict(param_dict, config.p_centroiders, "_ParamCentroider__")
        param_dict.update({"ncentroiders": len(config.p_centroiders)})

    for k in param_dict.keys():
        if isinstance(param_dict[k], list):
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
        if(isinstance(attr, np.ndarray)):
            save_hdf5(filename, i, attr)
        elif(isinstance(attr, list)):
            if(isinstance(attr[0], np.ndarray)):
                for k,data in enumerate(attr):
                    save_hdf5(filename, i + str(k), data)
            else:
                f.attrs.create(i, attr)
        else:
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

    if "dm" not in matricesToLoad:
        df = pandas.read_hdf(savepath + "matricesDataBase.h5", "dm")
        ind = len(df.index)
        filename = savepath + "mat/dm_" + commit + "_" + str(ind) + ".h5"
        create_file_attributes(filename, param_dict)
        updateDataBase(filename, savepath, "dm")

    if "imat" not in matricesToLoad:
        df = pandas.read_hdf(savepath + "matricesDataBase.h5", "imat")
        ind = len(df.index)
        filename = savepath + "mat/imat_" + commit + "_" + str(ind) + ".h5"
        create_file_attributes(filename, param_dict)
        updateDataBase(filename, savepath, "imat")


def initDataBase(savepath, param_dict):
    """ Initialize and create the database for all the saved matrices. This database
    will be placed on the top of the savepath and be named matricesDataBase.h5.

    Args:

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

    Args:

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

    Args:

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

    Args:

        config : (module) : simulation parameters

        matricesToLoad : (dictionary) :  matrices that will be load and their path
    """
    dataBase = pandas.read_hdf(savepath + "matricesDataBase.h5", "A")
    param2test = [
            "_ParamAtmos__r0", "_ParamAtmos__seeds", "_ParamAtmos__L0",
            "_ParamAtmos__alt", "_ParamTel__diam", "_ParamTel__cobs",
            "_ParamGeom__pupdiam", "_ParamGeom__zenithangle", "_ParamTarget__xpos",
            "_ParamTarget__ypos", "_ParamWfs__xpos", "_ParamWfs__ypos"
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

    Args:

        config : (module) : simulation parameters

        matricesToLoad : (dictionary) :  matrices that will be load and their path
    """
    dataBase = pandas.read_hdf(savepath + "matricesDataBase.h5", "imat")

    param2test = [
            "_ParamTel__diam", "_ParamTel__t_spiders", "_ParamTel__spiders_type",
            "_ParamTel__pupangle", "_ParamTel__referr", "_ParamTel__std_piston",
            "_ParamTel__std_tt", "_ParamTel__type_ap", "_ParamTel__nbrmissing",
            "_ParamTel__cobs", "_ParamGeom__pupdiam", "nwfs", "_ParamWfs__type",
            "_ParamWfs__nxsub", "_ParamWfs__npix", "_ParamWfs__pixsize",
            "_ParamWfs__fracsub", "_ParamWfs__xpos", "_ParamWfs__ypos",
            "_ParamWfs__Lambda", "_ParamWfs__dms_seen", "_ParamWfs__fssize",
            "_ParamWfs__fstop", "_ParamWfs__pyr_ampl", "_ParamWfs__pyr_loc",
            "_ParamWfs__pyr_npts", "_ParamWfs__pyr_pup_sep", "_ParamWfs__pyrtype",
            "ndms", "_ParamDm__type", "_ParamDm__alt", "_ParamDm__coupling",
            "_ParamDm__margin_in", "_ParamDm__margin_out", "_ParamDm__nact",
            "_ParamDm__nkl", "_ParamDm__type_kl", "_ParamDm__push4imat",
            "_ParamDm__thresh", "_ParamDm__unitpervolt", "ncentroiders",
            "_ParamCentroider__type", "_ParamCentroider__nmax",
            "_ParamCentroider__nwfs", "_ParamCentroider__sizex",
            "_ParamCentroider__sizey", "_ParamCentroider__thresh",
            "_ParamCentroider__type_fct", "_ParamCentroider__weights",
            "_ParamCentroider__width"
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

    Args:

        config : (module) : simulation parameters

        matricesToLoad : (dictionary) :  matrices that will be load and their path
    """
    dataBase = pandas.read_hdf(savepath + "matricesDataBase.h5", "dm")

    param2test = [
            "_ParamTel__diam", "_ParamTel__t_spiders", "_ParamTel__spiders_type",
            "_ParamTel__pupangle", "_ParamTel__referr", "_ParamTel__std_piston",
            "_ParamTel__std_tt", "_ParamTel__type_ap", "_ParamTel__nbrmissing",
            "_ParamTel__cobs", "_ParamGeom__pupdiam", "nwfs", "_ParamWfs__type",
            "_ParamWfs__nxsub", "_ParamWfs__npix", "_ParamWfs__pixsize",
            "_ParamWfs__fracsub", "_ParamWfs__xpos", "_ParamWfs__ypos",
            "_ParamWfs__Lambda", "_ParamWfs__dms_seen", "_ParamWfs__fssize",
            "_ParamWfs__fstop", "_ParamWfs__pyr_ampl", "_ParamWfs__pyr_loc",
            "_ParamWfs__pyr_npts", "_ParamWfs__pyrtype", "_ParamWfs__pyr_pup_sep",
            "ndms", "_ParamDm__type", "_ParamDm__alt", "_ParamDm__coupling",
            "_ParamDm__margin_in", "_ParamDm__margin_out", "_ParamDm__nkl",
            "_ParamDm__nact", "_ParamDm__type_kl", "_ParamDm__push4imat",
            "_ParamDm__thresh", "_ParamDm__unitpervolt"
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
    if "A" not in matricesToLoad:
        validInStore(store, savepath, "A")
    if "dm" not in matricesToLoad:
        validInStore(store, savepath, "dm")
    if "imat" not in matricesToLoad:
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
    #import shesha.config as conf

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
        config.p_wfss.append(config.ParamWfs())
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
            config.p_dms.append(config.ParamDm())
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
            config.p_centroiders.append(config.ParamCentroider())
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
            config.p_controllers.append(config.ParamController())
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

    Args:

        filename: (str) : name of the file to read from

        datasetName: (str) : name of the dataset to read (default="dataset")
    """

    f = h5py.File(filename, "r")
    data = f[datasetName][:]
    f.close()
    return data


def load_AB_from_dataBase(database, ind):
    """ Read and return A, B, istx and isty from the database

    Args:

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

    Args:

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

    Args:

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

    Args:

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

    Args:

        database: (dict): dictionary containing paths to matrices to load
    """
    print("loading", database["imat"])
    f = h5py.File(database["imat"], 'r')
    imat = f["imat"][:]
    f.close()

    return imat


def save_imat_in_dataBase(imat):
    """ Save the DM geometry in the database

    Args:

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
