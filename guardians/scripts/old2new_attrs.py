import h5py
from glob import glob

old2new_dict = {
        # Loop params
        "niter": "_ParamLoop__niter",
        "ittime": "_ParamLoop__ittime",
        # Geom params
        "zenithangle": "_ParamGeom__zenithangle",
        "pupdiam": "_ParamGeom__pupdiam",
        # Telescope params
        "tel_diam": "_ParamTel__diam",
        "cobs": "_ParamTel__cobs",
        "t_spiders": "_ParamTel__t_spiders",
        "spiders_type": "_ParamTel__spiders_type",
        "type_ap": "_ParamTel__type_ap",
        "referr": "_ParamTel__referr",
        "pupangle": "_ParamTel__pupangle",
        "nbrmissing": "_ParamTel__nbrmissing",
        "std_piston": "_ParamTel__std_piston",
        "std_tt": "_ParamTel__std_tt",
        # Atmos params
        "r0": "_ParamAtmos__r0",
        "nscreens": "_ParamAtmos__nscreens",
        "frac": "_ParamAtmos__frac",
        "atm.alt": "_ParamAtmos__alt",
        "windspeed": "_ParamAtmos__windspeed",
        "winddir": "_ParamAtmos__winddir",
        "L0": "_ParamAtmos__L0",
        "seeds": "_ParamAtmos__seeds",
        # Target params
        "ntargets": "_ParamTarget__ntargets",
        "target.xpos": "_ParamTarget__xpos",
        "target.ypos": "_ParamTarget__ypos",
        "target.Lambda": "_ParamTarget__Lambda",
        "target.mag": "_ParamTarget__mag",
        "target.dms_seen": "_ParamTarget__dms_seen",
        #WFS params
        "type": "_ParamWfs__type",
        "nxsub": "_ParamWfs__nxsub",
        "npix": "_ParamWfs__npix",
        "pixsize": "_ParamWfs__pixsize",
        "fracsub": "_ParamWfs__fracsub",
        "wfs.xpos": "_ParamWfs__xpos",
        "wfs.ypos": "_ParamWfs__ypos",
        "wfs.Lambda": "_ParamWfs__Lambda",
        "gsmag": "_ParamWfs__gsmag",
        "optthroughput": "_ParamWfs__optthroughput",
        "zerop": "_ParamWfs__zerop",
        "noise": "_ParamWfs__noise",
        "atmos_seen": "_ParamWfs__atmos_seen",
        "dms_seen": "_ParamWfs__dms_seen",
        "beamsize": "_ParamWfs__beamsize",
        "fssize": "_ParamWfs__fssize",
        "fstop": "_ParamWfs__fstop",
        "gsalt": "_ParamWfs__gsalt",
        "laserpower": "_ParamWfs__laserpower",
        "lgsreturnperwatt": "_ParamWfs__lgsreturnperwatt",
        "lltx": "_ParamWfs__lltx",
        "llty": "_ParamWfs__llty",
        "open_loop": "_ParamWfs__open_loop",
        "proftype": "_ParamWfs__proftype",
        "pyr_ampl": "_ParamWfs__pyr_ampl",
        "pyr_loc": "_ParamWfs__pyr_loc",
        "pyr_npts": "_ParamWfs__pyr_npts",
        "pyr_pup_sep": "_ParamWfs__pyr_pup_sep",
        "pyrtype": "_ParamWfs__pyrtype",
        #DM params
        # "type": "_ParamDm__type_dm",
        "dm.alt": "_ParamDm__alt",
        "coupling": "_ParamDm__coupling",
        "nkl": "_ParamDm__nkl",
        "kl_type": "_ParamDm__type_kl",
        "pupoffset": "_ParamDm__pupoffset",
        "nact": "_ParamDm__nact",
        "push4imat": "_ParamDm__push4imat",
        "dm.thresh": "_ParamDm__thresh",
        "unitpervolt": "_ParamDm__unitpervolt",
        #Centroider params
        # "type": "_ParamCentroider__type",
        "nmax": "_ParamCentroider__nmax",
        "centro.nwfs": "_ParamCentroider__nwfs",
        "sizex": "_ParamCentroider__sizex",
        "sizey": "_ParamCentroider__sizey",
        "centroider.thresh": "_ParamCentroider__thresh",
        "type_fct": "_ParamCentroider__type_fct",
        "weights": "_ParamCentroider__weights",
        "width": "_ParamCentroider__width",
        # Control params
        # "type": "_ParamController__type",
        "TTcond": "_ParamController__TTcond",
        "cured_ndivs": "_ParamController__cured_ndivs",
        "delay": "_ParamController__delay",
        "gain": "_ParamController__gain",
        "maxcond": "_ParamController__maxcond",
        "modopti": "_ParamController__modopti",
        "ndm": "_ParamController__ndm",
        "nmodes": "_ParamController__nmodes",
        "nrec": "_ParamController__nrec",
        "gmin": "_ParamController__gmin",
        "gmax": "_ParamController__gmax",
        "ngain": "_ParamController__ngain",
        "control.nwfs": "_ParamController__nwfs",
        "ndms": "ndms",
        "nwfs": "nwfs",
        "ncontrollers": "ncontrollers",
        "simulname": "simulname",
        "revision": "revision",
        "ncentroiders": "ncentroiders",
        "hyst": "hyst",
        "margin": "margin",
        "validity": "validity"
}

files = glob("/home/fferreira/Data/correlation/*.h5")

for ff in files:
    f = h5py.File(ff, 'r+')
    if "_ParamAtmos__r0" not in f.attrs.keys():
        for k in f.attrs.keys():
            try:
                f.attrs[old2new_dict[k]] = f.attrs[k]
                del f.attrs[k]
            except BaseException:
                print(ff)
                print(k)
    f.close()
