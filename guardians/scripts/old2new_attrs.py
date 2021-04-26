import h5py
from glob import glob

old2new_dict = {
        # Loop params
        "niter": "_Param_loop__niter",
        "ittime": "_Param_loop__ittime",
        # Geom params
        "zenithangle": "_Param_geom__zenithangle",
        "pupdiam": "_Param_geom__pupdiam",
        # Telescope params
        "tel_diam": "_Param_tel__diam",
        "cobs": "_Param_tel__cobs",
        "t_spiders": "_Param_tel__t_spiders",
        "spiders_type": "_Param_tel__spiders_type",
        "type_ap": "_Param_tel__type_ap",
        "referr": "_Param_tel__referr",
        "pupangle": "_Param_tel__pupangle",
        "nbrmissing": "_Param_tel__nbrmissing",
        "std_piston": "_Param_tel__std_piston",
        "std_tt": "_Param_tel__std_tt",
        # Atmos params
        "r0": "_Param_atmos__r0",
        "nscreens": "_Param_atmos__nscreens",
        "frac": "_Param_atmos__frac",
        "atm.alt": "_Param_atmos__alt",
        "windspeed": "_Param_atmos__windspeed",
        "winddir": "_Param_atmos__winddir",
        "L0": "_Param_atmos__L0",
        "seeds": "_Param_atmos__seeds",
        # Target params
        "ntargets": "_Param_target__ntargets",
        "target.xpos": "_Param_target__xpos",
        "target.ypos": "_Param_target__ypos",
        "target.Lambda": "_Param_target__Lambda",
        "target.mag": "_Param_target__mag",
        "target.dms_seen": "_Param_target__dms_seen",
        #WFS params
        "type": "_Param_wfs__type",
        "nxsub": "_Param_wfs__nxsub",
        "npix": "_Param_wfs__npix",
        "pixsize": "_Param_wfs__pixsize",
        "fracsub": "_Param_wfs__fracsub",
        "wfs.xpos": "_Param_wfs__xpos",
        "wfs.ypos": "_Param_wfs__ypos",
        "wfs.Lambda": "_Param_wfs__Lambda",
        "gsmag": "_Param_wfs__gsmag",
        "optthroughput": "_Param_wfs__optthroughput",
        "zerop": "_Param_wfs__zerop",
        "noise": "_Param_wfs__noise",
        "atmos_seen": "_Param_wfs__atmos_seen",
        "dms_seen": "_Param_wfs__dms_seen",
        "beamsize": "_Param_wfs__beamsize",
        "fssize": "_Param_wfs__fssize",
        "fstop": "_Param_wfs__fstop",
        "gsalt": "_Param_wfs__gsalt",
        "laserpower": "_Param_wfs__laserpower",
        "lgsreturnperwatt": "_Param_wfs__lgsreturnperwatt",
        "lltx": "_Param_wfs__lltx",
        "llty": "_Param_wfs__llty",
        "open_loop": "_Param_wfs__open_loop",
        "proftype": "_Param_wfs__proftype",
        "pyr_ampl": "_Param_wfs__pyr_ampl",
        "pyr_loc": "_Param_wfs__pyr_loc",
        "pyr_npts": "_Param_wfs__pyr_npts",
        "pyr_pup_sep": "_Param_wfs__pyr_pup_sep",
        "pyrtype": "_Param_wfs__pyrtype",
        #DM params
        "type": "_Param_dm__type_dm",
        "dm.alt": "_Param_dm__alt",
        "coupling": "_Param_dm__coupling",
        "nkl": "_Param_dm__nkl",
        "kl_type": "_Param_dm__type_kl",
        "pupoffset": "_Param_dm__pupoffset",
        "nact": "_Param_dm__nact",
        "push4imat": "_Param_dm__push4imat",
        "dm.thresh": "_Param_dm__thresh",
        "unitpervolt": "_Param_dm__unitpervolt",
        #Centroider params
        "type": "_Param_centroider__type",
        "nmax": "_Param_centroider__nmax",
        "centro.nwfs": "_Param_centroider__nwfs",
        "sizex": "_Param_centroider__sizex",
        "sizey": "_Param_centroider__sizey",
        "centroider.thresh": "_Param_centroider__thresh",
        "type_fct": "_Param_centroider__type_fct",
        "weights": "_Param_centroider__weights",
        "width": "_Param_centroider__width",
        # Control params
        "type": "_Param_controller__type",
        "TTcond": "_Param_controller__TTcond",
        "cured_ndivs": "_Param_controller__cured_ndivs",
        "delay": "_Param_controller__delay",
        "gain": "_Param_controller__gain",
        "maxcond": "_Param_controller__maxcond",
        "modopti": "_Param_controller__modopti",
        "ndm": "_Param_controller__ndm",
        "nmodes": "_Param_controller__nmodes",
        "nrec": "_Param_controller__nrec",
        "gmin": "_Param_controller__gmin",
        "gmax": "_Param_controller__gmax",
        "ngain": "_Param_controller__ngain",
        "control.nwfs": "_Param_controller__nwfs",
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
    if not "_Param_atmos__r0" in f.attrs.keys():
        for k in f.attrs.keys():
            try:
                f.attrs[old2new_dict[k]] = f.attrs[k]
                del f.attrs[k]
            except:
                print(ff)
                print(k)
    f.close()
