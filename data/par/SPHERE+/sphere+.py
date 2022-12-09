#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

@author: fv
"""

# /!\ : This symbol marks the parameters that must not be changed.
# /?\ : This symbol marks the parameters that must be chosen among a list.

import shesha.config as conf
import numpy as np

#simul_name = "sphere+"
layout = "layoutDeSpherePlus"

# loop
p_loop = conf.Param_loop()
p_loop.set_niter(5000)          #     number of loops
                                # /?\ second stage frequency
p_loop.set_ittime(1./3000.)     # second loop at 3 kHz
# p_loop.set_ittime(1./2000.)   # second loop at 2 kHz
# p_loop.set_ittime(1./1000.)   # second loop at 1 kHz
p_loop.set_devices([0, 1, 2, 3])

# geom
p_geom = conf.Param_geom()
p_geom.set_zenithangle(0.)

# tel
p_tel = conf.Param_tel()
p_tel.set_diam(8.0)            # /!\  VLT diameter
p_tel.set_cobs(0.14)           # /!\  central obstruction
p_tel.set_type_ap("VLT")       # /!\  VLT pupil
p_tel.set_t_spiders(0.00625)   # /!\  spider width = 5 cm

# atmos
p_atmos = conf.Param_atmos()
p_atmos.set_r0(0.14)          #     Fried parameters @ 500 nm
p_atmos.set_nscreens(1)       # /!\ Number of layers
p_atmos.set_frac([1.0])       # /!\ Fraction of atmosphere (100% = 1)
p_atmos.set_alt([0.0])        # /!\ Altitude(s) in meters
p_atmos.set_windspeed([8])    #     Wind speed of layer(s) in m/s
p_atmos.set_winddir([45])     # /!\ Wind direction in degrees
p_atmos.set_L0([25])          #     Outer scale in meters

# target
p_target = conf.Param_target()
p_targets = [p_target]

p_target.set_xpos(0.)         # /!\ On axis
p_target.set_ypos(0.)         # /!\ On axis
p_target.set_Lambda(1.65)     # /!\ H Band
p_target.set_mag(6.)          # /!\

# wfs
p_wfs0 = conf.Param_wfs(roket=True)
p_wfss = [p_wfs0]

p_wfs0.set_type("pyrhr")        # /!\ pyramid
p_wfs0.set_nxsub(50)            #     number of pixels
p_wfs0.set_fracsub(0.0001)      #     threshold on illumination fraction for valid pixel
p_wfs0.set_Lambda(1.2)          #     wavelength
p_wfs0.set_gsmag(6.)
# p_wfs0.set_gsmag(6. + 2.5 * np.log10(3 / 2))  # at 2 kHz
# p_wfs0.set_gsmag(6. + 2.5 * np.log10(3 / 1))  # at 1 kHz
p_wfs0.set_zerop(1.e11)
p_wfs0.set_optthroughput(0.5)
p_wfs0.set_noise(0.1)           #     readout noise
p_wfs0.set_xpos(0.)             # /!\ On axis
p_wfs0.set_ypos(0.)             # /!\ On axis
rMod = 3.                       # Modulation radius, in lam/D units
p_wfs0.set_pyr_ampl(rMod)
nbPtMod = int(np.ceil(int(rMod * 2 * 3.141592653589793) / 4.) * 4)
p_wfs0.set_pyr_npts(nbPtMod) 
p_wfs0.set_pyr_pup_sep(p_wfs0.nxsub) # separation between the 4 images of the pyramid 
p_wfs0.set_fstop("round")
p_wfs0.set_fssize(1.5)          # Size of the field stop
p_wfs0.set_atmos_seen(1)        # /!\

# dm
p_dm0 = conf.Param_dm()
p_dms = [p_dm0]

p_dm0.set_type("pzt")           # /!\
p_dm0.set_thresh(-200)          # /!\ to get all Boston actuators
p_dm0.set_alt(0.)               # /!\
p_dm0.set_unitpervolt(1.)       # /!\
p_dm0.set_push4imat(1.0e-3)
# select Boston 24x24 or 32x32
p_dm0.set_file_influ_fits('Boston24x24.fits')   # /?\ Boston 24*24 or Boston 32*32
# p_dm0.set_file_influ_fits('Boston32x32.fits')

# centroiders
p_centroider0 = conf.Param_centroider()
p_centroiders = [p_centroider0]

p_centroider0.set_nwfs(0)           # /!\
p_centroider0.set_type("maskedpix")

# controllers
p_controller0 = conf.Param_controller()
p_controllers = [p_controller0]

p_controller0.set_type("generic")   # /?\ generic => must do manual imat.
# p_controller0.set_type("ls")
p_controller0.set_nwfs([0])         # /!\
p_controller0.set_ndm([0])       # /!\
p_controller0.set_maxcond(5.)       #     determines the number of modes to be filtered
p_controller0.set_delay(2)          # 
# for instance at 2 or 1 kHz :
# p_controller0.set_delay(2 * 2/3)  # at 2 kHz
# p_controller0.set_delay(2 * 1/3)  # at 1 kHz
p_controller0.set_gain(0.3)
p_controller0.set_calpix_name("compass2_calPix")
p_controller0.set_loopdata_name("compass2_loopData")
