#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Friday 7th of February 2020

@author: nour
"""

import shesha.config as conf
import numpy as np

simul_name = "sphere+"
layout = "layoutDeFab_PYR"

# loop
p_loop = conf.Param_loop()
p_loop.set_niter(5000)
p_loop.set_ittime(1./3000.)
# p_loop.set_ittime(1./2000.)
# p_loop.set_ittime(1./1000.)
p_loop.set_devices([0, 1, 2, 3])
# geom
p_geom = conf.Param_geom()

p_geom.set_zenithangle(0.)

# tel
p_tel = conf.Param_tel()
p_tel.set_diam(8.0)
p_tel.set_cobs(0.14)
p_tel.set_type_ap("VLT")
p_tel.set_t_spiders(0.00625)

# atmos
p_atmos = conf.Param_atmos()
p_atmos.set_r0(0.14)
p_atmos.set_nscreens(1)
p_atmos.set_frac([1.0])
p_atmos.set_alt([0.0])
p_atmos.set_windspeed([8.0])
p_atmos.set_winddir([45])
p_atmos.set_L0([25])

# target
p_target = conf.Param_target()
p_targets = [p_target]
p_target.set_xpos(0.)
p_target.set_ypos(0.)
p_target.set_Lambda(1.65)
p_target.set_mag(6.)

# wfs
p_wfs0 = conf.Param_wfs(roket=True)
p_wfss = [p_wfs0]

p_wfs0.set_type("pyrhr")
p_wfs0.set_nxsub(50)
p_wfs0.set_fracsub(0.0001)
p_wfs0.set_Lambda(1.2)
p_wfs0.set_gsmag(6.)
# p_wfs0.set_gsmag(6. + 2.5 * np.log10(3 / 2))
# p_wfs0.set_gsmag(6. + 2.5 * np.log10(3 / 1))
p_wfs0.set_zerop(1.e11)
p_wfs0.set_optthroughput(0.5)
p_wfs0.set_noise(0.1)
p_wfs0.set_xpos(0.)
p_wfs0.set_ypos(0.)
rMod = 3.                                        # Modulation radius, in lam/D units
p_wfs0.set_pyr_ampl(rMod)
nbPtMod = int(np.ceil(int(rMod * 2 * 3.141592653589793) / 4.) * 4)
p_wfs0.set_pyr_npts(nbPtMod) 
p_wfs0.set_pyr_pup_sep(p_wfs0.nxsub) # separation between the 4 images of the pyramid 
p_wfs0.set_fstop("round")
p_wfs0.set_fssize(1.5) # Size of the field stop
p_wfs0.set_atmos_seen(1) # If False, the WFS don’t see the atmosphere layers

# dm
p_dm0 = conf.Param_dm()
p_dm1 = conf.Param_dm()
p_dms = [p_dm0, p_dm1]

p_dm0.set_type("pzt")         # /!\
p_dm0.set_thresh(-0.5)        # /!\ to get the SAXO 1377 active actuators
p_dm0.set_alt(0.)             # /!\
p_dm0.set_unitpervolt(1.)     # /!\
p_dm0.set_push4imat(0.180)    #     to displace ~ half a pixel
p_dm0.set_file_influ_fits('SAXO_HODM.fits')

p_dm1.set_type("tt")
p_dm1.set_alt(0.)
p_dm1.set_unitpervolt(1.)
p_dm1.set_push4imat(1.0e-3)

# centroiders
p_centroider0 = conf.Param_centroider()
p_centroiders = [p_centroider0]

p_centroider0.set_nwfs(0)
p_centroider0.set_type("maskedpix")

# controllers
p_controller0 = conf.Param_controller()
p_controllers = [p_controller0]

p_controller0.set_type("generic") # ls (classic easy simple) or generic
p_controller0.set_nwfs([0])
p_controller0.set_ndm([0, 1])
p_controller0.set_maxcond(5.)
p_controller0.set_delay(2)  # 2 frames at 3 kHz
# p_controller0.set_delay(2 * 2/3)  # at 2 kHz
# p_controller0.set_delay(2 * 1/3)  # at 1 kHz
p_controller0.set_gain(0.4)
p_controller0.set_calpix_name("compass2_calPix")
p_controller0.set_loopdata_name("compass2_loopData")
