#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Friday 7th of February 2020

@author: nour
"""

import shesha.config as conf


# loop
p_loop = conf.Param_loop()

p_loop.set_niter(5000)
p_loop.set_ittime(0.0005)  # =1/2000 - assuming loop at 2kHz
p_loop.set_devices([0, 1, 2, 3])
# geom
p_geom = conf.Param_geom()

p_geom.set_zenithangle(0.)

# tel
p_tel = conf.Param_tel()

p_tel.set_diam(8.2) # VLT diameter
p_tel.set_cobs(0.12)# TBC (central obstruction)

# atmos
# here we simulate the first stage of correction of ao188
p_atmos = conf.Param_atmos()

p_atmos.set_r0(0.16) # Fried parameters @ 500 nm
p_atmos.set_nscreens(1) # Number of layers
p_atmos.set_frac([1.0])
p_atmos.set_alt([0.0])
p_atmos.set_windspeed([20.0])
p_atmos.set_winddir([45])
p_atmos.set_L0([15]) # in meters. here we simulate ao188's precorrection. Layers outer scale

# target
p_target = conf.Param_target()
p_targets = [p_target]
p_target.set_xpos(0.)
p_target.set_ypos(0.)
p_target.set_Lambda(1.65)
p_target.set_mag(10.)

# wfs
p_wfs0 = conf.Param_wfs(roket=True)
p_wfss = [p_wfs0]

p_wfs0.set_type("pyrhr")
p_wfs0.set_nxsub(80) # TBC Number of pixels along the pupil diameter, NB. need more subaperture than nactu.
p_wfs0.set_fssize(1.5) # Size of the field stop
p_wfs0.set_fracsub(0.0001) # was 0.8 before Vincent 
p_wfs0.set_xpos(0.)
p_wfs0.set_ypos(0.)
p_wfs0.set_Lambda(0.5) # pyramid wavelength 
p_wfs0.set_gsmag(5.) # Guide star magnitude
p_wfs0.set_optthroughput(0.5) # Optiical throughput coefficient
p_wfs0.set_zerop(1.e11)
p_wfs0.set_noise(-1)
p_wfs0.set_fstop("round")
p_wfs0.set_pyr_npts(16) # Number of modulation point along the circle
p_wfs0.set_pyr_ampl(3) # Pyramid modulation amplitude (pyramid only)
p_wfs0.set_pyr_pup_sep(p_wfs0.nxsub) # separation between the 4 images of the pyramid 
p_wfs0.set_atmos_seen(1) # If False, the WFS don’t see the atmosphere layers



# dm
p_dm0 = conf.Param_dm()
p_dm1 = conf.Param_dm()
p_dms = [p_dm0, p_dm1]
p_dm0.set_type("pzt")
# nact = p_wfs0.nxsub + 1
nact = 60
p_dm0.set_nact(nact)
p_dm0.set_alt(0.) # Layers altitudes
p_dm0.set_thresh(0.3) # Threshold on response for selection of valid actuators. Expressed in fraction of the maximal response
p_dm0.set_coupling(0.2)
p_dm0.set_unitpervolt(0.01)
p_dm0.set_push4imat(100.) # Nominal voltage for imat = integration matrix = response matrix
p_dm0.set_margin_out(0.3) # pour adapter la taille de la pupille du DM a celle du WFS

p_dm1.set_type("tt")
p_dm1.set_alt(0.)
p_dm1.set_unitpervolt(0.0005) # Influence function sensitivity
p_dm1.set_push4imat(100)

# centroiders
p_centroider0 = conf.Param_centroider()
p_centroiders = [p_centroider0]

p_centroider0.set_nwfs(0)
p_centroider0.set_type("pyr")
# p_centroider0.set_type("corr")
# p_centroider0.set_type_fct("model")

# controllers
p_controller0 = conf.Param_controller()
p_controllers = [p_controller0]

p_controller0.set_type("generic") # ls (classic easy simple) or generic
p_controller0.set_nwfs([0])
p_controller0.set_ndm([0, 1])
p_controller0.set_maxcond(5.) # what determines the number of modes to be filtered
p_controller0.set_delay(1)
p_controller0.set_gain(0.4)
