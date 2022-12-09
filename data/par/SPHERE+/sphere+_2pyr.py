#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: FV
"""

import shesha.config as conf
import numpy as np
simul_name = "sphere+2stages"
layout = "layoutDeCharlesSPHEREplus"
# loop
p_loop = conf.Param_loop()

p_loop.set_niter(5000)           # number of loop iterations
p_loop.set_ittime(1./2000.)      # = 1/3000 - assuming loop at 3 kHz
p_loop.set_devices([0, 1, 2, 3]) # GPU numbers
# geom
p_geom = conf.Param_geom()
# p_geom.set_pupdiam(1024)
p_geom.set_zenithangle(0.)

# tel
p_tel = conf.Param_tel()

p_tel.set_diam(8.0)         # VLT diameter
p_tel.set_type_ap("VLT")    # VLT pupil
p_tel.set_cobs(0.14)        # central obstruction
p_tel.set_t_spiders(0.00625)

# atmos
p_atmos = conf.Param_atmos()

p_atmos.set_r0(0.14)         # Fried parameters @ 500 nm
p_atmos.set_nscreens(1)      # Number of layers
p_atmos.set_frac([1.0])      # Fraction of atmosphere (100%=1)
p_atmos.set_alt([0.0])       # Altitude(s) in meters
p_atmos.set_windspeed([15])   # wind speed of layer (s) in m/s
p_atmos.set_winddir([45])    # wind direction in degrees
p_atmos.set_L0([25])         # in meters

# target
p_target = conf.Param_target()
p_targets = [p_target]
p_target.set_xpos(0.)
p_target.set_ypos(0.)
p_target.set_Lambda(1.65) # H Band
p_target.set_mag(6.)

# wfs : SAXO SH but a pyramid
p_wfs0 = conf.Param_wfs()

p_wfs0.set_type("pyrhr")
p_wfs0.set_nxsub(50)                # TBC Number of pixels along the pupil diameter, NB. need more subaperture than nactu.
p_wfs0.set_fssize(4)                # Size of the field stop
p_wfs0.set_fracsub(0.0001)          # was 0.8 before Vincent
p_wfs0.set_xpos(0.)
p_wfs0.set_ypos(0.)
p_wfs0.set_Lambda(0.7)
p_wfs0.set_gsmag(6.)
p_wfs0.set_optthroughput(0.25)      # still unknown
p_wfs0.set_zerop(1e11)
p_wfs0.set_noise(0.2)
p_wfs0.set_fstop("round")
rMod = 20.                           # Modulation radius, in lam/D units
nbPtMod = int(np.ceil(int(rMod * 2 * 3.141592653589793) / 4.) * 4)
p_wfs0.set_pyr_npts(nbPtMod)         # Number of modulation point along the circle
p_wfs0.set_pyr_ampl(rMod)            # Pyramid modulation amplitude (pyramid only)
p_wfs0.set_pyr_pup_sep(p_wfs0.nxsub) # separation between the 4 images of the pyramid
p_wfs0.set_atmos_seen(1)             # If False, the WFS don’t see the atmosphere layers
p_wfs0.set_dms_seen(np.array([0,2])) # If False, the WFS don’t see the atmosphere layers


# wfs : near infrared pyramid
p_wfs1 = conf.Param_wfs()

p_wfs1.set_type("pyrhr")
p_wfs1.set_nxsub(50)                 # TBC Number of pixels along the pupil diameter, NB. need more subaperture than nactu.
p_wfs1.set_fssize(2.5)               # Size of the field stop
p_wfs1.set_fracsub(0.0001)           # was 0.8 before Vincent 
p_wfs1.set_xpos(0.)
p_wfs1.set_ypos(0.)
p_wfs1.set_Lambda(1.2)               # pyramid wavelength 
p_wfs1.set_gsmag(6.)                 # Guide star magnitude
p_wfs1.set_optthroughput(0.5)        # Optical throughput coefficient
p_wfs1.set_zerop(1.e11)
p_wfs1.set_noise(0.2)
p_wfs1.set_fstop("round")
rMod = 3.                            # Modulation radius, in lam/D units
nbPtMod = int(np.ceil(int(rMod * 2 * 3.141592653589793) / 4.) * 4)
p_wfs1.set_pyr_npts(nbPtMod)         # Number of modulation point along the circle
p_wfs1.set_pyr_ampl(rMod)            # Pyramid modulation amplitude (pyramid only)
p_wfs1.set_pyr_pup_sep(50)           # separation between the 4 images of the pyramid
p_wfs1.set_atmos_seen(1)             # If False, the WFS don’t see the atmosphere layers


p_wfss = [p_wfs0, p_wfs1]


# dm (waiting for the custom HODM)
p_dm0 = conf.Param_dm()
p_dm1 = conf.Param_dm()
p_dm2 = conf.Param_dm()

p_dm0.set_type("pzt")
nact = 41
p_dm0.set_nact(nact)
p_dm0.set_alt(0.)           # Layers altitudes
p_dm0.set_thresh(0.3)       # Threshold on response for selection of valid actuators. Expressed in fraction of the maximal response
p_dm0.set_coupling(0.3)
p_dm0.set_unitpervolt(1.0)
p_dm0.set_push4imat(1.0e-3) # Nominal voltage for imat = integration matrix = response matrix
p_dm0.set_margin_out(0.3)   # pour adapter la taille de la pupille du DM a celle du WFS

p_dm1.set_type("pzt")
nact = 24
p_dm1.set_nact(nact)
p_dm1.set_alt(0.)           # Layers altitudes
p_dm1.set_thresh(0.3)       # Threshold on response for selection of valid actuators. Expressed in fraction of the maximal response
p_dm1.set_coupling(0.3)
p_dm1.set_unitpervolt(1.0)
p_dm1.set_push4imat(1.0e-3) # Nominal voltage for imat = integration matrix = response matrix
p_dm1.set_margin_out(0.3)   # pour adapter la taille de la pupille du DM a celle du WFS

p_dm2.set_type("tt")
p_dm2.set_alt(0.)
p_dm2.set_unitpervolt(1.)   # Influence function sensitivity
p_dm2.set_push4imat(0.005)

p_dms = [p_dm0, p_dm1, p_dm2]

# centroiders
p_centroider0 = conf.Param_centroider()
p_centroider1 = conf.Param_centroider()
p_centroiders = [p_centroider0, p_centroider1]

p_centroider0.set_nwfs(0)
p_centroider0.set_type("maskedpix")
p_centroider1.set_nwfs(1)
p_centroider1.set_type("maskedpix")
# p_centroider0.set_type("corr")
# p_centroider0.set_type_fct("model")

# controllers
p_controller0 = conf.Param_controller()
p_controllers = [p_controller0]

p_controller0.set_type("generic") # ls (classic easy simple) or generic
p_controller0.set_nwfs([0, 1])
p_controller0.set_ndm([0, 1, 2])
p_controller0.set_maxcond(5.) # what determines the number of modes to be filtered
p_controller0.set_delay(1.5)
p_controller0.set_gain(0.3)
#p_controller0.set_nstates(6)
