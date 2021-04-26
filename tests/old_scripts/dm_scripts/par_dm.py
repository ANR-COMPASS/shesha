# -*- coding: utf-8 -*-
"""
Created on Wed Nov  9 14:03:29 2016

@author: sdurand
"""

#import min :
import shesha as ao
import numpy as np

simul_name = "dm_init"

nact = 17  # number of actuator
pupdiam = 500  # size of DM

# note available for norms != 0 :
alt = 0.


def calcul_size_support_dmpzt(nact, pupdiam):
    """
    This fonction is available just for alt =0 or/and norms = 0
    """

    ssize = int(2**np.ceil(np.log2(pupdiam) + 1))
    cent = ssize / 2 + 0.5
    pitch = int(pupdiam / (nact - 1))  #--> for wfs_xpos and ypos = 0
    extent = pitch * (nact + 5)
    n1 = np.floor(cent - extent / 2)
    n2 = np.ceil(cent + extent / 2)
    taille = n2 - n1 + 1

    return taille


#geometry param :
p_geom = ao.Param_geom()

p_geom.set_pupdiam(pupdiam)  # size of dm in support (pixel)
p_geom.set_apod(0)  #booleen 1 = actif 0 = inactif

#telescope param :

p_tel = ao.Param_tel()

# These values are mandatory
# for alt = 0 or/and norm = 0 this value is not use for compute dm support size

# diam is not use for pzt dm if alt=0 or/and norms=0
# diam is not use for kl dm
p_tel.set_diam(8.0)  #--> use for tiptilt and dm_h5 (and dm support size)

# Cobs is not use for tiptilt dm
# cobs is not use for pzt if have no filter
p_tel.set_cobs(0.12)  #--> use for kl_miror and PZT filter

#dm param:

p_dm0 = ao.Param_dm()
p_dms = [p_dm0]
p_dm0.set_type("pzt")
p_dm0.set_nact(nact)
p_dm0.set_alt(alt)
p_dm0.set_thresh(0.3)
p_dm0.set_coupling(0.2)
p_dm0.set_unitpervolt(1.)
p_dm0.set_push4imat(1.)
p_dm0.set_margin_out(0)
