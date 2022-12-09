"""
This script is intended to be used with the saxo+ Manager ONLY!

The saxo+ manager can be instanciated using the follwing commands. 

> cd $HOME/compass/shesha/shesha/widgets/
> ipython -i widget_saxoplus.py ../../data/par/SPHERE+/sphere.py ../../data/par/SPHERE+/sphere+.py

The first argument (here ../../data/par/SPHERE+/sphere.py) point towards the saxo parameter configuration file
The second argument (here ../../data/par/SPHERE+/sphere+.py) point towards the saxo+ parameter configuration file


Once started the GUI interface should show 3 windows. 
1) is the main control window of the manager (which controls the 2 instance of COMPASS)
2) 1 GUI with a key displays of the first stage (WFS SH image, turbulence phase, DM shape, PSF at the saxo focal plane ect..) 
3) 1 GUI with a key displays of the first stage (WFS PYR image, residual phase from first stage, DM shape PSF at the saxo+ focal plane ect..) 

COMPASS methods can be accessed using either:

<wao.manager>  object is the main SAXO+ manager (which handles the synchronisation between the 2 AO stages)
<wao.wao1.supervisor> is the First stage compass supervisor
<wao.wao2.supervisor> is the Second stage compass supervisor


Author: F.V
"""

import numpy as np
import matplotlib.pyplot as plt
import astropy.io.fits as pfits
import os
from shesha.util.slopesCovariance import KLmodes

saxoplusmanager = wao.manager
saxo = wao.wao1.supervisor
saxoplus = wao.wao2.supervisor


"""
script for computing interaction matrix
and command matrix in saxo+ simulation
"""

wao.wao1.aoLoopClicked(False) # Make sure the loop controlled by the main GUI is stopped
wao.wao1.enable_atmos(False) # disabling turbulence
#Measuring ref slopes saxo
saxo.wfs.set_noise(0, -1) #removing noise on SH WFS (-1 = no noise)
for i in range(6): saxoplusmanager.next(seeAtmos=False); # check
saxo.rtc.reset_ref_slopes(0) # reset refslopes from saxo to 0 (check)
saxo.rtc.do_ref_slopes(0) # meaure and load reflopes from null phase
for i in range(6): saxoplusmanager.next(seeAtmos=False); # check



#Measuring ref slopes saxoplus
saxoplus.wfs.set_noise(0, -1) #removing noise on PYR WFS (-1 = no noise)
saxoplus.rtc.reset_ref_slopes(0) # reset refslopes from saxo to 0 (check)
for i in range(6): saxoplusmanager.next(seeAtmos=False); # check

saxoplus.rtc.do_ref_slopes(0) # meaure and load reflopes from null phase
for i in range(6): saxoplusmanager.next(seeAtmos=False); # check

# Computing modified KL basis (Gendron modal basis)
xpos0 = saxo.config.p_dms[0]._xpos # actus positions
ypos0 = saxo.config.p_dms[0]._ypos

xpos1 = saxoplus.config.p_dms[0]._xpos
ypos1 = saxoplus.config.p_dms[0]._ypos

nAllcommands0 = saxo.rtc.get_command(0).shape[0] # of total commands for cmat I.e HDOM + TT"

L0 = 25  # [m]
B0, l = KLmodes(xpos0, ypos0, L0, True) #basis on saxo stage
B1, l = KLmodes(xpos1, ypos1, L0, True) #basis on saxoplus stage

B0All = np.zeros((nAllcommands0,B0.shape[1]))
B0All[0:B0.shape[0],:] = B0

# interaction matrix on first stage
nModes0 = B0.shape[1]
ampli0 = 1.0e-2 #arbitraty unit
Nslopes0 = saxo.rtc.get_slopes(0).shape[0] # nb slopes of first stage
Nslopes1 = saxoplus.rtc.get_slopes(0).shape[0] # nb slopes of second stage
imat0 = np.zeros((Nslopes0, nModes0))
imat01 = np.zeros((Nslopes1, nModes0))


# Measuring imat on saxo. (both HODM and SH (imat0) + HODM and PYR (imat01))
for mode in trange(nModes0):
    volts = B0All[:, mode] * ampli0; 
    saxo.rtc.set_perturbation_voltage(0, "tmp", volts)
    for i in range(6): saxoplusmanager.next(seeAtmos=False); # check

    s1 = saxo.rtc.get_slopes(0) / ampli0
    s2 = saxoplus.rtc.get_slopes(0)  / ampli0
    imat0[:, mode] = s1.copy()
    imat01[:, mode] = s2.copy()
saxo.rtc.reset_perturbation_voltage(0) #remove ALL pertu voltages (rest the DM)
saxo.wfs.set_noise(0, saxo.config.p_wfss[0].get_noise()) #  sets the noise back to the config value



# Measuring imat on saxo+. ( Boston and PYR (imat1))
for i in range(6): saxoplusmanager.next(seeAtmos=False); # check
nmodes1 = B1.shape[1] # number of modes for second stage AO
ampli1 = 1.0e-2
imat1 = np.zeros((Nslopes1, nmodes1))

for mode in trange(nmodes1):
    volts = B1[:, mode] * ampli1; 
    saxoplus.rtc.set_perturbation_voltage(0, "tmp", volts)
    for i in range(6): saxoplusmanager.next(seeAtmos=False); # check
    s = saxoplus.rtc.get_slopes(0)  / ampli1
    imat1[:, mode] = s.copy()
saxoplus.rtc.reset_perturbation_voltage(0) #remove ALL pertu voltages (rest the DM)
saxoplus.wfs.set_noise(0, saxoplus.config.p_wfss[0].get_noise()) #  sets the noise back to the config value


# Computing command matrix first stage (HODM only). TT done by HODM. 
nControlled0 = 800
imat0f = imat0[:, :nControlled0].copy()
Dmp0 = np.dot(np.linalg.inv(np.dot(imat0f.T, imat0f)), imat0f.T) # modal command matrix
cmat0 = B0[:, :nControlled0].dot(Dmp0) # cmat for first stage only (HODM only)

cmatsaxo = np.zeros((nAllcommands0, Nslopes0)) #here cmat contains all DMs actuators beware
cmatsaxo[0:cmat0.shape[0],:] = cmat0 # filling HODM cmat on meta command matrix with all actus
saxo.rtc.set_command_matrix(0, cmatsaxo) # loading cmat in COMPASS (first stage)
gain = 0.4
saxo.rtc.set_gain(0, gain)

wao.wao1.enable_atmos(True) # enabling turbulence
saxo.rtc.close_loop(0) # closing loop on first stage
#saxo.rtc.open_loop(0) # opening loop



# Computing command matrices second stage. 
nControlled1 = 200
imat1f = imat1[:, :nControlled1].copy()
Dmp1 = np.dot(np.linalg.inv(np.dot(imat1f.T, imat1f)), imat1f.T) # modal command matrix
cmat1 = B1[:, :nControlled1].dot(Dmp1)


saxoplus.rtc.set_command_matrix(0, cmat1)
gain = 0.5
saxoplus.rtc.set_gain(0, gain)
wao.wao1.enable_atmos(True) # enabling turbulence
saxoplus.rtc.close_loop(0) # closing loop on second stage
#saxoplus.rtc.open_loop(0) # opening loop on second stage


# starting long exposure... 
nbiters = 300
print("starting long exposure")
plt.ion()
saxo.target.reset_strehl(0) # resetting long exposure SR on saxo
saxoplus.target.reset_strehl(0) # resetting long exposure SR on saxo+
for i in trange(nbiters): saxoplusmanager.next(seeAtmos=False); # exposure...
psfsaxo = saxo.target.get_tar_image(0, expo_type="le")
psfsaxoplus = saxoplus.target.get_tar_image(0, expo_type="le")

nsize = psfsaxo.shape[0]//2
cx = 60
plt.figure(1)
plt.imshow(np.log(psfsaxo[nsize-cx:nsize+cx,nsize-cx:nsize+cx]))
plt.figure(2)
plt.imshow(np.log(psfsaxoplus[nsize-cx:nsize+cx,nsize-cx:nsize+cx]))
print("SAXO SR = ", np.max(psfsaxo))
print("SAXO+ SR = " , np.max(psfsaxoplus))