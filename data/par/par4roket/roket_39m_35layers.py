import shesha.config as conf
import numpy as np

simul_name = ""
#loop
p_loop = conf.Param_loop()

p_loop.set_niter(40000)
p_loop.set_ittime(1 / 500.)  #=1/500
p_loop.set_devices([5])

#geom
p_geom = conf.Param_geom()

p_geom.set_zenithangle(0.)

#tel
p_tel = conf.Param_tel()

p_tel.set_diam(39.0)
p_tel.set_cobs(0.28)

#atmos
p_atmos = conf.Param_atmos()

altESO = np.array([
        30, 90, 150, 200, 245, 300, 390, 600, 1130, 1880, 2630, 3500, 4500, 5500, 6500,
        7500, 8500, 9500, 10500, 11500, 12500, 13500, 14500, 15500, 16500, 17500, 18500,
        19500, 20500, 21500, 22500, 23500, 24500, 25500, 26500
])
altESO = altESO.astype(int)

fracmed = [
        24.2, 12, 9.68, 5.9, 4.73, 4.73, 4.73, 4.73, 3.99, 3.24, 1.62, 2.6, 1.56, 1.04,
        1, 1.2, 0.4, 1.4, 1.3, 0.7, 1.6, 2.59, 1.9, 0.99, 0.62, 0.4, 0.25, 0.22, 0.19,
        0.14, 0.11, 0.06, 0.09, 0.05, 0.04
]
fracQ1 = [
        22.6, 11.2, 10.1, 6.4, 4.15, 4.15, 4.15, 4.15, 3.1, 2.26, 1.13, 2.21, 1.33, 0.88,
        1.47, 1.77, 0.59, 2.06, 1.92, 1.03, 2.3, 3.75, 2.76, 1.43, 0.89, 0.58, 0.36,
        0.31, 0.27, 0.2, 0.16, 0.09, 0.12, 0.07, 0.06
]
fracQ2 = [
        25.1, 11.6, 9.57, 5.84, 3.7, 3.7, 3.7, 3.7, 3.25, 3.47, 1.74, 3, 1.8, 1.2, 1.3,
        1.56, 0.52, 1.82, 1.7, 0.91, 1.87, 3.03, 2.23, 1.15, 0.72, 0.47, 0.3, 0.25, 0.22,
        0.16, 0.13, 0.07, 0.11, 0.06, 0.05
]
fracQ3 = [
        25.5, 11.9, 9.32, 5.57, 4.5, 4.5, 4.5, 4.5, 4.19, 4.04, 2.02, 3.04, 1.82, 1.21,
        0.86, 1.03, 0.34, 1.2, 1.11, 0.6, 1.43, 2.31, 1.7, 0.88, 0.55, 0.36, 0.22, 0.19,
        0.17, 0.12, 0.1, 0.06, 0.08, 0.04, 0.04
]
fracQ4 = [
        23.6, 13.1, 9.81, 5.77, 6.58, 6.58, 6.58, 6.58, 5.4, 3.2, 1.6, 2.18, 1.31, 0.87,
        0.37, 0.45, 0.15, 0.52, 0.49, 0.26, 0.8, 1.29, 0.95, 0.49, 0.31, 0.2, 0.12, 0.1,
        0.09, 0.07, 0.06, 0.03, 0.05, 0.02, 0.02
]

windESO = [
        5.5, 5.5, 5.1, 5.5, 5.6, 5.7, 5.8, 6, 6.5, 7, 7.5, 8.5, 9.5, 11.5, 17.5, 23, 26,
        29, 32, 27, 22, 14.5, 9.5, 6.3, 5.5, 6, 6.5, 7, 7.5, 8, 8.5, 9, 9.5, 10, 10
]

p_atmos.set_r0(0.15)
p_atmos.set_nscreens(len(altESO))
p_atmos.set_frac(fracQ3)
p_atmos.set_alt(altESO)
p_atmos.set_windspeed(windESO)
p_atmos.set_winddir(np.random.random(len(altESO)) * 360)
p_atmos.set_L0([100] * len(altESO))

#target
p_target = conf.Param_target()
p_targets = [p_target]
p_target.set_xpos(0.)
p_target.set_ypos(0.)
p_target.set_Lambda(1.65)
p_target.set_mag(10.)

#wfs
#p_wfs0= conf.Param_wfs()
#p_wfs1= conf.Param_wfs()
p_wfss = [conf.Param_wfs(roket=True)]

for i in range(len(p_wfss)):
    p_wfss[i].set_type("sh")
    p_wfss[i].set_nxsub(78)
    p_wfss[i].set_npix(6)
    p_wfss[i].set_pixsize(0.5)
    p_wfss[i].set_fracsub(0.8)
    p_wfss[i].set_xpos(5.)
    p_wfss[i].set_ypos(0.)
    p_wfss[i].set_Lambda(0.5)
    p_wfss[i].set_gsmag(8.)
    p_wfss[i].set_optthroughput(0.5)
    p_wfss[i].set_zerop(3e10)
    p_wfss[i].set_noise(3)
    p_wfss[i].set_atmos_seen(1)

#lgs parameters
#p_wfss[0].set_gsalt(90*1.e3)
#p_wfss[0].set_lltx(0)
#p_wfss[0].set_llty(0)
#p_wfss[0].set_laserpower(10)
#p_wfss[0].set_lgsreturnperwatt(1.e3)
#p_wfss[0].set_proftype("Exp")
#p_wfss[0].set_beamsize(0.8)

#dm
#p_dm0=conf.Param_dm()
#p_dm1=conf.Param_dm()
p_dms = [conf.Param_dm(), conf.Param_dm()]
p_dms[0].set_type("pzt")
nact = p_wfss[0].nxsub + 1
p_dms[0].set_nact(nact)
p_dms[0].set_alt(0.)
p_dms[0].set_thresh(0.3)
p_dms[0].set_coupling(0.2)
p_dms[0].set_unitpervolt(1.)
p_dms[0].set_push4imat(1.)
p_dms[0].set_influType("radialSchwartz")

p_dms[1].set_type("tt")
p_dms[1].set_alt(0.)
p_dms[1].set_unitpervolt(1.)
p_dms[1].set_push4imat(1.)

#centroiders
#p_centroider0=conf.Param_centroider()
p_centroiders = [conf.Param_centroider()]

for i in range(len(p_centroiders)):

    p_centroiders[i].set_nwfs(i)
    p_centroiders[i].set_type("cog")
    #p_centroiders[i].set_nmax(8)
    p_centroiders[i].set_thresh(0)

#p_centroider0.set_type("corr")
#p_centroider0.set_type_fct("model")

#controllers
p_controller1 = conf.Param_controller()
p_controllers = [p_controller1]

p_controller1.set_type("ls")
p_controller1.set_nwfs([0])
p_controller1.set_ndm([0, 1])
p_controller1.set_maxcond(150)
p_controller1.set_delay(0)
p_controller1.set_gain(0.3)
