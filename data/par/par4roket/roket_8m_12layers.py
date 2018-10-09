import shesha.config as conf

simul_name = ""
#loop
p_loop = conf.Param_loop()

p_loop.set_niter(20000)
p_loop.set_ittime(1 / 500.)  #=1/500
p_loop.set_devices([5])

#geom
p_geom = conf.Param_geom()

p_geom.set_zenithangle(0.)

#tel
p_tel = conf.Param_tel()

p_tel.set_diam(8.0)
p_tel.set_cobs(0.)

#atmos
p_atmos = conf.Param_atmos()

p_atmos.set_r0(0.16)
p_atmos.set_nscreens(12)
p_atmos.set_frac([
        0.261, 0.138, 0.081, 0.064, 0.105, 0.096, 0.085, 0.053, 0.048, 0.037, 0.021,
        0.011
])
p_atmos.set_alt([
        0.0, 100.0, 200.0, 300.0, 900.0, 1800.0, 4500.0, 7100, 11000, 12800, 14500, 16500
])
p_atmos.set_windspeed([13, 17, 5, 17, 10, 10, 8, 6, 14, 9, 8, 17])
p_atmos.set_winddir([345, -292, -115, -161, -179, -266, -208, 185, 265, 116, 6, 272])
p_atmos.set_L0([100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100])

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
    p_wfss[i].set_nxsub(40)
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
p_controller1.set_maxcond(20)
p_controller1.set_delay(0)
p_controller1.set_gain(0.3)
