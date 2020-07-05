import shesha.config as conf

simul_name = "bench_scao_sh_16x16_8pix"
layout = "layoutDeFab_SH"

# loop
p_loop = conf.Param_loop()

p_loop.set_niter(1000)
p_loop.set_ittime(0.002)  # =1/500

# geom
p_geom = conf.Param_geom()

p_geom.set_zenithangle(0.)

# tel
p_tel = conf.Param_tel()

p_tel.set_diam(4.0)
p_tel.set_cobs(0.2)

# atmos
p_atmos = conf.Param_atmos()

p_atmos.set_r0(0.16)
p_atmos.set_nscreens(1)
p_atmos.set_frac([1.0])
p_atmos.set_alt([0.0])
p_atmos.set_windspeed([10.])
p_atmos.set_winddir([45.])
p_atmos.set_L0([1.e5])

# target
p_target = conf.Param_target()
p_targets = [p_target]

# p_target.set_ntargets(1)
p_target.set_xpos(0.)
p_target.set_ypos(0.)
p_target.set_Lambda(1.65)
p_target.set_mag(10.)

# wfs
p_wfs0 = conf.Param_wfs(roket=True)
p_wfss = [p_wfs0]

p_wfs0.set_type("sh")
p_wfs0.set_nxsub(8)
p_wfs0.set_npix(8)
p_wfs0.set_pixsize(0.3)
p_wfs0.set_fracsub(0.8)
p_wfs0.set_xpos(0.)
p_wfs0.set_ypos(0.)
p_wfs0.set_Lambda(0.5)
p_wfs0.set_gsmag(8.)
p_wfs0.set_optthroughput(0.5)
p_wfs0.set_zerop(1.e11)
p_wfs0.set_noise(3.)
p_wfs0.set_atmos_seen(1)

# lgs parameters
# p_wfs0.set_gsalt(90*1.e3)
# p_wfs0.set_lltx(0)
# p_wfs0.set_llty(0)
# p_wfs0.set_laserpower(10)
# p_wfs0.set_lgsreturnperwatt(1.e3)
# p_wfs0.set_proftype("Exp")
# p_wfs0.set_beamsize(0.8)

# dm
p_dm0 = conf.Param_dm()
p_dm1 = conf.Param_dm()
p_dms = [p_dm0, p_dm1]
p_dm0.set_type("pzt")
p_dm0.set_file_influ_fits("test_custom_dm.fits")
p_dm0.set_alt(0.)
p_dm0.set_thresh(0.3)
p_dm0.set_unitpervolt(0.01)
p_dm0.set_push4imat(100.)
p_dm0.set_diam_dm_proj(4.1)

p_dm1.set_type("tt")
p_dm1.set_alt(0.)
p_dm1.set_unitpervolt(0.0005)
p_dm1.set_push4imat(10.)

# centroiders
p_centroider0 = conf.Param_centroider()
p_centroiders = [p_centroider0]

p_centroider0.set_nwfs(0)
p_centroider0.set_type("cog")
# p_centroider0.set_type("corr")
# p_centroider0.set_type_fct("model")

# controllers
p_controller0 = conf.Param_controller()
p_controllers = [p_controller0]

p_controller0.set_type("ls")
p_controller0.set_nwfs([0])
p_controller0.set_ndm([0, 1])
p_controller0.set_maxcond(1500.)
p_controller0.set_delay(1.)
p_controller0.set_gain(0.4)

p_controller0.set_modopti(0)
p_controller0.set_nrec(2048)
p_controller0.set_nmodes(216)
p_controller0.set_gmin(0.001)
p_controller0.set_gmax(0.5)
p_controller0.set_ngain(500)
