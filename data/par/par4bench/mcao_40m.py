import shesha.config as conf

simul_name = "mcao_40m_80_8pix"

# loop
p_loop = conf.ParamLoop()

p_loop.set_niter(5000)
p_loop.set_ittime(0.002)  # =1/500

# geom
p_geom = conf.ParamGeom()

p_geom.set_zenithangle(0.)

# tel
p_tel = conf.ParamTel()

p_tel.set_diam(40.0)
p_tel.set_cobs(0.12)

# atmos
p_atmos = conf.ParamAtmos()

p_atmos.set_r0(0.16)
p_atmos.set_nscreens(4)
p_atmos.set_frac([0.5, 0.2, 0.2, 0.1])
p_atmos.set_alt([0.0, 4499., 4500., 9000.])
p_atmos.set_windspeed([10.0, 10.0, 10.0, 10.0])
p_atmos.set_winddir([0., 10., 20., 25.])
p_atmos.set_L0([25., 25., 25., 25.])

# target
p_target = conf.ParamTarget()
p_targets = [p_target]
p_target.set_xpos(0.)
p_target.set_ypos(0.)
p_target.set_Lambda(1.65)
p_target.set_mag(10.)

# wfs
p_wfs1 = conf.ParamWfs()
p_wfs2 = conf.ParamWfs()
p_wfs3 = conf.ParamWfs()
p_wfs4 = conf.ParamWfs()
p_wfss = [p_wfs1, p_wfs2, p_wfs3, p_wfs4]

p_wfs1.set_type("sh")
p_wfs1.set_nxsub(80)
p_wfs1.set_npix(8)
p_wfs1.set_pixsize(0.3)
p_wfs1.set_fracsub(0.8)
p_wfs1.set_xpos(40.)
p_wfs1.set_ypos(40.)
p_wfs1.set_Lambda(0.5)
p_wfs1.set_gsmag(8.)
p_wfs1.set_optthroughput(0.5)
p_wfs1.set_zerop(1.e11)
p_wfs1.set_noise(1.)
p_wfs1.set_atmos_seen(1)

p_wfs2.set_type("sh")
p_wfs2.set_nxsub(86)
p_wfs2.set_npix(8)
p_wfs2.set_pixsize(0.3)
p_wfs2.set_fracsub(0.8)
p_wfs2.set_xpos(40.)
p_wfs2.set_ypos(-40.)
p_wfs2.set_Lambda(0.5)
p_wfs2.set_gsmag(8.)
p_wfs2.set_optthroughput(0.5)
p_wfs2.set_zerop(1.e11)
p_wfs2.set_noise(1.)
p_wfs2.set_atmos_seen(1)

p_wfs3.set_type("sh")
p_wfs3.set_nxsub(80)
p_wfs3.set_npix(8)
p_wfs3.set_pixsize(0.3)
p_wfs3.set_fracsub(0.8)
p_wfs3.set_xpos(-40.)
p_wfs3.set_ypos(40.)
p_wfs3.set_Lambda(0.5)
p_wfs3.set_gsmag(8.)
p_wfs3.set_optthroughput(0.5)
p_wfs3.set_zerop(1.e11)
p_wfs3.set_noise(1.)
p_wfs3.set_atmos_seen(1)

p_wfs4.set_type("sh")
p_wfs4.set_nxsub(80)
p_wfs4.set_npix(8)
p_wfs4.set_pixsize(0.3)
p_wfs4.set_fracsub(0.8)
p_wfs4.set_xpos(-40.)
p_wfs4.set_ypos(-40.)
p_wfs4.set_Lambda(0.5)
p_wfs4.set_gsmag(8.)
p_wfs4.set_optthroughput(0.5)
p_wfs4.set_zerop(1.e11)
p_wfs4.set_noise(1.)
p_wfs4.set_atmos_seen(1)

# lgs parameters
# p_wfs0.set_gsalt(90*1.e3)
# p_wfs0.set_lltx(0.)
# p_wfs0.set_llty(0.)
# p_wfs0.set_laserpower(10)
# p_wfs0.set_lgsreturnperwatt(1.e3)
# p_wfs0.set_proftype("Exp")
# p_wfs0.set_beamsize(0.8)

# dm
p_dm0 = conf.ParamDm()
p_dm1 = conf.ParamDm()
p_dm2 = conf.ParamDm()
p_dm3 = conf.ParamDm()
p_dms = [p_dm0, p_dm1, p_dm2, p_dm3]

p_dm0.set_type("pzt")
p_dm0.set_nact(81)
p_dm0.set_alt(0.)
p_dm0.set_thresh(0.3)
p_dm0.set_coupling(0.2)
p_dm0.set_unitpervolt(0.01)
p_dm0.set_push4imat(100.)

p_dm1.set_type("pzt")
p_dm1.set_nact(101)
p_dm1.set_alt(4500.)
p_dm1.set_thresh(0.)
p_dm1.set_coupling(0.2)
p_dm1.set_unitpervolt(0.01)
p_dm1.set_push4imat(100.)

p_dm2.set_type("pzt")
p_dm2.set_nact(131)
p_dm2.set_alt(9000.)
p_dm2.set_thresh(0.)
p_dm2.set_coupling(0.2)
p_dm2.set_unitpervolt(0.01)
p_dm2.set_push4imat(100.)

p_dm3.set_type("tt")
p_dm3.set_alt(0.)
p_dm3.set_unitpervolt(0.0005)
p_dm3.set_push4imat(10.)

# centroiders
p_centroider0 = conf.ParamCentroider()
p_centroider1 = conf.ParamCentroider()
p_centroider2 = conf.ParamCentroider()
p_centroider3 = conf.ParamCentroider()
p_centroiders = [p_centroider0, p_centroider1, p_centroider2, p_centroider3]

p_centroider0.set_nwfs(0)
p_centroider0.set_type("cog")

p_centroider1.set_nwfs(1)
p_centroider1.set_type("cog")

p_centroider2.set_nwfs(2)
p_centroider2.set_type("cog")

p_centroider3.set_nwfs(3)
p_centroider3.set_type("cog")

# controllers
p_controller0 = conf.ParamController()
p_controllers = [p_controller0]

p_controller0.set_type("mv")
p_controller0.set_nwfs([0, 1, 2, 3])
p_controller0.set_ndm([0, 1, 2, 3])
p_controller0.set_maxcond(1500.)
p_controller0.set_delay(1.)
p_controller0.set_gain(0.3)
