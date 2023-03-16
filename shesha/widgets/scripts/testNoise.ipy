from tqdm import trange
import matplotlib.pyplot as plt
plt.ion()


def testNoise(noise=0.9, niter=100, close_loop=True):

    s1 = np.zeros(((niter, int(wao.sim.config.p_controllers[0].nvalid * 2))))
    s2 = np.zeros(((niter, int(wao.sim.config.p_controllers[0].nvalid * 2))))
    sub1 = np.zeros(((niter, int(wao.sim.config.p_controllers[0].nvalid))))
    sub2 = np.zeros(((niter, int(wao.sim.config.p_controllers[0].nvalid))))
    dimim = wao.sim.wfs.get_pyrimg(0).shape[0]
    dimimHR = wao.sim.wfs.get_pyrimghr(0).shape[0]
    dimscreen = wao.sim.atm.get_screen(0).shape[0]
    im1 = np.zeros(((niter, dimim, dimim)))
    im2 = np.zeros(((niter, dimim, dimim)))
    #im1HR = np.zeros(((niter, dimimHR,dimimHR)))
    #im2HR = np.zeros(((niter, dimimHR,dimimHR)))
    screen1 = np.zeros(((niter, dimscreen, dimscreen)))
    screen2 = np.zeros(((niter, dimscreen, dimscreen)))

    wao.sim.rtc.set_open_loop(0, 1)
    wao.sim.atm.set_seed(0, 1234)
    wao.sim.atm.refresh_screen(0)
    wao.sim.wfs.set_noise(0, noise)
    wao.sim.dms.resetdm(b"pzt", 0.)
    wao.sim.dms.resetdm(b"tt", 0.)
    if close_loop:
        wao.sim.rtc.set_open_loop(0, 0)
    #s1, v1, t1, aiData1, psfLE1 = wao.record_ao_circular_buffer(niter)

    for k in trange(niter):
        wao.sim.next()
        s1[k, :] = wao.sim.rtc.get_centroids(0)
        sub1[k, :] = wao.sim.rtc.get_intensities(0)

        im1[k, :, :] = wao.sim.wfs.get_pyrimg(0)
        # im1HR[k,:,:] = wao.sim.wfs.get_pyrimghr(0)
        screen1[k, :, :] = wao.sim.atm.get_screen(0)
    wao.sim.rtc.set_open_loop(0, 1)
    wao.sim.atm.set_seed(0, 1234)
    wao.sim.atm.refresh_screen(0)
    wao.sim.wfs.set_noise(0, noise)
    wao.sim.dms.resetdm(b"pzt", 0.)
    wao.sim.dms.resetdm(b"tt", 0.)
    if close_loop:
        wao.sim.rtc.set_open_loop(0, 0)
    #s2, v2, t2, aiData2, psfLE2 = wao.record_ao_circular_buffer(niter)

    for k in trange(niter):
        wao.sim.next()
        s2[k, :] = wao.sim.rtc.get_centroids(0)
        sub2[k, :] = wao.sim.rtc.get_intensities(0)
        im2[k, :, :] = wao.sim.wfs.get_pyrimg(0)
        screen2[k, :, :] = wao.sim.atm.get_screen(0)
        # im2HR[k,:,:] = wao.sim.wfs.get_pyrimghr(0)

    screenErr = np.abs(screen2 - screen1).max() / screen1.max()
    imagesErr = np.abs(im2 - im1).max() / im1.max()
    subErr = np.abs(sub2 - sub1).max() / sub1.max()
    # imagesHRErr = np.abs(im2HR-im1HR).max()/im1HR.max()
    imagesHRErr = 0.
    slopesErr = np.abs(s2 - s1).max() / s1.max()
    print("Screen Error =" + str(screenErr))
    # print("Images HR Error ="+str(imagesHRErr))
    print("Images Error =" + str(imagesErr))
    print("Slopes Error =" + str(slopesErr))
    print("Subsum Error =" + str(subErr))

    #print(np.abs(psfLE2-psfLE1).max())
    #print(psfLE1.max())
    #print(psfLE2.max())
    return screenErr, imagesErr, imagesHRErr, slopesErr, subErr
    #plt.figure(1)
    #plt.clf()
    #plt.matshow(s2-s1, aspect="auto", fignum=1)


#plt.matshow(im2[-1,:,:]-im1[-1,:,:], aspect="auto", fignum=2)

errscreenList = []
imagesErrList = []
imagesHRErrList = []
slopesErrList = []
subErrList = []
for i in range(5):
    errscreen, imagesErr, imagesHRErr, slopesErr, subErr = testNoise(0.5, 200, True)
    errscreenList.append(errscreen)
    imagesErrList.append(imagesErr)
    imagesHRErrList.append(imagesHRErr)
    slopesErrList.append(slopesErr)
    subErrList.append(subErr)

plt.figure(1)
plt.clf()
plt.plot(errscreenList)
plt.title("Screen Error")

plt.figure(2)
plt.clf()
plt.plot(imagesErrList)
plt.title("images Error")

plt.figure(3)
plt.clf()
plt.plot(slopesErrList)
plt.title("slopes Error")

plt.figure(4)
plt.clf()
plt.plot(imagesHRErrList)
plt.title("images HR Error")
