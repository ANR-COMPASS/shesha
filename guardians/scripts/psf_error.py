import numpy as np
import matplotlib.pyplot as plt
from guardians import groot, gamora, drax
from rich.progress import track

filename = "/home/fferreira/Data/roket_8m_nssp40_dir135_speed10.h5"
spup = drax.get_pup(filename)

Cee = groot.compute_Cn_cpu(filename)
Cee += groot.compute_Calias(filename)
speed = np.array([15, 16, 17, 18, 19, 21, 22, 23, 24, 25]) - 10

Cab = groot.compute_Cerr(filename)
otftel, otf2, psfm, gpu = gamora.psf_rec_Vii(filename, fitting=False,
                                             cov=(Cee + Cab).astype(np.float32))
otf_fit, psf_fit = groot.compute_OTF_fitting(filename, otftel)
psfm = np.fft.fftshift(np.real(np.fft.ifft2(otf_fit * otf2 * otftel)))
psfm *= (psfm.shape[0] * psfm.shape[0] / float(np.where(spup)[0].shape[0]))

psfe = gamora.psf_rec_Vii(filename)
psf_compass = drax.get_tar_image(filename)
SR = []
EE5 = []
EE10 = []
EE20 = []

for s in track(speed):
    Cab = groot.compute_Cerr(filename, speed=np.array([s], dtype=np.float32))
    otftel, otf2, psfi, gpu = gamora.psf_rec_Vii(filename, fitting=False,
                                                 cov=(Cee + Cab).astype(np.float32))
    otf_fit, psf_fit = groot.compute_OTF_fitting(filename, otftel)
    psfi = np.fft.fftshift(np.real(np.fft.ifft2(otf_fit * otf2 * otftel)))
    psfi *= (psfm.shape[0] * psfm.shape[0] / float(np.where(spup)[0].shape[0]))
    SR.append(psfi.max())
    EE5.append(drax.ensquared_energy(filename, psfi, 5))
    EE10.append(drax.ensquared_energy(filename, psfi, 10))
    EE20.append(drax.ensquared_energy(filename, psfi, 20))

plt.figure()
plt.plot(speed, psf_compass.max() - SR)
plt.figure()
plt.plot(speed, drax.ensquared_energy(filename, psf_compass, 5) - EE5)
plt.plot(speed, drax.ensquared_energy(filename, psf_compass, 10) - EE10)
plt.plot(speed, drax.ensquared_energy(filename, psf_compass, 20) - EE20)
