import numpy as np
import matplotlib.pyplot as plt
import h5py
from guardians import groot, gamora
import os

filename = os.getenv("DATA_GUARDIAN") + "roket_8m_LE.h5"
Cab = groot.compute_Cerr(filename)
_, _, psfModel, _ = gamora.psf_rec_Vii(filename, fitting=False,
                                       cov=Cab.astype(np.float32))

f = h5py.File(filename, 'r')
tb = f["tomography"][:] + f["bandwidth"][:]

for k in range(10000, 201000, 10000):
    C = tb[:, :k].dot(tb[:, :k].T) / k
    _, _, psfC, _ = gamora.psf_rec_Vii(filename, fitting=False,
                                       covmodes=C.astype(np.float32))
    plt.matshow(
            np.log10(np.abs(psfC - psfModel)), vmin=np.log10(np.abs(psfModel)).min(),
            vmax=np.log10(np.abs(psfModel)).max())
    plt.title("niter = %d" % k)
    plt.colorbar()
