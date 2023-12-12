"""
Created on Wed Oct 5 14:28:23 2016

@author: fferreira
"""
import numpy as np
import h5py
import matplotlib.pyplot as plt
import matplotlib
from guardians import gamora, drax

datapath = "/home/fferreira/Data/"
fname_layers = "roket_8m_12layers_gamma1.h5"  # File with  all layers
buferr_ref = drax.get_err(datapath + fname_layers)
f_layers = h5py.File(datapath + fname_layers)
nlayers = f_layers.attrs["_ParamAtmos__nscreens"]

fname_layer_i = []
name = "roket_8m_12layers"
for i in range(nlayers):
    fname_layer_i.append(name + "_%d.h5" % (i))

files = []
for f in fname_layer_i:
    files.append(h5py.File(datapath + f))

print("--------------------------------------------")
print("file ", fname_layers, " :")
print("    nlayers : ", f_layers.attrs["_ParamAtmos__nscreens"])
print("    frac    : ", f_layers.attrs["_ParamAtmos__frac"])
print("--------------------------------------------")

nmodes = f_layers["P"][:].shape[0]
contributors = [
        "tomography", "bandwidth", "non linearity", "noise", "filtered modes", "aliasing"
]
Lambda_tar = f_layers.attrs["_ParamTarget__Lambda"][0]
fracs = f_layers.attrs["_ParamAtmos__frac"]
alts = f_layers.attrs["_ParamAtmos__alt"]
frac_per_layer = dict()
i = 0
for a in alts:
    frac_per_layer[a] = fracs[i]
    i += 1

frac = []
buferr_layers = drax.get_err(datapath + fname_layer_i[0]) * 0.
for k in range(len(files)):
    frac.append(frac_per_layer[files[k].attrs["_ParamAtmos__alt"][0]])
    buferr_layers += drax.get_err(datapath + fname_layer_i[k]) * np.sqrt(
            frac_per_layer[files[k].attrs["_ParamAtmos__alt"][0]])

C_layers = np.zeros((buferr_layers.shape[0], buferr_layers.shape[0]))
for k in range(len(files)):
    C_layers += (
            frac[k] * drax.get_covmat_contrib(datapath + fname_layer_i[k], contributors))
print("contributors : ", contributors)

# Column 1 : with correlation, column 2 : independence assumption
err_layers = np.zeros((nmodes, 2))

err_layer_i = np.zeros((nmodes, 2 * nlayers))

err_layers[:, 0] = drax.variance(f_layers, contributors, method="Default")
err_layers[:, 1] = drax.variance(f_layers, contributors, method="Independence")
atm_layer = 0
for f in files:
    err_layer_i[:, atm_layer] = drax.variance(f, contributors, method="Default")
    err_layer_i[:, atm_layer + 1] = drax.variance(f, contributors, method="Independence")
    atm_layer += 2

#err_layer1p2 = varianceMultiFiles([f_layer1,f_layer2], frac_per_layer, contributors)
inderr = np.zeros(nmodes)
derr = np.zeros(nmodes)
for atm_layer in range(nlayers):
    inderr += frac[atm_layer] * err_layer_i[:, 2 * atm_layer + 1]
    derr += frac[atm_layer] * err_layer_i[:, 2 * atm_layer]

otftel_ref, otf2_ref, psf_ref, gpu = gamora.psf_rec_Vii(datapath + fname_layers)
otftel_sum, otf2_sum, psf_sum, gpu = gamora.psf_rec_Vii(datapath + fname_layers,
                                                        err=buferr_layers)

# Plots
plt.figure(1)
plt.subplot(2, 1, 1)
plt.semilogy(err_layers[:, 1])
plt.semilogy(inderr)
plt.legend(["%d layers" % nlayers, "Layers sum"])
plt.xlabel("Modes #")
plt.ylabel("Variance [mic^2]")
plt.title("Variance with independence assumption")
plt.subplot(2, 1, 2)
plt.plot(drax.cumulativeSR(err_layers[:, 1], Lambda_tar))
plt.plot(drax.cumulativeSR(inderr, Lambda_tar))
plt.legend(["%d layers" % nlayers, "Layers sum"])
plt.xlabel("Modes #")
plt.ylabel("SR")
plt.title("Resulting SR")

plt.figure(2)
plt.subplot(2, 1, 1)
plt.semilogy(err_layers[:, 0])
plt.semilogy(derr)
plt.legend(["%d layers" % nlayers, "Layers sum"])
plt.xlabel("Modes #")
plt.ylabel("Variance [mic^2]")
plt.title("Variance with correlation")
plt.subplot(2, 1, 2)
plt.plot(drax.cumulativeSR(err_layers[:, 0], Lambda_tar))
plt.plot(drax.cumulativeSR(derr, Lambda_tar))
plt.legend(["%d layers" % nlayers, "Layers sum"])
plt.xlabel("Modes #")
plt.ylabel("SR")
plt.title("Resulting SR")

RASC = 180 / np.pi * 3600.
#pixsize = (Lambda_tar*1e-6 / 8. * RASC) * 16./64.
#lambda/(Nfft*pupdiam/D)
pixsize = Lambda_tar * 1e-6 / (psf_ref.shape[0] * 8. / 640) * RASC
x = (np.arange(psf_ref.shape[0]) - psf_ref.shape[0] / 2) * pixsize / (
        Lambda_tar * 1e-6 / 8. * RASC)
font = {'family': 'normal', 'weight': 'bold', 'size': 22}

matplotlib.rc('font', **font)

plt.figure()
plt.semilogy(x, psf_ref[psf_ref.shape[0] / 2, :], color="blue")
plt.semilogy(x, psf_sum[psf_sum.shape[0] / 2, :], color="red")
plt.xlabel("Angle [units of lambda/D]")
plt.ylabel("Normalized intensity")
plt.legend(["12-layers PSF", "Sum of 12 layers PSF"])
