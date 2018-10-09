"""
DRAX (Dedicated functions for Roket file Analysis and eXploitation)

Useful functions for ROKET file exploitation
"""

import numpy as np
import h5py
import pandas
import matplotlib.pyplot as plt
plt.ion()
from scipy.sparse import csr_matrix


def variance(f, contributors, method="Default"):
    """ Return the error variance of specified contributors
    params:
        f : (h5py.File) : roket hdf5 file opened with h5py
        contributors : (list of string) : list of the contributors
        method : (optional, default="Default") : if "Independence", the
                    function returns ths sum of the contributors variances.
                    If "Default", it returns the variance of the contributors sum
    """
    P = f["P"][:]
    nmodes = P.shape[0]
    swap = np.arange(nmodes) - 2
    swap[0:2] = [nmodes - 2, nmodes - 1]
    if (method == "Default"):
        err = f[contributors[0]][:] * 0.
        for c in contributors:
            err += f[c][:]
        return np.var(P.dot(err), axis=1)  #[swap]

    elif (method == "Independence"):
        nmodes = P.shape[0]
        v = np.zeros(nmodes)
        for c in contributors:
            v += np.var(P.dot(f[c][:]), axis=1)
        return v  #[swap]

    else:
        raise TypeError("Wrong method input")


def varianceMultiFiles(fs, frac_per_layer, contributors):
    """ Return the variance computed from the sum of contributors of roket
    files fs, ponderated by frac
    params:
        fs : (list) : list of hdf5 files opened with h5py
        frac_per_layer : (dict) : frac for each layer
        contributors : (list of string) : list of the contributors
    return:
        v : (np.array(dim=1)) : variance vector
    """
    f = fs[0]
    P = f["P"][:]
    nmodes = P.shape[0]
    swap = np.arange(nmodes) - 2
    swap[0:2] = [nmodes - 2, nmodes - 1]
    err = f[contributors[0]][:] * 0.
    for f in fs:
        frac = frac_per_layer[f.attrs["_Param_atmos__.alt"][0]]
        for c in contributors:
            err += np.sqrt(frac) * f[c][:]

    return np.var(P.dot(err), axis=1)  #[swap]


def cumulativeSR(v, Lambda_tar):
    """ Returns the cumulative Strehl ratio over the modes from the variance
    on each mode
    params:
        v : (np.array(dim=1)) : variance vector
    return:
        s : (np.array(dim=1)) : cumulative SR
    """
    s = np.cumsum(v)
    s = np.exp(-s * (2 * np.pi / Lambda_tar)**2)

    return s


def get_cumSR(filename):
    """
    Compute the SR over the modes from the variance
    on each mode

    :parameters:
        filename: (str): path to the ROKET file
    """
    f = h5py.File(filename, 'r')
    error_list = [
            "noise", "aliasing", "tomography", "filtered modes", "non linearity",
            "bandwidth"
    ]
    if (list(f.attrs.keys()).count("_Param_target__Lambda")):
        Lambda = f.attrs["_Param_target__Lambda"][0]
    else:
        Lambda = 1.65
    nactus = f["noise"][:].shape[0]
    niter = f["noise"][:].shape[1]
    P = f["P"][:]
    nmodes = P.shape[0]
    swap = np.arange(nmodes) - 2
    swap[0:2] = [nmodes - 2, nmodes - 1]
    data = np.zeros((nmodes, niter))
    data2 = np.zeros(nmodes)

    for i in error_list:
        data += np.dot(P, f[i][:])
        data2 += np.var(np.dot(P, f[i][:]), axis=1)

    data = np.var(data, axis=1)
    data = np.cumsum(data[swap])
    data = np.exp(-data * (2 * np.pi / Lambda)**2)
    data2 = np.cumsum(data2[swap])
    data2 = np.exp(-data2 * (2 * np.pi / Lambda)**2)
    data *= np.exp(-f["fitting"].value)
    data2 *= np.exp(-f["fitting"].value)

    SR2 = np.ones(nmodes) * f["SR2"].value
    SR = np.ones(nmodes) * f["SR"].value

    return data, data2, SR, SR2


def get_Btt(filename):
    """
    Return the Modes to Volt matrix
    :parameters:
        filename: (str): path to the ROKET file
    """
    f = h5py.File(filename, 'r')
    return f["Btt"][:]


def get_P(filename):
    """
    Return the Volt to Modes matrix
    :parameters:
        filename: (str): path to the ROKET file
    """
    f = h5py.File(filename, 'r')
    return f["P"][:]


def get_contribution(filename, contributor):
    """
    Return the variance of an error contributor

    :parameters:
        filename: (str): path to the ROKET file
        contributor: (str): contributor name
    :return:
        v: (np.array[ndim=1, dtype=np.float32]): variance of the contributor
    """
    f = h5py.File(filename, 'r')
    P = f["P"][:]
    nmodes = P.shape[0]
    swap = np.arange(nmodes) - 2
    swap[0:2] = [nmodes - 2, nmodes - 1]

    return np.var(np.dot(P, f[contributor][:]), axis=1)  #[swap]


def get_err_contributors(filename, contributors):
    """
    Return the sum of the specified contributors error buffers

    :parameters:
        filename: (str): path to the ROKET file
        contributors: (list): list of contributors
    :return:
        err: (np.ndarray[ndim=2,dtype=np.float32]): Sum of the error buffers
    """
    f = h5py.File(filename, 'r')
    # Get the sum of error contributors
    err = f["noise"][:] * 0.
    for c in contributors:
        err += f[c][:]
    f.close()

    return err


def get_err(filename):
    """
    Return the sum of all the error buffers

    :parameters:
        filename: (str): path to the ROKET file
    :return:
        err: (np.ndarray[ndim=2,dtype=np.float32]): Sum of the error buffers
    """

    f = h5py.File(filename, 'r')
    # Get the sum of error contributors
    err = f["noise"][:]
    err += f["aliasing"][:]
    err += f["tomography"][:]
    err += f["filtered modes"][:]
    err += f["non linearity"][:]
    err += f["bandwidth"][:]
    f.close()

    return err


def get_coverr_independence(filename):
    """
    Return the error covariance matrix considering statistical independence between contributors

    :parameters:
        filename: (str): path to the ROKET file
    :return:
        err: (np.ndarray[ndim=2,dtype=np.float32]): Covariance matrix
    """

    f = h5py.File(filename, 'r')
    # Get the sum of error contributors
    N = f["noise"][:].shape[1]
    err = f["noise"][:].dot(f["noise"][:].T)
    err += f["aliasing"][:].dot(f["aliasing"][:].T)
    err += f["tomography"][:].dot(f["tomography"][:].T)
    err += f["filtered modes"][:].dot(f["filtered modes"][:].T)
    err += f["non linearity"][:].dot(f["non linearity"][:].T)
    err += f["bandwidth"][:].dot(f["bandwidth"][:].T)
    f.close()

    return err / N


def get_coverr_independence_contributors(filename, contributors):
    """
    Return the error covariance matrix considering statistical independence between specified contributors

    :parameters:
        filename: (str): path to the ROKET file
        contributors: (list): list of contributors
    :return:
        err: (np.ndarray[ndim=2,dtype=np.float32]): Covariance matrix
    """

    f = h5py.File(filename, 'r')
    # Get the sum of error contributors
    N = f["noise"][:].shape[1]
    err = np.zeros((f["noise"][:].shape[0], f["noise"][:].shape[0]))
    for c in contributors:
        err += f[c][:].dot(f[c][:].T)

    f.close()

    return err / N


def get_covmat_contrib(filename, contributors, modal=True):
    """
    Return the covariance matrix of the specified contributors

    :parameters:
        filename: (str): path to the ROKET file
        contributor: (list): name of a contributor of the ROKET file
        modal: (bool): if True (default), return the matrix expressed in the modal basis
    :return:
        covmat: (np.ndarray(ndim=2, dtype=np.float32)): covariance matrix
    """
    h5f = h5py.File(filename, 'r')
    contrib = h5f["bandwidth"][:] * 0.
    for c in contributors:
        contrib += h5f[c][:]
    covmat = contrib.dot(contrib.T) / contrib.shape[1]
    if modal:
        P = h5f["P"][:]
        covmat = P.dot(covmat).dot(P.T)
    h5f.close()

    return covmat


def get_pup(filename):
    """
    Return the pupil saved in a ROKET file
    :parameters:
        filename: (str): path to the ROKET file
    :return:
        spup: (np.ndarray[ndim=2,dtype=np.float32]): pupil

    """
    f = h5py.File(filename, 'r')
    if (list(f.keys()).count("spup")):
        spup = f["spup"][:]
    else:
        indx_pup = f["indx_pup"][:]
        pup = np.zeros((f["dm_dim"].value, f["dm_dim"].value))
        pup_F = pup.flatten()
        pup_F[indx_pup] = 1.
        pup = pup_F.reshape(pup.shape)
        spup = pup[np.where(pup)[0].min():np.where(pup)[0].max() + 1,
                   np.where(pup)[1].min():np.where(pup)[1].max() + 1]

    f.close()
    return spup


def get_breakdown(filename):
    """
    Computes the error breakdown in nm rms from a ROKET file

    :parameters:
        filename: (str): path to the ROKET file
    :return:
        breakdown: (dict): dictionnary containing the error breakdown
    """
    f = h5py.File(filename, 'r')
    P = f["P"][:]
    noise = f["noise"][:]
    trunc = f["non linearity"][:]
    bp = f["bandwidth"][:]
    tomo = f["tomography"][:]
    aliasing = f["aliasing"][:]
    filt = f["filtered modes"][:]
    nmodes = P.shape[0]
    swap = np.arange(nmodes) - 2
    swap[0:2] = [nmodes - 2, nmodes - 1]
    N = np.var(P.dot(noise), axis=1)
    S = np.var(P.dot(trunc), axis=1)
    B = np.var(P.dot(bp), axis=1)
    T = np.var(P.dot(tomo), axis=1)
    A = np.var(P.dot(aliasing), axis=1)
    F = np.var(P.dot(filt), axis=1)
    C = np.var(P.dot(filt + noise + trunc + bp + tomo + aliasing), axis=1)
    inde = N + S + B + T + A + F

    if (list(f.attrs.keys()).count("_Param_target__Lambda")):
        Lambda = f.attrs["_Param_target__Lambda"][0]
    else:
        Lambda = 1.65

    print("noise :", np.sqrt(np.sum(N)) * 1e3, " nm rms")
    print("trunc :", np.sqrt(np.sum(S)) * 1e3, " nm rms")
    print("bp :", np.sqrt(np.sum(B)) * 1e3, " nm rms")
    print("tomo :", np.sqrt(np.sum(T)) * 1e3, " nm rms")
    print("aliasing :", np.sqrt(np.sum(A)) * 1e3, " nm rms")
    print("filt :", np.sqrt(np.sum(F)) * 1e3, " nm rms")
    print("fitting :",
          np.mean(np.sqrt(f["fitting"].value / ((2 * np.pi / Lambda)**2)) * 1e3),
          " nm rms")
    print("cross-terms :", np.sqrt(np.abs(np.sum(C) - np.sum(inde))) * 1e3, " nm rms")
    return {
            "noise":
                    np.sqrt(np.sum(N)) * 1e3,
            "non linearity":
                    np.sqrt(np.sum(S)) * 1e3,
            "bandwidth":
                    np.sqrt(np.sum(B)) * 1e3,
            "tomography":
                    np.sqrt(np.sum(T)) * 1e3,
            "aliasing":
                    np.sqrt(np.sum(A)) * 1e3,
            "filtered modes":
                    np.sqrt(np.sum(F)) * 1e3,
            "fitting":
                    np.mean(
                            np.sqrt(f["fitting"].value /
                                    ((2 * np.pi / Lambda)**2)) * 1e3)
    }


# def plotContributions(filename):
#     f = h5py.File(filename, 'r')
#     P = f["P"][:]
#     noise = f["noise"][:]
#     trunc = f["non linearity"][:]
#     bp = f["bandwidth"][:]
#     tomo = f["tomography"][:]
#     aliasing = f["aliasing"][:]
#     filt = f["filtered modes"][:]
#     nmodes = P.shape[0]
#     swap = np.arange(nmodes) - 2
#     swap[0:2] = [nmodes - 2, nmodes - 1]

#     plt.figure()
#     plt.plot(np.var(noise, axis=1), color="black")
#     plt.plot(np.var(trunc, axis=1), color="green")
#     plt.plot(np.var(bp, axis=1), color="red")
#     plt.plot(np.var(tomo, axis=1), color="blue")
#     plt.plot(np.var(aliasing, axis=1), color="cyan")
#     plt.plot(np.var(filt, axis=1), color="magenta")
#     plt.xlabel("Actuators")
#     plt.ylabel("Variance [microns^2]")
#     plt.title("Variance of estimated errors on actuators")
#     plt.legend([
#             "noise", "WFS non-linearity", "Bandwidth", "Anisoplanatism", "Aliasing",
#             "Filtered modes"
#     ])

#     plt.figure()
#     N = np.var(P.dot(noise), axis=1)
#     S = np.var(P.dot(trunc), axis=1)
#     B = np.var(P.dot(bp), axis=1)
#     T = np.var(P.dot(tomo), axis=1)
#     A = np.var(P.dot(aliasing), axis=1)
#     F = np.var(P.dot(filt), axis=1)
#     plt.plot(N[swap], color="black")
#     plt.plot(S[swap], color="green")
#     plt.plot(B[swap], color="red")
#     plt.plot(T[swap], color="blue")
#     plt.plot(A[swap], color="cyan")
#     plt.plot(F[swap], color="magenta")
#     plt.xlabel("Modes")
#     plt.ylabel("Variance [microns^2]")
#     plt.yscale("log")
#     plt.title("Variance of estimated errors on modal basis B")

#     if (list(f.attrs.keys()).count("_Param_target__Lambda")):
#         Lambda = f.attrs["_Param_target__Lambda"][0]
#     else:
#         Lambda = 1.65

#     print("noise :",
#           np.sqrt(np.sum(N)) * 1e3, " nm, ", "SR : ",
#           np.exp(-np.sum(N) * (2 * np.pi / Lambda)**2))
#     print("trunc :",
#           np.sqrt(np.sum(S)) * 1e3, " nm, ", "SR : ",
#           np.exp(-np.sum(S) * (2 * np.pi / Lambda)**2))
#     print("bp :",
#           np.sqrt(np.sum(B)) * 1e3, " nm, ", "SR : ",
#           np.exp(-np.sum(B) * (2 * np.pi / Lambda)**2))
#     print("tomo :",
#           np.sqrt(np.sum(T)) * 1e3, " nm, ", "SR : ",
#           np.exp(-np.sum(T) * (2 * np.pi / Lambda)**2))
#     print("aliasing :",
#           np.sqrt(np.sum(A)) * 1e3, " nm, ", "SR : ",
#           np.exp(-np.sum(A) * (2 * np.pi / Lambda)**2))
#     print("filt :",
#           np.sqrt(np.sum(F)) * 1e3, " nm, ", "SR : ",
#           np.exp(-np.sum(F) * (2 * np.pi / Lambda)**2))
#     print("fitting :",
#           np.sqrt(f["fitting"].value / ((2 * np.pi / Lambda)**2)) * 1e3, " nm, ",
#           "SR : ", np.exp(-f["fitting"].value))
#     #plt.legend(["noise","WFS non-linearity","Bandwidth","Anisoplanatism","Aliasing","Filtered modes"])


def plotCovCor(filename, maparico=None):
    """
    Displays the covariance and correlation matrix between the contributors
    :parameters:
        filename: (str): path to the ROKET file
        maparico: (str): (optional) matplotlib colormap to use

    """
    f = h5py.File(filename, 'r')
    cov = f["cov"][:]
    cor = f["cor"][:]

    labels = ["noise", "WF deviation", "aliasing", "filt. modes", "bandwidth", "aniso"]
    if (maparico is None):
        maparico = "viridis"
    x = np.arange(6)
    plt.matshow(cov, cmap=maparico)
    plt.colorbar()
    plt.xticks(x, labels, rotation="vertical")
    plt.yticks(x, labels)

    plt.matshow(cor, cmap=maparico)
    plt.colorbar()
    plt.xticks(x, labels, rotation="vertical")
    plt.yticks(x, labels)
    print("Total variance : ", cov.sum(), " microns^2")


def get_IF(filename):
    """
    Return the influence functions of the pzt and tt DM saved in a ROKET file
    :parameters:
       filename: (str): path to the ROKET file
    :return:
        IF: (csr_matrix): pzt influence function (sparse)
        T: (np.ndarray[ndim=2,dtype=np.float32]): tip tilt influence function
    """
    f = h5py.File(filename, 'r')
    IF = csr_matrix((f["IF.data"][:], f["IF.indices"][:], f["IF.indptr"][:]))
    if (list(f.keys()).count("TT")):
        T = f["TT"][:]
    else:
        T = IF[-2:, :].toarray()
        IF = IF[:-2, :]
    f.close()
    return IF, T.T.astype(np.float32)


def get_mode(filename, n):
    """
    Return the #n mode of the Btt modal basis contains in a ROKET file
    :parameters:
        filename: (str): path to the ROKET file
        n: (int): mode number
    :return:
        sc: (np.ndarray[ndim=2,dtype=np.float32]): mode #n of the Btt basis
    """
    f = h5py.File(filename, 'r')
    Btt = f["Btt"][:]
    IF, TT = get_IF(filename)
    dim = f["dm_dim"].value
    indx = f["indx_pup"][:]
    sc = np.zeros((dim, dim))
    sc = sc.flatten()
    mode = IF.T.dot(Btt[:-2, n])
    mode += TT.T.dot(Btt[-2:, n])
    sc[indx] = mode

    return sc.reshape((dim, dim))


def get_psf(filename):
    """
    Return the PSF computed by COMPASS saved in the ROKET file

    :parameters:
        filename: (str): path to the ROKET file
    :return:
        psf: (np.ndarray[ndim=2,dtype=np.float32]): PSF computed by COMPASS
    """
    f = h5py.File(filename, "r")
    psf = f["psf"][:]
    f.close()

    return psf


def getMap(filename, covmat):
    """
    Return the spatial representation of a covariance matrix expressed in the DM space
    :parameters:
        filename: (str): path to the ROKET file
        covmat: (np.ndarray[ndim=2,dtype=np.float32]): covariance matrix
    :return:
        Map: (np.ndarray[ndim=2,dtype=np.float32]): covariance map
    """
    f = h5py.File(filename, 'r')
    # nn, c'est, en gros un where(actus==valides)
    xpos = f["dm.xpos"][:]
    ypos = f["dm.ypos"][:]
    pitch = xpos[1] - xpos[0]
    nact = f.attrs["_Param_dm__nact"][0]
    x = ((xpos - xpos.min()) / pitch).astype(np.int32)
    y = ((ypos - ypos.min()) / pitch).astype(np.int32)
    nn = (x, y)

    #creation du tableau des decalages
    #xx et yy c'est les cood des actus valides
    #dx et dy c'est la matrice des differences de coordonnees, entre -nssp et +nssp
    xx = np.tile(np.arange(nact), (nact, 1))
    yy = xx.T
    dx = np.zeros((x.size, x.size), dtype=np.int32)
    dy = dx.copy()
    for k in range(x.size):
        dx[k, :] = xx[nn][k] - xx[nn]
        dy[k, :] = yy[nn][k] - yy[nn]

    # transformation des decalages en indice de tableau
    dx += (nact - 1)
    dy += (nact - 1)

    # transformation d'un couple de decalages (dx,dy) en un indice du tableau 'Map'
    Map = np.zeros((nact * 2 - 1, nact * 2 - 1)).flatten()
    div = Map.copy()
    ind = dy.flatten() + (nact * 2 - 1) * (dx.flatten())
    Cf = covmat.flatten()
    for k in range(ind.size):
        Map[ind[k]] += Cf[k]
        div[ind[k]] += 1

    div[np.where(div == 0)] = 1
    Map /= div

    return Map.reshape((nact * 2 - 1, nact * 2 - 1))


def SlopesMap(covmat, filename=None, nssp=None, validint=None):
    """
    Return a part of the spatial representation of a covariance matrix expressed in the slopes space.
    Need to be called 4 times to get the full map (XX, YY, XY, YX)

    :parameters:
        covmat: (np.ndarray[ndim=2,dtype=np.float32]): part of the covariance matrix
        filename: (str): (optional) path to the ROKET file
        nssp: (int): (optional) Number of ssp in the diameter
        validint: (float): (optional) Central obstruction as a ratio of D

    :return:
        Map: (np.ndarray[ndim=2,dtype=np.float32]): covariance map
    """
    if filename is not None:
        f = h5py.File(filename, 'r')
        nssp = f.attrs["_Param_wfs__nxsub"][0]
        validint = f.attrs["_Param_tel__cobs"]
        f.close()

    if nssp is None or validint is None:
        raise ValueError("nssp and validint not defined")

    nsub = covmat.shape[0]
    x = np.linspace(-1, 1, nssp)
    x, y = np.meshgrid(x, x)
    r = np.sqrt(x * x + y * y)

    rorder = np.sort(r.reshape(nssp * nssp))
    ncentral = nssp * nssp - np.sum(r >= validint, dtype=np.int32)
    validext = rorder[ncentral + nsub]
    valid = (r < validext) & (r >= validint)
    nn = np.where(valid)

    xx = np.tile(np.arange(nssp), (nssp, 1))
    yy = xx.T
    xx = xx[nn]
    yy = yy[nn]
    dx = np.zeros((xx.size, xx.size), dtype=np.int32)
    dy = dx.copy()
    for k in range(xx.size):
        dx[k, :] = xx[k] - xx
        dy[k, :] = yy[k] - yy

    # transformation des decalages en indice de tableau
    dx += (nssp - 1)
    dy += (nssp - 1)

    # transformation d'un couple de decalages (dx,dy) en un indice du tableau 'Map'
    Map = np.zeros((nssp * 2 - 1, nssp * 2 - 1)).flatten()
    div = Map.copy()
    ind = dy.flatten() + (nssp * 2 - 1) * (dx.flatten())
    Cf = covmat.flatten()
    for k in range(ind.size):
        Map[ind[k]] += Cf[k]
        div[ind[k]] += 1

    div[np.where(div == 0)] = 1
    Map /= div

    return Map.reshape((nssp * 2 - 1, nssp * 2 - 1))


def covFromMap(Map, nsub, filename=None, nssp=None, validint=None):
    """
    Return a part of the spatial representation of a covariance matrix expressed in the slopes space.
    Need to be called 4 times to get the full map (XX, YY, XY, YX)

    :parameters:
        covmat: (np.ndarray[ndim=2,dtype=np.float32]): part of the covariance matrix
        filename: (str): (optional) path to the ROKET file
        nssp: (int): (optional) Number of ssp in the diameter
        validint: (float): (optional) Central obstruction as a ratio of D

    :return:
        Map: (np.ndarray[ndim=2,dtype=np.float32]): covariance map
    """
    if filename is not None:
        f = h5py.File(filename, 'r')
        nssp = f.attrs["_Param_wfs__nxsub"][0]
        validint = f.attrs["_Param_tel__cobs"]
        f.close()

    if nssp is None or validint is None:
        raise ValueError("nssp and validint not defined")

    x = np.linspace(-1, 1, nssp)
    x, y = np.meshgrid(x, x)
    r = np.sqrt(x * x + y * y)

    rorder = np.sort(r.reshape(nssp * nssp))
    ncentral = nssp * nssp - np.sum(r >= validint, dtype=np.int32)
    validext = rorder[ncentral + nsub]
    valid = (r < validext) & (r >= validint)
    nn = np.where(valid)

    xx = np.tile(np.arange(nssp), (nssp, 1))
    yy = xx.T
    xx = xx[nn]
    yy = yy[nn]
    dx = np.zeros((xx.size, xx.size), dtype=np.int32)
    dy = dx.copy()
    for k in range(xx.size):
        dx[k, :] = xx[k] - xx
        dy[k, :] = yy[k] - yy

    # transformation des decalages en indice de tableau
    dx += (nssp - 1)
    dy += (nssp - 1)

    # transformation d'un couple de decalages (dx,dy) en un indice du tableau 'Map'
    covmat = np.zeros((nsub, nsub))
    ind = dy.flatten() + (nssp * 2 - 1) * (dx.flatten())
    Cf = covmat.flatten()
    Map = Map.flatten()
    for k in range(ind.size):
        Cf[k] = Map[ind[k]]

    return Cf.reshape((nsub, nsub))


def getCovFromMap(Map, nsub, filename=None, nssp=None, validint=None):
    """
    Return the full spatial representation of a covariance matrix expressed in the slopes space.

    :parameters:
        covmat: (np.ndarray[ndim=2,dtype=np.float32]): part of the covariance matrix
        filename: (str): (optional) path to the ROKET file
        nssp: (int): (optional) Number of ssp in the diameter
        validint: (float): (optional) Central obstruction as a ratio of D
    :return:
        Map: (np.ndarray[ndim=2,dtype=np.float32]): covariance map
    """
    if filename is not None:
        f = h5py.File(filename, 'r')
        nssp = f.attrs["_Param_wfs__nxsub"][0]
        f.close()
    mapSize = 2 * nssp - 1
    covmat = np.zeros((nsub, nsub))

    covmat[:nsub // 2, :nsub // 2] = covFromMap(Map[:mapSize, :mapSize], nsub // 2,
                                                filename=filename)
    covmat[nsub // 2:, nsub // 2:] = covFromMap(Map[mapSize:, mapSize:], nsub // 2,
                                                filename=filename)
    covmat[:nsub // 2, nsub // 2:] = covFromMap(Map[:mapSize, mapSize:], nsub // 2,
                                                filename=filename)
    covmat[nsub // 2:, :nsub // 2] = covFromMap(Map[mapSize:, :mapSize], nsub // 2,
                                                filename=filename)

    return covmat


def getSlopesMap(covmat, filename=None, nssp=None, validint=None):
    """
    Return the full spatial representation of a covariance matrix expressed in the slopes space.

    :parameters:
        covmat: (np.ndarray[ndim=2,dtype=np.float32]): part of the covariance matrix
        filename: (str): (optional) path to the ROKET file
        nssp: (int): (optional) Number of ssp in the diameter
        validint: (float): (optional) Central obstruction as a ratio of D
    :return:
        Map: (np.ndarray[ndim=2,dtype=np.float32]): covariance map
    """
    if filename is not None:
        f = h5py.File(filename, 'r')
        nssp = f.attrs["_Param_wfs__nxsub"][0]
        f.close()
    nsub = covmat.shape[0] // 2
    mapSize = 2 * nssp - 1
    Map = np.zeros((2 * mapSize, 2 * mapSize))

    Map[:mapSize, :mapSize] = SlopesMap(covmat[:nsub, :nsub], filename=filename,
                                        nssp=nssp, validint=validint)
    Map[:mapSize, mapSize:] = SlopesMap(covmat[:nsub, nsub:], filename=filename,
                                        nssp=nssp, validint=validint)
    Map[mapSize:, :mapSize] = SlopesMap(covmat[nsub:, :nsub], filename=filename,
                                        nssp=nssp, validint=validint)
    Map[mapSize:, mapSize:] = SlopesMap(covmat[nsub:, nsub:], filename=filename,
                                        nssp=nssp, validint=validint)

    return Map


def ensquare_PSF(filename, psf, N, display=False, cmap="jet"):
    """
    Return the ensquared PSF

    :parameters:
        filename: (str): path to the ROKET file
        psf: (np.ndarray[ndim=2,dtype=np.float32]): PSF to ensquare
        N: (int): size of the square in units of Lambda/D
        display: (bool): (optional) if True, displays also the ensquare PSF
        cmat: (str): (optional) matplotlib colormap to use
    :return:
        psf: (np.ndarray[ndim=2,dtype=np.float32]): the ensquared psf
    """
    f = h5py.File(filename, 'r')
    Lambda_tar = f.attrs["_Param_target__Lambda"][0]
    RASC = 180 / np.pi * 3600.
    pixsize = Lambda_tar * 1e-6 / (psf.shape[0] * f.attrs["_Param_tel__diam"] / f.attrs[
            "_Param_geom__pupdiam"]) * RASC
    x = (np.arange(psf.shape[0]) - psf.shape[0] / 2) * pixsize / (
            Lambda_tar * 1e-6 / f.attrs["_Param_tel__diam"] * RASC)
    w = int(N * (Lambda_tar * 1e-6 / f.attrs["_Param_tel__diam"] * RASC) / pixsize)
    mid = psf.shape[0] // 2
    psfe = np.abs(psf[mid - w:mid + w, mid - w:mid + w])
    if (display):
        plt.matshow(np.log10(psfe), cmap=cmap)
        plt.colorbar()
        xt = np.linspace(0, psfe.shape[0] - 1, 6).astype(np.int32)
        yt = np.linspace(-N, N, 6).astype(np.int32)
        plt.xticks(xt, yt)
        plt.yticks(xt, yt)

    f.close()
    return psf[mid - w:mid + w, mid - w:mid + w]


def ensquared_energy(filename, psf, N):
    """
    Return the ensquared energy in a box width of N * lambda/D

    :parameters:
        filename: (str): path to the ROKET file
        N: (int): size of the square in units of Lambda/D
    """
    return ensquare_PSF(filename, psf, N).sum() / psf.sum()


def cutsPSF(filename, psf, psfs):
    """
    Plots cuts of two PSF along X and Y axis for comparison
    :parameters:
        filename: (str): path to the ROKET file
        psf: (np.ndarray[ndim=2,dtype=np.float32]): first PSF
        psfs: (np.ndarray[ndim=2,dtype=np.float32]): second PSF
    """
    f = h5py.File(filename, 'r')
    Lambda_tar = f.attrs["_Param_target__Lambda"][0]
    RASC = 180 / np.pi * 3600.
    pixsize = Lambda_tar * 1e-6 / (psf.shape[0] * f.attrs["_Param_tel__diam"] / f.attrs[
            "_Param_geom__pupdiam"]) * RASC
    x = (np.arange(psf.shape[0]) - psf.shape[0] / 2) * pixsize / (
            Lambda_tar * 1e-6 / f.attrs["_Param_tel__diam"] * RASC)
    plt.figure()
    plt.subplot(2, 1, 1)
    plt.semilogy(x, psf[psf.shape[0] // 2, :], color="blue")
    plt.semilogy(x, psfs[psf.shape[0] // 2, :], color="red")
    plt.semilogy(x,
                 np.abs(psf[psf.shape[0] // 2, :] - psfs[psf.shape[0] // 2, :]),
                 color="green")
    plt.xlabel("X-axis angular distance [units of lambda/D]")
    plt.ylabel("Normalized intensity")
    plt.legend(["PSF exp", "PSF model", "Diff"])
    plt.xlim(-20, 20)
    plt.ylim(1e-7, 1)
    plt.subplot(2, 1, 2)
    plt.semilogy(x, psf[:, psf.shape[0] // 2], color="blue")
    plt.semilogy(x, psfs[:, psf.shape[0] // 2], color="red")
    plt.semilogy(x,
                 np.abs(psf[:, psf.shape[0] // 2] - psfs[:, psf.shape[0] // 2]),
                 color="green")
    plt.xlabel("Y-axis angular distance [units of lambda/D]")
    plt.ylabel("Normalized intensity")
    plt.legend(["PSF exp", "PSF model", "Diff"])
    plt.xlim(-20, 20)
    plt.ylim(1e-7, 1)
    f.close()


def compDerivativeCmm(filename=None, slopes=None, dt=1, dd=False, ss=False):
    """
    Compute d/dt(slopes)*slopes from ROKET buffer
    :parameters:
        filename: (str): (optional) path to the ROKET file
        slopes: (np.ndarray[ndim=2,dtype=np.float32]: (optional) Buffer of slopes arranged as (nsub x niter)
        dt: (int): (optionnal) dt in frames
        dd: (bool): (optionnal) if True, computes d/dt(slopes)*d/dt(slopes)
    :return:
        dCmm: (np.ndarray[ndim=2,dtype=np.float32]: covariance matrix of slopes with their derivative
    """
    if filename is not None:
        f = h5py.File(filename, 'r')
        slopes = f["slopes"][:]
        f.close()
    if slopes is not None:
        if dd:
            dCmm = (slopes[:, dt:] - slopes[:, :-dt]).dot(
                    (slopes[:, dt:] - slopes[:, :-dt]).T / 2)
        elif ss:
            dCmm = slopes[:, :-dt].dot(slopes[:, dt:].T)
        else:
            dCmm = (slopes[:, dt:] - slopes[:, :-dt]).dot(
                    (slopes[:, dt:] + slopes[:, :-dt]).T / 2)

        return dCmm / slopes[:, dt:].shape[1]


def compProfile(filename, nlayers):
    """
    Identify turbulent parameters (wind speed, direction and frac. of r0) from ROKET file

    :parameters:
        filename: (str): path to the ROKET file
        nlayers: (int): number of turbulent layers (maybe deduced in the future ?)
    """
    f = h5py.File(filename, "r")
    dt = f.attrs["_Param_loop__ittime"]
    dk = int(2 / 3 * f.attrs["_Param_tel__diam"] / 20 / dt)
    pdiam = f.attrs["_Param_tel__diam"] / f.attrs["_Param_wfs__nxsub"]

    mapC = getSlopesMap(compDerivativeCmm(filename, dt=dk), filename)
    size = mapC.shape[0] // 2
    minimap = mapC[size:, size:] + mapC[size:, :size] + mapC[:size,
                                                             size:] + mapC[:size, :size]

    ws = np.zeros(nlayers)
    wd = np.zeros(nlayers)
    frac = np.zeros(nlayers)

    for k in range(nlayers):
        plt.matshow(minimap)
        plt.title(str(k))
        x, y = np.where(minimap == minimap.max())
        x = int(x)
        y = int(y)
        print("max ", k, ": x=", x, " ; y=", y)
        frac[k] = minimap[x, y]
        r = np.linalg.norm([x - size / 2, y - size / 2]) * pdiam
        ws[k] = r / (dk * dt)
        wd[k] = np.arctan2(x - size / 2, y - size / 2) * 180 / np.pi
        if (wd[k] < 0):
            wd[k] += 360
        minimap[x - 2:x + 3, y - 2:y + 3] = 0
        minimap[(size - x - 1) - 2:(size - x - 1) + 3, (size - y - 1) - 2:
                (size - y - 1) + 3] = 0
    frac /= frac.sum()

    ind = np.argsort(f.attrs["_Param_atmos__frac"])[::-1]
    print("Real wind speed: ", f.attrs["_Param_atmos__windspeed"][ind].tolist())
    print("Estimated wind speed: ", ws.tolist())
    print("-----------------------------")
    print("Real wind direction: ", f.attrs["_Param_atmos__winddir"][ind].tolist())
    print("Estimated wind direction: ", wd.tolist())
    print("-----------------------------")
    print("Real frac: ", f.attrs["_Param_atmos__frac"][ind].tolist())
    print("Estimated frac: ", frac.tolist())
    print("-----------------------------")
    f.close()
