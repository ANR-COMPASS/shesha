'''
Basic utilities function
'''
import numpy as np


def rebin(a, shape):
    """
    TODO: docstring

    """

    sh = shape[0], a.shape[0] // shape[0], shape[1], a.shape[1] // shape[1]
    return np.mean(a.reshape(sh), axis=(1, 3))


def fft_goodsize(s):
    """find best size for a fft from size s

    :parameters:

         s: (int) size
    """
    return 2**(int(np.log2(s)) + 1)


def bin2d(data_in, binfact):
    """
    Returns the input 2D array "array", binned with the binning factor "binfact".
    The input array X and/or Y dimensions needs not to be a multiple of
    "binfact"; The final/edge pixels are in effect replicated if needed.
    This routine prepares the parameters and calls the C routine _bin2d.
    The input array can be of type int, float or double.
    Last modified: Dec 15, 2003.
    Author: F.Rigaut
    SEE ALSO: _bin2d

    :parmeters:

        data_in: (np.ndarray) : data to binned

        binfact: (int) : binning factor
    """
    if (binfact < 1):
        raise ValueError("binfact has to be >= 1")

    nx = data_in.shape[0]
    ny = data_in.shape[1]
    fx = int(np.ceil(nx / float(binfact)))
    fy = int(np.ceil(ny / float(binfact)))

    data_out = np.zeros((fx, fy), dtype=data_in.dtype)

    for i1 in range(fx):
        for j1 in range(fy):
            for i2 in range(binfact):
                for j2 in range(binfact):
                    i = i1 * binfact + i2
                    j = j1 * binfact + j2
                    if (i >= nx):
                        i = nx - 1
                    if (j >= ny):
                        j = ny - 1
                    data_out[i1, j1] += data_in[i, j]

    return data_out


#def MESH(Range, Dim):
#    """
#    TODO: docstring
#
#    """
#
#    last = (0.5 * Range - 0.25 / Dim)
#    step = (2 * last) / (Dim - 1)
#
#    return np.tile(np.arange(Dim) * step - last, (Dim, 1))


def pad_array(A, N):
    """
    TODO: docstring

    """

    S = A.shape
    D1 = (N - S[0]) // 2
    D2 = (N - S[1]) // 2
    padded = np.zeros((N, N))
    padded[D1:D1 + S[0], D2:D2 + S[1]] = A
    return padded


def dist(dim, xc=-1, yc=-1):
    """
    TODO: docstring

    """

    if (xc < 0):
        xc = int(dim / 2.)
    else:
        xc -= 1.
    if (yc < 0):
        yc = int(dim / 2.)
    else:
        yc -= 1.

    dx = np.tile(np.arange(dim) - xc, (dim, 1))
    dy = np.tile(np.arange(dim) - yc, (dim, 1)).T

    d = np.sqrt(dx**2 + dy**2)
    return d


def makegaussian(size, fwhm, xc=-1, yc=-1, norm=0):
    """
    Returns a centered gaussian of specified size and fwhm.
    norm returns normalized 2d gaussian

    :param size: (int) :

    :param fwhm: (float) :

    :param xc: (float) : (optional) center position on x axis

    :param yc: (float) : (optional) center position on y axis

    :param norm: (int) : (optional) normalization
    """
    tmp = np.exp(-(dist(size, xc, yc) / (fwhm / 1.66))**2.)
    if (norm > 0):
        tmp = tmp / (fwhm**2. * 1.140075)
    return tmp


def rotate3d(im, ang, cx=-1, cy=-1, zoom=1.0):
    """Rotates an image of an angle "ang" (in DEGREES).

    The center of rotation is cx,cy.
    A zoom factor can be applied.

    (cx,cy) can be omitted :one will assume one rotates around the
    center of the image.
    If zoom is not specified, the default value of 1.0 is taken.

    modif dg : allow to rotate a cube of images with one angle per image

    :parameters:

        im: (np.ndarray[ndim=3,dtype=np.float32_t]) : array to rotate

        ang: (np.ndarray[ndim=1,dtype=np.float32_t]) : rotation angle  (in degrees)

        cx: (float) : (optional) rotation center on x axis (default: image center)

        cy: (float) : (optional) rotation center on x axis (default: image center)

        zoom: (float) : (opional) zoom factor (default =1.0)
"""

    # TODO test it
    if (zoom == 0):
        zoom = 1.0
    if (ang.size == 1):
        if (zoom == 1.0 and ang[0] == 0.):
            return im

    ang *= np.pi / 180

    nx = im.shape[1]
    ny = im.shape[2]

    if (cx < 0):
        cx = nx / 2 + 1
    if (cy < 0):
        cy = ny / 2 + 1

    x = np.tile(np.arange(nx) - cx + 1, (ny, 1)).T / zoom
    y = np.tile(np.arange(ny) - cy + 1, (nx, 1)) / zoom

    rx = np.zeros((nx, ny, ang.size)).astype(np.int64)
    ry = np.zeros((nx, ny, ang.size)).astype(np.int64)
    wx = np.zeros((nx, ny, ang.size)).astype(np.float32)
    wy = np.zeros((nx, ny, ang.size)).astype(np.float32)

    ind = np.zeros((nx, ny, ang.size)).astype(np.int64)

    imr = np.zeros((im.shape[0], im.shape[1], im.shape[2])).\
        astype(np.float32)

    for i in range(ang.size):
        matrot = np.array([[np.cos(ang[i]), -np.sin(ang[i])],
                           [np.sin(ang[i]), np.cos(ang[i])]], dtype=np.float32)
        wx[:, :, i] = x * matrot[0, 0] + y * matrot[1, 0] + cx
        wy[:, :, i] = x * matrot[0, 1] + y * matrot[1, 1] + cy

    nn = np.where(wx < 1)
    wx[nn] = 1.
    nn = np.where(wy < 1)
    wy[nn] = 1.

    nn = np.where(wx > (nx - 1))
    wx[nn] = nx - 1
    nn = np.where(wy > (ny - 1))
    wy[nn] = ny - 1

    rx = wx.astype(np.int64)  # partie entiere
    ry = wy.astype(np.int64)
    wx -= rx  # partie fractionnaire
    wy -= ry

    ind = rx + (ry - 1) * nx
    if (ang.size > 1):
        for i in range(ang.size):
            ind[:, :, i] += i * nx * ny

    imr.flat = \
        (im.flatten()[ind.flatten()] *
         (1 - wx.flatten()) +
            im.flatten()[ind.flatten() + 1] * wx.flatten())\
        * (1 - wy.flatten()) + \
        (im.flatten()[ind.flatten() + nx] * (1 - wx.flatten()) +
         im.flatten()[ind.flatten() + nx + 1] * wx.flatten()) * wy.flatten()

    return imr


def rotate(im, ang, cx=-1, cy=-1, zoom=1.0):
    """Rotates an image of an angle "ang" (in DEGREES).

    The center of rotation is cx,cy.
    A zoom factor can be applied.

    (cx,cy) can be omitted :one will assume one rotates around the
    center of the image.
    If zoom is not specified, the default value of 1.0 is taken.

    :parameters:

        im: (np.ndarray[ndim=3,dtype=np.float32_t]) : array to rotate

        ang: (float) : rotation angle (in degrees)

        cx: (float) : (optional) rotation center on x axis (default: image center)

        cy: (float) : (optional) rotation center on x axis (default: image center)

        zoom: (float) : (opional) zoom factor (default =1.0)
    """
    # TODO merge it with rotate3d or see if there is any np.rotate or other...
    if (zoom == 0):
        zoom = 1.0
    if (zoom == 1.0 and ang == 0.):
        return im

    ang *= np.pi / 180
    nx = im.shape[1]
    ny = im.shape[2]

    if (cx < 0):
        cx = nx / 2 + 1
    if (cy < 0):
        cy = ny / 2 + 1

    x = np.tile(np.arange(nx) - cx + 1, (ny, 1)).T / zoom
    y = np.tile(np.arange(ny) - cy + 1, (nx, 1)) / zoom

    ind = np.zeros((nx, ny, ang.size))

    imr = np.zeros((im.shape[0], im.shape[1], im.shape[2])).\
        astype(np.float32)

    matrot = np.array([[np.cos(ang), -np.sin(ang)], [np.sin(ang), np.cos(ang)]])

    wx = x * matrot[0, 0] + y * matrot[1, 0] + cx
    wy = x * matrot[0, 1] + y * matrot[1, 1] + cy

    wx = np.clip(wx, 1., nx - 1)
    wy = np.clip(wy, 1., ny - 1)
    wx, rx = np.modf(wx)
    wy, ry = np.modf(wy)

    ind = rx + (ry - 1) * nx

    imr = (im[ind] * (1 - wx) + im[ind + 1] * wx) * (1 - wy) + \
        (im[ind + nx] * (1 - wx) + im[ind + nx + 1] * wx) * wy

    return imr
