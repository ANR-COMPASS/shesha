## @package   shesha.util.utilities
## @brief     Basic utilities function
## @author    COMPASS Team <https://github.com/ANR-COMPASS>
## @version   5.2.0
## @date      2020/05/18
## @copyright GNU Lesser General Public License
#
#  This file is part of COMPASS <https://anr-compass.github.io/compass/>
#
#  Copyright (C) 2011-2019 COMPASS Team <https://github.com/ANR-COMPASS>
#  All rights reserved.
#  Distributed under GNU - LGPL
#
#  COMPASS is free software: you can redistribute it and/or modify it under the terms of the GNU Lesser
#  General Public License as published by the Free Software Foundation, either version 3 of the License,
#  or any later version.
#
#  COMPASS: End-to-end AO simulation tool using GPU acceleration
#  The COMPASS platform was designed to meet the need of high-performance for the simulation of AO systems.
#
#  The final product includes a software package for simulating all the critical subcomponents of AO,
#  particularly in the context of the ELT and a real-time core based on several control approaches,
#  with performances consistent with its integration into an instrument. Taking advantage of the specific
#  hardware architecture of the GPU, the COMPASS tool allows to achieve adequate execution speeds to
#  conduct large simulation campaigns called to the ELT.
#
#  The COMPASS platform can be used to carry a wide variety of simulations to both testspecific components
#  of AO of the E-ELT (such as wavefront analysis device with a pyramid or elongated Laser star), and
#  various systems configurations such as multi-conjugate AO.
#
#  COMPASS is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the
#  implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
#  See the GNU Lesser General Public License for more details.
#
#  You should have received a copy of the GNU Lesser General Public License along with COMPASS.
#  If not, see <https://www.gnu.org/licenses/lgpl-3.0.txt>.

import importlib
import sys, os

import numpy as np


def rebin(a, shape):
    """
    TODO: docstring

    """

    sh = shape[0], a.shape[0] // shape[0], shape[1], a.shape[1] // shape[1]
    return np.mean(a.reshape(sh), axis=(1, 3))


def fft_goodsize(s):
    """find best size for a fft from size s

    Args:

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
    if (yc < 0):
        yc = int(dim / 2.)

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


def generate_square(radius: float, density: float = 1.):
    """ Generate modulation points positions following a square pattern

    Args:
        radius : (float) : half the length of a side in lambda/D

        density : (float), optional) : number of psf per lambda/D. Default is 1

    Returns:
        cx : (np.ndarray) : X-positions of the modulation points

        cy : (np.ndarray) : Y-positions of the modulation points
    """
    x = np.linspace(-radius, radius, 1 + 2 * int(radius * density))
    cx, cy = np.meshgrid(x, x, indexing='ij')
    cx = cx.flatten()
    cy = cy.flatten()
    return (cx, cy)


def generate_circle(radius: float, density: float = 1.):
    """ Generate modulation points positions following a circular pattern
s
    Args:
        radius : (float) : half the length of a side in lambda/D

        density : (float), optional) : number of psf per lambda/D. Default is 1

    Returns:
        cx : (np.ndarray) : X-positions of the modulation points

        cy : (np.ndarray) : Y-positions of the modulation points
    """
    cx, cy = generate_square(radius, density)
    r = cx * cx + cy * cy <= radius**2
    return (cx[r], cy[r])

def generate_pseudo_source(radius: float, additional_psf=0, density=1.):
    """ Used to generate a pseudo source for PYRWFS

    Args:
        radius : (float) : TODO description

        additional_psf : (int) :TODO description

        density : (float, optional) :TODO description

    Returns:
        ox : TODO description & explicit naming

        oy : TODO description & explicit naming

        w : TODO description & explicit naming

        xc : TODO description & explicit naming

        yc : TODO description & explicit naming
    """
    struct_size = (1 + 2 * additional_psf)**2
    center_x, center_y = generate_square(additional_psf, density)
    center_weight = (1 + 2 * int(additional_psf * density))**2 * [1]
    center_size = 1 + 2 * int(additional_psf * density)

    weight_edge = [(1 + 2 * int(radius * density) - center_size) // 2]
    xc, yc = generate_circle(radius, density)
    for k in range(additional_psf):
        line_length = np.sum(yc == (k + 1))
        print(line_length)
        weight_edge.append((line_length - center_size) // 2)

    edge_dist = (radius + additional_psf) // 2
    V_edge_x = []
    V_edge_y = []
    V_edge_weight = []
    for m in [-1, 1]:
        V_edge_x.append(0)
        V_edge_y.append(m * edge_dist)
        V_edge_weight.append(weight_edge[0])
    for k, val in enumerate(weight_edge[1:]):
        for l in [-1, 1]:
            for m in [-1, 1]:
                V_edge_x.append(l * (k + 1) * density)
                V_edge_y.append(m * edge_dist)
                V_edge_weight.append(val)
    H_edge_x = []
    H_edge_y = []
    H_edge_weight = []
    for m in [-1, 1]:
        H_edge_x.append(m * edge_dist)
        H_edge_y.append(0)
        H_edge_weight.append(weight_edge[0])
    for k, val in enumerate(weight_edge[1:]):
        for l in [-1, 1]:
            for m in [-1, 1]:
                H_edge_x.append(m * edge_dist)
                H_edge_y.append(l * (k + 1) * density)
                H_edge_weight.append(val)
    pup_cent_x = []
    pup_cent_y = []
    pup_cent_weight = 4 * [(len(xc) - 2 * np.sum(H_edge_weight) - struct_size) / 4]
    pup_cent_dist = int(edge_dist // np.sqrt(2))
    for l in [-1, 1]:
        for m in [-1, 1]:
            pup_cent_x.append(l * pup_cent_dist)
            pup_cent_y.append(m * pup_cent_dist)
    ox = np.concatenate((center_x, V_edge_x, H_edge_x, pup_cent_x))
    oy = np.concatenate((center_y, V_edge_y, H_edge_y, pup_cent_y))
    w = np.concatenate((center_weight, V_edge_weight, H_edge_weight,
                        pup_cent_weight))
    return (ox, oy, w, xc, yc)


def first_non_zero(array: np.ndarray, axis: int, invalid_val: int = -1) -> np.ndarray:
    """ Find the first non zero element of an array

    Args:
        array : (np.ndarray) : input array

        axis : (int) : axis index

        invalid_val : (int, optional) : Default is -1

    Returns:
        non_zeros_pos : (np.ndarray) : Index of the first non-zero element
                                        for each line or column following the axis
    """
    mask = array != 0
    return np.where(mask.any(axis=axis), mask.argmax(axis=axis), invalid_val)


# def rotate3d(im, ang, cx=-1, cy=-1, zoom=1.0):
#     """Rotates an image of an angle "ang" (in DEGREES).

#     The center of rotation is cx,cy.
#     A zoom factor can be applied.

#     (cx,cy) can be omitted :one will assume one rotates around the
#     center of the image.
#     If zoom is not specified, the default value of 1.0 is taken.

#     modif dg : allow to rotate a cube of images with one angle per image

#     Args:

#         im: (np.ndarray[ndim=3,dtype=np.float32_t]) : array to rotate

#         ang: (np.ndarray[ndim=1,dtype=np.float32_t]) : rotation angle  (in degrees)

#         cx: (float) : (optional) rotation center on x axis (default: image center)

#         cy: (float) : (optional) rotation center on x axis (default: image center)

#         zoom: (float) : (opional) zoom factor (default =1.0)
# """

#     # TODO test it
#     if (zoom == 0):
#         zoom = 1.0
#     if (ang.size == 1):
#         if (zoom == 1.0 and ang[0] == 0.):
#             return im

#     ang *= np.pi / 180

#     nx = im.shape[1]
#     ny = im.shape[2]

#     if (cx < 0):
#         cx = nx / 2 + 1
#     if (cy < 0):
#         cy = ny / 2 + 1

#     x = np.tile(np.arange(nx) - cx + 1, (ny, 1)).T / zoom
#     y = np.tile(np.arange(ny) - cy + 1, (nx, 1)) / zoom

#     rx = np.zeros((nx, ny, ang.size)).astype(np.int64)
#     ry = np.zeros((nx, ny, ang.size)).astype(np.int64)
#     wx = np.zeros((nx, ny, ang.size)).astype(np.float32)
#     wy = np.zeros((nx, ny, ang.size)).astype(np.float32)

#     ind = np.zeros((nx, ny, ang.size)).astype(np.int64)

#     imr = np.zeros((im.shape[0], im.shape[1], im.shape[2])).\
#         astype(np.float32)

#     for i in range(ang.size):
#         matrot = np.array([[np.cos(ang[i]), -np.sin(ang[i])],
#                            [np.sin(ang[i]), np.cos(ang[i])]], dtype=np.float32)
#         wx[:, :, i] = x * matrot[0, 0] + y * matrot[1, 0] + cx
#         wy[:, :, i] = x * matrot[0, 1] + y * matrot[1, 1] + cy

#     nn = np.where(wx < 1)
#     wx[nn] = 1.
#     nn = np.where(wy < 1)
#     wy[nn] = 1.

#     nn = np.where(wx > (nx - 1))
#     wx[nn] = nx - 1
#     nn = np.where(wy > (ny - 1))
#     wy[nn] = ny - 1

#     rx = wx.astype(np.int64)  # partie entiere
#     ry = wy.astype(np.int64)
#     wx -= rx  # partie fractionnaire
#     wy -= ry

#     ind = rx + (ry - 1) * nx
#     if (ang.size > 1):
#         for i in range(ang.size):
#             ind[:, :, i] += i * nx * ny

#     imr.flat = \
#         (im.flatten()[ind.flatten()] *
#          (1 - wx.flatten()) +
#             im.flatten()[ind.flatten() + 1] * wx.flatten())\
#         * (1 - wy.flatten()) + \
#         (im.flatten()[ind.flatten() + nx] * (1 - wx.flatten()) +
#          im.flatten()[ind.flatten() + nx + 1] * wx.flatten()) * wy.flatten()

#     return imr

# def rotate(im, ang, cx=-1, cy=-1, zoom=1.0):
#     """Rotates an image of an angle "ang" (in DEGREES).

#     The center of rotation is cx,cy.
#     A zoom factor can be applied.

#     (cx,cy) can be omitted :one will assume one rotates around the
#     center of the image.
#     If zoom is not specified, the default value of 1.0 is taken.

#     Args:

#         im: (np.ndarray[ndim=3,dtype=np.float32_t]) : array to rotate

#         ang: (float) : rotation angle (in degrees)

#         cx: (float) : (optional) rotation center on x axis (default: image center)

#         cy: (float) : (optional) rotation center on x axis (default: image center)

#         zoom: (float) : (opional) zoom factor (default =1.0)
#     """
#     # TODO merge it with rotate3d or see if there is any np.rotate or other...
#     if (zoom == 0):
#         zoom = 1.0
#     if (zoom == 1.0 and ang == 0.):
#         return im

#     ang *= np.pi / 180
#     nx = im.shape[1]
#     ny = im.shape[2]

#     if (cx < 0):
#         cx = nx / 2 + 1
#     if (cy < 0):
#         cy = ny / 2 + 1

#     x = np.tile(np.arange(nx) - cx + 1, (ny, 1)).T / zoom
#     y = np.tile(np.arange(ny) - cy + 1, (nx, 1)) / zoom

#     ind = np.zeros((nx, ny, ang.size))

#     imr = np.zeros((im.shape[0], im.shape[1], im.shape[2])).\
#         astype(np.float32)

#     matrot = np.array([[np.cos(ang), -np.sin(ang)], [np.sin(ang), np.cos(ang)]])

#     wx = x * matrot[0, 0] + y * matrot[1, 0] + cx
#     wy = x * matrot[0, 1] + y * matrot[1, 1] + cy

#     wx = np.clip(wx, 1., nx - 1)
#     wy = np.clip(wy, 1., ny - 1)
#     wx, rx = np.modf(wx)
#     wy, ry = np.modf(wy)

#     ind = rx + (ry - 1) * nx

#     imr = (im[ind] * (1 - wx) + im[ind + 1] * wx) * (1 - wy) + \
#         (im[ind + nx] * (1 - wx) + im[ind + nx + 1] * wx) * wy

#     return imr
