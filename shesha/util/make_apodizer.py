## @package   shesha.util.make_apodizer
## @brief     make_apodizer function
## @author    COMPASS Team <https://github.com/ANR-COMPASS>
## @version   4.4.1
## @date      2011/01/28
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

import numpy as np
from scipy.ndimage import interpolation as interp

from . import utilities as util


def make_apodizer(dim, pupd, filename, angle):
    """TODO doc

    :parameters:

        (int) : im:

        (int) : pupd:

        (str) : filename:

        (float) : angle:
    """

    print("Opening apodizer")
    print("reading file:", filename)
    pup = np.load(filename)
    A = pup.shape[0]

    if (A > dim):
        raise ValueError("Apodizer dimensions must be smaller.")

    if (A != pupd):
        # use misc.imresize (with bilinear)
        print("TODO pup=bilinear(pup,pupd,pupd)")

    if (angle != 0):
        # use ndimage.interpolation.rotate
        print("TODO pup=rotate2(pup,angle)")
        pup = interp.rotate(pup, angle, reshape=False, order=2)

    reg = np.where(util.dist(pupd) > pupd / 2.)
    pup[reg] = 0.

    pupf = np.zeros((dim, dim), dtype=np.float32)

    if (dim != pupd):
        if ((dim - pupd) % 2 != 0):
            pupf[(dim - pupd + 1) / 2:(dim + pupd + 1) / 2, (dim - pupd + 1) /
                 2:(dim + pupd + 1) / 2] = pup

        else:
            pupf[(dim - pupd) / 2:(dim + pupd) / 2, (dim - pupd) / 2:(dim + pupd) /
                 2] = pup

    else:
        pupf = pup

    pupf = np.abs(pupf).astype(np.float32)

    return pupf
