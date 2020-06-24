## @package   shesha.config.config_setter
## @brief     Utility functions for enforcing types in a property setter
## @author    COMPASS Team <https://github.com/ANR-COMPASS>
## @version   4.4.2
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


def enforce_int(n):
    if not isinstance(n, int):
        raise TypeError("Value should be integer.")
    return n


def enforce_float(f):
    if not (isinstance(f, float) or isinstance(f, int)):
        raise TypeError("Value should be float.")
    return float(f)


def enforce_or_cast_bool(b):
    if isinstance(b, bool):
        return b
    if isinstance(b, (int, float)):
        if b == 0:
            return False
        elif b == 1:
            return True
        else:
            raise ValueError("Will not cast non 0/1 int or float to boolean.")
    raise TypeError("Will only cast int and float to booleans.")


def enforce_array(data, size, dtype=np.float32, scalar_expand=False):
    # Scalar case
    if isinstance(data, (int, float, complex)):
        data = [data]

    # Singleton case
    if len(data) == 1:
        if scalar_expand or size == 1:
            return np.full(size, data[0], dtype=dtype)
        else:
            raise TypeError("This non-singleton array cannot " + \
                            "be initialized with a scalar.")

    if len(data) != size:
        raise TypeError("Input argument has wrong number of elements.")
    if isinstance(data, np.ndarray) and len(data.shape) > 1:
        raise TypeError("Multidimensional ndarray input is not allowed")
    if isinstance(data,
                  list) and not all([isinstance(x, (float, int, complex))
                                     for x in data]):
        raise TypeError("Input list may only contain numerical values.")

    # OK, now let's try it
    return np.array(data, dtype=dtype)


def enforce_arrayMultiDim(data, shape, dtype=np.float32):
    if not isinstance(data, np.ndarray):
        raise TypeError("Input argument must be a np.ndarray")
    else:
        doRaise = False
        if len(data.shape) != len(shape):
            doRaise = True
        for (i, j) in zip(data.shape, shape):
            if j != -1 and i != j:
                doRaise = True
        if doRaise:
            raise TypeError(
                    "Input has wrong dimensions, expect multi dimensional arrays")

    return np.array(data, dtype=dtype)
