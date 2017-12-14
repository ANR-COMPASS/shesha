'''
Utility functions for enforcing types in a property setter
'''

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
        if data.shape != shape:
            raise TypeError(
                    "Input has wrong dimensions, expect multi dimensional arrays")

    return np.array(data, dtype=dtype)
