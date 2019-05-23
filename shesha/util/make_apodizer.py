''' @package shesha.util.make_apodizer
make_apodizer function
'''

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
