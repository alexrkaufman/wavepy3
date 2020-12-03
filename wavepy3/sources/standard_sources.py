"""
A library of standard sources as a convenience to users.

May be better to call this something like apertures???

TODO improve variable names for readability.
"""
import numpy as np


def square_ap(x, D=1):
    """square_ap

    :param x: meshgrid of coords
    :param D: diameter (Default value = 1)

    """

    x = np.abs(x)

    output = np.ones(x.shape)

    output[x > (D / 2)] = 0
    output[x == (D / 2)] = 0.5

    return np.prod(output, axis=0)


def circ_ap(x, y, radius=1):
    """

    :param x: x meshgrid
    :param y: y meshgrid
    :param radius: the radius of the aperture (Default value = 1)

    """

    r = np.sqrt(x**2 + y**2)

    circ = np.ones(r.shape)
    circ[r > radius] = 0

    return circ
