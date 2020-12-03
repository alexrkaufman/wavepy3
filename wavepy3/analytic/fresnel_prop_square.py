"""
Analytic result for the Fresnel propagation of a square aperture
through vacuum.

TODO: update to conform to python style guides more effectively
"""
from numpy import sqrt
from scipy.special import fresnel


def fresnel_prop_square(x_n, y_n, D_0, wvl, Z):
    """
    A function for the analytic result of the Fresnel propagation
    of a square aperture through vacuum.

    :param x_n: array of x coordinates in terminal plane
    :param y_n: array of y coordinates in terminal plane
    :param D_0: Side length of the square aperture.
    :param wvl: Wavelength of light.
    :param Z: Propagation distance

    :returns: the analytic fresnel propagation result for the coordinates given
    """
    N_F = (D_0 / 2)**2 / (wvl * Z)

    X = x_n / sqrt(wvl * Z)
    Y = y_n / sqrt(wvl * Z)

    alpha1 = -sqrt(2) * (sqrt(N_F) + X)
    alpha2 = sqrt(2) * (sqrt(N_F) - X)
    beta1 = -sqrt(2) * (sqrt(N_F) + Y)
    beta2 = sqrt(2) * (sqrt(N_F) - Y)

    sa1, ca1 = fresnel(alpha1)
    sa2, ca2 = fresnel(alpha2)
    sb1, cb1 = fresnel(beta1)
    sb2, cb2 = fresnel(beta2)

    return (1 / (2 * 1j)
            * ((ca2 - ca1) + 1j * (sa2 - sa1))
            * ((cb2 - cb1) + 1j * (sb2 - sb1)))
