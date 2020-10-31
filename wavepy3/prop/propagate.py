import numpy as np
from numpy import (exp, pi, meshgrid, linspace, sqrt)
from numpy.fft import (fft2, fftshift, ifft2, ifftshift)
from ..atmos import Atmos


def split_step(field_in, wvl, delta_0, delta_n, z_prop, atmos=None):

    n_gridpts = len(field_in)
    nx = np.linspace(-n_gridpts / 2, n_gridpts / 2, n_gridpts,
                     endpoint=False)
    nx, ny = np.meshgrid(nx, nx)

    k = 2 * pi / wvl

    nsq = nx**2 + ny**2
    boundary_width = 0.47 * n_gridpts
    boundary_supergaussian = np.exp(-nsq**8 / boundary_width**16)

    delta_z = z_prop[1:] - z_prop[:-1]

    prop_frac = z_prop / z_prop[-1]
    delta = (1 - prop_frac) * delta_0 + prop_frac * delta_n
    m = delta[1:] / delta[:-1]
    x0 = nx * delta[0]
    y0 = ny * delta[0]
    r0sq = x0**2 + y0**2

    if atmos is None:
        atmos = Atmos(n_gridpts, z_prop, delta[0], delta[-1],
                      screen_method_name='vacuum')
    else:
        raise Exception("Only vacuum propagation is implemented so far. "
                        + "Remove atmos input.")

    Q1 = np.exp(1j * k / 2 * (1 - m[0]) / delta_z[0] * r0sq)
    field_in = field_in * Q1

    for (dx, dx_ratio, dz, phz) in zip(delta, m, delta_z, atmos):

        deltaf = 1 / (n_gridpts * dx)

        f_x = nx * deltaf
        f_y = ny * deltaf
        fsq = f_x**2 + f_y**2

        Q2 = np.exp(-1j * pi**2 * 2 * dz / dx_ratio / k * fsq)

        field_in = (boundary_supergaussian
                    * ift2(Q2 * ft2(field_in / dx_ratio
                                    * np.exp(phz), dx), deltaf))

    xn = nx * delta[-1]
    yn = ny * delta[-1]

    rnsq = xn**2 + yn**2

    Q3 = np.exp(1j * k / 2 * (m[-1] - 1) / (m[-1] * delta_z[-1]) * rnsq)

    Uout = Q3 * field_in * np.exp(atmos[-1])

    return Uout


def ft2(g, dx):
    """ Schmidt implementation of DFT2

    TODO This needs better documentation.
    Why this instead of np version of fft2?
    """

    G = fftshift(fft2(fftshift(g))) * dx**2

    return G


def ift2(G, df):
    """ Schmidt implementation of DIFT2

    TODO This needs better documentation for the same reason as ft2.
    """

    N = len(G)
    g = ifftshift(ifft2(ifftshift(G))) * (N * df)**2

    return g


def super_gaussian_boundary(rad, width):
    return exp(-(rad / width)**16.0)
