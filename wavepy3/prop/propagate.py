import matplotlib.pyplot as plt

from numpy import (exp, pi, meshgrid, linspace, sqrt)
from numpy.fft import (fft2, fftshift, ifft2, ifftshift)
from ..atmos import Atmos

def split_step(field_in, wvl, dx_0, dx_n, proplocs, atmos=None):

    n_gridpts = len(field_in)
    nx = linspace(-n_gridpts/2, n_gridpts/2, n_gridpts)
    nx, ny = meshgrid(nx, nx)

    k = 2 * pi / wvl

    # generate sgb
    sgb = super_gaussian_boundary(sqrt(nx**2 + ny**2), 0.55 * n_gridpts)

    n_prop = len(proplocs)
    dz_prop = proplocs[1:] - proplocs[:-1]
    prop_frac = proplocs / proplocs[-1]

    sampling = (dx_n - dx_0) * prop_frac + dx_0
    samplingratio = sampling[1:] / sampling[:-1]

    if atmos is None:
        atmos = Atmos(n_gridpts, sampling, screen_method='vacuum')
    else:
        raise Exception("Only vacuum propagation is implemented so far. "
                        + "Remove atmos input.")

    r_0 = sqrt(nx**2 + ny**2) * dx_0

    #  Initial Propagation from source plane to first screen location
    Q1 = exp(1j * (k / (2*dz_prop[0]))
                * (1 - samplingratio[0])) * (r_0**2)

    Uin = field_in * Q1 * exp(1j * atmos[0])

    for (dz, dx, dx_ratio, phz) in zip(
            dz_prop, sampling, samplingratio, atmos):

        UinSpec = ft2(Uin / dx_ratio, dx)

        # Set spatial frequencies at propagation plane
        deltaf = 1/(n_gridpts * dx)
        fX = nx * deltaf
        fY = ny * deltaf
        fsq = fX**2 + fY**2

        # Quadratic Phase Factor
        Q2 = exp(-1j * pi * wvl * dz * fsq / dx_ratio)

        Uin = ift2(Q2 * UinSpec, deltaf)

        Uin = Uin * sgb * exp(1j * phz)

    r_n = sqrt(nx**2 + ny**2) * dx_n

    Q3 = exp(1j * (k / (2 * dz_prop[-1]))
             * (samplingratio[-1] - 1) / samplingratio[-1]
             * r_n**2)

    Uout = Q3 * Uin

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

    N = G.shape[0]
    g = ifftshift(ifft2(ifftshift(G))) * (N * df)**2

    return g


def super_gaussian_boundary(rad, width):
    return exp(-(rad / width)**16.0)
