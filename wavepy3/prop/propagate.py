import numpy as np
from numpy import (exp, pi, meshgrid, linspace, sqrt)
from numpy.fft import (fft2, fftshift, ifft2, ifftshift)
from ..atmos import Atmos


def split_step(Uin, wvl, delta1, deltan, z, atmos=None):

    N = len(Uin)
    nx = np.linspace(-N / 2, N / 2, N)
    nx, ny = np.meshgrid(nx, nx)

    k = 2 * pi / wvl

    nsq = nx**2 + ny**2
    w = 0.47 * N
    sg = np.exp(-nsq**8 / w**16)

    n = len(z)

    delta_z = z[1:] - z[:-1]

    alpha = z / z[-1]
    delta = (1 - alpha) * delta1 + alpha * deltan
    m = delta[1:] / delta[:-1]
    x1 = nx * delta[0]
    y1 = ny * delta[0]
    r1sq = x1**2 + y1**2

    if atmos is None:
        atmos = Atmos(N, z, delta[0], delta[-1], screen_method_name='vacuum')
    else:
        raise Exception("Only vacuum propagation is implemented so far. "
                        + "Remove atmos input.")

    Q1 = np.exp(1j * k / 2 * (1 - m[0]) / delta_z[0] * r1sq)
    Uin = Uin * Q1

    for idx in range(n - 1):

        deltaf = 1 / (N * delta[idx])
        fX = nx * deltaf
        fY = ny * deltaf
        fsq = fX**2 + fY**2

        Dz = delta_z[idx]

        Q2 = np.exp(-1j * pi**2 * 2 * Dz / m[idx] / k * fsq)

        Uin = (sg * np.exp(atmos[idx])
               * ift2(Q2
                      * ft2(Uin / m[idx], delta[idx]), deltaf))

    xn = nx * delta[-1]
    yn = ny * delta[-1]

    rnsq = xn**2 + yn**2

    Q3 = np.exp(1j * k / 2 * (m[-1] - 1) / (m[-1] * Dz) * rnsq)

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

    N = len(G)
    g = ifftshift(ifft2(ifftshift(G))) * (N * df)**2

    return g


def super_gaussian_boundary(rad, width):
    return exp(-(rad / width)**16.0)
