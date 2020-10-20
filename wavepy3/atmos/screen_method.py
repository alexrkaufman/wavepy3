import numpy as np
from numpy.random import default_rng
from numpy import pi
from . import utils

rng = default_rng()


def vacuum(n_gridpts):
    return np.ones([n_gridpts, n_gridpts])


def ft_sh_phase_screen(N, dx, n_subharm, r0, psd_fn):
    ''' phase screen with subharmonic methods

    This computes phase screens including subharmonics.

    TODO improve documentation
    '''

    phz_hi = ft_phase_screen(N, dx, r0, psd_fn)
    phz_lo = ft_sh(N, dx, n_subharm, r0, psd_fn)

    return phz_hi + phz_lo


def ft_phase_screen(N, dx, r0, psd):
    """ft phase screen

    fourier transform phase screen without subharmonics
    on their own these do not do a good job approximating atmosphere

    TODO improve documentation
    """

    df = 1 / (N * dx)  # Frequency grid spacing.
    fx = np.linspace(-N / 2, N / 2, N, endpoint=False) * df

    # create frequency grid and compute frequency radii
    fx, fy = np.meshgrid(fx, fx)
    f = np.sqrt(fx**2 + fy**2)

    # setup PSD
    psd_phi = psd(r0, f)
    psd_phi[N // 2, N // 2] = 0

    # get random Fourier coefficients
    cn = ((rng.standard_normal([N, N]) + 1j * rng.standard_normal([N, N]))
          * np.sqrt(psd_phi) * df)

    phz = np.real(utils.ift2(cn, 1))

    return phz


def ft_sh(N, dx, n_subharm, r0, psd):
    ''' ft_sh

    subharmonics for fourier transform phase screens

    TODO Improve documentation
    '''

    D = N * dx

    x = np.linspace(-D/2, D/2-dx, N)
    x, y = np.meshgrid(x, x)

    phz_lo = np.zeros([N, N])

    for p in range(1, n_subharm + 1):
        df = 1 / (D * 3**p)

        fx = np.linspace(-1, 1, 3)
        fx, fy = np.meshgrid(fx, fx)

        f = np.sqrt(fx**2 + fy**2)

        psd_phi = psd(r0, f)
        psd_phi[1, 1] = 0

        cn = (complex(rng.normal(3), rng.normal(3))
              * np.sqrt(psd_phi) * df)

        sub_harmonics = np.zeros([N, N])

        for i in range(3):
            for j in range(3):
                sub_harmonics = (sub_harmonics + cn[i, j]
                                 * np.exp(1j * 2 * pi
                                          * (fx[i, j] * x + fy[i, j] * y)))

        phz_lo = phz_lo + sub_harmonics

    phz_lo = np.real(phz_lo) - np.mean(np.real(phz_lo))
    return phz_lo
