import numpy as np

fft2 = np.fft.fft2
ifft2 = np.fft.ifft2
fftshift = np.fft.fftshift
ifftshift = np.fft.ifftshift


def cart2pol(x, y):

    r = np.sqrt(x**2 + y**2)
    phi = np.arctan2(y, x)
    return (r, phi)


def pol2cart(r, phi):

    x = r * np.cos(phi)
    y = r * np.sin(phi)
    return (x, y)


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

def super_gaussian_boundary(r_coord, N):

    # Construction of Super Gaussian Boundary
    rad = r_coord * (N)
    w = 0.55 * N
    sg = np.exp(-((rad / w)**16.0))

    return sg
