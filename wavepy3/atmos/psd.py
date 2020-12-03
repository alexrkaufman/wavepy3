"""
A set of power spectral density functions.
These functions all exist to set up functions
of coherence diameter and frequency that can be passed
to phase screen methods.

Functions here should only return functions of coherence
diameter and frequency so they can be used generally in any
phase screen generation method.

TODO Implement the Tatarskii spectrum. [1]
TODO implement the exponential spectrum [1]
"""
import numpy as np


def modified_vonkarman(L0, l0):
    """modified_vonKarman

    This is the modified von Karman PSD function [1].
    The von Karman PSD is equivalent to fm = 'inf'.
    The Kolmogorov PSD is equivalent to fm = 'inf' and f0 = 0.

    :param L0: outer length scale (average size of large eddies)
    :param l0: inner length scale (average size of smallest eddies)

    :returns: A psd function of r0 and frequency.

    TODO Cite the paper where this function was first defined as [1].
    .. [1] ???
    .. [2] L. C. Andrews and R. L. Phillips, Laser beam propagation
       through random media, 2nd ed. Bellingham, Wash: SPIE Press, 2005.

    """

    f0 = 1 / L0
    try:
        fm = 5.92 / l0 / (2 * np.pi)
    except ZeroDivisionError:
        fm = float('inf')

    def fun(r0, f):
        """

        :param r0:
        :param f:

        """

        if fm == float('inf'):
            f_ratio = 0
        else:
            f_ratio = f / fm

        psd_phi = (0.023 * r0**(-5/3) * np.exp(-f_ratio**2)
                   / (f**2 + f0**2)**(11/6))

        N = len(psd_phi)

        psd_phi[N // 2, N // 2] = 0

        return psd_phi

    return fun


def vonkarman(L0):
    """vonKarman

    wrapper for modified_vonKarman w/ l0=0

    :param L0: The outer length scale. (average size of large eddies)

    :returns: a function of coherence diameter (r0) and frequency (f)

    """
    return modified_vonkarman(L0, l0=0)


def kolmogorov():
    """kolmogorov

    This returns the kolmogorov psd function.
    It is a wrapper for modified_vonKarman w/ L0 = inf and l0 = 0.

    :returns: a function of coherence diameter and frequency describing
    the kolmogorov power spectral density.

    """
    return modified_vonkarman(L0=float('inf'), l0=0)

# I do not know the proper name of this psd
def wavepy_og(L0, l0, c_one, theta, aniso, c, alpha):
    """The psd implemented in the original wavepy
    I do not know it's name or if it has one.
    Based on the original wavepy.validate() routine in WavePy
    this psd is able to generate screens with average structure
    function close to theory

    :param L0:
    :param l0:
    :param c_one:
    :param theta:
    :param aniso:
    :param c:
    :param alpha:

    """
    pass
