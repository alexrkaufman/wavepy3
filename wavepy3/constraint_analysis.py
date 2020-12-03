"""
This module provides a quick interface for plotting the constraint
analysis graph described in [1].

TODO Improve documentation.
     > Primarily specific reasons for why these constraints are defined
       the way they are should be provided.

.. [1] J. D. Schmidt, Numerical Simulation of Optical Wave Propagation
   With Examples in Matlab. SPIE Press, 2010.

"""
import numpy as np
import matplotlib.pyplot as plt


def constraint_analysis(wvl, z, D_0, D_n, dx_0=np.linspace(1e-9, 0.5e-3, 1000),
                        R=None, r_0=float('inf'), c=2):
    """
    This function generates a plot with applicable constraints for simulations
    using the split_step method as defined in [1].

    :param wvl: The wavelength to be used
    :param z: the maximum propagation distance.
    :param D_0: The pupil diameter in the initial plane.
    :param D_n: The pupil diameter in the terminal plane.
    :param dx_0: A range of potential grid spacing values.
    (Default value = np.linspace(1e-9, 0.5e-3, 1000))
    :param R:  (Default value = None)
    :param r_0: coherence diameter  (Default value = float('inf'))
    :param c: sort of a fudge factor  (Default value = 2)

    :returns: nothing. shows a plot of the constraints to help you decide what
    parameters to use to get good results.

    """

    if R is None:
        R = z

    if r_0 is not float('inf'):
        D_0 = D_0 + c * wvl * z / r_0
        D_n = D_n + c * wvl * z / r_0

    constraint1 = _constraint1(wvl, z, D_0, D_n)
    constraint2 = _constraint2(wvl, z, D_0, D_n)
    constraint3 = _constraint3(wvl, z, D_0, R)

    dx_n_max = constraint1(dx_0)
    c3_max, c3_min = constraint3(dx_0)

    dx_0, dx_n = np.meshgrid(dx_0, 2 * dx_0)
    log_nprop = np.log2(constraint2(dx_0, dx_n))

    plt.plot(dx_0[0, :], dx_n_max)
    plt.plot(dx_0[0, :], c3_max)
    plt.plot(dx_0[0, :], c3_min)
    plt.contour(dx_0, dx_n, log_nprop, list(range(9)))
    plt.xlim(0, np.amax(dx_0))
    plt.ylim(0, np.amax(dx_n))
    plt.show()


def _constraint1(wvl, z, D_0, D_n):
    """_constraint1

    This places an upper limit on dx_n.

    :param wvl: wavelength
    :param z: propagation distance
    :param D_0: pupil diameter in initial plane
    :param D_n: pupil diameter in terminal plane

    :returns: a function of grid spacing in the initial plane

    """
    def fun(dx_0):
        """

        :param dx_0:

        """
        return (wvl * z - D_n * dx_0) / D_0
    return fun


def _constraint2(wvl, z, D_0, D_n):
    """_constraint2

    This places a lower bound on the number of propagations.

    :param wvl: wavelength
    :param z: propagation distance
    :param D_0: pupil diameter in initial plane
    :param D_n: pupil diameter in terminal plane

    :returns: a function of initial and terminal plane grid spacings.
    """

    def fun(dx_0, dx_n):
        """

        :param dx_0:
        :param dx_n:

        """
        return D_0 / (2 * dx_0) + D_n / (2 * dx_n) \
            + wvl * z / (2 * dx_0 * dx_n)
    return fun

def _constraint3(wvl, z, D_0, R):
    """_constraint3

    places additional upper and lower bounds on terminal plane grid spacing

    :param wvl: wavelength
    :param z: propagation distance
    :param D_0: pupil diameter in initial plane
    :param R: The radius of curvature of the wavefront.

    :returns: a function of the initial plane grid spacing
    """

    def c3_max(dx_0):
        """

        :param dx_0:

        """
        return (1 + z / R) * dx_0 + wvl * z / D_0

    def c3_min(dx_0):
        """

        :param dx_0:

        """
        return (1 + z / R) * dx_0 - wvl * z / D_0

    def getmaxandmin(dx_0):
        """

        :param dx_0:

        """
        return (c3_max(dx_0), c3_min(dx_0))

    return getmaxandmin
