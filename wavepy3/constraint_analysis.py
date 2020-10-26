import numpy as np
import matplotlib.pyplot as plt

def constraint_analysis(wvl, z, D_0, D_n, dx_0=np.linspace(1e-9, 0.5e-3, 1000),
                        R=None, r_0=float('inf'), c=2):

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


# constraint 1 places an upper limit on dx_n
def _constraint1(wvl, z, D_0, D_n):
    def fun(dx_0):
        return (wvl * z - D_n * dx_0) / D_0
    return fun


# constraint 2 puts a lower bound on nprop
def _constraint2(wvl, z, D_0, D_n):
    def fun(dx_0, dx_n):
        return D_0 / (2 * dx_0) + D_n / (2 * dx_n) \
            + wvl * z / (2 * dx_0 * dx_n)
    return fun

# constraint 3 puts additional upper and lower bounds on dx_n

def _constraint3(wvl, z, D_0, R):

    def c3_max(dx_0):
        return (1 + z / R) * dx_0 + wvl * z / D_0

    def c3_min(dx_0):
        return (1 + z / R) * dx_0 - wvl * z / D_0

    def getmaxandmin(dx_0):
        return (c3_max(dx_0), c3_min(dx_0))

    return getmaxandmin
