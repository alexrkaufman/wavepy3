import numpy as np


def modified_vonkarman(L0, l0):
    ''' modified_vonKarman
    - return psd function of r0 and f

    This is the modified von Karman PSD function.
    The von Karman PSD is equivalent to fm = 'inf'.
    The Kolmogorov PSD is equivalent to fm = 'inf' and f0 = 0.
    '''

    f0 = 1 / L0
    try:
        fm = 5.92 / l0 / (2 * np.pi)
    except ZeroDivisionError:
        fm = float('inf')

    def fun(r0, f):

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
    '''
    vonKarman psd
    - return psd function of r0 and f

    wrapper for modified_vonKarman w/ l0=0
    '''
    return modified_vonkarman(L0, l0=0)


def kolmogorov():
    '''
    kolmogorov psdfn
    - return psd function of r0 and f

    wrapper for modified_vonKarman w/ L0 = inf and l0 = 0
    '''
    return modified_vonkarman(L0=float('inf'), l0=0)

# I do not know the proper name of this psd
def wavepy_og(L0, l0, c_one, theta, aniso, c, alpha):
    '''
    The psd implemented in the original wavepy
    I do not know it's name or if it has one.
    Based on the original wavepy.validate() routine in WavePy
    this psd is able to generate screens with average structure
    function close to theory
    '''
    pass 
