import numpy as np

'''
These psd's return functions of frequency with the given parameters
'''

def modified_vonkarman(L0, l0):
    ''' vonKarman

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

        return psd_phi

    return fun


def vonkarman(L0):
    return modified_vonkarman(L0, l0=0)


def kolmogorov():
    return modified_vonkarman(L0=float('inf'), l0=0)

# I do not know the proper name of this psd
def wavepy_og():
    pass
