import numpy as np


def modified_vonKarman_psd(r0, f, fm, f0):
    ''' vonKarman_psd

    This is the modified von Karman PSD function.
    The von Karman PSD is equivalent to fm = 'inf'.
    The Kolmogorov PSD is equivalent to fm = 'inf' and f0 = 0.
    '''

    if fm == float('inf'):
        f_ratio = 0
    else:
        f_ratio = f / fm

    psd_phi = (0.023 * r0**(-5/3) * np.exp(-(f_ratio)**2)
               / (f**2 + f0**2)**(11/6))

    return psd_phi


def vonKarman_psd(r0, f, f0):
    return modified_vonKarman_psd(r0, f, float('inf'), f0)


def kolmogorov_psd(r0, f):
    return modified_vonKarman_psd(r0, f, float('inf'), 0)
