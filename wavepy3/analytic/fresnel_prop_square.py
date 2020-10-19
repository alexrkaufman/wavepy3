from numpy import sqrt
from scipy.special import fresnel

def fresnel_prop_square(x_n, y_n, D_0, wvl, Z):
    N_F = (D_0)**2 / (wvl * Z)

    X = x_n / sqrt(wvl * Z)
    Y = y_n / sqrt(wvl * Z)

    alpha1 = -sqrt(2) * (sqrt(N_F) + X)
    alpha2 = sqrt(2) * (sqrt(N_F) - X)
    beta1 = -sqrt(2) * (sqrt(N_F) + Y)
    beta2 = sqrt(2) * (sqrt(N_F) - Y)

    ca1, sa1 = fresnel(alpha1)
    ca2, sa2 = fresnel(alpha2)
    cb1, sb1 = fresnel(beta1)
    cb2, sb2 = fresnel(beta2)

    return (1 / (2 * 1j)
            * ((ca2 - ca1) + 1j * (sa2 - sa1))
            * ((cb2 - cb1) + 1j * (sb2 - sb1)))
