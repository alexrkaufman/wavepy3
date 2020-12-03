# atmos_test
# Copied from the original wavepy and broken into
# a few functions for slight improvement of readability

# TODO make this more readable
# TODO Add axes to grid. ``

import numpy as np
import matplotlib.pyplot as plt
import wavepy3 as wp

def main():
    n_screens = 5

    D = 1
    N = 256
    dx = 5e-3
    z = np.linspace(0, 10e3, n_screens)
    r0scrn = 0.5 * D / 20

    atmos_parms_FT = {
        'psd_name': 'modified_vonKarman',
        'screen_method_name': 'ft',
        'r0': r0scrn,
        'l0': 0.01,
        'L0': 100,
        'wvl': 1e-6,
    }

    atmos_parms_SH = {
        'psd_name': 'modified_vonKarman',
        'screen_method_name': 'ft_sh',
        'n_subharm': 4,
        'r0': r0scrn,
        'l0': 0.01,
        'L0': 100,
        'wvl': 1e-6,
    }

    atmos_FT = wp.Atmos(N, z, dx, dx, **atmos_parms_FT)
    atmos_SH = wp.Atmos(N, z, dx, dx, **atmos_parms_SH)

    phz_FT = np.zeros((N, N))
    phz_FT_temp = phz_FT
    phz_SH = np.zeros((N, N))
    phz_SH_temp = phz_SH

    # Generating multiple phase screens
    for j in range(0, n_screens):
        phz_FT_temp = atmos_FT.screen[j]
        # using phase screens from ^ so that time isn't wasted generating
        # screens for the SubHarmonic case
        phz_SH_temp = atmos_SH.screen[j]

        phz_FT_temp = StructFunc(phz_FT_temp, D, N)
        phz_SH_temp = StructFunc(phz_SH_temp, D, N)
        phz_FT = phz_FT + phz_FT_temp
        phz_SH = phz_SH + phz_SH_temp

    # Averaging the runs and correct bin size
    phz_FT = phz_FT / n_screens
    phz_SH = phz_SH / n_screens
    m, n = np.shape(phz_FT)
    centerX = round(m / 2) + 1

    phz_FT_disp = np.ones(N // 2)
    phz_FT_disp = phz_FT[:, centerX]
    phz_SH_disp = np.ones(N // 2)
    phz_SH_disp = phz_SH[:, centerX]

    phz_FT_disp = phz_FT_disp[0:(N // 2)]
    phz_FT_disp = phz_FT_disp[::-1]
    phz_SH_disp = phz_SH_disp[0:(N // 2)]
    phz_SH_disp = phz_SH_disp[::-1]

    # array of values for normalized r to plot x-axis
    cent_dist = np.zeros(N // 2)
    r_size = (0.5 * D) / (0.5 * N)
    for i in range(0, (N // 2)):
        cent_dist[i] = (i * r_size) / (r0scrn)

    # Defining theoretical equation
    theory_val = np.zeros(N // 2)
    theory_val = 6.88 * (cent_dist)**(5.0/3.0)

    # Plotting 3 options,  with
    # blue=theory,  green=FT,  and red=SH in current order
    plt.plot(cent_dist, theory_val)
    plt.plot(cent_dist, phz_FT_disp)
    plt.plot(cent_dist, phz_SH_disp)
    plt.legend(['theory', 'FT', 'FT+SH'])
    plt.xlim((0, 10))
    plt.ylim((0, 400))
    plt.show()

def StructFunc(ph, D, N):
    mask = make_pupil(D / 4, D, N)
    delta = D / len(ph)

    N_size = np.shape(ph)
    ph = ph * mask

    P = np.fft.fftshift(np.fft.fft2(np.fft.fftshift(ph)))*(delta**2)
    S = np.fft.fftshift(np.fft.fft2(np.fft.fftshift(ph**2)))*(delta**2)
    W = np.fft.fftshift(np.fft.fft2(np.fft.fftshift(mask)))*(delta**2)
    delta_f = 1/(N_size[0]*delta)

    fft_size_a = np.shape(W*np.conjugate(W))
    w2 = (np.fft.ifftshift(
          np.fft.ifft2(
              np.fft.ifftshift(
                  W*np.conjugate(W))))
          * ((fft_size_a[0] * delta_f)**2))

    fft_size_b = np.shape(np.real(S * np.conjugate(W)) - np.abs(P)**2)
    D = 2 * (
            (np.fft.ifftshift(
                np.fft.ifft2(
                    np.fft.ifftshift(
                        np.real(S * np.conjugate(W))
                        - np.abs(P)**2))))
            * ((fft_size_b[0]*delta_f)**2))

    D = D/w2

    D = np.abs(D) * mask

    return D

def make_pupil(d, D, N):
    boundary1 = -d / 2
    boundary2 = d / 2
    A = np.linspace(boundary1, boundary2, N)
    A = np.array([A] * N)  # horizontal distance map created
    base = np.linspace(boundary1, boundary2, N)

    set_ones = np.ones(N)  # builds array of length N filled with ones
    B = np.array([set_ones] * N)

    for i in range(0,  len(base)):
        B[i] = B[i] * base[i]  # vertical distance map created

    A = A.reshape(N, N)
    B = B.reshape(N, N)  # arrays reshaped into matrices

    x_coord = A**2
    y_coord = B**2

    rad_dist = np.sqrt(x_coord + y_coord)  # define radial distance

    mask = []
    for row in rad_dist:
        for val in row:
            if val < d:
                mask.append(1.0)
            elif val > d:
                mask.append(0.0)
            elif val == d:
                mask.append(0.5)
                mask = np.array([mask])

    # mask created and reshaped into a matrix
    mask = np.reshape(mask, (N, N))

    return mask  # returns the pupil mask as the whole function's output


if __name__ == '__main__':
    main()
