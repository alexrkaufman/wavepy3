# This is how easy it should be to propagate a signal
import wavepy3 as wp
import numpy as np
import matplotlib.pyplot as plt


def main():

    D_0 = 2e-3
    D_n = 6e-3
    wvl = 1e-6
    z = 2
    dx_0 = D_0 / 30
    dx_n = D_n / 30
    N = 128
    n_prop = 5
    z = z * np.linspace(0, 1, n_prop + 1)

    x = np.linspace(-N * dx_0/2, N * dx_0/2, N)
    x, y = np.meshgrid(x, x)

    field_in = rect(x / D_0) * rect(y / D_0)
    field_out = wp.propagate(field_in, wvl, dx_0, dx_n, z)

    I_in = np.absolute(field_in)**2
    I_out = np.absolute(field_out)**2

    field_analytic = wp.analytic.fresnel_prop_square(x[64, :], 0, 30 * dx_0,
                                                     wvl, z[-1])
    I_analytic = np.absolute(field_analytic)**2

    fig, ax = plt.subplots(1, 2)
    ax[0].imshow(I_in)
    ax[1].imshow(I_out)
    plt.show()

    fig, ax = plt.subplots(1, 2)
    sim_out = I_out[:, N // 2]
    analytic_out = I_analytic
    ax[0].plot(x[N // 2, :], sim_out / np.amax(sim_out))
    ax[0].plot(x[N // 2, :], analytic_out / np.amax(analytic_out))
    ax[1].plot(x[N // 2, :], np.angle(field_out[:, N // 2]))
    ax[1].plot(x[N // 2, :], np.angle(field_analytic))
    plt.show()


def rect(x):
    return (np.abs(x) < 1).astype(int)


if __name__ == '__main__':
    main()
