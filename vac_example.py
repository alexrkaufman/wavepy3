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

    x_0 = np.linspace(-N / 2, N / 2, N, endpoint=False) * dx_0
    x_0, y_0 = np.meshgrid(x_0, x_0)

    x_n = np.linspace(-N / 2, N / 2, N, endpoint=False) * dx_n
    x_n, y_n = np.meshgrid(x_n, x_n)

    field_in = rect(x_0 / D_0) * rect(y_0 / D_0)
    field_out = wp.propagate(field_in, wvl, dx_0, dx_n, z)

    I_in = np.absolute(field_in)**2
    I_out = np.absolute(field_out)**2

    field_analytic = wp.analytic.fresnel_prop_square(x_n[N // 2, :], 0, D_0,
                                                     wvl, z[-1])
    I_analytic = np.absolute(field_analytic)**2

    fig, ax = plt.subplots(2, 2) #creates tuple fig#, [[ax1, ax2], [ax3 ,ax4]]
    ax[0][0].imshow(I_in)
    ax[0][1].imshow(I_out)

    sim_out = I_out[:, N // 2]
    analytic_out = I_analytic
    ax[1][0].scatter(x_n[N // 2, :], sim_out / np.amax(sim_out),
                     marker='x', color='k',)
    ax[1][0].plot(x_n[N // 2, :], analytic_out / np.amax(analytic_out))
    ax[1][1].scatter(x_n[N // 2, :], np.angle(field_out[:, N // 2]),
                     marker='x', color='k')
    ax[1][1].plot(x_n[N // 2, :], np.angle(field_analytic))
    plt.show()


def rect(x, D=1):
    x = np.abs(x)

    rect = np.ones(x.shape)
    rect[x > (D / 2)] = 0
    rect[x == (D / 2)] = 0.5

    return rect


if __name__ == '__main__':
    main()
