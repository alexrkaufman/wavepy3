"""
This should show how to propagate a signal through vacuum.
Will also display a plot so that users can see agreement between
theory and simulation for a square aperture source.
"""
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
    xy_0 = np.meshgrid(x_0, x_0)

    x_n = np.linspace(-N / 2, N / 2, N, endpoint=False) * dx_n

    field_in = wp.sources.square_ap(xy_0, D=D_0)

    field_out = wp.split_step(field_in, wvl, dx_0, dx_n, z)

    field_analytic = wp.analytic.fresnel_prop_square(x_n, 0, D_0,
                                                     wvl, z[-1])

    validation_plots(field_in, field_out, field_analytic, x_0, x_n)


def validation_plots(field_in, field_out, field_analytic, x_0, x_n):
    """
    This is just a helper function to provide a nice display
    of some plots that show the user agreement between the
    numerical and analytic results.

    :param field_in: The input field.
    :param field_out: The numerical output field.
    :param field_analytic: A 1D slice of the analytic solution.
    :param x_0: A 1D array of the object plane.
    :param x_n: A 1D array of the x and y axes in the terminal plane.

    :returns: Nothing.
    """

    N = len(field_in)

    I_in = np.absolute(field_in)**2
    I_out = np.absolute(field_out)**2

    I_analytic = np.absolute(field_analytic)**2

    fig, ax = plt.subplots(2, 2)
    ax[0][0].imshow(I_in, extent=[x_0[0], x_0[-1], x_0[0], x_0[-1]])
    ax[0][1].imshow(I_out)

    sim_out = I_out[:, N // 2]
    analytic_out = I_analytic
    ax[1][0].scatter(x_n, sim_out / np.amax(sim_out),
                     marker='x', color='k',)
    ax[1][0].plot(x_n, analytic_out / np.amax(analytic_out))
    ax[1][1].scatter(x_n, np.angle(field_out[:, N // 2]),
                     marker='x', color='k')
    ax[1][1].plot(x_n, np.angle(field_analytic))

    plt.show()


if __name__ == '__main__':
    main()
