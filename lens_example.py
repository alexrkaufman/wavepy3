import wavepy3 as wp3
import numpy as np
import matplotlib.pyplot as plt

def main():
    D_0 = 3e-3
    wvl = 1.55e-9
    N = 128
    focal_length = 25e-3
    dx_0 = D_0 / 30
    n_prop = 5
    z = np.abs(focal_length) * np.linspace(0, 1, n_prop + 1)
    waist_beam = D_0  # mm

    x_0 = np.linspace(-N / 2, N / 2, N, endpoint=False) * dx_0
    x_mesh, y_mesh = np.meshgrid(x_0, x_0)

    lens_pos = wp3.ThinLens(0, focal_length)

    field_in = np.exp(-(x_mesh**2 + y_mesh**2) / waist_beam**2)

    field_atlens = wp3.split_step(field_in, wvl, dx_0, dx_0, z)
    field_afterlens = lens_pos.apply(field_atlens, wvl, x_mesh, y_mesh)
    field_out_pos = wp3.split_step(field_afterlens, wvl, dx_0, dx_0, z)

    lens_neg = wp3.ThinLens(0, -focal_length)
    field_atlens = wp3.split_step(field_in, wvl, dx_0, dx_0, z)
    field_afterlens = lens_neg.apply(field_atlens, wvl, x_mesh, y_mesh)
    field_out_neg = wp3.split_step(field_afterlens, wvl, dx_0, dx_0, z)

    fig, ax = plt.subplots(5, 3)
    ax[0][0].imshow(np.abs(field_in)**2)
    ax[0][1].plot(x_0, np.abs(field_in[64, :])**2)
    ax[0][2].plot(x_0, np.angle(field_in[64, :]))
    ax[1][0].imshow(np.abs(field_atlens)**2)
    ax[1][1].plot(x_0, np.abs(field_atlens[64, :])**2)
    ax[1][2].plot(x_0, np.angle(field_atlens[64, :]))
    ax[2][0].imshow(np.abs(field_afterlens)**2)
    ax[2][1].plot(x_0, np.abs(field_afterlens[64, :])**2)
    ax[2][2].plot(x_0, np.angle(field_afterlens[64, :]))
    ax[3][0].imshow(np.abs(field_out_pos)**2)
    ax[3][1].plot(x_0, np.abs(field_out_pos[64, :])**2)
    ax[3][2].plot(x_0, np.angle(field_out_pos[64, :]))
    ax[4][0].imshow(np.abs(field_out_neg)**2)
    ax[4][1].plot(x_0, np.abs(field_out_neg[64, :])**2)
    ax[4][2].plot(x_0, np.angle(field_out_neg[64, :]))

    plt.show()


if __name__ == '__main__':
    main()
