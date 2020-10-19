# atmos_test
# This script is meant to be a test to display graphs similar
# to the schmidt validation graphs for atmospheric conditions.
# create a phase screen and plot it

import matplotlib.pyplot as plt
import wavepy3 as wp

D = 2
N = 256
dx = D / N

atmos_parms = {
    'psd': 'vacuum',
    'screen_method': 'vacuum',
    'Cn2': 1e-16,
    'l0': 0.01,
    'L0': 100,
    'wvl': 1e-6
}

z = [0, 50e3]

a1 = wp.Atmos(256, z, [dx, dx], **atmos_parms)

for screen in a1.screens:
    plt.imshow(screen)
    plt.colorbar()
    plt.show()
