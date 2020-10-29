# atmos_test
# This script is meant to be a test to display graphs similar
# to the schmidt validation graphs for atmospheric conditions.
# create a phase screen and plot it

import matplotlib.pyplot as plt
import numpy as np
import wavepy3 as wp

D = 2
N = 256
dx = D / N
z = np.linspace(0, 1, 3)

atmos_parms = {
    'screen_method_name': 'vacuum',
}

a1 = wp.Atmos(256, z, dx, dx, **atmos_parms)

for screen in a1.screen:
    plt.imshow(screen)
    plt.colorbar()
    plt.show()

atmos_parms = {
    'screen_method_name': 'ft',
    'psd_name': 'modified_vonKarman',
    'r0': 0.2,
    'L0': 100,
    'l0': 0.01
}

a1 = wp.Atmos(256, z, dx, dx, **atmos_parms)

for screen in a1.screen:
    plt.imshow(screen)
    plt.colorbar()
    plt.show()
