import numpy as np
import wavepy3 as wp

wvl = 1e-6
z_max = 2
D_0 = 2e-3
D_n = 6e-3

dx_0 = np.linspace(1e-9, 6e-4, 1000)

wp.constraint_analysis(wvl, z_max, D_0, D_n, dx_0=dx_0)
