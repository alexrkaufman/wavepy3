import numpy as np
from numpy.random import default_rng
from .utils import ft2, ift2

rng = default_rng()

pi = np.pi


class Atmos:
    """Object representing Atmosphere

    TODO improve documentation
    """

    def __init__(self, n_gridpts, dx_sampling, **settings):
        '''
        N - number of grid points
            right now we only do square grids
        z - a list containing the locations of each phase screen [m]
        dx - a list containing the grid spacing of each phase screen [m]
        atmos_parms - kwargs defining necessary atmospheric parameters
         - this should be formatted as follows
        atmos_parms = {'screen_method': method, 'psd': psd,
                       'parms': {dict of parms}}
        '''

        # there should be a check on atmos params based on psd and
        # screen_method to make sure the user has defined all the
        # appropriate values

        self.n_gridpts = n_gridpts
        self.dx_sampling = dx_sampling
        self.screen_method = settings['screen_method']

        try:
            self.atmos_parms = settings['atmos_parms']
        except KeyError:
            self.atmos_parms = {}

        try:
            self.psd_type = settings['psd']
        except KeyError:
            self.psd_type = None

        self.screen = [self.phase_screen(n_gridpts, **self.atmos_parms)
                       for i in range(len(dx_sampling))]

    def psd(self, **atmos_parms):
        ''' psd function

        The psd function is meant to alias whichever psd someone wants to use.
        The idea is that this can be swapped out at will and we wont have to
        change other code.
        '''
        # TODO spin psds out into their own files (way way down the line)
        psd_dict = {
            'vonKarman': self.__vonKarman_psd,
            'wavepy_og': self.__wavepy_og,
        }

        psdfn = psd_dict[self.psd_type]

        return psdfn(**atmos_parms)

    def phase_screen(self, n_gridpts, **atmos_parms):
        ''' Phase screen calculation

        This should be the generic top level function similar to psd.

        TODO Improve documentation
        TODO Make this generic like the psd function
        '''
        screen_method_dict = {
            'ft_sh': self.ft_sh_phase_screen,
            'ft': self.ft_phase_screen,
            'vacuum': self.vacuum
        }

        phase_screenfn = screen_method_dict[self.screen_method]

        return phase_screenfn(n_gridpts, **atmos_parms)

    def ft_sh_phase_screen(self, N, dx, r0, L0=float('inf'), l0=0):
        ''' phase screen with subharmonic methods

        This computes phase screens including subharmonics.

        TODO improve documentation
        '''

        phz_hi = self.ft_phase_screen(N, dx, r0, L0, l0)
        phz_lo = self.ft_sh(N, dx, r0, L0, l0)

        return phz_hi + phz_lo

    def ft_phase_screen(self, N, dx, r0, L0=float('inf'), l0=0):
        """ft phase screen

        fourier transform phase screen without subharmonics
        on their own these do not do a good job approximating atmosphere

        TODO improve documentation
        """

        df = 1 / (N * dx)  # Frequency grid spacing.
        fx = np.linspace(-N / 2, N / 2, N, endpoint=False) * df

        # create frequency grid and compute frequency radii
        fx, fy = np.meshgrid(fx, fx)
        f = np.sqrt(fx**2 + fy**2)

        f0 = 1 / L0

        try:
            fm = 5.92 / l0 / (2 * pi)
        except ZeroDivisionError:
            fm = float('inf')

        # setup PSD
        psd_phi = self.psd(r0, f, fm, f0)
        psd_phi[N // 2, N // 2] = 0

        # get random Fourier coefficients
        cn = ((rng.standard_normal([N, N]) + 1j * rng.standard_normal([N, N]))
              * np.sqrt(psd_phi) * df)

        phz = np.real(ift2(cn, 1))

        return phz

    def ft_sh(self, N, dx, r0, L0=float('inf'), l0=0):
        ''' ft_sh

        subharmonics for fourier transform phase screens

        TODO Improve documentation
        '''

        D = N * dx

        x = np.linspace(-D/2, D/2-dx, N)
        x, y = np.meshgrid(x, x)

        phz_lo = np.zeros([N, N])

        for p in [1, 2, 3]:
            df = 1 / (D * 3**p)

            fx = np.linspace(-1, 1, 3)
            fx, fy = np.meshgrid(fx, fx)

            f = np.sqrt(fx**2 + fy**2)
            f0 = 1 / L0

            try:
                fm = 5.92 / l0 / (2 * pi)
            except ZeroDivisionError:
                fm = float('inf')

            psd_phi = self.psd(r0, f, fm, f0)
            psd_phi[1, 1] = 0

            cn = (complex(rng.normal(3), rng.normal(3))
                  * np.sqrt(psd_phi) * df)

            sub_harmonics = np.zeros([N, N])

            for i in range(3):
                for j in range(3):
                    sub_harmonics = (sub_harmonics + cn[i, j]
                                     * np.exp(1j * 2 * pi
                                              * (fx[i, j] * x + fy[i, j] * y)))

            phz_lo = phz_lo + sub_harmonics

        phz_lo = np.real(phz_lo) - np.mean(np.real(phz_lo))
        return phz_lo

    def __get_derived_parms(self, z, wvl, Cn2):

        k = 2 * pi / wvl

        # This is the plane wave r0
        r0_pw = (0.423 * k**2 * Cn2 * z) ** (-3.0 / 5.0)
        print(r0_pw)

        return {
            'r0': r0_pw
            # 'theta0': theta0,
            # 'rytov_sq': rytov_sq
        }

    def __wavepy_og(self):
        # TODO make this do what wavepy did originally
        # it may require updating the ft_phase_screen and _ft_sh methods
        # and its possible that this approach has all the same params and
        # then some so we could just set some of the parameters for the
        # von karman psd

        # extract params from info

        # create phase screens
        pass

    def vacuum(self, *args):
        return np.ones([self.n_gridpts, self.n_gridpts])

    @staticmethod
    def __vonKarman_psd(r0, f, fm, f0):
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

    def __getitem__(self, key):
        return self.screen[key]
