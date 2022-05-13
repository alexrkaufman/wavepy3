import numpy as np
from numpy import pi


class Lens():
    '''
    TODO Should there be a thick lens?
    '''
    pass


class ThinLens(Lens):
    """
    Object representing a lens.


    This should represent a thin lens with a size and a focal length.
    TODO Improve documentation.
    """

    def __init__(self, diameter, focal_length):
        self.diameter = diameter
        self.focal_length = focal_length

    def _lens_phase(self, wvl, x_mesh, y_mesh):

        return -1j * pi * (x_mesh**2 + y_mesh**2) / (wvl * self.focal_length)

    def apply(self, field, wvl, x_mesh, y_mesh):

        t_lens = np.exp(self._lens_phase(wvl, x_mesh, y_mesh))

        return t_lens * field
