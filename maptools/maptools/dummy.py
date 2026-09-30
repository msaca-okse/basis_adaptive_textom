import pickle
import numpy as np
import maptools

def load_dset(path):
    with open(path, "rb") as f:
        dset = pickle.load(f)
    return dset

class dset(object):
    """Minimal stand-in for ImageD11.sinograms.dataset.DataSet.

    Holds just what the ImageD11.sinograms.point_by_point module needs (omega, dty, the parameter file
    and the omega/dty binning), so that point-by-point indexing can run on a merged peak file without
    the per-scan HDF5 layout that a full DataSet expects.
    """

    def __init__(self, colf, parfile, h5path, savefile, ny, nomega):
        self.parfile = parfile
        self.omega = colf.omega.copy()
        self.dty = colf.dty.copy()
        self.icolfile = savefile + "_pbp.h5"
        self.path = savefile + "_dummy_dset.h5"
        self.shape = (ny, nomega)
        self.guessbins()

    def update_colfile_pars(self, cf, phase_name=None):  # adapted from ImageD11.sinograms.dataset
        """Load parameters and update geometry for colfile"""
        cf.parameters.loadparameters(self.parfile, phase_name=phase_name)
        cf.updateGeometry()

    def guessbins(self):  # adapted from ImageD11.sinograms.dataset
        ny, nomega = self.shape
        self.omin = maptools.constants.OMEGAMIN
        self.omax = maptools.constants.OMEGAMAX
        if (self.omax - self.omin) > 360:
            # multi-turn scan...
            self.omin = 0.0
            self.omax = 360.0
            self.omega_for_bins = self.omega % 360
        else:
            self.omega_for_bins = self.omega
        self.ostep = np.round( (self.omax - self.omin) / (nomega - 1), 5 )
        self.ymin = maptools.constants.DTYMIN
        self.ymax = maptools.constants.DTYMAX
        if ny > 1:
            self.ystep = (self.ymax - self.ymin) / (ny - 1)
        else:
            self.ystep = 1
        self.obincens = np.linspace(self.omin, self.omax, nomega)
        self.ybincens = np.linspace(self.ymin, self.ymax, ny)
        self.obinedges = np.linspace(
            self.omin - self.ostep / 2, self.omax + self.ostep / 2, nomega + 1
        )
        self.ybinedges = np.linspace(
            self.ymin - self.ystep / 2, self.ymax + self.ystep / 2, ny + 1
        )

    def save(self):
        with open(self.path, "wb") as f:
            pickle.dump(self, f, protocol=pickle.HIGHEST_PROTOCOL)
