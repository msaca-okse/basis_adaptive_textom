import pickle
import numpy as np
import maptools

def load_dset(path):
    with open(path, "rb") as f:
        dset = pickle.load(f)
    return dset

class dset(object):
    """This is a dummpy wrapper for the dataset object in ImageD11.sinograms.dataset
    allowing the use of the same functions in the ImageD11.sinograms.point_by_point module
    without having to setup h5 file paths and other things. This is a hack for running PBP
    when we only have a merged peak file.
    """

    def __init__(self, colf, parfile, h5path, savefile, ny, nomega):
        # self.icolfile
        self.parfile = parfile
        # self.dsfile
        self.omega = colf.omega.copy()
        self.dty = colf.dty.copy()
        self.icolfile = savefile + "_pbp.h5"
        self.path = savefile + "_dummy_dset.h5"

        # nysteps = len(scans)
        # ystep = h5glob[scans[1]]["eiger/frames/y"][0] - h5glob[scans[0]]["eiger/frames/y"][0]
        # y0 = h5glob[scans[len(scans)//2]]["eiger/frames/y"][0]
        # ymin = h5glob[scans[0]]["eiger/frames/y"][0]
        # nomsteps = len(h5glob[scans[0]]['eiger/frames/omega'])
        # omstep = h5glob[scans[0]]['eiger/frames/omega'][1] - h5glob[scans[0]]['eiger/frames/omega'][0]
        # ypositions = [h5glob[scan]['eiger/frames/y'][0] for scan in scans]

        self.shape = (ny, nomega)

        self.guessbins()

    def update_colfile_pars(
        self, cf, phase_name=None
    ):  # stolen from ImageD11.sinograms.dataset
        """Load parameters and update geometry for colfile"""
        cf.parameters.loadparameters(self.parfile, phase_name=phase_name)
        cf.updateGeometry()

    def guessbins(self):  # stolen from ImageD11.sinograms.dataset
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
        # values 0, 1, 2
        # shape = 3
        # step = 1
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
