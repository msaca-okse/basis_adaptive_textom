import os
import ImageD11.cImageD11
import ImageD11.unitcell
import numpy as np

from .paths import MASKFILE, PROCESS, PARAMETERS_AL1050

import ImageD11.parameters


_pars = ImageD11.parameters.read_par_file(PARAMETERS_AL1050).parameters
LATTICE_PARAMETERS = [  _pars["cell__a"],
                        _pars["cell__b"],
                        _pars["cell__c"],
                        _pars["cell_alpha"],
                        _pars["cell_beta"],
                        _pars["cell_gamma"]]
SYMMETRY = _pars["cell_lattice_[P,A,B,C,I,F,R]"]
UNITCELL = ImageD11.unitcell.unitcell(LATTICE_PARAMETERS, SYMMETRY)



# defines which h5 file corresponds to which im_x range (i.e dty range)
SCANRANGES = {
    "scan-0048.h5": (-0.75, -0.59),
    "scan-0049.h5": (-0.61, -0.45),
    "scan-0050.h5": (-0.47, -0.31),
    "scan-0051.h5": (-0.33, -0.17),
    "scan-0052.h5": (-0.19, -0.03),
    "scan-0053.h5": (-0.05, 0.11),
    "scan-0054.h5": (0.09, 0.25),
    "scan-0055.h5": (0.23, 0.39),
    "scan-0056.h5": (0.37, 0.53),
    "scan-0057.h5": (0.51, 0.67),
    "scan-0058.h5": (0.65, 0.75),
}

DETECTOR_O11 = "-1"
PILATUS_PIXELSIZE = 172  # microns
PILATUS_MASK = np.load(MASKFILE)
OMEGASTEP = 0.11996  # degrees
DTYSTEP = 0.01  # mm
OMEGAMAX = 360.12  # degrees
OMEGAMIN = 0.0  # degrees
DTYMAX = 0.75  # mm
DTYMIN = -0.75  # mm
NUMBER_OF_OMEGA_STEPS = 3003
NUMBER_OF_OMEGA_STEPS_IN_360_RANGE = 3002
NUMBER_OF_DTY_STEPS = 151
PILATUS_SHAPE = (1679, 1475)
DTYPOSITIONS = np.linspace(DTYMIN, DTYMAX, NUMBER_OF_DTY_STEPS) # mm
DTYPRECISION = 3

def load_rotation_axis_dty_shift():
    path = os.path.join(PROCESS, 'al1050_15pct_center_slice/rotation_axis_com.npy')
    if os.path.isfile(path):
        return float(np.load(path))
    else:
        return None

ROTATION_AXIS_DTY_SHIFT = load_rotation_axis_dty_shift()


PEAK_FEATURE_TITLES = [
    "s_1",
    "s_I",
    "s_I2",
    "s_fI",
    "s_ffI",
    "s_sI",
    "s_ssI",
    "s_sfI",
    "s_oI",
    "s_ooI",
    "s_soI",
    "s_foI",
    "mx_I",
    "mx_I_f",
    "mx_I_s",
    "mx_I_o",
    "bb_mx_f",
    "bb_mx_s",
    "bb_mx_o",
    "bb_mn_f",
    "bb_mn_s",
    "bb_mn_o",
    "avg_i",
    "f_raw",
    "s_raw",
    "o_raw",
    "m_ss",
    "m_ff",
    "m_oo",
    "m_sf",
    "m_so",
    "m_fo",
]

PEAK_FEATURE_COLS = [getattr(ImageD11.cImageD11, p) for p in PEAK_FEATURE_TITLES]

if __name__ == "__main__":
    pass
