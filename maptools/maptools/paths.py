import os

ROOT = "/dtu/3d-imaging-center/projects/2025_QIM_BlackBeauty/raw_data_extern/2025_Danmax_Al1050"
PROCESS = os.path.join(ROOT, "process")
RAW = os.path.join(ROOT, "raw")
CODE = os.path.join("/zhome/71/c/146676/texture_tomography")

OMEGAMOTOR = "/entry/measurement/tom_ry"
DTYMOTOR = "/entry/measurement/im_x"
FRAMES = "/entry/measurement/pilatus"
MASKFILE = os.path.join(PROCESS, "mask.npy")
PARAMETERS_AL1050 = os.path.join(PROCESS, "al1050_15pct_center_slice", "al1050.par")

# defines which h5 file with segmented peaks corresponds to which scan number
RAW_PEAKFILES = {
    "scan-0048": os.path.join(
        PROCESS, "al1050_15pct_center_slice", "segmented_peaks_scan-0048.h5"
    ),
    "scan-0049": os.path.join(
        PROCESS, "al1050_15pct_center_slice", "segmented_peaks_scan-0049.h5"
    ),
    "scan-0050": os.path.join(
        PROCESS, "al1050_15pct_center_slice", "segmented_peaks_scan-0050.h5"
    ),
    "scan-0051": os.path.join(
        PROCESS, "al1050_15pct_center_slice", "segmented_peaks_scan-0051.h5"
    ),
    "scan-0052": os.path.join(
        PROCESS, "al1050_15pct_center_slice", "segmented_peaks_scan-0052.h5"
    ),
    "scan-0053": os.path.join(
        PROCESS, "al1050_15pct_center_slice", "segmented_peaks_scan-0053.h5"
    ),
    "scan-0054": os.path.join(
        PROCESS, "al1050_15pct_center_slice", "segmented_peaks_scan-0054.h5"
    ),
    "scan-0055": os.path.join(
        PROCESS, "al1050_15pct_center_slice", "segmented_peaks_scan-0055.h5"
    ),
    "scan-0056": os.path.join(
        PROCESS, "al1050_15pct_center_slice", "segmented_peaks_scan-0056.h5"
    ),
    "scan-0057": os.path.join(
        PROCESS, "al1050_15pct_center_slice", "segmented_peaks_scan-0057.h5"
    ),
    "scan-0058": os.path.join(
        PROCESS, "al1050_15pct_center_slice", "segmented_peaks_scan-0058.h5"
    ),
}

if __name__ == "__main__":
    pass
