import threading
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime

import h5py
import numpy as np
from ImageD11.frelon_peaksearch import worker
from tqdm import tqdm

from maptools import constants


def segment_frame(raw_image, options):
    # this one is for finding the params.
    image_worker = worker(**options)
    raw_image[raw_image < 0] = 0
    raw_image = raw_image.astype(np.uint32)
    goodpeaks = image_worker.peaksearch(img=raw_image, omega=0)
    fc, sc = goodpeaks[
        :, 23:25
    ].T  # 23 and 24 are the columns for fc and sc from blob properties
    return image_worker, fc, sc, len(fc)


def merge_dict(pks):
    for key in pks:
        pks[key] = np.concatenate(pks[key])
    return pks


def peaks_list_to_dict(feature_table, dty):
    peaks_2d_dict = {title: [] for title in constants.PEAK_FEATURE_TITLES}
    for feature in feature_table:
        for title, col in zip(
            constants.PEAK_FEATURE_TITLES, constants.PEAK_FEATURE_COLS
        ):
            peaks_2d_dict[title].append(feature[:, col])
    peaks_2d_dict = merge_dict(peaks_2d_dict)
    peaks_2d_dict["dty"] = (
        np.ones_like(peaks_2d_dict[constants.PEAK_FEATURE_TITLES[0]]) * dty
    )
    return peaks_2d_dict


_tls = threading.local()


def _get_worker(options):
    if not hasattr(_tls, "w"):
        _tls.w = worker(**options)
    return _tls.w


def _peaksearch_idx(idx, options, frames, omegas, mask):
    w = _get_worker(options)
    return w.peaksearch(img=frames[idx] * mask, omega=omegas[idx])


def segment_single_scan(
    frames,
    omegas,
    dty,
    start_index,
    end_index,
    options,
    max_workers=None,
):
    dty = float(dty)
    feature_table = []
    mask = options["maskfile"] == 0
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        it = ex.map(
            lambda i: _peaksearch_idx(
                i,
                options,
                frames,
                omegas,
                mask,
            ),
            range(start_index, end_index),
        )
        for ft in tqdm(it, total=end_index - start_index):
            feature_table.append(ft)
    return peaks_list_to_dict(feature_table, dty)


def write_pks_to_h5(
    path, key, data_dict, options, start, end, raw_file, overwrite=False
):
    with h5py.File(path, "a") as f:
        if key in f:
            if not overwrite:
                return
            del f[key]

        g = f.create_group(key)

        for name, arr in data_dict.items():
            g.create_dataset(name, data=np.asarray(arr))

        g.attrs["updated"] = datetime.now().isoformat()

        g.create_dataset("frame_start_index", data=start)
        g.create_dataset("frame_end_index", data=end)
        g.create_dataset("path_to_raw_frames", data=raw_file)

        sg = g.create_group("segmenter_params")
        for on, ov in options.items():
            if ov is None:
                continue
            if isinstance(ov, str):
                dt = h5py.string_dtype("utf-8")
                sg.create_dataset(on, data=np.array(ov, dtype=dt), dtype=dt)
            else:
                sg.create_dataset(on, data=ov)


if __name__ == "__main__":
    pass
