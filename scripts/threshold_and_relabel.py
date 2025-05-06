import zarr
import numpy as np
import sys
from skimage.measure import label
from skimage.filters import threshold_otsu
from funlib.persistence.arrays import open_ds, prepare_ds
from funlib.geometry import Roi, Coordinate


def thresh_relabel(pred, thresh="otsu"):

    if thresh == "otsu":
        threshold_value = threshold_otsu(pred)
    else:
        threshold_value = thresh*255

    binarized_array = (pred > threshold_value)#.astype(np.uint32)

    labeled_array = label(binarized_array)

    return labeled_array


if __name__ == "__main__":
    affs_file = sys.argv[1]
    affs_ds = sys.argv[2]
    seg_ds = sys.argv[3]
    
    try:
        thresh = float(sys.argv[3])
    except:
        thresh = "otsu"

    seg_file = affs_file

    # load affs
    affs = open_ds(f'{affs_file}/{affs_ds}', 'r')
    
    roi = affs.roi
    #roi = Roi(affs.roi.shape // 2, affs.roi.shape // 3).snap_to_grid(affs.voxel_size)
    affs_array = affs.to_ndarray(roi) / 255.0
    affs_array = np.mean(affs_array, axis=0)

    labels = thresh_relabel(affs_array, thresh).astype(np.uint32)

    print(labels.shape)

    out = zarr.open(seg_file,'a')
    out[seg_ds] = labels
    out[seg_ds].attrs['offset'] = roi.offset
    out[seg_ds].attrs['voxel_size'] = affs.voxel_size
    out[seg_ds].attrs['axis_names'] = ["z","y","x"]
    out[seg_ds].attrs['units'] = ["nm","nm","nm"]

    # # prepare
    # out_seg = prepare_ds(
    #     store = f'{seg_file}/{seg_ds}',
    #     shape = labels.shape,
    #     offset = roi.offset,
    #     voxel_size = affs.voxel_size,
    #     dtype = np.uint32,
    #     axis_names= ["z","y","x"],
    #     units = ["nm","nm","nm"]
    # )

    # # write
    # out_seg[roi] = labels
