import mwatershed as mws
import numpy as np
import sys
import zarr
from scipy.ndimage import gaussian_filter
from funlib.persistence.arrays import open_ds, prepare_ds
from funlib.geometry import Roi, Coordinate
from skimage.morphology import remove_small_objects

def watershed(
        affs,
        neighborhood,
        bias,
):

    # add random noise
    random_noise = np.random.randn(*affs.shape) * 0.01

    # # add smoothed affs, to solve a similar issue to the random noise. We want to bias
    # # towards processing the central regions of objects first.
    # logging.info("Smoothing affs")
    smoothed_affs: np.ndarray = (
             gaussian_filter(affs, sigma=(0, 1, 2, 2))
             - 0.5
    ) * 0.05

    affs = (affs + random_noise + smoothed_affs).astype(np.float32)
    affs = np.clip(affs, 0.0, 1.0)

    print("Running mutex watershed..")
    pred_labels = mws.agglom(
        np.array(
            [
                affs[i] + bias[i] for i in range(len(neighborhood))
            ]
        ).astype(np.float64),
        neighborhood,
    )

    return pred_labels.astype(np.uint32)

if __name__ == "__main__":

    in_zarr = sys.argv[1] # path to zarr!
    affs_ds = sys.argv[2] # name of affs dataset inside zarr
    out_seg_ds = sys.argv[3] # name of output segmentation

    #neighborhood = [[1,0,0], [0,1,0], [0,0,1]]
    #bias = [-0.9, -0.9, -0.9]

    neighborhood = [[1,0,0], [0,1,0], [0,0,1]] #, [2,0,0], [0,5,0], [0,0,5]]
    bias = [-0.99, -0.99, -0.99] #, -0.5, -0.5, 0.5]

    # open
    affs = open_ds(f'{in_zarr}/{affs_ds}')
    roi = Roi(affs.roi.shape // 4, affs.roi.shape // 2).snap_to_grid(affs.voxel_size)
    #roi = affs.roi
    #print(roi, affs.roi)
    affs_array = affs.to_ndarray(roi)[:3] / 255.0
    print(np.max(affs_array),np.min(affs_array))

    # watershed
    seg_array = watershed(affs_array, neighborhood, bias)

    print("Filtering small objects..")
    # remove small objects
    filtered_seg = remove_small_objects(
        seg_array.astype(np.uint32), min_size=20, connectivity=1
    )

    print("Writing..")
    out = zarr.open(in_zarr,'a')
    out[out_seg_ds] = filtered_seg.astype(np.uint32)
    out[out_seg_ds].attrs['offset'] = roi.offset
    out[out_seg_ds].attrs['voxel_size'] = affs.voxel_size
    out[out_seg_ds].attrs['axis_names'] = ["z","y","x"]
    out[out_seg_ds].attrs['units'] = ["nm","nm","nm"]

    # prepare
#    out_seg = prepare_ds(
#        store = f'{in_zarr}/{out_seg_ds}',
#        shape = seg_array.shape,
#        offset = roi.offset,
#        voxel_size = affs.voxel_size,
#        dtype = np.uint32,
#        axis_names= ["z","y","x"],
#        units = ["nm","nm","nm"]
#    )
#
#    # write
#    out_seg[roi] = seg_array.astype(np.uint32)
