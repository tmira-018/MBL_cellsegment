import numpy as np
import sys
from funlib.persistence import open_ds, prepare_ds
from skimage.measure import regionprops


if __name__ == "__main__":

    seg_f = sys.argv[1] # path to seg zarr!
    seg_ds = sys.argv[2] # name of segmentation dataset inside zarr
    out_f = seg_f
    out_ds = sys.argv[3]
    # dust_filter = int(sys.argv[4]) # name of output segmentation


    seg = open_ds(f'{seg_f}/{seg_ds}', mode='r')


    seg_array = seg.to_ndarray(seg.roi)

    # if dust_filter > 0:
    #     print("Filtering small objects..")
    #     # remove small objects
    #     seg_array = remove_small_objects(
    #         seg_array.astype(np.uint32), min_size=dust_filter, connectivity=1
    #     )

    print("calculating sizes...")
    regions = regionprops(seg_array)
    label_size_dict = {r.label: r.area for r in regions}

    mean = np.mean(list(label_size_dict.values()))
    std_dev = np.std(list(label_size_dict.values()))

    print(mean, std_dev)

    # outlier_labels = [
    #     label
    #     for label, size in label_size_dict.items()
    #     if abs(size - mean) > 1 * std_dev
    # ]

    outlier_labels = []
    for section in seg_array:
        # get  unique ID with largest count in section
        unique, counts = np.unique(section, return_counts=True)
        counts = dict(zip(unique, counts))
        largest_id = max(counts, key=counts.get)
        # add to outlier list
        outlier_labels.append(largest_id)


    seg_array[np.isin(seg_array, outlier_labels)] = 0

    seg_array = seg_array.astype(np.uint32)


    # prepare
    out_seg = prepare_ds(
        store = f'{out_f}/{out_ds}',
        shape = seg_array.shape,
        offset = seg.roi.offset,
        voxel_size = seg.voxel_size,
        dtype = np.uint32,
        axis_names= ["z","y","x"],
        units = ["nm","nm","nm"]
    )

    # write
    out_seg[seg.roi] = seg_array