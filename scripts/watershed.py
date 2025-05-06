import sys
import logging
import numpy as np
import os
import waterz

from scipy.ndimage import label, measurements, distance_transform_edt, maximum_filter, gaussian_filter
from skimage.segmentation import watershed

from funlib.persistence.arrays import open_ds, prepare_ds
from funlib.geometry import Roi, Coordinate


def watershed_from_boundary_distance(
        boundary_distances,
        boundary_mask,
        id_offset=0,
        min_seed_distance=10):

    max_filtered = maximum_filter(boundary_distances, min_seed_distance)
    maxima = max_filtered==boundary_distances
    seeds, n = label(maxima)

    print(f"Found {n} fragments")

    if n == 0:
        return np.zeros(boundary_distances.shape, dtype=np.uint64), id_offset

    seeds[seeds!=0] += id_offset

    fragments = watershed(
        boundary_distances.max() - boundary_distances,
        seeds,
        mask=boundary_mask)
    
    if boundary_mask is not None:
        fragments *= boundary_mask

    ret = (fragments.astype(np.uint64), n + id_offset)

    return ret


def watershed_from_affinities(
    affs,
    max_affinity_value=1.0,
    fragments_in_xy=True,
    background_mask=False,
    mask_thresh=0.1,
    return_seeds=False,
    min_seed_distance=10,
):
    """Extract initial fragments from affinities using a watershed
    transform. Returns the fragments and the maximal ID in it.
    Returns:
        (fragments, max_id)
        or
        (fragments, max_id, seeds) if return_seeds == True"""
   
    # # add random noise
    # random_noise = np.random.randn(*affs.shape) * 0.01

    # # add smoothed affs, to solve a similar issue to the random noise. We want to bias
    # # towards processing the central regions of objects first.
    # logging.info("Smoothing affs")
    # smoothed_affs: np.ndarray = (
    #         gaussian_filter(affs, sigma=(0, 1, 2, 2))
    #         - 0.5
    # ) * 0.05

    # affs = (affs + random_noise + smoothed_affs).astype(np.float32)
    # affs = np.clip(affs, 0.0, 1.0)

    if fragments_in_xy:

        mean_affs = 0.5 * (affs[-1] + affs[-2]) # affs are (c,z,y,x)
        depth = mean_affs.shape[0]

        fragments = np.zeros(mean_affs.shape, dtype=np.uint64)
        if return_seeds:
            seeds = np.zeros(mean_affs.shape, dtype=np.uint64)

        id_offset = 0
        for z in range(depth):

            boundary_mask = mean_affs[z] > mask_thresh * max_affinity_value
            boundary_distances = distance_transform_edt(boundary_mask)

            if background_mask is False:
                boundary_mask = None

            ret = watershed_from_boundary_distance(
                boundary_distances,
                boundary_mask,
                id_offset=id_offset,
                min_seed_distance=min_seed_distance,
            )

            fragments[z] = ret[0]
            if return_seeds:
                seeds[z] = ret[2]

            id_offset = ret[1]

        ret = (fragments, id_offset)
        if return_seeds:
            ret += (seeds,)

    else:

        boundary_mask = np.mean(affs, axis=0) > mask_thresh * max_affinity_value
        boundary_distances = distance_transform_edt(boundary_mask)

        if background_mask is False:
            boundary_mask = None

        ret = watershed_from_boundary_distance(
            boundary_distances, boundary_mask, return_seeds, min_seed_distance=min_seed_distance
        )

        fragments = ret[0]

    return ret

def get_segmentation(affinities, threshold):

    fragments = watershed_from_affinities(affinities)[0]
    thresholds = [threshold]

    generator = waterz.agglomerate(
        affs=affinities.astype(np.float32),
        fragments=fragments,
        thresholds=thresholds,
    )

    segmentation = next(generator)

    return segmentation

if __name__ == "__main__":
    affs_file = sys.argv[1]
    affs_ds = sys.argv[2]
    seg_ds = sys.argv[3]
    
    try:
        threshold = float(sys.argv[4])
    except:
        threshold = 0.5

    seg_file = affs_file

    # load affs
    affs = open_ds(f'{affs_file}/{affs_ds}', 'r')
    print(affs_file)
    print(affs_ds)
    
    #roi = affs.roi
    #roi = Roi(affs.roi.shape // 2, affs.roi.shape // 3).snap_to_grid(affs.voxel_size)
    
    offset = Coordinate([40, 113, 189]) * affs.voxel_size
    end = Coordinate([119, 303, 614]) * affs.voxel_size
    roi = Roi((offset),(end - offset))

    affs_array = affs.to_ndarray(roi) / 255.0

    seg_array = get_segmentation(affs_array[:3], threshold)

    # prepare
    out_seg = prepare_ds(
        store = f'{seg_file}/{seg_ds}',
        shape = seg_array.shape,
        offset = roi.offset,
        voxel_size = affs.voxel_size,
        dtype = np.uint32,
        axis_names= ["z","y","x"],
        units = ["nm","nm","nm"]
    )

    # write
    out_seg[roi] = seg_array