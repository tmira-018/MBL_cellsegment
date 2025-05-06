import numpy as np
from funlib.evaluate import detection_scores, rand_voi
from scipy.ndimage import remove_small_objects
import sys
from funlib.persistence.arrays import open_ds
from pprint import pprint


def evaluate(
    gt,
    seg,
    dust_filter=20
):

    # filter
    filtered_seg = remove_small_objects(
        seg.astype(np.int64), min_size=dust_filter, connectivity=1
    )

    # run eval
    detection = detection_scores(gt, filtered_seg)
    vois = rand_voi(gt, filtered_seg)

    return detection, vois

if __name__ == "__main__":

    gt_f = sys.argv[1] # path to gt labels zarr
    gt_ds = sys.argv[2] # name of gt labels dataset
    seg_f = sys.argv[3] # path to seg zarr!
    seg_ds = sys.argv[4] # name of segmentation dataset inside zarr
    dust_filter = int(sys.argv[5]) # name of output segmentation

    # open
    gt = open_ds(f'{gt_f}/{gt_ds}', mode='r')
    seg = open_ds(f'{seg_f}/{seg_ds}', mode='r')

    roi = gt.roi.intersect(seg.roi)

    gt_array = gt.to_ndarray(roi)
    seg_array = seg.to_ndarray(roi)

    # run
    detection, vois = evaluate(gt_array, seg_array, dust_filter)

    # print
    print(f"Scores for {seg_f}/{seg_ds} against {seg_gt}/{gt_ds}:")
    pprint(detection)
    pprint(vois)



