import numpy as np
import sys
from scipy.ndimage import find_objects
from funlib.persistence.arrays import open_ds, prepare_ds
from funlib.geometry import Coordinate, Roi


def perform_bounding_box_crop(image):
    """
    Perform bounding box cropping on the image.

    Args:
        image (np.ndarray): Input image array.

    Returns:
        Tuple[np.ndarray, List[int]]: Cropped image and bounding box offset.
    """
    bbox = find_objects(image > 0)[0]
    cropped_image = image[bbox]
    bbox_offset = [x.start for x in bbox]
    return cropped_image, bbox_offset


if __name__ == "__main__":

    in_f = sys.argv[1] # path to the zarr container
    in_ds = sys.argv[2] # name of labels dataset inside zarr
    out_ds  = sys.argv[3] # name of cropped output

    # open ds
    labels = open_ds(f'{in_f}/{in_ds}','r')
    voxel_size = labels.voxel_size

    labels_arr = labels.to_ndarray(labels.roi)

    # do bbox
    cropped_labels_arr, voxel_offset = perform_bounding_box_crop(labels_arr)
    new_roi = Roi(Coordinate(voxel_offset), cropped_labels_arr.shape) * voxel_size

    # prepare output
    new_arr = prepare_ds(
        store = f'{in_f}/{out_ds}',
        shape = cropped_labels_arr.shape,
        offset = new_roi.offset,
        voxel_size = voxel_size,
        dtype = np.uint32,
        axis_names= ['z','y','x'],
        units = ['nm','nm','nm']
    )

    # write
    new_arr[new_roi] = cropped_labels_arr.astype(np.uint32)


