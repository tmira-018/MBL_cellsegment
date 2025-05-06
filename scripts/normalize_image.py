import zarr
import numpy as np
import sys
from skimage.exposure import equalize_adapthist
from funlib.persistence.arrays import open_ds, prepare_ds

in_f = sys.argv[1] # path to zarr container
in_ds = sys.argv[2] # name of image dataset inside zarr container
out_ds = sys.argv[3] # name of new normalized image daa\taset

# load image
ds = open_ds(f'{in_f}/{in_ds}','a')
arr = ds.to_ndarray(ds.roi)

# # normalize to 0 mean and 1 std
# mean, std = np.mean(arr), np.std(arr)
# arr = (arr - mean) / std

# now make it go from 0 to 1
arr = equalize_adapthist(arr)

# now make it go from 0 and 255
arr *= 255

# write it
new_arr = prepare_ds(
    store = f'{in_f}/{out_ds}',
    shape = arr.shape,
    offset = ds.offset,
    voxel_size = ds.voxel_size,
    dtype = np.uint8,
    axis_names= ['z','y','x'],
    units = ['nm','nm','nm']
)

new_arr[ds.roi] = arr