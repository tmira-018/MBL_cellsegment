import zarr
import numpy as np
import sys
import daisy
from functools import partial
from funlib.persistence.arrays import open_ds, prepare_ds


def clahe_block(in_array, out_array, block):
    import numpy as np
    from skimage.exposure import equalize_adapthist as clahe
    from funlib.persistence import Array

    # read
    in_data = in_array.to_ndarray(block.read_roi, fill_value=0)

    # clahe
    out_data = clahe(in_data)

    # make it [0, 255]
    out_data *= 255

    # write as np.uint8
    out_data_array = Array(out_data,block.read_roi.offset,out_array.voxel_size)
    out_array[block.write_roi] = out_data_array.to_ndarray(block.write_roi).astype(np.uint8)


if __name__ == "__main__":

    in_f = sys.argv[1] # path to zarr container
    in_ds = sys.argv[2] # name of image dataset inside zarr container
    out_ds = sys.argv[3] # name of new normalized image daa\taset

    # load image
    in_ds = open_ds(f'{in_f}/{in_ds}','a')

    # prepare output
    out_ds = prepare_ds(
        store = f'{in_f}/{out_ds}',
        shape = in_ds.shape,
        offset = in_ds.offset,
        voxel_size = in_ds.voxel_size,
        dtype = np.uint8,
        axis_names= ['z','y','x'],
        units = ['nm','nm','nm']
    )

    # block i/o shapes
    read_size = in_ds.chunk_shape * 2 * in_ds.voxel_size
    write_size = in_ds.chunk_shape * in_ds.voxel_size
    read_roi = daisy.Roi((0,0,0), read_size) - (write_size / 2)
    write_roi = daisy.Roi((0,0,0), write_size)

    # create blockwise task
    task = daisy.Task(
        task_id='ClaheBlockwise',
        total_roi=in_ds.roi,
        read_roi=read_roi,
        write_roi=write_roi,
        process_function=partial(clahe_block, in_ds, out_ds),
        read_write_conflict=True,
        num_workers=10,
        max_retries=0,
        fit='shrink'
    )
    

    daisy.run_blockwise([task])