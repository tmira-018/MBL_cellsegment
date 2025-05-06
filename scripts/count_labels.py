import numpy as np
import zarr
import sys

def count(in_zarr, in_ds):

    # load the array
    #print("in_zarr",in_zarr)
    #print("in_ds", in_ds)
    in_store = f'{in_zarr}'
    #print("in_store",in_store)pyt
    in_array = zarr.open(in_store,'r')
    uniques = np.unique(in_array)

    return len(uniques) - 1


if __name__ == "__main__":

    in_zarr = sys.argv[1]
    in_ds = sys.argv[2]

    print("number of instances: ", count(in_zarr,in_ds))