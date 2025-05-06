import os 
from aicsimageio import AICSImage
import sys
import glob
from funlib.persistence import prepare_ds
from funlib.geometry import Coordinate, Roi
import zarr
import argparse
import numpy as np
from typing import List, Tuple

#Load czi
def load_czi(input_path):

    img = AICSImage(input_path)
    img_data = img.get_image_data("ZYX")
    
    return img_data

#Save to Zarr #array #path to zarr #dataset name

def create_zarr(
    img_path: str,
    out_zarr: str,
    out_ds: str,
    out_dtype: np.dtype,
    out_voxel_size: List[int],
    voxel_offset: List[int]
) -> None:
    """
    Create a Zarr dataset from an input image.

    Args:
        img_path (str): Path to the input image file.
        out_zarr (str): Output Zarr file path.
        out_ds (str): Output dataset name.
        out_dtype (np.dtype): Output data type.
        out_voxel_size (List[int]): Output voxel size.
        voxel_offset (List[int]): Voxel offset.
    """
    image = load_czi(img_path)
    print(f"Loaded image shape: {image.shape}, voxel offset: {voxel_offset}")

    voxel_size = Coordinate(out_voxel_size)
    total_roi = Roi(Coordinate(voxel_offset), image.shape) * voxel_size

    out_image_ds = prepare_ds(
        store = f"{out_zarr}/{out_ds}",
        shape = image.shape,
        voxel_size = voxel_size,
        offset = Coordinate(voxel_offset) * voxel_size,
        axis_names = ["z","y","x"],
        units = ["nm","nm","nm"],
        dtype = out_dtype,
        compressor=zarr.get_codec({"id": "blosc"})
    )

    print(f"Writing {out_ds} to {out_zarr}..")
    out_image_ds[total_roi] = image


def main():
    parser = argparse.ArgumentParser(description="Convert an image to a Zarr dataset.")
    parser.add_argument("data_directory", help="Path to the input image file")
    parser.add_argument("out_ds_name", help="Output dataset name")
    parser.add_argument("out_z_res", type=int, help="Output Z resolution")
    parser.add_argument("out_yx_res", type=int, help="Output Y and X resolution")
    parser.add_argument("dtype", type=str, help="output datatype for array")
    parser.add_argument("--voxel_offset", type=int, nargs=3, default=[0, 0, 0], 
                        help="Voxel offset (default: [0, 0, 0])")

    args = parser.parse_args()

    out_voxel_size = [args.out_z_res, args.out_yx_res, args.out_yx_res]
    out_dtype = np.dtype(args.dtype) #np.uint8 if any(x in args.out_ds_name for x in ['image', 'img', 'raw', 'mask']) else np.uint32

    
    for x in glob.glob(args.data_directory + '/*/*.czi'):
        print(x)

        create_zarr(
            x,
            os.path.split(x)[0] + '/volume.zarr',
            args.out_ds_name,
            out_dtype,
            out_voxel_size,
            args.voxel_offset
        )

if __name__ == "__main__":
    main()
