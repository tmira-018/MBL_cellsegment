import argparse
from typing import List, Tuple
import numpy as np
from tifffile import imread
from scipy.ndimage import find_objects
from funlib.persistence import prepare_ds
from funlib.geometry import Coordinate, Roi
import zarr
import os 
import glob

def load_and_preprocess_image(img_path: str, out_dtype: np.dtype) -> np.ndarray:
    """
    Load and preprocess the input image.

    Args:
        img_path (str): Path to the input image file.
        out_dtype (np.dtype): Desired output data type.

    Returns:
        np.ndarray: Preprocessed image array.
    """
    if img_path.endswith('tif') or img_path.endswith('tiff'):
        image = imread(img_path)
    elif img_path.endswith('.npy'):
        image = np.load(img_path, allow_pickle=True).item()['masks']
    else:
        raise RuntimeError("Unimplemented data format")

    # print(np.max(image), np.min(image), image.dtype)

    if len(image.shape) > 3:
        raise ValueError("Input image should be single-channel.")
    
    # if out_dtype == np.uint8 and image.dtype != np.uint8:
    #     image = (image // 256).astype(np.uint8) 
    # else:
    image = image.astype(out_dtype)
    
    return image

def perform_bounding_box_crop(image: np.ndarray) -> Tuple[np.ndarray, List[int]]:
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

def create_zarr_dataset(
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
    print(out_dtype)
    image = load_and_preprocess_image(img_path, out_dtype)
    print(f"Loaded image shape: {image.shape}, voxel offset: {voxel_offset}")

    # if out_dtype != np.uint8:
    #     perform_crop = input("\nPerform bounding box crop? (y/n, default: y): ").lower().strip() != 'n'
    #     if perform_crop:
    #         image, bbox_offset = perform_bounding_box_crop(image)
    #         voxel_offset = [a + b for a, b in zip(voxel_offset, bbox_offset)]
    #         print(f"Image shape after bounding box: {image.shape}, new voxel offset: {voxel_offset}")

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
        compressor=zarr.get_codec({"id": "blosc"}),
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

#for loop to create zarrs  
    for x in glob.glob(args.data_directory + '/*'):
        print(x)
        if x.endswith('_'):
            print('skip')
        else: 
            if x.endswith('315'):
                print('skip')
            else:
                try:
                    create_zarr_dataset(
                        x + f'/{os.path.basename(x)}.tif',
                        x + '/volume.zarr',
                        args.out_ds_name,
                        out_dtype,
                        out_voxel_size,
                        args.voxel_offset
                    )
                except:
                    print('no data')

if __name__ == "__main__":
    main()
