
import os 
from aicsimageio import AICSImage
import tifffile as tiff
import sys
import glob

def convert_czi_to_tif(input_folder : str, output_folder: str):
    # Ensure the output folder exists
    os.makedirs(output_folder, exist_ok=True)
    # List all .czi files in the input folder
    czi_files = glob.glob(input_folder + '*/*.czi')
    print(czi_files)

czi_files = [f for f in os.listdir(input_folder) if f.endswith('.czi')]

for czi_file in czi_files:
    # Construct full file paths
    input_path = os.path.join(input_folder, czi_file)
    output_path = os.path.join(input_folder, os.path.splitext(czi_file)[0] + '.tif')

    # Read the .czi file
    img = AICSImage(input_path)
    img_data = img.get_image_data("ZYX")
    
    # Save as .tif file
    tiff.imwrite(output_path, img_data)
print("Converted", czi_file, "to tifs")

if __name__ == "__main__":
    
    input_folder = sys.argv[1]
    output_folder = sys.argv[2]

    convert_czi_to_tif(input_folder, output_folder)