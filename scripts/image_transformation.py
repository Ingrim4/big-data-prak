# i need to take in a list of .png images (used) and a folder of .tif images into a list
# then i need to substract the list of used.png from the list of .tif images (other file ending)
# then i need to save the remaining list of .tif images as .png images in a new folder

import os
from PIL import Image
import json

"""# Function to read the JSON file and extract "file_upload" values
def get_file_uploads(json_file):
    with open(json_file) as file:
        data = json.load(file)
    used_png_list = [item['file_upload'] for item in data]
    return used_png_list"""

def get_file_uploads(json_file):
    with open(json_file) as file:
        data = json.load(file)
    used_png_list = []
    for item in data:
        filename = item['file_upload']
        filename = filename.split("-", 1)[1]  # Split until the first "-" character
        used_png_list.append(filename)
        #print(filename)
    return used_png_list


def convert_tif_to_png(tif_folder, used_png_list, output_folder, skipped):
    # Create the output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)

    # Load the list of used PNG files
    used_png_files = set(used_png_list)
    print(len(used_png_files))

    # Iterate over the TIFF files in the given folder
    for filename in os.listdir(tif_folder):
        if filename.endswith(".tif"):
            tif_path = os.path.join(tif_folder, filename)

            # Check if the TIFF file is in the used PNG list
            filename = filename.lstrip("{")  # Remove curly brackets at start if present
            if filename.endswith("}.tif"): # Remove trailing curly bracket if present
                filename = filename.replace("}.tif", ".tif")  # Remove ".tif" extension temporarily

            #print(filename)
            if filename.replace(".tif", ".png") in used_png_files:
                skipped += 1
                continue  # Skip the file if it is in the used list


            # Open the TIFF file and convert it to PNG
            with Image.open(tif_path) as image:
                # Create the output filename by replacing the extension
                output_filename = filename.replace(".tif", ".png")
                output_path = os.path.join(output_folder, output_filename)

                # Save the image as PNG
                image.save(output_path)
    print("Conversion completed!")
    return skipped



if __name__ == "__main__":
    tif_folder = "BigDataPraktikumDaecher"
    #output_folder = "ValidationPNGImages" #only json_file1 (training)
    output_folder = "TestingPNGImages" #json_file1 (training) and json_file2 (validation)
    json_file1 = 'used_png.json' #training data json
    json_file2 = 'Validate_COCO_154.json' #validation data json

    # Get the list of "file_upload" values
    used_png_list = get_file_uploads(json_file1)
    used_png_list.extend(get_file_uploads(json_file2))
    print(len(used_png_list))
    
    skipped = 0
    skipped = convert_tif_to_png(tif_folder, used_png_list, output_folder, skipped)
    print(f"{skipped} skipped images since they were already used in Label studio")