import os
import random
import shutil

def select_random_images(source_folder, destination_folder, num_images):
    # Create the destination folder if it doesn't exist
    os.makedirs(destination_folder, exist_ok=True)

    # Get the list of PNG files in the source folder
    png_files = [file for file in os.listdir(source_folder) if file.endswith(".png")]

    # Filter PNG files based on size (>32 KB)
    filtered_files = [file for file in png_files if os.path.getsize(os.path.join(source_folder, file)) > 32 * 1024]

    # Select random images
    selected_files = random.sample(filtered_files, num_images)

    # Copy the selected images to the destination folder
    for file in selected_files:
        source_path = os.path.join(source_folder, file)
        destination_path = os.path.join(destination_folder, file)
        shutil.copy2(source_path, destination_path)

    print(f"{num_images} random images selected and copied to {destination_folder}.")

if __name__ == "__main__":
    # Set the source folder, destination folder, and number of images to select
    source_folder = "TestingPNGImages"
    num_images = 128  # Change this value as desired
    destination_folder = f"{source_folder}Selected{num_images}"

    # Call the function to select and copy random images
    select_random_images(source_folder, destination_folder, num_images)
