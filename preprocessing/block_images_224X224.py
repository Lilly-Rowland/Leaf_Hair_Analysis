import os
from PIL import Image

def split_image_into_tiles(image_path, tile_size=224):
    # Open the image
    with Image.open(image_path) as img:
        # Get the dimensions of the image
        width, height = img.size
        
        # Calculate the number of rows and columns
        num_rows = height // tile_size
        num_cols = width // tile_size
        
        # Iterate over each tile
        for row in range(num_rows):
            for col in range(num_cols):
                # Calculate the coordinates of the current tile
                left = col * tile_size
                upper = row * tile_size
                right = left + tile_size
                lower = upper + tile_size
                
                # Crop the tile from the image
                tile = img.crop((left, upper, right, lower))
                
                # Save the tile with the specified name
                tile_name = os.path.splitext(os.path.basename(image_path))[0] + f"_{row}_{col}.png"
                tile.save(os.path.join("leaf_tiles", tile_name))

def split_images_in_directory(input_folder):
    # Create the output directory if it doesn't exist
    os.makedirs("leaf_tiles", exist_ok=True)
    
    # Iterate through all files in the input directory
    for file_name in os.listdir(input_folder):
        if file_name.endswith("009-RemNE11xSo_T139_9.png"):
        #if file_name.endswith(".png"):
            # Get the path to the current image file
            image_path = os.path.join(input_folder, file_name)
            
            # Split the image into tiles
            split_image_into_tiles(image_path)

# Specify the input folder containing the images
input_folder = "leaf_samples"

# Split the images in the input folder into tiles
split_images_in_directory(input_folder)


if __name__ == "__main__":
    # Specify the input folder containing the images
    input_folder = "leaf_samples"

    # Split the images in the input folder into tiles
    split_images_in_directory(input_folder)