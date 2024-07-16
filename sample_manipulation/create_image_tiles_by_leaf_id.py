import os
from PIL import Image

def is_hidden_or_system_directory(directory):
    return directory.startswith('.') or directory in ('System Volume Information', '$RECYCLE.BIN')

def split_image_into_tiles(image_path, leaf_folder, tile_size=224):
    # Open the image
    with Image.open(image_path) as img:
        # Get the dimensions of the image
        width, height = img.size
        
        # Calculate the number of rows and columns
        num_rows = height // tile_size
        num_cols = width // tile_size

        positions = []
        
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
                tile_path = os.path.join(leaf_folder, tile_name)
                tile.save(tile_path)
                
                positions.append((row, col, tile_path))

        return positions, img.size


def split_images_in_directory(input_folder, output_folder):

    # Iterate through all files in the input directory
    for file_name in os.listdir(input_folder):
        if file_name.endswith(".png") and not (is_hidden_or_system_directory(file_name)):
            # Get the path to the current image file
            image_path = os.path.join(input_folder, file_name)
            leaf_folder = os.path.join(output_folder, os.path.basename(image_path)[:-4])
            if os.path.exists(leaf_folder):
                continue
            os.makedirs(leaf_folder, exist_ok=True)
            # Split the image into tiles
            split_image_into_tiles(image_path, leaf_folder)

if __name__ == "__main__":
    # Specify the input folder containing the images
    input_folder = "/Volumes/Image Data /Repository06032024_DM_ 6-8-2024_3dpi_1"
    output_folder = "/Volumes/Image Data /Tiled_Repository06032024_DM_ 6-8-2024_3dpi_1"

    # Split the images in the input folder into tiles
    split_images_in_directory(input_folder, output_folder)
