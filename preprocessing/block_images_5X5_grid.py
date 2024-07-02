import os
from PIL import Image

def split_image(image_path, output_dir):
    # Open the image
    img = Image.open(image_path)
    img_width, img_height = img.size
    
    # Define the dimensions of each tile
    tile_width = img_width // 5
    tile_height = img_height // 5
    
    # Get the base name of the image file (without extension)
    base_name = os.path.splitext(os.path.basename(image_path))[0]
    
    # Create the output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Split the image into a 5x5 grid
    for row in range(5):
        for col in range(5):
            # Calculate the bounding box of the tile
            left = col * tile_width
            upper = row * tile_height
            right = (col + 1) * tile_width
            lower = (row + 1) * tile_height
            
            # Crop the image to the bounding box
            tile = img.crop((left, upper, right, lower))
            
            # Save the tile with the appropriate naming convention
            tile_name = f"{base_name}_{row}_{col}.png"
            tile_path = os.path.join(output_dir, tile_name)
            tile.save(tile_path)

def process_images(input_dir, output_dir):
    # Get a list of all image files in the input directory
    image_files = [f for f in os.listdir(input_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    # Process each image file
    for image_file in image_files:
        image_path = os.path.join(input_dir, image_file)
        split_image(image_path, output_dir)

if __name__ == "__main__":
    # Define the input and output directories
    input_dir = "leaf_samples"
    output_dir = "leaf_samples/blocked_images"
    
    # Process the images
    process_images(input_dir, output_dir)
