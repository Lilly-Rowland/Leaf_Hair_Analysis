from PIL import Image
import os

# Function to split RGB image into CMYK channels
def split_cmyk(image):
    cmyk_image = image.convert('CMYK')
    c, m, y, k = cmyk_image.split()
    return c, m, y, k

# Directory path for input images
input_dir = "/home/lgr4641/Desktop/Leaf_Hair_Analysis/leaves_to_inference"

# Directory path for output channels
output_dir = "/home/lgr4641/Desktop/Leaf_Hair_Analysis/CMYK_channels"

# Ensure output directory exists
os.makedirs(output_dir, exist_ok=True)

# Process each image in the input directory
for img_name in os.listdir(input_dir):
    # Open image and convert to RGB mode if it's not already
    img_path = os.path.join(input_dir, img_name)
    image = Image.open(img_path).convert('RGB')

    # Split image into CMYK channels
    c, m, y, k = split_cmyk(image)

    # Save each channel separately
    c.save(os.path.join(output_dir, f"c_{img_name}"))
    m.save(os.path.join(output_dir, f"m_{img_name}"))
    y.save(os.path.join(output_dir, f"y_{img_name}"))
    k.save(os.path.join(output_dir, f"k_{img_name}"))

    print(f"CMYK channels saved for {img_name}")
