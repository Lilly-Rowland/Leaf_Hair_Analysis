from PIL import Image

def calculate_non_transparent_pixels(image_path):
    # Open the image
    image = Image.open(image_path).convert("RGBA")  # Ensure image has an alpha channel

    # Get the data of the image
    data = image.getdata()

    # Count non-transparent pixels
    non_transparent_pixels = sum(1 for pixel in data if pixel[3] != 0)

    return non_transparent_pixels

# Example usage
image_path = "312-sNiagara_T163_15.png"  # Replace with your actual image path
num_non_transparent_pixels = calculate_non_transparent_pixels(image_path)
print(f'The number of non-transparent pixels in the image is: {num_non_transparent_pixels}')

#3850201