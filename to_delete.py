from PIL import Image, ImageOps

# Open an image file
input_image_path = 'whole_leaf_masks/reconstructed_mask_007-PI588601_15-11.png'
image = Image.open(input_image_path)

# Invert the image colors
inverted_image = ImageOps.invert(image.convert('RGB'))

# Save the inverted image
output_image_path = 'path_to_save_inverted_image.jpg'
inverted_image.save(output_image_path)

# Optionally, show the image
inverted_image.show()
