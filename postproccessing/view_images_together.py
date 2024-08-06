import os
import cv2
import numpy as np
from PIL import Image

def create_image_grid(image_folder, images_per_page=25, thumb_size=(100, 100), grid_size=(5, 5), padding=10):
    images = [os.path.join(image_folder, f) for f in os.listdir(image_folder) if f.endswith(('png', 'jpg', 'jpeg', 'bmp'))]
    images.sort()
    total_pages = (len(images) + images_per_page - 1) // images_per_page

    for page in range(total_pages):
        start_index = page * images_per_page
        end_index = min(start_index + images_per_page, len(images))
        grid_img = create_grid_image(images[start_index:end_index], thumb_size, grid_size, padding)
        cv2.imshow(f'Page {page + 1}', grid_img)
        cv2.waitKey(0)  # Press any key to move to the next page
        cv2.destroyAllWindows()

def create_grid_image(images, thumb_size, grid_size, padding):
    grid_height = grid_size[0] * (thumb_size[1] + padding) - padding
    grid_width = grid_size[1] * (thumb_size[0] + padding) - padding
    grid_img = np.zeros((grid_height, grid_width, 3), dtype=np.uint8)

    for i, img_path in enumerate(images):
        img = Image.open(img_path)
        img.thumbnail(thumb_size)
        img = np.array(img)

        # If the image has an alpha channel, remove it
        if img.shape[2] == 4:
            img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)

        # Create a blank image with the desired thumbnail size
        thumb_img = np.zeros((thumb_size[1], thumb_size[0], 3), dtype=np.uint8)

        # Calculate the position to place the image in the center of the thumbnail
        y_offset = (thumb_size[1] - img.shape[0]) // 2
        x_offset = (thumb_size[0] - img.shape[1]) // 2

        # Place the resized image in the center of the blank thumbnail
        thumb_img[y_offset:y_offset + img.shape[0], x_offset:x_offset + img.shape[1], :] = img

        row = i // grid_size[1]
        col = i % grid_size[1]
        y = row * (thumb_size[1] + padding)
        x = col * (thumb_size[0] + padding)
        grid_img[y:y + thumb_size[1], x:x + thumb_size[0], :] = thumb_img

    return grid_img

if __name__ == "__main__":
    image_folder = "cropped_leaves"  # Change this to your image directory
    create_image_grid(image_folder)
