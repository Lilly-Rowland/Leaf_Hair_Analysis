import cv2

def convert_rgb_to_hsv(input_image_path, output_image_path):
    # Read the RGB image
    rgb_image = cv2.imread(input_image_path)

    # Check if the image was loaded successfully
    if rgb_image is None:
        print(f"Error: Could not load image at {input_image_path}")
        return

    # Convert the RGB image to HSV
    hsv_image = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2HSV)

    # Save the HSV image
    cv2.imwrite(output_image_path, hsv_image)
    print(f"HSV image saved at {output_image_path}")

if __name__ == "__main__":
    # Define the input and output image paths
    input_image_path = "leaf_samples/blocked_images/034-sLutie_T141_5_0_2.png"
    output_image_path = "new_hsv.png"

    # Convert the image
    convert_rgb_to_hsv(input_image_path, output_image_path)
