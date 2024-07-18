# import numpy as np
# import os
# import cv2

# color_threshold = 100
# for img_name in os.listdir("/home/lgr4641/Desktop/Leaf_Hair_Analysis/leaves_to_inference"):
#     img_path = os.path.join("/home/lgr4641/Desktop/Leaf_Hair_Analysis/leaves_to_inference", img_name)
    
#     # Open image
#     image = cv2.imread(img_path)

#     # Convert to LAB color space
#     lab_image = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    
#     l_channel, a_channel, b_channel = cv2.split(lab_image)

#     cv2.imwrite(f"/home/lgr4641/Desktop/Leaf_Hair_Analysis/LAB_imgs/b_{img_name}", b_channel)

#     # Apply a color threshold to filter out certain colors in LAB space
#     lab_mask = np.zeros_like(lab_image[:,:,0], dtype=np.uint8)
#     lab_mask[(lab_image[:,:,1] >= color_threshold) & (lab_image[:,:,2] >= color_threshold)] = 255

#     # Save LAB channels separately
#     l_channel, a_channel, b_channel = cv2.split(lab_image)

#     cv2.imwrite(f"/home/lgr4641/Desktop/Leaf_Hair_Analysis/LAB_imgs/threshold_b_{img_name}", b_channel)

import numpy as np
import os
from PIL import Image, ImageCms
import cv2

color_threshold = 140
for img_name in os.listdir("/home/lgr4641/Desktop/Leaf_Hair_Analysis/leaves_to_inference"):
    img_path = os.path.join("/home/lgr4641/Desktop/Leaf_Hair_Analysis/leaves_to_inference", img_name)
    
    im = Image.open(img_path).convert('RGB')

    # Convert to Lab colourspace
    srgb_p = ImageCms.createProfile("sRGB")
    lab_p  = ImageCms.createProfile("LAB")

    rgb2lab = ImageCms.buildTransformFromOpenProfiles(srgb_p, lab_p, "RGB", "LAB")
    Lab = ImageCms.applyTransform(im, rgb2lab)
    # Save original b_channel
    _, _, b_channel = Lab.split()
    

    b_channel_np = np.array(b_channel)

    _, thresholded_b_channel_np = cv2.threshold(b_channel_np, color_threshold, 255, cv2.THRESH_BINARY)

    thresholded_b_channel = Image.fromarray(thresholded_b_channel_np)

    # Save original b_channel
    b_channel.save(os.path.join("/home/lgr4641/Desktop/Leaf_Hair_Analysis/LAB_imgs/", f"{img_name}"))

    # Save thresholded b_channel
    thresholded_b_channel.save(os.path.join("/home/lgr4641/Desktop/Leaf_Hair_Analysis/LAB_imgs/", f"threshold_b_{img_name}"))

    # # Apply a color threshold to filter out certain colors in LAB space
    # lab_mask = np.zeros_like(lab_image[:,:,0], dtype=np.uint8)
    # lab_mask[(lab_image[:,:,1] >= color_threshold) & (lab_image[:,:,2] >= color_threshold)] = 255

    # # Apply the mask to b_channel
    # thresholded_b_channel = cv2.bitwise_and(b_channel, b_channel, mask=lab_mask)
    
    # # Save thresholded b_channel
    # cv2.imwrite(f"/home/lgr4641/Desktop/Leaf_Hair_Analysis/LAB_imgs/threshold_b_{img_name}", thresholded_b_channel)
