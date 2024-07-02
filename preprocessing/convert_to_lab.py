import numpy as np
from PIL import Image, ImageCms
import os

for img in os.listdir("leaf_samples/blocked_images"):
    # Open image and discard alpha channel which makes wheel round rather than square
    im = Image.open("leaf_samples/blocked_images/" + img).convert('RGB')

    # Convert to Lab colourspace
    srgb_p = ImageCms.createProfile("sRGB")
    lab_p  = ImageCms.createProfile("LAB")

    rgb2lab = ImageCms.buildTransformFromOpenProfiles(srgb_p, lab_p, "RGB", "LAB")
    Lab = ImageCms.applyTransform(im, rgb2lab)

    # Split into constituent channels so we can save 3 separate greyscales
    l, a, b = Lab.split()

    l.save("leaf_samples/LAB_blocked_samples/lab_" + img)
    # a.save('a.png')
    # b.save('b.png')