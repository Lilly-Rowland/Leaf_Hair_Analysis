import numpy as np
from PIL import Image, ImageCms
import os

for img in os.listdir("/home/lgr4641/Desktop/Leaf_Hair_Analysis/leaves_to_inference"):
    # Open image and discard alpha channel which makes wheel round rather than square
    im = Image.open("/home/lgr4641/Desktop/Leaf_Hair_Analysis/leaves_to_inference/" + img).convert('RGB')

    # Convert to Lab colourspace
    srgb_p = ImageCms.createProfile("sRGB")
    lab_p  = ImageCms.createProfile("LAB")

    rgb2lab = ImageCms.buildTransformFromOpenProfiles(srgb_p, lab_p, "RGB", "LAB")
    Lab = ImageCms.applyTransform(im, rgb2lab)
    

    # Split into constituent channels so we can save 3 separate greyscales
    l, a, b = Lab.split()
    

    # l.save("/home/lgr4641/Desktop/Leaf_Hair_Analysis/LAB_imgs/l" + img)
    # a.save("/home/lgr4641/Desktop/Leaf_Hair_Analysis/LAB_imgs/a" + img)
    b.save("/home/lgr4641/Desktop/Leaf_Hair_Analysis/LAB_imgs/b" + img)