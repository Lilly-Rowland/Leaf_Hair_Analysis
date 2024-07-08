from pycocotools.coco import COCO
import numpy as np

def get_freqs(coco_file, inverse):
    # Initialize COCO API for instance annotations
    coco = COCO(coco_file)  # Replace with your COCO JSON file path

    # Define the class names you are interested in
    class_names = ['leaf_hair']

    # Initialize a dictionary to store class frequencies
    class_freq = {class_name: 0 for class_name in class_names}
    class_freq['no_leaf_hair'] = 0  # Initialize for pixels not identified as leaf hair
    total_pixels = 0
    # Load annotations and count pixels for each class
    img_ids = coco.getImgIds()
    for img_id in img_ids:
        ann_ids = coco.getAnnIds(imgIds=img_id)
        anns = coco.loadAnns(ann_ids)
        for ann in anns:
            if 'segmentation' in ann:
                mask = coco.annToMask(ann)  # Convert annotation to segmentation mask
                
                # Assume 'hair' is the only category in your annotations
                # Determine which pixels are 'leaf hair' (assumed to be class 1) and which are not (class 0)
                leaf_hair_pixels = np.sum(mask == 1)
                no_leaf_hair_pixels = np.sum(mask == 0)  # Assuming 'hair' is class 0
                
                class_freq['leaf_hair'] += leaf_hair_pixels
                class_freq['no_leaf_hair'] += no_leaf_hair_pixels
                total_pixels += leaf_hair_pixels + no_leaf_hair_pixels

    # Print class frequencies (number of pixels)
    print('Class frequencies (number of pixels):')
    freqs = [0] * 2
    i = 0
    for class_name, freq in class_freq.items():
        print(f'{class_name}: {freq/total_pixels}')
        freqs[i] = freq/total_pixels
        if inverse:
            freqs[i] = 1 - freqs[i]
        i += 1

    print(freqs)
    return(freqs)

if __name__ == '__main__':
    get_freqs("Data/combined_coco.json")