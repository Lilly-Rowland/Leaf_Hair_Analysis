import json
import os

def load_coco_annotation(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)
    return data

def save_coco_annotation(data, file_path):
    with open(file_path, 'w') as f:
        json.dump(data, f, indent=4)

def combine_coco_annotations(folder_path, output_file):
    combined_coco = {
        'images': [],
        'annotations': [],
        'categories': []
    }
    
    image_id_offset = 0
    annotation_id_offset = 0
    
    # Iterate over all JSON files in the folder
    id_path = {}

    for file_name in os.listdir(folder_path):
        if file_name.endswith('.json') and not file_name.startswith('._'):
            file_path = os.path.join(folder_path, file_name)
            coco = load_coco_annotation(file_path)
            
            # If this is the first file, take its categories
            if not combined_coco['categories']:
                combined_coco['categories'] = coco['categories']
            
            # Update image file paths and IDs
            for image in coco['images']:
                image['id'] += image_id_offset
                file_name = image["file_name"]
                leaf_folder = file_name[:file_name.find('_', (file_name.find('_', file_name.find('_') + 1) + 1))]
                image['path'] = f'datasets/{leaf_folder}/{os.path.basename(file_name)}'
                combined_coco['images'].append(image)
                id_path[image['id']] = image['path']
            
            # Update annotation IDs and image IDs
            for annotation in coco['annotations']:
                annotation['id'] += annotation_id_offset
                annotation['image_id'] += image_id_offset
                #annotation['file_path'] = id_path[annotation['image_id']]
                if annotation['image_id'] in id_path:
                    annotation['file_path'] = id_path[annotation['image_id']]
                else:
                    raise KeyError(f"No file name found for image_id {annotation['image_id']}")
                combined_coco['annotations'].append(annotation)
            
            # Update offsets for next file
            image_id_offset += len(coco['images'])
            annotation_id_offset += len(coco['annotations'])
    
    # Save the combined annotations to the output file
    save_coco_annotation(combined_coco, output_file)

if __name__ == "__main__":
    folder_path = '/Volumes/Image Data /COCO_Annotations'
    output_file = '/Volumes/Image Data /combined_coco.json'
    
    combine_coco_annotations(folder_path, output_file)
    print(f"Combined COCO annotation file saved to {output_file}")
