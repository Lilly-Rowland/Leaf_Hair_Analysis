import os
import csv

def create_csv_from_folders(root_dir, output_csv):
    # List to store image data
    image_data = []

    # Walk through the directory structure
    for seedling_char in os.listdir(root_dir):
        seedling_char_path = os.path.join(root_dir, seedling_char)
        if os.path.isdir(seedling_char_path):
            for leaf in os.listdir(seedling_char_path):
                leaf_path = os.path.join(seedling_char_path, leaf)
                if os.path.isdir(leaf_path):
                    for file in os.listdir(leaf_path):
                        if file.lower().endswith('.png'):
                            full_path = os.path.join(leaf_path, file)
                            image_data.append([full_path, seedling_char, leaf])

    # Write the data to a CSV file
    with open(output_csv, 'w', newline='') as csvfile:
        csvwriter = csv.writer(csvfile)
        # Write header
        csvwriter.writerow(['image_url', 'seedling_characteristic', 'leaf'])
        # Write image data
        for row in image_data:
            csvwriter.writerow(row)

# Set your root directory and output CSV file
root_directory = 'path/to/your/blocked_images'
output_csv_file = 'output.csv'

# Create the CSV file
create_csv_from_folders(root_directory, output_csv_file)
