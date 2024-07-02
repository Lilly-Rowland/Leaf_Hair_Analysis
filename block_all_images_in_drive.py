import os
import create_image_tiles_by_leaf_id
import json

def load_checkpoint(checkpoint_file):
    if os.path.exists(checkpoint_file):
        with open(checkpoint_file, 'r') as f:
            return json.load(f)
    return {}

def save_checkpoint(checkpoint_file, data):
    with open(checkpoint_file, 'w') as f:
        json.dump(data, f)

def is_hidden_or_system_directory(directory):
    return directory.startswith('.') or directory in ('System Volume Information', '$RECYCLE.BIN')

def create_folders_in_new_directory(source_dir, destination_dir, checkpoint_file):
    checkpoint_data = load_checkpoint(checkpoint_file)

    if not os.path.exists(destination_dir):
        os.makedirs(destination_dir)
    
    for item in os.listdir(source_dir):
        if is_hidden_or_system_directory(item):
            continue  # Skip hidden or system directories

        seed_char_path = os.path.join(source_dir, item)
        if not os.path.isdir(seed_char_path):
            continue  # Skip non-directory items

        folder_name = "sc_" + item[-8:]
        
        if item in checkpoint_data and checkpoint_data[item] == "completed":
            continue

        for day in os.listdir(seed_char_path):
            if is_hidden_or_system_directory(day):
                continue  # Skip hidden or system directories

            if day.endswith("3dpi"):
                folder_name_with_3dpi = folder_name + "_3dpi_"
                day_path = os.path.join(seed_char_path, day)

                for tray in os.listdir(day_path):
                    if is_hidden_or_system_directory(tray):
                        continue  # Skip hidden or system directories

                    tray_path = os.path.join(day_path, tray)
                    if not os.path.isdir(tray_path):
                        continue  # Skip non-directory items

                    folder_name_with_tray = folder_name_with_3dpi + tray
                    tray_folder_path = os.path.join(destination_dir, folder_name_with_tray)

                    if not os.path.exists(tray_folder_path):
                        os.makedirs(tray_folder_path, exist_ok=True)
                    
                    checkpoint_data[item] = tray
                    save_checkpoint(checkpoint_file, checkpoint_data)
                    
                    create_image_tiles_by_leaf_id.split_images_in_directory(tray_path, tray_folder_path)
                    
                    checkpoint_data[item] = "completed"
                    save_checkpoint(checkpoint_file, checkpoint_data)

source_directory = "/Volumes/LeafHair"
destination_directory = "/Volumes/LeafHair/Blocked_Images"
checkpoint_file = "checkpoint.json"

create_folders_in_new_directory(source_directory, destination_directory, checkpoint_file)
