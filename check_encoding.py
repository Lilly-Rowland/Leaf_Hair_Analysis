import json
import os

def check_encoding(folder_path):
    for file_name in os.listdir(folder_path):
        if file_name.endswith('.json'):
            file_path = os.path.join(folder_path, file_name)
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    json.load(f)
                print(f"{file_name} - OK")
            except UnicodeDecodeError as e:
                print(f"{file_name} - Encoding Error: {e}")

if __name__ == "__main__":
    folder_path = '/Volumes/Image Data /COCO_Annotations'
    check_encoding(folder_path)
