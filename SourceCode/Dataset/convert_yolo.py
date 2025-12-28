import os
import cv2
import numpy as np
from tqdm import tqdm
from PIL import Image

# Configuration
DATASET_ROOT = 'MSRS'
SETS = ['train', 'test']
LABEL_FOLDER = 'Segmentation_labels'
OUTPUT_LABEL_FOLDER = 'labels'

# Class Mapping
# MSRS ID -> YOLO ID
# 1: car, 2: person, 3: bike
CLASS_MAPPING = {
    1: 0, # car
    2: 1, # person
    3: 2  # bike
}
YOLO_CLASSES = ['car', 'person', 'bike']

def convert_to_yolo_format(img_width, img_height, x, y, w, h):
    x_center = (x + w / 2) / img_width
    y_center = (y + h / 2) / img_height
    norm_w = w / img_width
    norm_h = h / img_height
    return x_center, y_center, norm_w, norm_h

def process_set(subset):
    print(f"Processing {subset} set...")
    mask_dir = os.path.join(DATASET_ROOT, subset, LABEL_FOLDER)
    output_dir = os.path.join(DATASET_ROOT, subset, OUTPUT_LABEL_FOLDER)
    
    os.makedirs(output_dir, exist_ok=True)
    
    files = os.listdir(mask_dir)
    for filename in tqdm(files):
        if not filename.lower().endswith(('.png', '.jpg', '.jpeg', '.tif')):
            continue
            
        file_path = os.path.join(mask_dir, filename)
        
        # Load mask
        # Open as PIL and convert to numpy ensuring we get the IDs
        try:
            mask = np.array(Image.open(file_path))
        except Exception as e:
            print(f"Error reading {file_path}: {e}")
            continue

        height, width = mask.shape[:2]
        
        yolo_labels = []
        
        # Check for each class we care about
        for msrs_id, yolo_id in CLASS_MAPPING.items():
            # Create binary mask for this class
            class_mask = (mask == msrs_id).astype(np.uint8)
            
            # Find contours
            contours, _ = cv2.findContours(class_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            for cnt in contours:
                x, y, w, h = cv2.boundingRect(cnt)
                
                # Filter small artifacts (optional)
                if w < 5 or h < 5:
                    continue
                    
                xc, yc, nw, nh = convert_to_yolo_format(width, height, x, y, w, h)
                
                # YOLO format: class_id xc yc w h
                yolo_labels.append(f"{yolo_id} {xc:.6f} {yc:.6f} {nw:.6f} {nh:.6f}")
        
        # Save output
        txt_filename = os.path.splitext(filename)[0] + '.txt'
        txt_path = os.path.join(output_dir, txt_filename)
        
        with open(txt_path, 'w') as f:
            if yolo_labels:
                f.write('\n'.join(yolo_labels))
            else:
                # Create empty file if no objects found (important for YOLO)
                pass

    print(f"Finished {subset} set. Labels saved to {output_dir}")

def generate_yaml():
    yaml_content = f"""
path: ../SourceCode/Dataset/MSRS # dataset root dir
train: train/ir  # train images (relative to 'path') 
val: test/ir     # val images (relative to 'path')
test: test/ir    # test images (optional)

# Classes
names:
  0: car
  1: person
  2: bike
"""
    with open('MSRS/msrs_detection.yaml', 'w') as f:
        f.write(yaml_content)
    print("Generated MSRS/msrs_detection.yaml")

if __name__ == '__main__':
    # Ensure raw directory MSRS exists relative to script
    if not os.path.exists(os.path.join(DATASET_ROOT, 'train')):
        print(f"Error: {DATASET_ROOT} not found. Please run this script from SourceCode/Dataset/")
        exit(1)
        
    process_set('train')
    process_set('test')
    generate_yaml()
