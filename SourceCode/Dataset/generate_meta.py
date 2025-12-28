import os
import random

DATASET_ROOT = 'MSRS'
META_DIR = os.path.join(DATASET_ROOT, 'meta')
os.makedirs(META_DIR, exist_ok=True)

def generate_list(subset, mode_name):
    # Use IR images as the source of truth
    img_dir = os.path.join(DATASET_ROOT, subset, 'ir')
    if not os.path.exists(img_dir):
        print(f"Error: {img_dir} does not exist.")
        return

    files = sorted([f for f in os.listdir(img_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.tif', '.bmp'))])
    
    txt_path = os.path.join(META_DIR, f'{mode_name}.txt')
    with open(txt_path, 'w') as f:
        f.write('\n'.join(files))
    
    print(f"Generated {txt_path} with {len(files)} images.")

if __name__ == '__main__':
    # train -> train
    generate_list('train', 'train')
    # test -> val (TarDAL uses 'val' for validation)
    generate_list('test', 'val')
    # test -> test (optional)
    generate_list('test', 'test')
