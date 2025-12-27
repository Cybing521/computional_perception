import torch

def check():
    print("--- Checkpoint Keys (TT) ---")
    try:
        ckpt = torch.load('weights/v1/tardal-tt.pth', map_location='cpu')
        keys = list(ckpt.keys())
        print(keys[:10])
        
        has_yolo = any('model.0' in k for k in keys)
        print(f"Has YOLO keys: {has_yolo}")
        
    except Exception as e:
        print(f"Error: {e}")

if __name__ == '__main__':
    check()
