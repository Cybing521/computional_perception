import torch
from module.fuse.generator import Generator

def check():
    print("--- Model Keys ---")
    model = Generator(dim=32, depth=3)
    model_keys = set(model.state_dict().keys())
    # print first 5
    print(list(model_keys)[:5])
    
    print("\n--- Checkpoint Keys ---")
    try:
        ckpt = torch.load('weights/v1/tardal-ct.pth', map_location='cpu')
        
        if 'fuse' in ckpt:
            print("Found 'fuse' key in ckpt, using it.")
            ckpt = ckpt['fuse']
        else:
            print("checkpoint keys:", list(ckpt.keys())[:5])
            
        ckpt_keys = set(ckpt.keys())
        print(list(ckpt_keys)[:5])
        
        print("\n--- Mismatch ---")
        missing = model_keys - ckpt_keys
        unexpected = ckpt_keys - model_keys
        
        print(f"Missing (In Model, Not in File): {len(missing)}")
        if missing: print(list(missing)[:5])
        
        print(f"Unexpected (In File, Not in Model): {len(unexpected)}")
        if unexpected: print(list(unexpected)[:5])
            
    except Exception as e:
        print(f"Error loading checkpoint: {e}")

if __name__ == '__main__':
    check()
