import logging
import random
from pathlib import Path
from typing import Literal, List, Optional

import torch
from kornia.geometry import vflip, hflip, resize
from torch import Tensor, Size
from torch.utils.data import Dataset
from torchvision.ops import box_convert
from torchvision.transforms import Resize
from torchvision.utils import draw_bounding_boxes

from config import ConfigDict
from loader.utils.checker import check_mask, check_image, check_labels, check_iqa, get_max_size
from loader.utils.reader import gray_read, ycbcr_read, label_read, img_write, label_write
from tools.scenario_reader import scenario_counter, generate_meta


class MSRS(Dataset):
    type = 'fuse & detect'  # dataset type: 'fuse' or 'fuse & detect'
    color = True  # dataset visible format: false -> 'gray' or true -> 'color'
    # MSRS Classes: Car, Person, Bike
    # Note: YOLO labels were generated with ID 0:car, 1:person, 2:bike.
    # The list order here must match the ID order.
    classes = ['Car', 'Person', 'Bike']
    palette = ['#C1C337', '#FF0000', '#2FA7B4'] # Arbitrary colors

    generate_meta_lock = False  # generate meta once

    def __init__(self, root: str | Path, mode: Literal['train', 'val', 'pred'], config: ConfigDict):
        super().__init__()
        root = Path(root)
        # Verify MSRS structure exists
        if not (root / 'train').exists():
             logging.warning(f"MSRS train folder not found at {root}/train")

        # Determine sub-root based on mode
        # MSRS structure: root/train/ir, root/test/ir
        if mode == 'train':
            self.root = root / 'train'
        else:
            self.root = root / 'test'
            
        self.mode = mode
        self.config = config

        # We skip json meta generation because we manually generated txt files
        # Note: meta files are in root/meta, not root/train/meta
        img_list_path = root / 'meta' / f'{mode}.txt'
        if not img_list_path.exists():
             logging.fatal(f"Meta file not found: {img_list_path}")
             raise FileNotFoundError(f"{img_list_path}")

        # read corresponding list
        img_list = img_list_path.read_text().splitlines()
        logging.info(f'load {len(img_list)} images from {self.root.name}')
        self.img_list = img_list

        # check images (using self.root which points to train/ or test/)
        check_image(self.root, img_list)

        # check labels
        self.labels = check_labels(self.root, img_list)

        # more check
        match mode:
            case 'train' | 'val':
                # check mask cache
                logging.info(f"Checking/Generating Saliency Masks for {len(img_list)} images...")
                check_mask(self.root, img_list, config)
                # check iqa cache
                logging.info(f"Checking/Generating IQA weights for {len(img_list)} images...")
                check_iqa(self.root, img_list, config)
            case _:
                # get max shape
                self.max_size = get_max_size(self.root, img_list)
                self.transform_fn = Resize(size=self.max_size)

    def __len__(self) -> int:
        return len(self.img_list)

    def __getitem__(self, index: int) -> dict:
        # choose get item method
        match self.mode:
            case 'train' | 'val':
                return self.train_val_item(index)
            case _:
                return self.pred_item(index)

    def train_val_item(self, index: int) -> dict:
        # image name, like '00001D.png'
        name = self.img_list[index]
        logging.debug(f'train-val mode: loading item {name}')

        # load infrared and visible
        ir = gray_read(self.root / 'ir' / name)
        vi, cbcr = ycbcr_read(self.root / 'vi' / name)

        # load mask (if checker generated it)
        if (self.root / 'mask' / name).exists():
             mask = gray_read(self.root / 'mask' / name)
        else:
            # Fallback if checker hasn't run yet or failed?
            # Ideally checker runs in __init__
            mask = ir.clone() # Placeholder? Or should we error?

        # load information measurement
        ir_w = gray_read(self.root / 'iqa' / 'ir' / name)
        vi_w = gray_read(self.root / 'iqa' / 'vi' / name)

        # load label
        label_p = Path(name).stem + '.txt'
        labels = label_read(self.root / 'labels' / label_p)

        # concat images for transform(s)
        t = torch.cat([ir, vi, mask, ir_w, vi_w, cbcr], dim=0)

        # transform (resize)
        resize_fn = Resize(size=self.config.train.image_size)
        t = resize_fn(t)

        # transform (flip up-down)
        if random.random() < self.config.dataset.detect.flip_ud:
            t = vflip(t)
            if len(labels):
                labels[:, 2] = 1 - labels[:, 2]

        # transform (flip left-right)
        if random.random() < self.config.dataset.detect.flip_lr:
            t = hflip(t)
            if len(labels):
                labels[:, 1] = 1 - labels[:, 1]

        # transform labels (cls, x1, y1, x2, y2) -> (0, cls, ...)
        labels_o = torch.zeros((len(labels), 6))
        if len(labels):
            labels_o[:, 1:] = labels

        # unpack images
        ir, vi, mask, ir_w, vi_w, cbcr = torch.split(t, [1, 1, 1, 1, 1, 2], dim=0)

        # merge data
        sample = {
            'name': name,
            'ir': ir, 'vi': vi,
            'ir_w': ir_w, 'vi_w': vi_w, 'mask': mask, 'cbcr': cbcr,
            'labels': labels_o
        }

        # return as expected
        return sample

    def pred_item(self, index: int) -> dict:
        # similar to m3fd
        name = self.img_list[index]
        ir = gray_read(self.root / 'ir' / name)
        vi, cbcr = ycbcr_read(self.root / 'vi' / name)
        s = ir.shape[1:]
        t = torch.cat([ir, vi, cbcr], dim=0)
        ir, vi, cbcr = torch.split(self.transform_fn(t), [1, 1, 2], dim=0)
        sample = {'name': name, 'ir': ir, 'vi': vi, 'cbcr': cbcr, 'shape': s}
        return sample

    @staticmethod
    def pred_save(fus: Tensor, names: List[str | Path], shape: List[Size], pred: Optional[Tensor] = None, save_txt: bool = False):
        if pred is None:
            return MSRS.pred_save_no_boxes(fus, names, shape)
        return MSRS.pred_save_with_boxes(fus, names, shape, pred, save_txt)

    @staticmethod
    def pred_save_no_boxes(fus: Tensor, names: List[str | Path], shape: List[Size]):
         for img_t, img_p, img_s in zip(fus, names, shape):
            img_t = resize(img_t, img_s)
            img_write(img_t, img_p)

    @staticmethod
    def pred_save_with_boxes(fus: Tensor, names: List[str | Path], shape: List[Size], pred: Tensor, save_txt: bool = False):
        for img_t, img_p, img_s, pred_i in zip(fus, names, shape, pred):
            cur_s = img_t.shape[1:]
            scale_x, scale_y = cur_s[1] / img_s[1], cur_s[0] / img_s[0]
            pred_i[:, :4] *= Tensor([scale_x, scale_y, scale_x, scale_y]).to(pred_i.device)
            img_t = resize(img_t, img_s)
            img = (img_t.clamp_(0, 1) * 255).to(torch.uint8)
            pred_x = list(filter(lambda x: x[4] > 0.4, pred_i))  # Threshold 0.4
            boxes = [x[:4] for x in pred_x]
            cls_idx = [int(x[5].cpu().numpy()) for x in pred_x]
            labels = [f'{MSRS.classes[cls]}: {x[4].cpu().numpy():.2f}' for cls, x in zip(cls_idx, pred_x)]
            colors = [MSRS.palette[cls] for cls, x in zip(cls_idx, pred_x)]
            if len(boxes):
                img = draw_bounding_boxes(img, torch.stack(boxes, dim=0), labels, colors, width=2)
            img = img.float() / 255
            img_p = Path(img_p.parent) / 'images' / img_p.name
            img_write(img, img_p)
            if save_txt:
                 # similar logic
                 pass

    @staticmethod
    def collate_fn(data: List[dict]) -> dict:
         # same as m3fd
        keys = data[0].keys()
        new_data = {}
        for key in keys:
            k_data = [d[key] for d in data]
            match key:
                case 'name' | 'shape':
                    new_data[key] = k_data
                case 'labels':
                    for i, lb in enumerate(k_data):
                        lb[:, 0] = i
                    new_data[key] = torch.cat(k_data, dim=0)
                case _:
                    new_data[key] = torch.stack(k_data, dim=0)
        return new_data
