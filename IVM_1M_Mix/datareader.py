



from dis import Instruction
from torch.utils.data import Dataset, ConcatDataset
import json
import numpy as np
import torch
import io
from torchvision import transforms
from mmengine import fileio
from PIL import Image
from torchvision.transforms import functional as F
import random
import cv2

def read_masks(mask_path):
    img = cv2.imread(mask_path)
    heatmap = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    heatmap = cv2.convertScaleAbs(heatmap, alpha=1.5, beta=0)
    if heatmap.max() - heatmap.min() >= 1e-4: heatmap = (heatmap - heatmap.min()) / (heatmap.max() -heatmap.min())
    else: heatmap = heatmap / heatmap.max()
    return heatmap

def read_image(image_path):
    return Image.open(io.BytesIO(np.frombuffer(fileio.get(fileio.join_path(image_path)), np.uint8))).convert('RGB')


class RandomResizeCropPaddingWithMask(transforms.RandomResizedCrop):
    def __call__(self, img, mask):
        size = 1024
        mask = torch.from_numpy(mask).unsqueeze(0)
        i, j, h, w = self.get_params(img, self.scale, self.ratio)
        img = F.resized_crop(img, i, j, h, w, (size,size), self.interpolation)
        mask = F.resized_crop(mask, i, j, h, w, (size,size), self.interpolation)
        return img, mask.squeeze()


class IVMDataReader(Dataset):
    def __init__(self, 
        meta_file,
        image_root,
        mask_root
        ):
        super().__init__()
        with open(meta_file) as f:
            self.metas = f.readlines()
        random.shuffle(self.metas)
        self.image_root = image_root
        self.mask_root = mask_root
        self.data_augmentation = RandomResizeCropPaddingWithMask(1024, scale=(0.8, 1.0))

    def __len__(self):
        return len(self.metas)

    def __getitem__(self, index):
        try:
            item = json.loads(self.metas[index])
            uuid = item['uuid']
            if not 'image' in item.keys(): item['image'] = f"img_{uuid}.jpg"
            image = read_image(fileio.join_path(self.image_root, item['image']))
            mask = read_masks(fileio.join_path(self.mask_root,  f"mask_{uuid}.jpg"))
            image, mask = self.data_augmentation(image, mask)
            return (image, mask), random.choice(item['instruction'])
        except:
            print(f"Error when read {item}")
            return self.__getitem__((index+1) % self.__len__())


