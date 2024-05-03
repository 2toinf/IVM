import torch
from PIL import Image
from transformers import pipeline
import mmengine.fileio as fileio
import io
import numpy as np
import utils
from tqdm import tqdm
import cv2


class Segmentation_Auto_Processor:
    def __init__(self, 
                 image_root,
                 save_root,
                 mask_model = "facebook/sam-vit-huge",
                 device = "cuda" if torch.cuda.is_available() else "cpu") -> None:
        '''
        Get Panoptic Segments and CLIP scores, with SAM model and CLIP model, respectively
        Args:
        save_root (str): local or ceph path to store the processed data
        img_size_limit (int): Checkpoint filename.
        file_client_args (dict, optional): Arguments to instantiate
            a FileClient. See :class:`mmengine.fileio.FileClient`
            for details.Default: None.
        
        '''
        self.device = device
        self.image_root = image_root
        if self.device == 'cuda':
            self.device = f'cuda:{utils.get_rank() % torch.cuda.device_count()}'
        self.dtype = torch.float16 if device=='cuda' else torch.float32
        self.sam_model = pipeline("mask-generation", model=mask_model, device=self.device)
        print("============load sam model==========")
        self.save_dir = save_root
        print(f"============Data will be saved in \"{self.save_dir}\"=================")

    def read_image(self, img_path):
        value = fileio.get(img_path)
        img_bytes = np.frombuffer(value, np.uint8)
        buff = io.BytesIO(img_bytes)
        img = Image.open(buff).convert('RGB')
        return img

    @torch.no_grad()
    def procees_image(self, image_path: str):
        
        img = self.read_image(fileio.join_path(self.image_root, image_path))
        image_name = image_path.split('/')[-1].split('.')[0]
        image_path = '/'.join(image_path.split('/')[:-1])
        # if fileio.exists(fileio.join_path(self.save_dir, image_path, f'{image_name}.npz')): return
        # else: print(fileio.join_path(self.save_dir, image_path, f'{image_name}.npz'))
        
        masks = self.sam_model(img, points_per_batch=1)['masks']
        joint_mask = np.zeros_like(masks[0]).astype(np.uint32)
        for idx, mask in tqdm(enumerate(masks)):
            joint_mask[mask] = idx + 1

        assert (joint_mask >= 0).all()
        with io.BytesIO() as f:
            np.savez_compressed(f, joint_mask)
            fileio.put(f.getvalue(), fileio.join_path(self.save_dir, image_path, f'{image_name}.npz'))
        
        












