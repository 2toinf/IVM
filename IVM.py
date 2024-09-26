import cv2
from model.IVM import IVM
import torchvision.transforms.functional as TF
import torch
from PIL import Image
import numpy as np
from torch.nn import functional as F
import os
import gdown
import accelerate
import torch.nn as nn
from typing import Type, Tuple, List


def _download(url: str, name: str,root: str):
    os.makedirs(root, exist_ok=True)

    download_target = os.path.join(root, name)
    if os.path.exists(download_target) and not os.path.isfile(download_target):
        raise RuntimeError(f"{download_target} exists and is not a regular file")
    if os.path.isfile(download_target):
        return download_target
    gdown.download(url, download_target, quiet=False)
    return download_target

def load(ckpt_path, low_gpu_memory = False):
    url = "https://drive.google.com/uc?export=download&id=1OyVci6rAwnb2sJPxhObgK7AvlLYDLLHw"
    sam_ckpt = _download(url, "sam_vit_h_4b8939.pth", os.path.expanduser(f"~/.cache/IVM/Sam"))
    ckpt = torch.load(ckpt_path, map_location="cpu")
    model = IVM(sam_model=sam_ckpt)
    model.load_state_dict(ckpt, strict=False)
    if low_gpu_memory: return accelerate.cpu_offload(model, "cuda")
    else: return model.cuda()


def auto_postprocess(mask, dilate_kernel_rate = 0.05):
    # TODO: Need to be adjusted according to different datasets 
    dilate_kernel_size = int(mask.shape[0] * dilate_kernel_rate) * 2 + 1
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(dilate_kernel_size,dilate_kernel_size)) #ksize=7x7,
    mask = cv2.dilate(mask, kernel,iterations=1).astype(np.float32)
    mask = cv2.GaussianBlur(mask, (dilate_kernel_size, dilate_kernel_size), 0)[:,:,np.newaxis]
    return mask

@torch.no_grad()
def forward_batch(
    model, 
    image, # list of PIL.Image
    instruction: List[str], # list of instruction
    threshold: float = 0.1, # threshold for pixel reserve/drop
    do_crop = False,
    overlay_color = (255,255,255)
):
    ori_sizes = [img.size for img in image]
    ori_images = [np.asarray(img).astype(np.float32) for img in image]
    masks = model.generate_batch([img.resize((1024, 1024)) for img in image], instruction)

    result = []
    for mask, ori_image, ori_size in zip(masks, ori_images, ori_sizes):
        mask = torch.sigmoid(F.interpolate(
            mask.unsqueeze(0),
            (ori_size[1], ori_size[0]),
            mode="bilinear",
            align_corners=False,
        )[0, 0, :, :]).detach().cpu().numpy().astype(np.float32)[:,:,np.newaxis]
        if threshold > mask.max(): mask += threshold # fail to find the target, reserve the full image
        mask = auto_postprocess((mask > threshold).astype(np.float32))

        if len(ori_image.shape) < 3: ori_image = ori_image[:,:,np.newaxis].repeat(3,-1)
        
        processed_image = ori_image * mask + torch.tensor(overlay_color, dtype=torch.uint8).repeat(ori_size[1], ori_size[0], 1).numpy() * (1 - mask)
        if do_crop:
            try:
                y_indices, x_indices = np.where(mask[:,:,0] > 0)
                x_min, x_max = x_indices.min(), x_indices.max()
                y_min, y_max = y_indices.min(), y_indices.max()
                processed_image = processed_image[y_min:y_max+1, x_min:x_max+1]
            except:
                print("Warning, unable to crop a sample, reserve whole image")
        result.append(processed_image)
    return result