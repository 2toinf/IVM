import cv2
from model.LISA_vanilla import LISA
from model.FAM import FAM
import torchvision.transforms.functional as TF
import torch
from PIL import Image
import numpy as np
from torch.nn import functional as F
import os
import gdown
import accelerate
import torch.nn as nn
from typing import Type, Tuple
def _download(url: str, name: str,root: str):
    os.makedirs(root, exist_ok=True)

    download_target = os.path.join(root, name)
    if os.path.exists(download_target) and not os.path.isfile(download_target):
        raise RuntimeError(f"{download_target} exists and is not a regular file")
    if os.path.isfile(download_target):
        return download_target
    gdown.download(url, download_target, quiet=False)
    return download_target

def load(ckpt_path, type = "lisa", low_gpu_memory = False):

    url = "https://drive.google.com/uc?export=download&id=1OyVci6rAwnb2sJPxhObgK7AvlLYDLLHw"
    model_path = _download(url, "sam_vit_h_4b8939.pth", os.path.expanduser(f"~/.cache/SeeWhatYouNeed/Sam"))
    model = DeployModel_LISA(
        ckpt_path = ckpt_path,
        sam_ckpt=model_path,
        offload_languageencoder=low_gpu_memory
    )
    return model



class DeployModel_LISA(nn.Module):
    def __init__(self,
            ckpt_path,
            sam_ckpt,
            offload_languageencoder = True
        ):
        super().__init__()
        self.model = LISA(
            sam_model=sam_ckpt
        )
        ckpt = torch.load(ckpt_path, map_location="cpu")
        print(ckpt.keys())
        print(self.model.load_state_dict(ckpt, strict=False))
        self.model = self.model.half()
        if offload_languageencoder:
            self.model.prompt_encoder = accelerate.cpu_offload(self.model.prompt_encoder , 'cuda')
        else:
            self.model.prompt_encoder = self.model.prompt_encoder.cuda()
        self.model.image_encoder = self.model.image_encoder.cuda()
        self.model.pixel_mean = self.model.pixel_mean.cuda()
        self.model.pixel_std = self.model.pixel_std.cuda()
        self.model.mask_decoder = self.model.mask_decoder.cuda()

    @torch.no_grad()
    def forward_batch(
        self, 
        image, # list of PIL.Image
        instruction, # list of instruction
        blur_kernel_size = 201,
        range_threshold = 0.5,
        boxes_threshold = 0.5,
        dilate_kernel_rate = 0.1,
        min_reserved_ratio = 0.2,
        fill_color=(255, 255, 255)
    ):
        ori_sizes = [img.size for img in image]
        ori_images = [np.asarray(img).astype(np.float32) for img in image]
        masks = self.model.generate_batch([img.resize((1024, 1024)) for img in image], instruction)

        
        soft = []
        blur_image = []
        highlight_image = []
        cropped_blur_img = []
        cropped_highlight_img = []
        rgba = []
        for mask, ori_image, ori_size in zip(masks, ori_images, ori_sizes):
            mask = torch.sigmoid(F.interpolate(
                mask.unsqueeze(0),
                (ori_size[1], ori_size[0]),
                mode="bilinear",
                align_corners=False,
            )[0, 0, :, :]).detach().cpu().numpy().astype(np.float32)[:,:,np.newaxis]
            dilate_kernel_size = int(ori_size[0] * dilate_kernel_rate)
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(dilate_kernel_size,dilate_kernel_size)) #ksize=7x7,
            mask = cv2.dilate(mask,kernel,iterations=1).astype(np.float32)
            mask = cv2.GaussianBlur(mask, (dilate_kernel_size, dilate_kernel_size), 0)[:,:,np.newaxis]
            if mask.max() - mask.min() > range_threshold:
                mask = (mask - mask.min()) / (mask.max() - mask.min()) * (1 - min_reserved_ratio)
            else:
                mask = np.ones_like(mask) * (1 - min_reserved_ratio)
            
            soft.append(mask)
            rgba.append(np.concatenate((ori_image, mask * 255), axis=-1))
            blur_image.append(mask * ori_image + (1-mask) * cv2.GaussianBlur(ori_image, (blur_kernel_size, blur_kernel_size), 0)) 
            highlight_image.append(ori_image * (mask + min_reserved_ratio) + torch.tensor(fill_color, dtype=torch.uint8).repeat(ori_size[1], ori_size[0], 1).numpy() * (1 - min_reserved_ratio - mask))
            
            
            try:
                y_indices, x_indices = np.where(mask[:,:,0] > boxes_threshold)
                x_min, x_max = x_indices.min(), x_indices.max()
                y_min, y_max = y_indices.min(), y_indices.max()
                cropped_blur_img.append(blur_image[-1][y_min:y_max+1, x_min:x_max+1])
                cropped_highlight_img.append(highlight_image[-1][y_min:y_max+1, x_min:x_max+1])
            except:
                cropped_blur_img.append(blur_image[-1])
                cropped_highlight_img.append(highlight_image[-1])
        return {
            'soft': soft,
            'blur_image': blur_image,
            'highlight_image': highlight_image,
            'cropped_blur_img': cropped_blur_img,
            'cropped_highlight_img': cropped_highlight_img,
            'rgba_image': rgba
        }


    @torch.no_grad()
    def forward(
        self, 
        image: Image,
        instruction: str,
        blur_kernel_size = 401,
        crop_threshold = 0.5,
        range_threshold = 0.5,
        dilate_kernel_rate = 0.1,
        min_reserved_ratio = 0.1,
        fill_color=(255, 255, 255)):
        
        ori_size = image.size
        ori_image = np.asarray(image).astype(np.float32)
        masks = self.model.generate(image.resize((1024, 1024)), instruction)
        masks = torch.sigmoid(F.interpolate(
            masks,
            (ori_size[1], ori_size[0]),
            mode="bilinear",
            align_corners=False,
        )[0, 0, :, :]).detach().cpu().numpy().astype(np.uint8)[:,:,np.newaxis]
        dilate_kernel_size = int(ori_size[0] * dilate_kernel_rate)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(dilate_kernel_size,dilate_kernel_size)) #ksize=7x7,
        masks = cv2.dilate(masks,kernel,iterations=1).astype(np.float32)
        masks = cv2.GaussianBlur(masks, (dilate_kernel_size, dilate_kernel_size), 0)[:,:,np.newaxis]
        if masks.max() - masks.min() > range_threshold:
            masks = (masks - masks.min()) / (masks.max() - masks.min()) * (1 - min_reserved_ratio)
        else:
            masks = np.ones_like(masks)* (1 - min_reserved_ratio)

        rgba = np.concatenate((ori_image, masks * 255), axis=-1)
        ori_blurred_image = cv2.GaussianBlur(ori_image, (blur_kernel_size, blur_kernel_size), 0)  
        blur_image = masks * ori_image + (1-masks) * ori_blurred_image

        fill_tensor = torch.tensor(fill_color, dtype=torch.uint8).repeat(image.size[1], image.size[0], 1)
        highlight_image = ori_image * (masks + min_reserved_ratio) + fill_tensor.numpy() * (1 - min_reserved_ratio - masks)
        try:
            y_indices, x_indices = np.where(masks[:,:,0] > crop_threshold)
            # 计算裁剪边界
            x_min, x_max = x_indices.min(), x_indices.max()
            y_min, y_max = y_indices.min(), y_indices.max()

            # 根据边界裁剪图片
            cropped_blur_img = blur_image[y_min:y_max+1, x_min:x_max+1]
            cropped_highlight_img = highlight_image[y_min:y_max+1, x_min:x_max+1]
        except:
            cropped_blur_img = blur_image
            cropped_highlight_img = highlight_image
        return {
            'soft': masks,
            'blur_image': blur_image,
            'highlight_image': highlight_image,
            'cropped_blur_img': cropped_blur_img,
            'cropped_highlight_img': cropped_highlight_img,
            'alpha_image': rgba
        }





