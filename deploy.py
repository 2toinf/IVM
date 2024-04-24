import cv2
from model.LISA import LISAWithDiscriminator
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
def _download(url: str, name: str,root: str):
    os.makedirs(root, exist_ok=True)

    download_target = os.path.join(root, name)
    if os.path.exists(download_target) and not os.path.isfile(download_target):
        raise RuntimeError(f"{download_target} exists and is not a regular file")
    if os.path.isfile(download_target):
        return download_target
    gdown.download(url, download_target, quiet=False)
    return download_target

def load(ckpt_path, type = "fam", low_gpu_memory = False):

    url = "https://drive.google.com/uc?export=download&id=1OyVci6rAwnb2sJPxhObgK7AvlLYDLLHw"
    model_path = _download(url, "sam_vit_h_4b8939.pth", os.path.expanduser(f"~/.cache/SeeWhatYouNeed/Sam"))
    if type == 'lisa':
        model = DeployModel_LISA(
            ckpt_path = ckpt_path,
            sam_ckpt=model_path,
            offload_languageencoder=low_gpu_memory
        )
    else:
        model = DeployModel_FAM(
            ckpt_path = ckpt_path,
            sam_ckpt=model_path
        ).cuda().half()
    return model




class DeployModel_FAM(nn.Module):
    def __init__(self,
            ckpt_path,
            sam_ckpt
        ):
        super().__init__()
        self.model = FAM(
            sam_model=sam_ckpt
        )
        ckpt = torch.load(ckpt_path, map_location="cpu")
        print(self.model.load_state_dict(ckpt, strict=False))


    @torch.no_grad()
    def forward(self, 
        image: Image,
        instruction: str,
        blur_kernel_size = 201,
        threshold = 0.8):
        
        ori_size = image.size
        ori_image = np.asarray(image).astype(np.float32)
        images = TF.to_tensor(image.resize((1024, 1024))).unsqueeze(0).cuda().half()
        images_for_sam = (images - self.model.pixel_mean.unsqueeze(0)) / self.model.pixel_std.unsqueeze(0)
        image_embeddings = self.model.image_encoder(images_for_sam)
        language_embeddings = self.model.prompt_encoder([instruction], images)
        masks = self.model.mask_decoder(image_embeddings, language_embeddings)
        masks = torch.sigmoid(F.interpolate(
            masks,
            (ori_size[1], ori_size[0]),
            mode="bilinear",
            align_corners=False,
        )[0, 0, :, :]).detach().cpu().numpy().astype(np.float32)[:,:,np.newaxis]
        mask_output = np.where(masks > threshold, masks, 0)

        rgba = np.concatenate((ori_image, masks * 255), axis=-1)
        ori_blurred_image = cv2.GaussianBlur(ori_image, (blur_kernel_size, blur_kernel_size), 0)  
        blur_image = mask_output * ori_image + (1-mask_output) * ori_blurred_image
        highlight_image = ori_image * mask_output 
        
        y_indices, x_indices = np.where(mask_output[:,:,0] > 0)

        # 计算裁剪边界
        x_min, x_max = x_indices.min(), x_indices.max()
        y_min, y_max = y_indices.min(), y_indices.max()

        # 根据边界裁剪图片
        cropped_blur_img = blur_image[y_min:y_max+1, x_min:x_max+1]
        cropped_highlight_img = highlight_image[y_min:y_max+1, x_min:x_max+1]
        return {
            'soft': masks,
            'hard': mask_output,
            'blur_image': blur_image,
            'highlight_image': highlight_image,
            'cropped_blur_img': cropped_blur_img,
            'cropped_highlight_img': cropped_highlight_img,
            'alhpa_image': rgba
        }


class DeployModel_LISA(nn.Module):
    def __init__(self,
            ckpt_path,
            sam_ckpt,
            offload_imageencoder = False, 
            offload_languageencoder = True
        ):
        super().__init__()
        self.model = LISAWithDiscriminator(
            sam_model=sam_ckpt
        )
        ckpt = torch.load(ckpt_path, map_location="cpu")
        print(self.model.load_state_dict(ckpt, strict=False))
        self.model = self.model.half()
        if offload_imageencoder:
            self.model.image_encoder  = accelerate.cpu_offload(self.model.image_encoder , 'cuda')
        else:
            self.model.image_encoder = self.model.image_encoder.cuda()
        if offload_languageencoder:
            self.model.prompt_encoder = accelerate.cpu_offload(self.model.prompt_encoder , 'cuda')
        else:
            self.model.prompt_encoder = self.model.prompt_encoder.cuda()
        self.model.pixel_mean = self.model.pixel_mean.cuda()
        self.model.pixel_std = self.model.pixel_std.cuda()
        self.model.mask_decoder = self.model.mask_decoder.cuda()


    @torch.no_grad()
    def forward(self, 
        image: Image,
        instruction: str,
        blur_kernel_size = 201,
        threshold = 0.8):
        
        ori_size = image.size
        ori_image = np.asarray(image).astype(np.float32)
        images = TF.to_tensor(image.resize((1024, 1024))).unsqueeze(0).cuda().half()
        images_for_sam = (images - self.model.pixel_mean.unsqueeze(0)) / self.model.pixel_std.unsqueeze(0)
        image_embeddings = self.model.image_encoder(images_for_sam)
        language_embeddings = self.model.prompt_encoder([instruction], images)
        masks = self.model.mask_decoder(image_embeddings, language_embeddings)
        masks = torch.sigmoid(F.interpolate(
            masks,
            (ori_size[1], ori_size[0]),
            mode="bilinear",
            align_corners=False,
        )[0, 0, :, :]).detach().cpu().numpy().astype(np.float32)[:,:,np.newaxis]
        mask_output = np.where(masks > threshold, masks, 0)

        rgba = np.concatenate((ori_image, masks * 255), axis=-1)
        ori_blurred_image = cv2.GaussianBlur(ori_image, (blur_kernel_size, blur_kernel_size), 0)  
        blur_image = mask_output * ori_image + (1-mask_output) * ori_blurred_image
        highlight_image = ori_image * mask_output 
        
        y_indices, x_indices = np.where(mask_output[:,:,0] > 0)

        # 计算裁剪边界
        x_min, x_max = x_indices.min(), x_indices.max()
        y_min, y_max = y_indices.min(), y_indices.max()

        # 根据边界裁剪图片
        cropped_blur_img = blur_image[y_min:y_max+1, x_min:x_max+1]
        cropped_highlight_img = highlight_image[y_min:y_max+1, x_min:x_max+1]
        return {
            'soft': masks,
            'hard': mask_output,
            'blur_image': blur_image,
            'highlight_image': highlight_image,
            'cropped_blur_img': cropped_blur_img,
            'cropped_highlight_img': cropped_highlight_img,
            'alhpa_image': rgba
        }



