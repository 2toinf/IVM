import cv2
import model
from model.fam import FilterAnything, T5Encoder
from timm import create_model
import torch
from PIL import Image
import numpy as np
from torch.nn import functional as F
import os
import gdown

def _download(url: str, name: str,root: str):
    os.makedirs(root, exist_ok=True)

    download_target = os.path.join(root, name)
    if os.path.exists(download_target) and not os.path.isfile(download_target):
        raise RuntimeError(f"{download_target} exists and is not a regular file")
    if os.path.isfile(download_target):
        return download_target
    gdown.download(url, download_target, quiet=False)
    return download_target

def load(ckpt_path, device = "cuda"):


    url = "https://drive.google.com/uc?export=download&id=1pyx7jo8kKjNvPZ1kqk00u9Ujla0-YFYT"
    model_path = _download(url, "sam_vit_b_01ec64.pth", os.path.expanduser(f"~/.cache/SeeWhatYouNeed/Sam"))
    model = FamDeploy(
        decoder_ckpt=ckpt_path,
        backbone_name="sam_backbone_base",
        language_encoder_path = "google-t5/t5-base",
        backbone_pretrained_path = model_path,
        device=device
    )
    return model


class FamDeploy:
    def __init__(self,
            decoder_ckpt,
            decoder_name = 'decoder_base',
            backbone_name = 'sam_backbone_base',
            backbone_pretrained_path = "~/.cache/torch/hub/checkpoints/sam_vit_b_01ec64.pth",
            language_encoder_path = "google-t5/t5-base",
            device = "cuda"
        ):
        backbone = create_model(
            backbone_name,
            pretrained_ckpt = backbone_pretrained_path
        )
        language_encoder = T5Encoder(
            language_encoder_path, device = device
        )
        decoder = create_model(
            decoder_name
        )
        ckpt = torch.load(decoder_ckpt, map_location="cpu")
        if 'model' in ckpt: ckpt = ckpt['model']
        decoder.load_state_dict(ckpt)
        self.model = FilterAnything(
            backbone,
            language_encoder,
            decoder
        ).to(device)
        self.device = device


    @torch.no_grad()
    def __call__(self, 
        image: Image,
        instrution: str,
        blur_kernel_size = 201,
        threshold = 0.8):
        
        ori_size = image.size
        ori_image = np.asarray(image).astype(np.float32)
        image = self.model.preprocess(torch.from_numpy(np.asarray(image.resize((1024, 1024))).astype(np.float32)).to(self.device).permute(2, 0, 1).unsqueeze(0))
        image_embeddings = self.model.image_encoder(image)
        language_embeddings = self.model.prompt_encoder.embed_text([instrution],False)
        masks = self.model.mask_decoder(image_embeddings, language_embeddings)
        masks = self.model.postprocess_masks(masks, (1024, 1024))
        masks = F.interpolate(
            masks,
            (ori_size[1], ori_size[0]),
            mode="bilinear",
            align_corners=False,
        )
        masks = masks.detach().cpu().numpy()
        human_mask = masks[0,2].astype(np.float32)
        human_mask = ((human_mask - human_mask.min()) / (human_mask.max() - human_mask.min()))[:,:,np.newaxis]
        human_mask = np.where(human_mask > threshold, human_mask, 0)

        rgba = np.concatenate((ori_image, human_mask * 255), axis=-1)
        ori_blurred_image = cv2.GaussianBlur(ori_image, (blur_kernel_size, blur_kernel_size), 0)  
        blur_image = human_mask * ori_image + (1-human_mask) * ori_blurred_image
        highlight_image = ori_image * human_mask 
        
        y_indices, x_indices = np.where(human_mask[:,:,0] > 0)

        # 计算裁剪边界
        x_min, x_max = x_indices.min(), x_indices.max()
        y_min, y_max = y_indices.min(), y_indices.max()

        # 根据边界裁剪图片
        cropped_blur_img = blur_image[y_min:y_max+1, x_min:x_max+1]
        cropped_highlight_img = highlight_image[y_min:y_max+1, x_min:x_max+1]
        return {
            'soft': masks[0, 0],
            'hard': human_mask,
            'blur_image': blur_image,
            'highlight_image': highlight_image,
            'cropped_blur_img': cropped_blur_img,
            'cropped_highlight_img': cropped_highlight_img,
            'alhpa_image': rgba
        }



