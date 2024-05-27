# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


from syslog import LOG_SYSLOG
from PIL import Image
import torch
from torch import nn
from torch.nn import functional as F
from torchvision.transforms import functional as TF
from typing import Any, List, Tuple
from functools import partial

from model.decoder import ConditionedMaskDecoder
from model.discriminator import Discriminator
from model.llava_encoder import LLaVAEncoder
from model.sam_components.image_encoder import ImageEncoderViT
from model.sam_components.transformer import TwoWayTransformer
import numpy as np
def dice_loss(
	inputs: torch.Tensor,
	targets: torch.Tensor,
	scale=1000,  # 100000.0,
	eps=1e-6,
):
    inputs = inputs.sigmoid()
    inputs = inputs.flatten(1, 2)
    targets = targets.flatten(1, 2)
    numerator = 2 * (inputs / scale * targets).sum(-1)
    denominator = (inputs / scale).sum(-1) + (targets / scale).sum(-1)
    loss = 1 - (numerator + eps) / (denominator + eps)
    return loss


def sigmoid_ce_loss(inputs: torch.Tensor, targets: torch.Tensor):
    loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
    return loss.flatten(1, 2).mean(1)


class IVM(nn.Module):

    def __init__(
        self,
        sam_model = "/mnt/lustre/zhengjinliang/.cache/torch/hub/checkpoints/sam_vit_h_4b8939.pth",
        pixel_mean: List[float] = [0.485, 0.456, 0.406] ,#[123.675, 116.28, 103.53],
        pixel_std: List[float] = [0.229, 0.224, 0.225], #[58.395, 57.12, 57.375],
    ) -> None:
        """
        SAM predicts object masks from an image and input prompts.

        Arguments:
          image_encoder (ImageEncoderViT): The backbone used to encode the
            image into image embeddings that allow for efficient mask prediction.
          prompt_encoder (PromptEncoder): Encodes various types of input prompts.
          mask_decoder (MaskDecoder): Predicts masks from the image embeddings
            and encoded prompts.
          pixel_mean (list(float)): Mean values for normalizing pixels in the input image.
          pixel_std (list(float)): Std values for normalizing pixels in the input image.
        """
        super().__init__()
        self.image_encoder = ImageEncoderViT(
            depth=32,
            embed_dim=1280,
            img_size=1024,
            mlp_ratio=4,
            norm_layer=partial(torch.nn.LayerNorm, eps=1e-6),
            num_heads=16,
            patch_size=16,
            qkv_bias=True,
            use_rel_pos=True,
            global_attn_indexes=[7, 15, 23, 31],
            window_size=14,
            out_chans=256,
        )
        ckpt = torch.load(sam_model, map_location="cpu")
        backbone_ckpt = {}
        for key, value in ckpt.items():
            if 'image_encoder.' in key:
                backbone_ckpt[key.replace('image_encoder.', '')] = value
        self.image_encoder.load_state_dict(backbone_ckpt)
        self.image_encoder.requires_grad_(False)

        self.prompt_encoder = LLaVAEncoder()
        
        self.mask_decoder = ConditionedMaskDecoder(
            transformer_dim=256,
            transformer=TwoWayTransformer(
                depth=2,
                embedding_dim=256,
                mlp_dim=2048,
                num_heads=8,
            ),
            learnable_token_num=1
        )
        self.register_buffer("pixel_mean", torch.Tensor(pixel_mean).view(-1, 1, 1), False)
        self.register_buffer("pixel_std", torch.Tensor(pixel_std).view(-1, 1, 1), False)

    @property
    def device(self) -> Any:
        return self.pixel_mean.device


    def generate(self, images, instruction):
        image_embeddings = self.image_encoder(self.preprocess(images).unsqueeze(0))
        language_embeddings = self.prompt_encoder([instruction], [images])
        masks = self.mask_decoder(image_embeddings, language_embeddings)
        mask_output = self.postprocess_masks(masks)
        return mask_output
    
    def generate_batch(self, images, instruction):
        image_embeddings = self.image_encoder(torch.stack([self.preprocess(img) for img in images], dim = 0))
        language_embeddings = self.prompt_encoder(instruction, images)
        masks = self.mask_decoder(image_embeddings, language_embeddings)
        mask_output = self.postprocess_masks(masks)
        return mask_output

    def forward(
        self,
        images, # List[PIL.Image]
        gt_mask, # B H W
        instruction, # List(str)
        data_label
    ):
        image_embeddings = self.image_encoder(torch.stack([self.preprocess(image) for image in images]))
        language_embeddings = self.prompt_encoder(instruction, images)
        masks = self.mask_decoder(image_embeddings, language_embeddings)
        mask_output = self.postprocess_masks(masks)[:, 0, :, :]
        loss_weight = 0.2 + 0.8 * data_label.to(mask_output.device, mask_output.dtype)
        generator_ce_loss = sigmoid_ce_loss(mask_output, gt_mask.to(mask_output.device, mask_output.dtype)) 
        generator_dice_loss = dice_loss(mask_output, gt_mask.to(mask_output.device, mask_output.dtype))

        return {
            'loss': torch.mean((generator_ce_loss + generator_dice_loss) * loss_weight),
            'generator_ce_loss': torch.mean(generator_ce_loss).item(),
            'generator_dice_loss': torch.mean(generator_dice_loss).item()
        }


    def postprocess_masks(
        self,
        masks: torch.Tensor
    ) -> torch.Tensor:
        """
        Remove padding and upscale masks to the original image size.

        Arguments:
          masks (torch.Tensor): Batched masks from the mask_decoder,
            in BxCxHxW format.
          input_size (tuple(int, int)): The size of the image input to the
            model, in (H, W) format. Used to remove padding.
          original_size (tuple(int, int)): The original size of the image
            before resizing for input to the model, in (H, W) format.

        Returns:
          (torch.Tensor): Batched masks in BxCxHxW format, where (H, W)
            is given by original_size.
        """
        masks = F.interpolate(
            masks,
            (self.image_encoder.img_size, self.image_encoder.img_size),
            mode="bilinear",
            align_corners=False,
        )
        return masks

    def preprocess(self, x: Image) -> torch.Tensor:
        """Normalize pixel values and pad to a square input."""
        # Normalize colors
        x = TF.to_tensor(x).to(self.pixel_mean.device, self.pixel_mean.dtype)
        x = (x - self.pixel_mean) / self.pixel_std
        return x
