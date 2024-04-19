# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
from torch import nn
from torch.nn import functional as F
from timm.models.registry import register_model
from typing import Any, List, Tuple
from functools import partial

from fam.model.decoder import ConditionedMaskDecoder
from fam.model.T5_encoder import T5Encoder
from fam.model.sam_components.image_encoder import ImageEncoderViT
from fam.model.sam_components.transformer import TwoWayTransformer



def dice_loss(
	inputs: torch.Tensor,
	targets: torch.Tensor,
	scale=1000,  # 100000.0,
	eps=1e-6,
):
	"""
	Compute the DICE loss, similar to generalized IOU for masks
	Args:
		inputs: A float tensor of arbitrary shape.
				The predictions for each example.
		targets: A float tensor with the same shape as inputs. Stores the binary
				 classification label for each element in inputs
				(0 for the negative class and 1 for the positive class).
	"""
	inputs = inputs.sigmoid()
	inputs = inputs.flatten(1, 2)
	targets = targets.flatten(1, 2)
	numerator = 2 * (inputs / scale * targets).sum(-1)
	denominator = (inputs / scale).sum(-1) + (targets / scale).sum(-1)
	loss = 1 - (numerator + eps) / (denominator + eps)
	return loss.mean()

def sigmoid_ce_loss(
	inputs: torch.Tensor,
	targets: torch.Tensor,
):
	"""
	Args:
		inputs: A float tensor of arbitrary shape.
				The predictions for each example.
		targets: A float tensor with the same shape as inputs. Stores the binary
				 classification label for each element in inputs
				(0 for the negative class and 1 for the positive class).
	Returns:
		Loss tensor
	"""
	loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="mean")
	return loss




class FilterAnything(nn.Module):

    def __init__(
        self,
        image_encoder,
        prompt_encoder,
        mask_decoder,
        pixel_mean: List[float] = [123.675, 116.28, 103.53],
        pixel_std: List[float] = [58.395, 57.12, 57.375],
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
        self.image_encoder = image_encoder
        self.prompt_encoder = prompt_encoder
        self.mask_decoder = mask_decoder
        self.register_buffer("pixel_mean", torch.Tensor(pixel_mean).view(-1, 1, 1), False)
        self.register_buffer("pixel_std", torch.Tensor(pixel_std).view(-1, 1, 1), False)

    @property
    def device(self) -> Any:
        return self.pixel_mean.device
    

    @torch.no_grad()
    def forward_inference(
        self,
        batched_input,
        task_id = 0
    ):
        images = torch.stack([self.preprocess(x["image"]) for x in batched_input], dim=0)
        image_embeddings = self.image_encoder(images)
        language_embeddings = self.prompt_encoder.embed_text([x['instruction'] for x in batched_input], False)
        masks = self.mask_decoder(image_embeddings, language_embeddings)[:, task_id]
        masks = self.postprocess_masks(masks, images.shape[-2:])
        return masks

    def forward_train(
        self,
        images,
        instruction,
        mixture_heatmap,
        gt_mask,
        task_id
    ):
        with torch.no_grad():
            image_embeddings = self.image_encoder(images).detach()
            language_embeddings = self.prompt_encoder.embed_text(instruction, False).detach()
        masks = self.mask_decoder(image_embeddings, language_embeddings)
        return self.cal_loss(masks,mixture_heatmap, gt_mask, task_id)



    def cal_loss(self, masks, mixture_heatmap, gt_mask, task_id):
        masks = F.interpolate(
            masks,
            (self.image_encoder.img_size, self.image_encoder.img_size),
            mode="bilinear",
            align_corners=False,
        )
        specific_token = masks[torch.arange(masks.shape[0]), task_id, :, :]
        joint_token = masks[:, 0, :, :]
        return F.mse_loss(joint_token, mixture_heatmap),  \
        sigmoid_ce_loss(specific_token, gt_mask) + dice_loss(specific_token, gt_mask)


    def postprocess_masks(
        self,
        masks: torch.Tensor,
        input_size: Tuple[int, ...]
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
        masks = masks[..., : input_size[0], : input_size[1]]
        return masks

    def preprocess(self, x: torch.Tensor) -> torch.Tensor:
        """Normalize pixel values and pad to a square input."""
        # Normalize colors
        x = (x - self.pixel_mean) / self.pixel_std

        # Pad
        h, w = x.shape[-2:]
        padh = self.image_encoder.img_size - h
        padw = self.image_encoder.img_size - w
        x = F.pad(x, (0, padw, 0, padh))
        return x
