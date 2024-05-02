# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


from syslog import LOG_SYSLOG
import torch
from torch import nn
from torch.nn import functional as F

from typing import Any, List, Tuple
from functools import partial
from timm.models.layers import Mlp
from model.decoder import ConditionedMaskDecoder
from model.discriminator import Discriminator
from model.T5_encoder import T5Encoder
from model.lora_sam import LoRA_Sam
from model.sam_components.image_encoder import ImageEncoderViT
from model.sam_components.transformer import TwoWayTransformer
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
    loss = F.binary_cross_entropy_with_logits(inputs.flatten(1, 2), targets.flatten(1, 2), reduction='none').mean(-1)
    return loss


class FAM(nn.Module):

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
        # self.image_encoder = LoRA_Sam(self.image_encoder, r = 4)

        self.prompt_encoder = T5Encoder()
        self.prompt_encoder.requires_grad_(False)
        self.prompt_linear = Mlp(768, 768, 256)
        self.mask_decoder = ConditionedMaskDecoder(
            transformer_dim=256,
            transformer=TwoWayTransformer(
                depth=6,
                embedding_dim=256,
                mlp_dim=2048,
                num_heads=8,
            ),
            learnable_token_num=64
        )
        self.discriminator = Discriminator(
            transformer_dim=256,
            transformer=TwoWayTransformer(
                depth=3,
                embedding_dim=256,
                mlp_dim=2048,
                num_heads=8,
            )
        )
        self.register_buffer("pixel_mean", torch.Tensor(pixel_mean).view(-1, 1, 1), False)
        self.register_buffer("pixel_std", torch.Tensor(pixel_std).view(-1, 1, 1), False)

    @property
    def device(self) -> Any:
        return self.pixel_mean.device

    def forward(
        self,
        expert_images, # B C H W
        expert_instruction, # [str]
        expert_mask, # B H W
        nonexpert_images, # B C H W
        nonexpert_instruction, # [str]
        nonexpert_mask, # B H W
        eta = 0.001
    ):
        images = torch.cat((expert_images, nonexpert_images), 0).to(self.pixel_mean.dtype)
        instruction = expert_instruction + nonexpert_instruction
        gt_mask = torch.cat((expert_mask, nonexpert_mask), 0).to(self.pixel_mean.dtype)
        # images_for_sam = ((images - self.pixel_mean.unsqueeze(0)) / self.pixel_std.unsqueeze(0)).detach()
        image_embeddings = self.image_encoder(images)
        language_embeddings = self.prompt_linear(self.prompt_encoder(instruction))
        masks = self.mask_decoder(image_embeddings, language_embeddings)
        mask_output = F.interpolate(
            masks,
            (self.image_encoder.img_size, self.image_encoder.img_size),
            mode="bilinear",
            align_corners=False,
        )[:, 0, :, :]
        prob = torch.where(gt_mask > 0.5, mask_output.sigmoid(), 1 - mask_output.sigmoid()).detach()

        loss_weight = self.discriminator(image_embeddings.detach(), language_embeddings.detach(), prob.unsqueeze(1))

        expert_target = torch.cat((torch.ones(len(expert_instruction), 1), torch.zeros(len(nonexpert_instruction), 1)), 0).cuda()
        discriminator_loss = F.binary_cross_entropy_with_logits(loss_weight, expert_target)


        with torch.no_grad():
            loss_weight = torch.sigmoid(loss_weight)
            expert_loss_weight, nonexpert_loss_weight = torch.chunk(loss_weight, 2, 0)
            expert_mask_output, nonexpert_mask_output = torch.chunk(mask_output, 2, 0)
            applied_loss_weight = torch.clip(loss_weight, 0.1, 0.9).detach()
        generator_ce_loss = torch.mean(applied_loss_weight.detach() * sigmoid_ce_loss(mask_output, gt_mask))
        generator_dice_loss = torch.mean(applied_loss_weight.detach() * dice_loss(mask_output, gt_mask))

        return {
            'loss': eta * (discriminator_loss) + generator_ce_loss + generator_dice_loss,
            'discriminator_loss':  discriminator_loss.item(),
            'expert_generator_ce_loss': sigmoid_ce_loss(expert_mask_output, expert_mask).mean().item(),
            'expert_generator_dice_loss': dice_loss(expert_mask_output, expert_mask).mean().item(),
            'none_expert_generator_ce_loss': sigmoid_ce_loss(nonexpert_mask_output, nonexpert_mask).mean().item(),
            'none_expert_generator_dice_loss': dice_loss(nonexpert_mask_output, nonexpert_mask).mean().item(),
            'nonexpert_loss_weight': nonexpert_loss_weight.detach().mean().item(),
            'expert_loss_weight': expert_loss_weight.detach().mean().item()
        }
