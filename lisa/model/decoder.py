# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
from torch import nn
from torch.nn import functional as F

from typing import  Tuple, Type
from model.sam_components.prompt_encoder import PositionEmbeddingRandom
from model.sam_components.mask_decoder import MLP, LayerNorm2d
from model.sam_components.transformer import TwoWayTransformer
from timm.models.registry import register_model

class ConditionedMaskDecoder(nn.Module):
    def __init__(
        self,
        *,
        transformer_dim: int,
        transformer: nn.Module,
        activation: Type[nn.Module] = nn.GELU,
        image_embedding_size:int = (64, 64),
        learnable_token_num = 32
    ):
        """
        Predicts masks given an image and prompt embeddings, using a
        transformer architecture.

        Arguments:
          transformer_dim (int): the channel dimension of the transform
          transformer (nn.Module): the transformer used to predict masks
          activation (nn.Module): the type of activation to use when
            upscaling masks
        image_embedding_size (tuple(int, int)): The spatial size of the
            image embedding, as (H, W).
        """
        super().__init__()
        self.image_embedding_size = image_embedding_size
        self.transformer_dim = transformer_dim
        self.transformer = transformer
        self.pe_layer = PositionEmbeddingRandom(transformer_dim // 2)
        self.learnable_token_num = learnable_token_num
        self.mask_tokens = nn.Embedding(learnable_token_num, transformer_dim)
        self.output_upscaling = nn.Sequential(
            nn.ConvTranspose2d(transformer_dim, transformer_dim // 4, kernel_size=2, stride=2),
            LayerNorm2d(transformer_dim // 4),
            activation(),
            nn.ConvTranspose2d(transformer_dim // 4, transformer_dim // 8, kernel_size=2, stride=2),
            activation(),
        )
        
        self.output_hypernetworks_mlps = MLP(transformer_dim, transformer_dim, transformer_dim // 8, 3)

    def forward(
        self,
        image_embeddings: torch.Tensor,
        condition_embeddings: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Predict masks given image and prompt embeddings.

        Arguments:
          image_embeddings (torch.Tensor): the embeddings from the image encoder
          condition_embeddings (torch.Tensor): the embeddings of the language condition
        Returns:
          torch.Tensor: batched predicted masks
        """

        
        # generate pe
        image_pe = self.pe_layer(self.image_embedding_size,  torch.float16).unsqueeze(0)

        # Concatenate output tokens
        output_tokens = self.mask_tokens.weight
        output_tokens = output_tokens.unsqueeze(0).expand(condition_embeddings.size(0), -1, -1)
        tokens = torch.cat((output_tokens, condition_embeddings), dim=1)

        # Expand per-image data in batch direction to be per-mask
        # src = torch.repeat_interleave(image_embeddings, repeats=tokens.shape[0], dim=0)
        # pos_src = torch.repeat_interleave(image_pe, tokens.shape[0], dim=0)
        b, c, h, w = image_embeddings.shape

        # Run the transformer
        hs, src = self.transformer(image_embeddings, image_pe, tokens)
        mask_tokens_out = torch.mean(hs[:, :self.learnable_token_num, :], dim=1, keepdim=True)

        # Upscale mask embeddings and predict masks using the mask tokens
        src = src.transpose(1, 2).view(b, c, h, w)
        upscaled_embedding = self.output_upscaling(src)

        b, c, h, w = upscaled_embedding.shape

        masks = self.output_hypernetworks_mlps(mask_tokens_out) @ upscaled_embedding.view(b, c, h*w) # b, n, h * w
        return masks.view(b, 1, h, w)
 