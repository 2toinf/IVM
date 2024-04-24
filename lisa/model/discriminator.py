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
class Discriminator(nn.Module):
    def __init__(
        self,
        *,
        transformer_dim: int,
        transformer: nn.Module,
        activation: Type[nn.Module] = nn.GELU,
        image_embedding_size:int = (64, 64),
        mask_in_chans = 16
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
        self.discriminator_token = nn.Embedding(1, transformer_dim)
        self.mask_downscaling = nn.Sequential(
            nn.Conv2d(1, mask_in_chans // 4, kernel_size=2, stride=2),
            LayerNorm2d(mask_in_chans // 4),
            activation(),
            nn.Conv2d(mask_in_chans // 4, mask_in_chans, kernel_size=2, stride=2),
            LayerNorm2d(mask_in_chans),
            activation(),
            nn.Conv2d(mask_in_chans, transformer_dim, kernel_size=1),
        )
        self.discriminator_head = MLP(transformer_dim, transformer_dim, 1, 3)

    def forward(
        self,
        image_embeddings: torch.Tensor,
        condition_embeddings: torch.Tensor,
        mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Predict masks given image and prompt embeddings.

        Arguments:
          image_embeddings (torch.Tensor): the embeddings from the image encoder
          condition_embeddings (torch.Tensor): the embeddings of the language condition
        Returns:
          torch.Tensor: batched predicted masks
        """

        
        mask = self.mask_downscaling(mask)
        # generate pe
        image_pe = self.pe_layer(self.image_embedding_size,  torch.float16).unsqueeze(0)

        # Concatenate output tokens
        output_tokens = self.discriminator_token.weight
        output_tokens = output_tokens.unsqueeze(0).expand(condition_embeddings.size(0), -1, -1)
        tokens = torch.cat((output_tokens, condition_embeddings), dim=1)

        # Run the transformer
        hs, _ = self.transformer(image_embeddings, image_pe, tokens)
        discriminator_token_out = self.discriminator_head(hs[:, 0, :])
        
        return discriminator_token_out
        
