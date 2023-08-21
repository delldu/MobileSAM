# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
from torch import nn
from torch.nn import functional as F
from SAM.common import LayerNorm2d
from SAM.transformer import TwoWayTransformer

from typing import List, Tuple, Type

import pdb

class MaskDecoder(nn.Module):
    def __init__(self,
        transformer_dim=256,
        prompt_embed_dim=256,
        num_multimask_outputs=3,
        iou_head_depth=3,
        iou_head_hidden_dim=256,
    ):
        super().__init__()
        # transformer_dim = 256
        # num_multimask_outputs = 3
        # iou_head_depth = 3
        # iou_head_hidden_dim = 256

        self.transformer_dim = transformer_dim
        self.transformer = TwoWayTransformer(depth=2, embedding_dim=prompt_embed_dim, mlp_dim=2048, num_heads=8)
        self.num_multimask_outputs = num_multimask_outputs
        self.iou_token = nn.Embedding(1, transformer_dim)
        self.num_mask_tokens = num_multimask_outputs + 1
        self.mask_tokens = nn.Embedding(self.num_mask_tokens, transformer_dim)

        self.output_upscaling = nn.Sequential(
            nn.ConvTranspose2d(transformer_dim, transformer_dim // 4, kernel_size=2, stride=2),
            LayerNorm2d(transformer_dim // 4),
            nn.GELU(),
            nn.ConvTranspose2d(transformer_dim // 4, transformer_dim // 8, kernel_size=2, stride=2),
            nn.GELU(),
        )
        self.output_hypernetworks_mlps = nn.ModuleList(
            [
                MLP(transformer_dim, transformer_dim, transformer_dim // 8, 3) for i in range(self.num_mask_tokens)
            ]
        )

        self.iou_prediction_head = MLP(
            transformer_dim, iou_head_hidden_dim, self.num_mask_tokens, iou_head_depth
        )

    def forward(self, image_embeddings, image_pe, sparse_prompt_embeddings, dense_prompt_embeddings,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Predict masks given image and prompt embeddings.

        Returns:
          batched predicted masks, batched predictions of mask quality
        """
        # image_embeddings.size() -- [1, 256, 64, 64]
        # image_pe.size() -- [1, 256, 64, 64]
        # sparse_prompt_embeddings.size() -- [64, 2, 256]
        # dense_prompt_embeddings.size() -- [64, 256, 64, 64]

        masks, iou_pred = self.predict_masks(
            image_embeddings=image_embeddings,
            image_pe=image_pe,
            sparse_prompt_embeddings=sparse_prompt_embeddings,
            dense_prompt_embeddings=dense_prompt_embeddings,
        )
        # masks.size() -- [64, 4, 256, 256]
        # iou_pred.size() -- [64, 4]

        # Select the correct mask or masks for output
        mask_slice = slice(1, None)
        masks = masks[:, mask_slice, :, :]
        iou_pred = iou_pred[:, mask_slice]
        return masks, iou_pred

    def predict_masks(self, image_embeddings, image_pe, sparse_prompt_embeddings,
        dense_prompt_embeddings,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Predicts masks. See 'forward' for more details."""
        # Concatenate output tokens
        output_tokens = torch.cat([self.iou_token.weight, self.mask_tokens.weight], dim=0)
        output_tokens = output_tokens.unsqueeze(0).expand(sparse_prompt_embeddings.size(0), -1, -1)
        tokens = torch.cat((output_tokens, sparse_prompt_embeddings), dim=1)

        # Expand per-image data in batch direction to be per-mask
        src = torch.repeat_interleave(image_embeddings, tokens.shape[0], dim=0)
        src = src + dense_prompt_embeddings
        pos_src = torch.repeat_interleave(image_pe, tokens.shape[0], dim=0)
        b, c, h, w = src.shape

        # image_embeddings.size() -- [1, 256, 64, 64]
        # image_pe.size() -- [1, 256, 64, 64]
        # sparse_prompt_embeddings.size() -- [64, 2, 256]
        # dense_prompt_embeddings.size() -- [64, 256, 64, 64]
        # src.size() -- [64, 256, 64, 64]
        # pos_src.size() -- [64, 256, 64, 64]
        # tokens.size() -- [64, 7, 256]


        # Run the transformer
        hs, src = self.transformer(src, pos_src, tokens)
        iou_token_out = hs[:, 0, :]
        mask_tokens_out = hs[:, 1 : (1 + self.num_mask_tokens), :]

        # Upscale mask embeddings and predict masks using the mask tokens
        src = src.transpose(1, 2).view(b, c, h, w)
        upscaled_embedding = self.output_upscaling(src)
        hyper_in_list: List[torch.Tensor] = []

        # for i in range(self.num_mask_tokens):
        #     hyper_in_list.append(self.output_hypernetworks_mlps[i](mask_tokens_out[:, i, :]))
        for i, m in enumerate(self.output_hypernetworks_mlps):
            hyper_in_list.append(m(mask_tokens_out[:, i, :]))

        hyper_in = torch.stack(hyper_in_list, dim=1)
        b, c, h, w = upscaled_embedding.shape
        masks = (hyper_in @ upscaled_embedding.view(b, c, h * w)).view(b, -1, h, w)

        # Generate mask quality predictions
        iou_pred = self.iou_prediction_head(iou_token_out)

        return masks, iou_pred


# Lightly adapted from
# https://github.com/facebookresearch/MaskFormer/blob/main/mask_former/modeling/transformer/transformer_predictor.py # noqa
class MLP(nn.Module):
    def __init__(self, input_dim = 256, hidden_dim = 256, output_dim = 32, num_layers = 3):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))
        # self.sigmoid_output = sigmoid_output

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


if __name__ == "__main__":
    model = MaskDecoder()
    model = torch.jit.script(model)
    print(model)
    # ==> OK