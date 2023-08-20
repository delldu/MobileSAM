# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
import torch
from torch import nn

from typing import Optional, Tuple, Type

from .common import LayerNorm2d
import pdb

class PromptEncoder(nn.Module):
    def __init__(self,
        embed_dim: int = 256,
        image_embedding_size: Tuple[int, int] = (64, 64),
        input_image_size: Tuple[int, int] = (1024, 1024),
        mask_in_chans: int = 16,
        activation: Type[nn.Module] = nn.GELU,
    ):
        """
        Encodes prompts for input to SAM's mask decoder.

        Arguments:
          embed_dim (int): The prompts' embedding dimension
          image_embedding_size (tuple(int, int)): The spatial size of the image embedding, as (H, W).
          input_image_size (int): The padded size of the image as input to the image encoder, as (H, W).
          mask_in_chans (int): The number of hidden channels used for encoding input masks.
        """
        super().__init__()
        # embed_dim = 256
        # image_embedding_size = (64, 64)
        # input_image_size = (1024, 1024)
        # mask_in_chans = 16

        self.embed_dim = embed_dim
        self.input_image_size = input_image_size
        self.image_embedding_size = image_embedding_size # (64, 64)
        self.pe_layer = PositionEmbeddingRandom(embed_dim // 2) # embed_dim -- 256

        self.num_point_embeddings: int = 4  # pos/neg point + 2 box corners
        point_embeddings = [nn.Embedding(1, embed_dim) for i in range(self.num_point_embeddings)]
        self.point_embeddings = nn.ModuleList(point_embeddings)
        self.not_a_point_embed = nn.Embedding(1, embed_dim)

        self.mask_input_size = (4 * image_embedding_size[0], 4 * image_embedding_size[1]) # (64, 64)
        self.mask_downscaling = nn.Sequential(
            nn.Conv2d(1, mask_in_chans // 4, kernel_size=2, stride=2),
            LayerNorm2d(mask_in_chans // 4),
            activation(),
            nn.Conv2d(mask_in_chans // 4, mask_in_chans, kernel_size=2, stride=2),
            LayerNorm2d(mask_in_chans),
            activation(),
            nn.Conv2d(mask_in_chans, embed_dim, kernel_size=1),
        )
        self.no_mask_embed = nn.Embedding(1, embed_dim)

    def get_dense_pe(self):
        """
        Returns the positional encoding used to encode point prompts,
        applied to a dense set of points the shape of the image encoding.
        """
        # ==> pdb.set_trace()
        return self.pe_layer(self.image_embedding_size).unsqueeze(0) # self.image_embedding_size -- (64, 64)

    def _embed_points(self, points, labels, pad: bool):
        """Embeds point prompts."""
        # points.size() -- [64, 1, 2]
        # labels.size() -- [64, 1]
        # pad = True

        points = points + 0.5  # Shift to center of pixel
        if pad: # True
            padding_point = torch.zeros((points.shape[0], 1, 2), device=points.device)
            padding_label = -torch.ones((labels.shape[0], 1), device=labels.device)
            points = torch.cat([points, padding_point], dim=1)
            labels = torch.cat([labels, padding_label], dim=1)
        point_embedding = self.pe_layer.forward_with_coords(points, self.input_image_size)
        point_embedding[labels == -1] = 0.0
        point_embedding[labels == -1] += self.not_a_point_embed.weight
        point_embedding[labels == 0] += self.point_embeddings[0].weight
        point_embedding[labels == 1] += self.point_embeddings[1].weight

        # ==> pdb.set_trace()
        return point_embedding # size() --[64, 2, 256]

    def _embed_boxes(self, boxes):
        """Embeds box prompts."""
        pdb.set_trace()

        boxes = boxes + 0.5  # Shift to center of pixel
        coords = boxes.reshape(-1, 2, 2)
        corner_embedding = self.pe_layer.forward_with_coords(coords, self.input_image_size)
        corner_embedding[:, 0, :] += self.point_embeddings[2].weight
        corner_embedding[:, 1, :] += self.point_embeddings[3].weight
        return corner_embedding

    def _embed_masks(self, masks):
        """Embeds mask inputs."""
        pdb.set_trace()

        mask_embedding = self.mask_downscaling(masks)
        return mask_embedding

    def _get_batch_size(self,
        points: Optional[Tuple[torch.Tensor, torch.Tensor]],
        boxes: Optional[torch.Tensor],
        masks: Optional[torch.Tensor],
    ) -> int:
        """
        Gets the batch size of the output given the batch size of the input prompts.
        """
        if points is not None: # True
            return points[0].shape[0]
        elif boxes is not None: # False
            return boxes.shape[0]
        elif masks is not None: # False
            return masks.shape[0]
        else:
            return 1

    def _get_device(self) -> torch.device:
        return self.point_embeddings[0].weight.device

    def forward(self,
        points: Optional[Tuple[torch.Tensor, torch.Tensor]],
        boxes: Optional[torch.Tensor],
        masks: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Embeds different types of prompts, returning both sparse and dense
        embeddings.

        Arguments:
          points (tuple(torch.Tensor, torch.Tensor) or none): point coordinates and labels to embed.
          boxes (torch.Tensor or none): boxes to embed
          masks (torch.Tensor or none): masks to embed

        Returns:
          torch.Tensor: sparse embeddings for the points and boxes, with shape
            BxNx(embed_dim), where N is determined by the number of input points and boxes.
          torch.Tensor: dense embeddings for the masks, in the shape Bx(embed_dim)x(embed_H)x(embed_W)
        """
        bs = self._get_batch_size(points, boxes, masks)
        sparse_embeddings = torch.empty((bs, 0, self.embed_dim), device=self._get_device())
        if points is not None: # True
            coords, labels = points
            point_embeddings = self._embed_points(coords, labels, pad=(boxes is None))
            sparse_embeddings = torch.cat([sparse_embeddings, point_embeddings], dim=1)

        if boxes is not None: # False ???
            box_embeddings = self._embed_boxes(boxes)
            sparse_embeddings = torch.cat([sparse_embeddings, box_embeddings], dim=1)
        if masks is not None: # False
            dense_embeddings = self._embed_masks(masks)
        else:
            dense_embeddings = self.no_mask_embed.weight.reshape(1, -1, 1, 1).expand(
                bs, -1, self.image_embedding_size[0], self.image_embedding_size[1]
            )

        return sparse_embeddings, dense_embeddings


class PositionEmbeddingRandom(nn.Module):
    """
    Positional encoding using random spatial frequencies.
    """

    def __init__(self, num_pos_feats: int = 128):
        super().__init__()
        self.register_buffer("positional_encoding_gaussian_matrix", torch.randn((2, num_pos_feats)))

    def pe_encoding(self, coords):
        """Positionally encode points that are normalized to [0,1]."""
        # assuming coords are in [0, 1]^2 square and have d_1 x ... x d_n x 2 shape
        coords = 2 * coords - 1
        coords = coords @ self.positional_encoding_gaussian_matrix
        coords = 2 * np.pi * coords
        # outputs d_1 x ... x d_n x C shape
        return torch.cat([torch.sin(coords), torch.cos(coords)], dim=-1)

    def forward(self, size: Tuple[int, int]):
        """Generate positional encoding for a grid of the specified size."""
        h, w = size
        grid = torch.ones((h, w), device=self.positional_encoding_gaussian_matrix.device, dtype=torch.float32)
        y_embed = grid.cumsum(dim=0) - 0.5
        x_embed = grid.cumsum(dim=1) - 0.5
        y_embed = y_embed / h
        x_embed = x_embed / w

        pe = self.pe_encoding(torch.stack([x_embed, y_embed], dim=-1))
        return pe.permute(2, 0, 1)  # C x H x W

    def forward_with_coords(self, coords_input, image_size: Tuple[int, int] ):
        """Positionally encode points that are not normalized to [0,1]."""
        coords = coords_input.clone()
        coords[:, :, 0] = coords[:, :, 0] / image_size[1]
        coords[:, :, 1] = coords[:, :, 1] / image_size[0]
        return self.pe_encoding(coords.to(torch.float))  # B x N x C
