'''
 * The Segment Anything Model (SAM)
'''

import os

import torch
from torch import nn
from torch.nn import functional as F

from .image_encoder import TinyViT
from .mask_decoder import MaskDecoder
from .prompt_encoder import PromptEncoder
from .transformer import TwoWayTransformer

from typing import Any, Dict, List, Tuple

import pdb


class MobileSAM(nn.Module):
    mask_threshold: float = 0.0
    def __init__(self, pixel_mean=[123.675, 116.28, 103.53], pixel_std=[58.395, 57.12, 57.375]):
        super().__init__()

        prompt_embed_dim = 256
        image_size = 1024
        vit_patch_size = 16
        image_embedding_size = image_size // vit_patch_size # 64

        self.image_encoder = TinyViT(
            img_size=1024, 
            in_chans=3, 
            num_classes=1000,
            embed_dims=[64, 128, 160, 320],
            depths=[2, 2, 6, 2],
            num_heads=[2, 4, 5, 10],
            window_sizes=[7, 7, 14, 7],
            mlp_ratio=4.,
            drop_rate=0.,
            drop_path_rate=0.0,
            mbconv_expand_ratio=4.0,
            local_conv_size=3,
        )

        self.prompt_encoder = PromptEncoder(
            embed_dim=prompt_embed_dim,
            image_embedding_size=(image_embedding_size, image_embedding_size),
            input_image_size=(image_size, image_size),
            mask_in_chans=16,
        )

        self.mask_decoder = MaskDecoder(
            num_multimask_outputs=3,
            transformer_dim=prompt_embed_dim,
            iou_head_depth=3,
            iou_head_hidden_dim=256,
        )

        self.register_buffer("pixel_mean", torch.Tensor(pixel_mean).view(-1, 1, 1), False)
        self.register_buffer("pixel_std", torch.Tensor(pixel_std).view(-1, 1, 1), False)

        self.load_weights()

    # @property
    # def device(self) -> Any:
    #     return self.pixel_mean.device

    def forward(self, image):
        original_size = (image.shape[2], image.shape[3])
        image = self.preprocess(image)

        input_size = (image.shape[2], image.shape[3])

        image_embeddings = self.image_encoder(image)

        point_coords = torch.randn(32, 1, 2).to(image.device)
        point_coords = point_coords * 1024
        point_labels = torch.ones(32, 1).to(image.device)
        points = (point_coords, point_labels)
        boxes = None
        masks = None

        sparse_embeddings, dense_embeddings = self.prompt_encoder(
            points=points,
            boxes=boxes,
            masks=masks,
        )
        low_res_masks, iou_predictions = self.mask_decoder(
            image_embeddings=image_embeddings,
            image_pe=self.prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
        )
        masks = self.postprocess_masks(
            low_res_masks,
            input_size=input_size,
            original_size=original_size,
        )
        masks = masks > self.mask_threshold

        return masks


    def postprocess_masks(self, masks, input_size: Tuple[int, ...], original_size: Tuple[int, ...]):
        # masks.size() -- [64, 3, 256, 256]
        # input_size -- (1024, 1024)
        # original_size -- (1024, 1024)

        # self.image_encoder.img_size -- 1024
        masks = F.interpolate(
            masks,
            (self.image_encoder.img_size, self.image_encoder.img_size),
            mode="bilinear",
            align_corners=False,
        )
        masks = masks[..., : input_size[0], : input_size[1]]
        masks = F.interpolate(masks, original_size, mode="bilinear", align_corners=False)
        return masks

    def preprocess(self, x):
        # x.size() -- [1, 3, 1024, 1024], x.dtype=uint8

        x = (x - self.pixel_mean) / self.pixel_std
        # Pad
        h, w = x.shape[-2:]
        padh = self.image_encoder.img_size - h
        padw = self.image_encoder.img_size - w
        x = F.pad(x, (0, padw, 0, padh))

        return x


    def load_weights(self, model_path="models/SAM.pth"):
        cdir = os.path.dirname(__file__)
        checkpoint = model_path if cdir == "" else cdir + "/" + model_path

        if os.path.exists(checkpoint):
            print(f"Loading model weight from {checkpoint} ...")
            self.load_state_dict(torch.load(checkpoint))
        else:
            print("-" * 32, "Warnning", "-" * 32)
            print(f"model weight file '{checkpoint}'' not exist !!!")

