# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
import torch

from segment_anything.modeling import Sam
from .utils.transforms import ResizeLongestSide
from .utils.debug import debug_var

from typing import Optional, Tuple

import pdb

class SamPredictor:
    def __init__(self, sam_model: Sam):
        """
        Uses SAM to calculate the image embedding for an image, and then
        allow repeated, efficient mask prediction given prompts.

        Arguments:
          sam_model (Sam): The model to use for mask prediction.
        """
        super().__init__()
        self.model = sam_model
        self.transform = ResizeLongestSide(sam_model.image_encoder.img_size) # sam_model.image_encoder.img_size -- 1024
        self.reset_image()


    def set_image(self, image: np.ndarray, image_format: str = "RGB"):
        """
        Calculates the image embeddings for the provided image, allowing
        masks to be predicted with the 'predict' method.

        Arguments:
          image (np.ndarray): The image for calculating masks. Expects an
            image in HWC uint8 format, with pixel values in [0, 255].
          image_format (str): The color format of the image, in ['RGB', 'BGR'].
        """
        assert image_format in ["RGB", "BGR"], f"image_format must be in ['RGB', 'BGR'], is {image_format}."
        if image_format != self.model.image_format:
            image = image[..., ::-1]

        # Transform the image to the form expected by the model
        input_image = self.transform.apply_image(image)
        input_image_torch = torch.as_tensor(input_image, device=self.device)
        input_image_torch = input_image_torch.permute(2, 0, 1).contiguous()[None, :, :, :]

        # input_image_torch.size() -- torch.Size([1, 3, 1024, 1024]), uin8, pixel value 0 - 255

        self.set_torch_image(input_image_torch, image.shape[:2])

    @torch.no_grad()
    def set_torch_image(self, transformed_image, original_image_size: Tuple[int, ...]):
        """
        Calculates the image embeddings for the provided image, allowing
        masks to be predicted with the 'predict' method. Expects the input
        image to be already transformed to the format expected by the model.

        Arguments:
          transformed_image: The input image, with shape
            1x3xHxW, which has been transformed with ResizeLongestSide.
          original_image_size (tuple(int, int)): The size of the image
            before transformation, in (H, W) format.
        """
        assert (
            len(transformed_image.shape) == 4
            and transformed_image.shape[1] == 3
            and max(*transformed_image.shape[2:]) == self.model.image_encoder.img_size
        ), f"set_torch_image input must be BCHW with long side {self.model.image_encoder.img_size}."
        self.reset_image()

        self.original_size = original_image_size
        self.input_size = tuple(transformed_image.shape[-2:])

        # xxxx8888 !!! Step 1
        input_image = self.model.preprocess(transformed_image)
        # tensor [input_image] size: [1, 3, 1024, 1024] , min: tensor(-2.0837, device='cuda:0') , max: tensor(2.6400, device='cuda:0')
        self.features = self.model.image_encoder(input_image)
        # tensor [self.features] size: [1, 256, 64, 64] , min: tensor(-0.6405, device='cuda:0') , max: tensor(0.5725, device='cuda:0')
        # self.features -- image_embeddings in Sam.forward

        self.is_image_set = True


    @torch.no_grad()
    def predict_torch(self,
        point_coords: Optional[torch.Tensor],
        point_labels: Optional[torch.Tensor],
        boxes: Optional[torch.Tensor] = None,
        mask_input: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Predict masks for the given input prompts, using the currently set image.
        Input prompts are batched torch tensors and are expected to already be
        transformed to the input frame using ResizeLongestSide.
        """
        if not self.is_image_set:
            raise RuntimeError("An image must be set with .set_image(...) before mask prediction.")

        if point_coords is not None: # True for point_coords.size() -- [64, 1, 2]
            points = (point_coords, point_labels) # labels.size() -- [64, 1]
        else:
            points = None

        # Embed prompts
        # xxxx8888 !!! Step 2
        # tuple [points] len: 2
        # [boxes] value: None
        # [mask_input] value: None

        # Sam.forward: prompt_encoder, mask_decoder
        sparse_embeddings, dense_embeddings = self.model.prompt_encoder(points=points, boxes=boxes, masks=mask_input)
        # tensor [sparse_embeddings] size: [64, 2, 256] , min: tensor(-1.3537, device='cuda:0') , max: tensor(1.3647, device='cuda:0')
        # tensor [dense_embeddings] size: [64, 256, 64, 64] , min: tensor(-0.2179, device='cuda:0') , max: tensor(0.1378, device='cuda:0')

        # xxxx8888 !!! Step 3
        low_res_masks, iou_predictions = self.model.mask_decoder(
            image_embeddings=self.features,
            image_pe=self.model.prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
        )
        # tensor [low_res_masks] size: [64, 3, 256, 256] , min: tensor(-43.5383, device='cuda:0') , max: tensor(24.3619, device='cuda:0')
        # tensor [iou_predictions] size: [64, 3] , min: tensor(0.4337, device='cuda:0') , max: tensor(0.9843, device='cuda:0')

        # Upscale the masks to the original image resolution
        masks = self.model.postprocess_masks(low_res_masks, self.input_size, self.original_size)
        # tensor [masks] size: [64, 3, 683, 1024] , min: tensor(-41.8604, device='cuda:0') , max: tensor(19.8630, device='cuda:0')

        return masks, iou_predictions, low_res_masks

    @property
    def device(self) -> torch.device:
        return self.model.device

    def reset_image(self) -> None:
        """Resets the currently set image."""
        self.is_image_set = False
        self.features = None
        self.orig_h = None
        self.orig_w = None
        self.input_h = None
        self.input_w = None
