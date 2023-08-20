# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
import torch
from torchvision.ops.boxes import batched_nms, box_area  # type: ignore

from typing import Any, Dict, List, Optional, Tuple

from .modeling import Sam
from .predictor import SamPredictor
from .utils.amg import (
    MaskData,
    area_from_rle,
    batch_iterator,
    batched_mask_to_box,
    box_xyxy_to_xywh,
    build_all_layer_point_grids,
    calculate_stability_score,
    coco_encode_rle,
    generate_crop_boxes,
    is_box_near_crop_edge,
    mask_to_rle_pytorch,
    remove_small_regions,
    rle_to_mask,
    uncrop_boxes_xyxy,
    uncrop_masks,
    uncrop_points,
)
from .utils.debug import debug_var

import pdb

class SamAutomaticMaskGenerator:
    def __init__(
        self,
        model: Sam,
        points_per_side: Optional[int] = 32,
        points_per_batch: int = 64,
        pred_iou_thresh: float = 0.88, # 0.86
        stability_score_thresh: float = 0.95, # 0.92
        stability_score_offset: float = 1.0,
        box_nms_thresh: float = 0.7,
        crop_n_layers: int = 0, # 1
        crop_nms_thresh: float = 0.7,
        crop_overlap_ratio: float = 512 / 1500,
        crop_n_points_downscale_factor: int = 1, # 2
        min_mask_region_area: int = 0, # 100, Requires open-cv to run post-processing
    ) -> None:
        # points_per_side = 32
        # points_per_batch = 64
        # pred_iou_thresh = 0.86
        # stability_score_thresh = 0.92
        # stability_score_offset = 1.0
        # box_nms_thresh = 0.7
        # crop_n_layers = 1
        # crop_nms_thresh = 0.7
        # crop_overlap_ratio = 0.3413333333333333
        # crop_n_points_downscale_factor = 2
        # min_mask_region_area = 100

        self.point_grids = build_all_layer_point_grids(
            points_per_side, # 32
            crop_n_layers, # 1
            crop_n_points_downscale_factor,
        )

        # if min_mask_region_area > 0:
        #     import cv2  # type: ignore # noqa: F401

        self.predictor = SamPredictor(model)
        self.points_per_batch = points_per_batch
        self.pred_iou_thresh = pred_iou_thresh
        self.stability_score_thresh = stability_score_thresh
        self.stability_score_offset = stability_score_offset
        self.box_nms_thresh = box_nms_thresh
        self.crop_n_layers = crop_n_layers
        self.crop_nms_thresh = crop_nms_thresh
        self.crop_overlap_ratio = crop_overlap_ratio
        self.crop_n_points_downscale_factor = crop_n_points_downscale_factor
        self.min_mask_region_area = min_mask_region_area

    @torch.no_grad()
    def generate(self, image: np.ndarray) -> List[Dict[str, Any]]:
        # ============ !!! Start from Here !!! ============
        # array [image] shape: (1024, 1024, 3) , min: 0 , max: 255

        mask_data = self.generate_masks(image)
        # mask_data --
        # array [iou_preds] shape: [55]
        # array [points] shape: [55, 2]
        # array [stability_score] shape: [55]
        # array [boxes] shape: [55, 4]
        # list [rles] length: 55
        # array [crop_boxes] shape: [55, 4]

        # Filter small disconnected regions and holes in masks
        if self.min_mask_region_area > 0: # self.min_mask_region_area -- 100
            mask_data = self.postprocess_small_regions(
                mask_data,
                self.min_mask_region_area,
                max(self.box_nms_thresh, self.crop_nms_thresh),
            )
            # mask_data --
            # array [iou_preds] shape: [49]
            # array [points] shape: [49, 2]
            # array [stability_score] shape: [49]
            # array [boxes] shape: [49, 4]
            # list [rles] length: 49
            # array [crop_boxes] shape: [49, 4]

        # Encode masks -- "binary_mask":
            mask_data["segmentations"] = [rle_to_mask(rle) for rle in mask_data["rles"]]
        # mask_data["segmentations"][0].shape -- (1024, 1024)

        # Write mask records
        curr_anns = []
        for idx in range(len(mask_data["segmentations"])):
            ann = {
                "segmentation": mask_data["segmentations"][idx],
                "area": area_from_rle(mask_data["rles"][idx]),
                "bbox": box_xyxy_to_xywh(mask_data["boxes"][idx]).tolist(),
                "predicted_iou": mask_data["iou_preds"][idx].item(),
                "point_coords": [mask_data["points"][idx].tolist()],
                "stability_score": mask_data["stability_score"][idx].item(),
                "crop_box": box_xyxy_to_xywh(mask_data["crop_boxes"][idx]).tolist(),
            }
            curr_anns.append(ann)

        # list [curr_anns] len: 133
        # last ann is dict:
        #     array [segmentation] shape: (1024, 1024) , min: False , max: True
        #     [area] value: 177944
        #     list [bbox] len: 4 , [0, 544, 1023, 479]
        #     [predicted_iou] value: 0.8825232982635498
        #     list [point_coords] len: 1 , [[912.0, 880.0]]
        #     [stability_score] value: 0.9287664294242859
        #     list [crop_box] len: 4 , [0, 0, 1024, 1024]

        return curr_anns

    def generate_masks(self, image: np.ndarray) -> MaskData:
        # array [generate_masks.image] shape: (1024, 1024, 3) , min: 0 , max: 255

        orig_size = image.shape[:2]
        crop_boxes, layer_idxs = generate_crop_boxes(
            orig_size, self.crop_n_layers, self.crop_overlap_ratio # self.crop_n_layers -- 1, self.crop_overlap_ratio -- 0.34
        )

        # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        # 
        data = MaskData()
        for crop_box, layer_idx in zip(crop_boxes, layer_idxs):
            crop_data = self._process_crop(image, crop_box, layer_idx, orig_size)
            data.cat(crop_data)
        #
        # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        # data --
        # tensor iou_preds size: [76]
        # tensor points size: [76, 2]
        # tensor stability_score size: [76]
        # tensor boxes size: [76, 4]
        # list rles length: 76
        # tensor crop_boxes size: [76, 4]

        # Remove duplicate masks between crops
        if len(crop_boxes) > 1: # True for len(crop_boxes) -- 5
            # Prefer masks from smaller crops
            scores = 1 / box_area(data["crop_boxes"])
            scores = scores.to(data["boxes"].device)
            keep_by_nms = batched_nms(
                data["boxes"].float(),
                scores,
                torch.zeros_like(data["boxes"][:, 0]),  # categories
                iou_threshold=self.crop_nms_thresh,
            )
            data.filter(keep_by_nms)

        data.to_numpy()

        return data

    def _process_crop(self,
        image: np.ndarray,
        crop_box: List[int],
        crop_layer_idx: int,
        orig_size: Tuple[int, ...],
    ) -> MaskData:
        # array [image] shape: (1024, 1024, 3) , min: 0 , max: 255
        # list [crop_box] len: 4 , [0, 0, 1024, 1024]
        # [crop_layer_idx] value: 0
        # tuple [orig_size] len: 2 , (1024, 1024)

        # Crop the image and calculate embeddings
        x0, y0, x1, y1 = crop_box
        cropped_im = image[y0:y1, x0:x1, :]
        cropped_im_size = cropped_im.shape[:2]
        self.predictor.set_image(cropped_im)

        # Get points for this crop
        points_scale = np.array(cropped_im_size)[None, ::-1]
        points_for_image = self.point_grids[crop_layer_idx] * points_scale

        # Generate masks for this crop in batches
        data = MaskData()
        # self.points_per_batch -- 64

        # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        #
        # (Pdb) self.points_per_batch -- 64
        # (Pdb) points_for_image
        # array([[  16.,   16.],
        #        [  48.,   16.],
        #        [  80.,   16.],
        #        ...,
        #        [ 944., 1008.],
        #        [ 976., 1008.],
        #        [1008., 1008.]])
        # (Pdb) points_for_image.shape -- (1024, 2)
        for (points,) in batch_iterator(self.points_per_batch, points_for_image):
            batch_data = self._process_batch(points, cropped_im_size, crop_box, orig_size)
            data.cat(batch_data)

            del batch_data
        #
        # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

        # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        self.predictor.reset_image()
        # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

        # Remove duplicates within this crop.
        keep_by_nms = batched_nms(
            data["boxes"].float(), # data["boxes"].size() -- [708, 4]
            data["iou_preds"], # size() -- [708]
            torch.zeros_like(data["boxes"][:, 0]),  # categories
            iou_threshold=self.box_nms_thresh,
        )
        data.filter(keep_by_nms)

        # Return to the original image frame
        data["boxes"] = uncrop_boxes_xyxy(data["boxes"], crop_box)
        data["points"] = uncrop_points(data["points"], crop_box)
        data["crop_boxes"] = torch.tensor([crop_box for _ in range(len(data["rles"]))])

        return data

    def _process_batch(self,
        points: np.ndarray,
        im_size: Tuple[int, ...],
        crop_box: List[int],
        orig_size: Tuple[int, ...],
    ) -> MaskData:
        # array [_process_batch.points] shape: (64, 2) , min: 16.0 , max: 1008.0
        # tuple [_process_batch.im_size] len: 2 , (1024, 1024)
        # list [_process_batch.crop_box] len: 4 , [0, 0, 1024, 1024]
        # tuple [_process_batch.orig_size] len: 2 , (1024, 1024)

        orig_h, orig_w = orig_size

        # Run model on this batch
        # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        transformed_points = self.predictor.transform.apply_coords(points, im_size)
        # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

        in_points = torch.as_tensor(transformed_points, device=self.predictor.device)
        in_labels = torch.ones(in_points.shape[0], dtype=torch.int, device=in_points.device)

        # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        masks, iou_preds, _ = self.predictor.predict_torch(
            in_points[:, None, :], # [64, 2] --> [64, 1, 2]
            in_labels[:, None], # [64] --> [64, 1]
        )
        # masks.size() -- [64, 3, 1024, 1024]
        # iou_preds.size() -- [64, 3]

        # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

        # Serialize predictions and store in MaskData
        data = MaskData(
            masks=masks.flatten(0, 1),
            iou_preds=iou_preds.flatten(0, 1),
            points=torch.as_tensor(points.repeat(masks.shape[1], axis=0)),
        )
        del masks
        # data -- <segment_anything.utils.amg.MaskData object>

        # Filter by predicted IoU
        if self.pred_iou_thresh > 0.0: # True for self.pred_iou_thresh -- 0.86
            keep_mask = data["iou_preds"] > self.pred_iou_thresh
            data.filter(keep_mask)

        # Calculate stability score
        data["stability_score"] = calculate_stability_score(
            data["masks"], self.predictor.model.mask_threshold, self.stability_score_offset
        )

        if self.stability_score_thresh > 0.0: #True for self.stability_score_thresh -- 0.92
            keep_mask = data["stability_score"] >= self.stability_score_thresh
            data.filter(keep_mask)

        # Threshold masks and calculate boxes
        data["masks"] = data["masks"] > self.predictor.model.mask_threshold
        # self.predictor.model.mask_threshold -- 0.0

        data["boxes"] = batched_mask_to_box(data["masks"]) # size() -- [14, 4]

        # Filter boxes that touch crop boundaries
        keep_mask = ~is_box_near_crop_edge(data["boxes"], crop_box, [0, 0, orig_w, orig_h]) # size() -- 14
        if not torch.all(keep_mask):
            data.filter(keep_mask)

        # Compress to RLE
        data["masks"] = uncrop_masks(data["masks"], crop_box, orig_h, orig_w)
        data["rles"] = mask_to_rle_pytorch(data["masks"]) # len(data["rles"]) -- 14, data["rles"][0].keys() -- ['size', 'counts']
        del data["masks"]

        return data

    @staticmethod
    def postprocess_small_regions(
        mask_data: MaskData, min_area: int, nms_thresh: float
    ) -> MaskData:
        """
        Removes small disconnected regions and holes in masks, then reruns
        box NMS to remove any new duplicates.
        """

        if len(mask_data["rles"]) == 0:
            return mask_data

        # Filter small disconnected regions and holes
        new_masks = []
        scores = []
        for rle in mask_data["rles"]:
            mask = rle_to_mask(rle)

            mask, changed = remove_small_regions(mask, min_area, mode="holes")
            unchanged = not changed
            mask, changed = remove_small_regions(mask, min_area, mode="islands")
            unchanged = unchanged and not changed

            new_masks.append(torch.as_tensor(mask).unsqueeze(0))
            # Give score=0 to changed masks and score=1 to unchanged masks
            # so NMS will prefer ones that didn't need postprocessing
            scores.append(float(unchanged))

        # Recalculate boxes and remove any new duplicates
        masks = torch.cat(new_masks, dim=0)
        boxes = batched_mask_to_box(masks)
        keep_by_nms = batched_nms(
            boxes.float(),
            torch.as_tensor(scores),
            torch.zeros_like(boxes[:, 0]),  # categories
            iou_threshold=nms_thresh,
        )

        # Only recalculate RLEs for masks that have changed
        for i_mask in keep_by_nms:
            if scores[i_mask] == 0.0:
                mask_torch = masks[i_mask].unsqueeze(0)
                mask_data["rles"][i_mask] = mask_to_rle_pytorch(mask_torch)[0]
                mask_data["boxes"][i_mask] = boxes[i_mask]  # update res directly
        mask_data.filter(keep_by_nms)

        return mask_data
