#!/usr/bin/env python
# coding: utf-8
#
# /************************************************************************************
# ***
# ***    Copyright Dell 2023, All Rights Reserved.
# ***
# ***    File Author: Dell, Fri 30 Jun 2023 03:14:33 AM CST
# ***
# ************************************************************************************/
#

import os
from tqdm import tqdm

import torch
import numpy as np
# import cv2
import todos
import pdb

import sys
sys.path.append("..")

from mobile_encoder.setup_mobile_sam import setup_model
from segment_anything import SamAutomaticMaskGenerator


def save_segment_result(masks, input_tensor, output_file):
    # type(masks) -- <class 'list'>, len(masks) -- 17
    B, C, H, W = input_tensor.size()

    masks = sorted(masks, key=(lambda x: x['area']), reverse=True)

    seg_numpy = np.ones((B, H, W, C))
    for mask in masks:
        # mask.keys() -- ['segmentation', 'area', 'bbox', 'predicted_iou', 'point_coords', 'stability_score', 'crop_box']
        m = mask['segmentation']
        mask_color = np.concatenate([np.random.random(3)])
        seg_numpy[:, m] = mask_color
        # pdb.set_trace()

    seg_tensor = torch.from_numpy(seg_numpy).permute(0, 3, 1, 2)
    output_tensor = 0.5 * input_tensor + 0.5 * seg_tensor    

    todos.data.save_tensor([input_tensor, output_tensor], output_file)


if __name__ == "__main__":
    input_files = "images/*.*"    
    output_dir = "output"

    # Create directory to store result
    todos.data.mkdir(output_dir)


    # Load model
    checkpoint = torch.load('../weights/mobile_sam.pt')
    model = setup_model()
    model.load_state_dict(checkpoint, strict=True)
    device = "cuda"
    model.to(device=device)
    model.eval()
    # mask_generator = SamAutomaticMaskGenerator(model)

    mask_generator = SamAutomaticMaskGenerator(
        model=model,
        points_per_side=32,
        pred_iou_thresh=0.86,
        stability_score_thresh=0.92,
        crop_n_layers=1,
        crop_n_points_downscale_factor=2,
        min_mask_region_area=100,  # Requires open-cv to run post-processing
    )


    # Load files
    image_filenames = todos.data.load_files(input_files)

    # Start predict
    progress_bar = tqdm(total=len(image_filenames))
    for filename in image_filenames:
        progress_bar.update(1)

        # Original input
        input_tensor = todos.data.load_tensor(filename)

        # image = cv2.imread(filename)
        # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = input_tensor.permute(0, 2, 3, 1).squeeze(0).numpy() * 255.0
        masks = mask_generator.generate(image.astype(np.uint8))


        output_filename = f"{output_dir}/{os.path.basename(filename)}"
        save_segment_result(masks, input_tensor, output_filename)

    # todos.model.reset_device()
