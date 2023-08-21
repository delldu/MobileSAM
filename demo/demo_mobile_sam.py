import numpy as np
import torch
import matplotlib.pyplot as plt
import cv2

import sys
sys.path.append("..")
from mobile_encoder.setup_mobile_sam import setup_model
from segment_anything import SamAutomaticMaskGenerator

import os
import pdb


def show_anns(anns, filename):
    if len(anns) == 0:
        return

    sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)
    ax = plt.gca()
    ax.set_autoscale_on(False)

    img = np.ones((sorted_anns[0]['segmentation'].shape[0], sorted_anns[0]['segmentation'].shape[1], 4))
    img[:,:,3] = 0
    for ann in sorted_anns:
        m = ann['segmentation']
        color_mask = np.concatenate([np.random.random(3), [0.35]])
        img[m] = color_mask

    plt.figure(figsize=(10,10))
    plt.imshow(img)
    plt.axis('off')
    plt.savefig(filename)
    plt.close()



if __name__ == "__main__":
    checkpoint = torch.load('../weights/mobile_sam.pt')
    sam = setup_model()
    sam.load_state_dict(checkpoint,strict=True)
    device = "cuda"
    sam.to(device=device)
    sam.eval()


    # mask_generator = SamAutomaticMaskGenerator(sam)
    mask_generator = SamAutomaticMaskGenerator(
        model=sam,
        points_per_side=32,
        pred_iou_thresh=0.86,
        stability_score_thresh=0.92,
        crop_n_layers=1,
        crop_n_points_downscale_factor=2,
        min_mask_region_area=100,  # Requires open-cv to run post-processing
    )


    for i in range(9):
        print("image:   ",i)
        image = cv2.imread('../demo/input_imgs/example'+str(i)+'.png')
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        masks = mask_generator.generate(image)

        filename = f"mobile_sam_result/example_{i}.png"
        show_anns(masks, filename)

