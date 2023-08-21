"""Image Recognize Anything Model Package."""  # coding=utf-8
#
# /************************************************************************************
# ***
# ***    Copyright Dell 2023(18588220928@163.com) All Rights Reserved.
# ***
# ***    File Author: Dell, Thu 13 Jul 2023 01:55:56 PM CST
# ***
# ************************************************************************************/
#

__version__ = "1.0.0"

import os
from tqdm import tqdm
from PIL import Image
import numpy as np

import torch
import todos
from . import sam
from torchvision.transforms import Compose, ToTensor
from pathlib import Path
import pdb


def create_model():
    """
    Create model
    """

    model = sam.MobileSAM()
    device = todos.model.get_device()
    model = model.to(device)
    model.eval()
    print(f"Running model on {device} ...")

    return model, device


def get_model():
    """Load jit script model."""

    model, device = create_model()
    # print(model)

    # https://github.com/pytorch/pytorch/issues/52286
    torch._C._jit_set_profiling_executor(False)
    # C++ Reference
    # torch::jit::getProfilingMode() = false;                                                                                                             
    # torch::jit::setTensorExprFuserEnabled(false);

    # model = torch.jit.script(model)
    # todos.data.mkdir("output")
    # if not os.path.exists("output/SAM.torch"):
    #     model.save("output/SAM.torch")

    return model, device


def predict(input_files, output_dir):
    # Create directory to store result
    todos.data.mkdir(output_dir)

    # load model
    model, device = get_model()
    transform = Compose([
        lambda image: image.convert("RGB"),
        ToTensor(),
    ])

    # load files
    image_filenames = todos.data.load_files(input_files)

    # start predict
    progress_bar = tqdm(total=len(image_filenames))
    for filename in image_filenames:
        progress_bar.update(1)

        image = Image.open(filename).convert("RGB")
        input_tensor = transform(image).unsqueeze(0).to(device)

        with torch.no_grad():
            output_tensor = model(input_tensor).cpu()

        output_file = f"{output_dir}/{os.path.basename(filename)}"
        todos.data.save_tensor([input_tensor, output_tensor], output_file)

    progress_bar.close()

    todos.model.reset_device()
