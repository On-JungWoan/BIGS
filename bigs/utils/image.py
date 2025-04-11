# Code based on 3DGS: https://github.com/graphdeco-inria/gaussian-splatting/blob/main/utils/image_utils.py
# License from 3DGS: https://github.com/graphdeco-inria/gaussian-splatting/blob/main/LICENSE.md

#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import os
import cv2
import torch
import pathlib
import numpy as np
from torchvision.utils import make_grid
from PIL import Image, ImageDraw, ImageFont
from typing import Optional, Union, List, BinaryIO


def rgb_to_grayscale(image):
    r, g, b = image[0], image[1], image[2]
    gray_image = 0.299 * r + 0.587 * g + 0.114 * b
    return gray_image

# Find the bounding box based on the binary mask.
def get_bounding_box(mask):
    rows = torch.any(mask, dim=1)
    cols = torch.any(mask, dim=0)
    ymin, ymax = torch.where(rows)[0][[0, -1]]
    xmin, xmax = torch.where(cols)[0][[0, -1]]
    return xmin, ymin, xmax, ymax

# Crop the image tensor
def crop_image(image, xmin, ymin, xmax, ymax):
    return image[:, ymin:ymax+1, xmin:xmax+1]

# remove o.l.
def remove_outliers(mask, blur_kernel_size=5, threshold_value=0):
    median_blur_mask = cv2.medianBlur(mask, blur_kernel_size)
    diff = cv2.absdiff(mask, median_blur_mask)
    _, outliers = cv2.threshold(diff, threshold_value, 255, cv2.THRESH_BINARY)
    clean_mask = np.where(outliers == 255, median_blur_mask, mask)

    return clean_mask

def crop_and_center_image(image, threshold_value=0.9, img_size=256, resize_img=False, return_mask=False):
    device = image.device
    
    # Apply a smooth threshold
    binary_mask = rgb_to_grayscale(image.detach().clone()) < threshold_value

    # remove o.l
    binary_mask = torch.from_numpy(remove_outliers(binary_mask.float().cpu().numpy(), threshold_value=0)).bool()
    
    if return_mask:
        # gaussian blur
        binary_mask = np.ascontiguousarray(binary_mask.float())[..., None]
        # cv2.imwrite('test.png', cv2.GaussianBlur(binary_mask, (199, 199), 100)*255)
        blured_mask = cv2.GaussianBlur(binary_mask, (99, 99), 100)
        binary_mask = torch.tensor(blured_mask != 0.)    
        
        return binary_mask

    # Get bounding box coordinates
    xmin, ymin, xmax, ymax = get_bounding_box(binary_mask)

    # Crop the image tensor
    cropped_image = crop_image(image, xmin, ymin, xmax, ymax)
    
    if resize_img:
        H = W = int(img_size/2)
        cropped_image = torch.nn.functional.interpolate(cropped_image.unsqueeze(0), size=(H, W), mode='bilinear', align_corners=False)[0]
    else:
        _, H, W = cropped_image.shape
    
    # import cv2
    # cv2.imwrite('test.png', background.permute(1,2,0).cpu().detach().numpy()*255)
    
    # make bg image
    background = torch.ones(3, img_size, img_size, device=device)
    
    # Calculate the coordinates to center the cropped image on the background
    top = (img_size - H) // 2
    left = (img_size - W) // 2
    
    background[:, top:top+H, left:left+W] = cropped_image
    
    if return_mask:
        raise Exception('Not implemented yet!')
        # gaussian blur
        binary_mask = np.ascontiguousarray(binary_mask.float())[..., None]
        # cv2.imwrite('test.png', cv2.GaussianBlur(binary_mask, (199, 199), 100)*255)
        blured_mask = cv2.GaussianBlur(binary_mask, (99, 99), 100)
        binary_mask = torch.tensor(blured_mask != 0.)
        
        return background, binary_mask
    else:
        return background


def mse(img1, img2):
    return (((img1 - img2)) ** 2).view(img1.shape[0], -1).mean(1, keepdim=True)


def psnr(img1, img2):
    mse = (((img1 - img2)) ** 2).view(img1.shape[0], -1).mean(1, keepdim=True)
    return 20 * torch.log10(1.0 / torch.sqrt(mse))


@torch.no_grad()
def normalize_depth(depth, min=None, max=None):
    if depth.shape[0] == 1:
        depth = depth[0]
    
    if min is None:
        min = depth.min()

    if max is None:
        max = depth.max()
        
    depth = (depth - min) / (max - min)
    depth = 1.0 - depth
    return depth


@torch.no_grad()
def save_image(
    tensor: Union[torch.Tensor, List[torch.Tensor]],
    fp: Union[str, pathlib.Path, BinaryIO],
    format: Optional[str] = None,
    text_labels: Optional[List[str]] = None,
    **kwargs,
) -> None:
    """
    Save a given Tensor into an image file.

    Args:
        tensor (Tensor or list): Image to be saved. If given a mini-batch tensor,
            saves the tensor as a grid of images by calling ``make_grid``.
        fp (string or file object): A filename or a file object
        format(Optional):  If omitted, the format to use is determined from the filename extension.
            If a file object was used instead of a filename, this parameter should always be used.
        **kwargs: Other arguments are documented in ``make_grid``.
    """
    
    grid = make_grid(tensor, **kwargs)
    txt_font = ImageFont.load_default()
    # Add 0.5 after unnormalizing to [0, 255] to round to nearest integer
    ndarr = grid.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to("cpu", torch.uint8).numpy()
    im = Image.fromarray(ndarr)
    draw = ImageDraw.Draw(im)
    draw.text((10, 10), text_labels, fill=(0, 0, 0), font=txt_font)
    im.save(fp, format=format)
    
    
def save_rgba_image(
    tensor: Union[torch.Tensor, List[torch.Tensor]],
    fp: Union[str, pathlib.Path, BinaryIO],
    format: Optional[str] = 'PNG',
    text_labels: Optional[List[str]] = None,
    **kwargs,
) -> None:
    os.makedirs(os.path.dirname(fp), exist_ok=True)
    
    grid = make_grid(tensor, **kwargs)
    txt_font = ImageFont.load_default()
    # Add 0.5 after unnormalizing to [0, 255] to round to nearest integer
    ndarr = grid.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to("cpu", torch.uint8).numpy()
    im = Image.fromarray(ndarr)
    if text_labels is not None:
        draw = ImageDraw.Draw(im)
        draw.text((10, 10), text_labels, fill=(0, 0, 0), font=txt_font)
    im.save(fp, format=format)
