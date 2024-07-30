import argparse
import cv2
import glob
import os
import torch
import requests
import numpy as np
from os import path as osp
import utils_image as util  # Assuming utils_image.py contains the proper functions
from rich.progress import track
from natsort import natsorted


def im2tensor(image, imtype=np.uint8, cent=1., factor=255./2.):
    return torch.Tensor((image / factor - cent)
                        [:, :, :, np.newaxis].transpose((3, 2, 0, 1)))

# Cam-Vid+
path = "xxx"   # predication_path  #long
gt_path = 'xxx'  # gt_path  #long
folders = os.listdir(path)

print(path)
all_psnr = []
all_ssim = []

for folder in folders:
    print(folder)
    imgs = natsorted(glob.glob(osp.join(path, folder, '*.png')))
    imgs_gt = natsorted(glob.glob(osp.join(gt_path, folder, '*.jpg')))
    
    for i in track(range(len(imgs))):
        output = cv2.imread(imgs[i])
        gt = cv2.imread(imgs_gt[i])
        if output.shape != gt.shape:
            print(output.shape, gt.shape)

        all_psnr.append(util.calculate_psnr(output, gt))
        all_ssim.append(util.calculate_ssim(output, gt))

# Calculate and print the overall average PSNR and SSIM
avg_psnr = np.mean(all_psnr)
avg_ssim = np.mean(all_ssim)

print('Average PSNR: ', avg_psnr)
print('Average SSIM: ', avg_ssim)
print(path)