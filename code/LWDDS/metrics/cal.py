import argparse
import cv2
import glob
import os
import torch
import requests
import numpy as np
from os import path as osp
import utils_image as util
from rich.progress import track
from natsort import natsorted

def im2tensor(image, imtype=np.uint8, cent=1., factor=255./2.):
    return torch.Tensor((image / factor - cent)
                        [:, :, :, np.newaxis].transpose((3, 2, 0, 1)))

path = "xxx"   # predication_path
gt_path = 'xxx'  # gt_path
folders = os.listdir(path)

print(path)
psnr = []
ssim = []


for folder in folders:
    print(folder)
    imgs = natsorted(glob.glob(osp.join(path, folder, '*.png')))
    imgs_gt = natsorted(glob.glob(osp.join(gt_path, folder, '*.png')))
    psnr_folder = []
    ssim_folder = []
   # lpips_folder = []
    
    for i in track(range(len(imgs))):
        output = cv2.imread(imgs[i])
        gt = cv2.imread(imgs_gt[i])
        if output.shape != gt.shape:
            print(output.shape, gt.shape)

        psnr_folder.append(util.calculate_psnr(output, gt))
        ssim_folder.append(util.calculate_ssim(output, gt))


    psnr.append(np.mean(psnr_folder))
    ssim.append(np.mean(ssim_folder))

    
print('psnr: ', np.mean(psnr))
print('ssim: ', np.mean(ssim))
print(path)
    