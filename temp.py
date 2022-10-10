import torch
import numpy as np
import cv2
from PIL import Image

def rand_bbox(size, lam):
    W = size[2]
    H = size[3]
    cut_rat = np.sqrt(1. - lam)
    cut_w = np.int(W * cut_rat)
    cut_h = np.int(H * cut_rat)

    # uniform
    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    return bbx1, bby1, bbx2, bby2

def cutmix_data(input,target,alpha=1.0):
    lam = np.random.beta(alpha, alpha)
    rand_index = torch.randperm(input.size()[0]).cuda()
    bbx1, bby1, bbx2, bby2 = rand_bbox(input.size(), lam)
    input[:, :, bbx1:bbx2, bby1:bby2] = input[rand_index, :, bbx1:bbx2, bby1:bby2]
    target[:, bbx1:bbx2, bby1:bby2] = target[rand_index,bbx1:bbx2, bby1:bby2]
    
    # compute output
    return input,target

def halfmix_data(input,target,alpha=1.0):
    rand_index = torch.randperm(input.size()[0]).cuda()
    size = input.size()
    W = size[2]
    H = size[3]
    if np.random.random()>0.5:
        input[:, :, 0:H//2] = input[rand_index, :, 0:H//2]
        target[:, :, 0:H//2] = target[rand_index, :, 0:H//2]
    else:
        input[:, :,:, 0:W//2] = input[rand_index, :,:, 0:W//2]
        target[:, :,:, 0:W//2] = target[rand_index, :,:, 0:W//2]
    
    # compute output
    return input,target