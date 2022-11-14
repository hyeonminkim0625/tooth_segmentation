import torch
import torch.nn as nn
import numpy as np
import torchvision.transforms.functional as TF
import torch.nn.functional as F
from tqdm import tqdm
from pathlib import Path
from glob import glob
import cv2

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

def cutmix_data(input_prev,input_after,target,alpha=1.0):
    lam = np.random.beta(alpha, alpha)
    rand_index = torch.randperm(input_prev.size()[0]).cuda()
    bbx1, bby1, bbx2, bby2 = rand_bbox(input_prev.size(), lam)
    input_prev[:, :, bbx1:bbx2, bby1:bby2] = input_prev[rand_index, :, bbx1:bbx2, bby1:bby2]
    input_after[:, :, bbx1:bbx2, bby1:bby2] = input_after[rand_index, :, bbx1:bbx2, bby1:bby2]
    target[:, bbx1:bbx2, bby1:bby2] = target[rand_index,bbx1:bbx2, bby1:bby2]
    
    # compute output
    return input_prev,input_after,target

def halfmix_data(input_prev,input_after ,target,alpha=1.0):
    rand_index = torch.randperm(input_prev.size()[0]).cuda()
    size = input_prev.size()
    W = size[2]
    H = size[3]
    if np.random.random()>0.5:
        input_prev[:, :, 0:H//2] = input_prev[rand_index, :, 0:H//2]
        input_after[:, :, 0:H//2] = input_after[rand_index, :, 0:H//2]
        target[:, :, 0:H//2] = target[rand_index, :, 0:H//2]
    else:
        input_prev[:, :,:, 0:W//2] = input_prev[rand_index, :,:, 0:W//2]
        input_after[:, :,:, 0:W//2] = input_after[rand_index, :,:, 0:W//2]
        target[:,:, 0:W//2] = target[rand_index,:, 0:W//2]
    
    # compute output
    return input_prev,input_after,target

class RandomFlip(torch.nn.Module):
    def __init__(self,p=0.5):
        super().__init__()
        self.p = p

    def forward(self, prev_image, after_image, label_image):
        if np.random.random()<self.p:
            after_image = TF.hflip(after_image)
            prev_image = TF.hflip(prev_image)
            label_image = TF.hflip(label_image)
            

        if np.random.random()<self.p:
            after_image = TF.vflip(after_image)
            prev_image = TF.vflip(prev_image)
            label_image = TF.vflip(label_image)
            
        return prev_image,after_image,label_image

class RandomCropResize(torch.nn.Module):
    def __init__(self,rate=0.8,p=0.5):
        super().__init__()
        self.p = p
        self.maxsize = 256
        self.minsize = int(rate*self.maxsize)
    
    def set_parameter(self):
        resize = np.random.randint(self.minsize,self.maxsize+1)
        #np.random.randint(100,101) -> 100
        #                   100, 102 -> 100, 101
        margin = max(self.maxsize - resize,1)
        new_y = np.random.randint(0,margin)
        new_x = np.random.randint(0,margin)

        return new_y, new_x, resize

    def forward(self, prev_image, after_image, label_image):
        if np.random.random() < self.p:
            y,x,size = self.set_parameter()
            after_image = after_image[:,:,y:y+size,x:x+size]
            after_image = F.interpolate(after_image,(self.maxsize,self.maxsize),mode='bilinear',align_corners=True)
            
            prev_image = prev_image[:,:,y:y+size,x:x+size]
            prev_image = F.interpolate(prev_image,(self.maxsize,self.maxsize),mode='bilinear',align_corners=True)

            label_image = label_image[:,:,y:y+size,x:x+size]
            label_image = F.interpolate(label_image,(self.maxsize,self.maxsize),mode='bilinear',align_corners=True)

        return prev_image,after_image,label_image

class RandomFlip(torch.nn.Module):
    def __init__(self,p=0.5):
        super().__init__()
        self.p = p

    def forward(self, prev_image, after_image, label_image):
        if np.random.random()<self.p:
            after_image = TF.hflip(after_image)
            prev_image = TF.hflip(prev_image)
            label_image = TF.hflip(label_image)
            
        if np.random.random()<self.p:
            after_image = TF.vflip(after_image)
            prev_image = TF.vflip(prev_image)
            label_image = TF.vflip(label_image)
            
        return prev_image,after_image,label_image

class RandomColorjitter(torch.nn.Module):
    def __init__(self,brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2,p=0.5):
        super().__init__()
        self.p = p
        self.brightness = brightness
        self.contrast = contrast
        self.saturation = saturation
        self.hue = hue

    def forward(self, prev_image, after_image, label_image):

        idx = torch.randperm(4)
        for i in idx:
            if np.random.random()<self.p and i==0:
                rate = 1 + np.random.random()/2.5 - 0.2
                after_image = TF.adjust_brightness(after_image,rate)
                prev_image = TF.adjust_brightness(prev_image,rate)

            if np.random.random()<self.p and i==1:
                rate = 1 + np.random.random()/2.5 - 0.2
                after_image = TF.adjust_contrast(after_image,rate)
                prev_image = TF.adjust_contrast(prev_image,rate)

            if np.random.random()<self.p and i==2:
                rate = 1 + np.random.random()/2.5 - 0.2
                after_image = TF.adjust_saturation(after_image,rate)
                prev_image = TF.adjust_saturation(prev_image,rate)
            
            if np.random.random()<self.p and i==3:
                rate = np.random.random()/2.5 - 0.2
                after_image = TF.adjust_hue(after_image,rate)
                prev_image = TF.adjust_hue(prev_image,rate)

        return prev_image,after_image,label_image

def calculate_iou(pred,target,num_classes):

    #pred_mask = np.argmax(pred,axis=0)
    #target_mask = np.argmax(target,axis=0)
    iou_list = []
    for i in range(0,num_classes):
        if torch.sum((torch.sum((pred[i]==True)|(target[i]==True)))) == 0:
            iou_score = np.nan
        else:
            iou_score = (torch.sum((pred[i]==True)&(target[i]==True))+ 1e-6) /(torch.sum((pred[i]==True)|(target[i]==True))+ 1e-6)
        iou_list.append(iou_score)
    
    return iou_list