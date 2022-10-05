import torch
import torchvision
from torch.utils.data import Dataset
from glob import glob
import PIL.Image
import numpy as np
from torchvision import transforms
import torchvision.transforms.functional as TF
import random
import albumentations as A
import cv2
import os
import csv
import pdb
import pandas as pd
from tqdm import tqdm
from pathlib import Path
import math
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
from PIL import Image
import pickle
from utils import RandomColorjitter,RandomCropResize,RandomFlip

class LEVIR_256(torch.utils.data.Dataset):
    def __init__(self,conf,mode):
        super(LEVIR_256, self).__init__()
        
        if(mode=='train'):
            txt_file = open("/data/LEVIR-CD-256/list/train.txt", "r")
            content_list = txt_file.readlines()
            content_list = [c.replace('\n','') for c in content_list]
            self.data = content_list
            self.mode='train'
            
            self.label_smooothing = conf.LABEL_SMOOTHING
            print("train ",len(self))

        elif(mode=='val'):
            txt_file = open("/data/LEVIR-CD-256/list/val.txt", "r")
            content_list = txt_file.readlines()
            content_list = [c.replace('\n','') for c in content_list]
            self.data = content_list
            self.mode='val'
            
            self.label_smooothing = conf.LABEL_SMOOTHING
            print("val ",len(self))
        
        elif(mode=='test'):
            txt_file = open("/data/LEVIR-CD-256/list/test.txt", "r")
            content_list = txt_file.readlines()
            content_list = [c.replace('\n','') for c in content_list]
            self.data = content_list
            self.mode='test'
            
            self.label_smooothing = conf.LABEL_SMOOTHING
            print("test ",len(self))
        
        self.resnet_mean = [0.485, 0.456, 0.406]
        self.resnet_std = [0.229, 0.224, 0.225]

        self.RandomCrop = RandomCropResize()
        self.RandomColorjitter = RandomColorjitter()
        self.RandomFlip = RandomFlip()
        
    def __getitem__(self,index):
        
        img_path = self.data[index]

        prev_img_path = '/data/LEVIR-CD-256/'+'A'+'/'+img_path
        after_img_path = '/data/LEVIR-CD-256/'+'B'+'/'+img_path
        label_img_path = '/data/LEVIR-CD-256/'+'label'+'/'+img_path

        prev_img = cv2.imread(prev_img_path)
        prev_img = cv2.cvtColor(prev_img, cv2.COLOR_BGR2RGB)
        after_img = cv2.imread(after_img_path)
        after_img = cv2.cvtColor(after_img, cv2.COLOR_BGR2RGB)

        label_img = cv2.imread(label_img_path,0)
        h,w = label_img.shape
        target = np.zeros((h,w,2))
        target[:,:,0] = (label_img==0)*1
        target[:,:,1] = (label_img==255)*1
        label_img = target.astype(np.float32)
        
        prev_img = torch.from_numpy(prev_img).permute(2,0,1).to(dtype=torch.float32).unsqueeze(0)
        after_img = torch.from_numpy(after_img).permute(2,0,1).to(dtype=torch.float32).unsqueeze(0)
        label_img = torch.from_numpy(label_img).permute(2,0,1).to(dtype=torch.float32).unsqueeze(0)

        prev_img = prev_img/255.0
        after_img = after_img/255.0

        if self.mode=='train':
            prev_img, after_img, label_img = self.RandomCrop(prev_img,after_img,label_img)
            prev_img, after_img, label_img = self.RandomFlip(prev_img,after_img,label_img)
            prev_img, after_img, label_img = self.RandomColorjitter(prev_img,after_img,label_img)

        img_dict = {}

        prev_img = prev_img.squeeze()
        after_img = after_img.squeeze()
        label_img = torch.argmax(label_img.squeeze(),dim=0).to(torch.int64)

        prev_img = TF.normalize(prev_img,mean=self.resnet_mean, std=self.resnet_std)
        after_img = TF.normalize(after_img, mean=self.resnet_mean, std=self.resnet_std)

        img_dict['prev_img'] = prev_img
        img_dict['after_img'] = after_img
        img_dict['label_img'] = label_img
        img_dict['path'] = img_path
    
        return img_dict

    def __len__(self):
        return len(self.data)