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
import torch.nn.functional as F
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
from albumentations.pytorch import ToTensorV2

class LEVIR_256(torch.utils.data.Dataset):
    def __init__(self,conf,mode):
        super(LEVIR_256, self).__init__()
        
        if(mode=='train'):
            txt_file = open("/data/LEVIR-CD-256/list/train.txt", "r")
            content_list = txt_file.readlines()
            content_list = [c.replace('\n','') for c in content_list]
            
            self.data = content_list

            self.mode='train'
            self.geometric_transform = A.Compose(
                [
                    A.Resize(conf.IMAGE_SIZE, conf.IMAGE_SIZE),
                    A.RandomSizedCrop(min_max_height=(int(0.75*conf.IMAGE_SIZE), conf.IMAGE_SIZE), height=conf.IMAGE_SIZE, width=conf.IMAGE_SIZE, p=0.5),
                    A.Flip(p=0.5),
                    A.RandomRotate90(p=0.5),],
                    additional_targets={'image0': 'image'}
            )
            self.transform = A.Compose(
                [
                    A.OneOf([
                       A.GaussNoise(p=1.0),
                       A.ISONoise(p=1.0)
                    ], p=0.5),
                    A.OneOf([
                        A.MedianBlur(blur_limit=3, p=1.0),
                        A.Blur(blur_limit=3, p=1.0),
                        A.CLAHE(p=1.0),
                        A.Equalize(p=1.0),
                    ], p=0.5),
                    A.ColorJitter(0.4,0.4,0.4,0.4),
                    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                    ToTensorV2(),
                ],
                
            )

            print(conf.LABEL_SMOOTHING)
            
            self.label_smooothing = conf.LABEL_SMOOTHING
            print(len(self.data))
            print("train ",len(self))

        elif(mode=='val'):
            txt_file = open("/data/LEVIR-CD-256/list/val.txt", "r")
            content_list = txt_file.readlines()
            content_list = [c.replace('\n','') for c in content_list]
            self.data = content_list
            self.mode='val'
            self.transform = A.Compose(
            [A.Resize(conf.IMAGE_SIZE, conf.IMAGE_SIZE),
            A.Normalize(mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225)), ToTensorV2()],
            additional_targets={'image0': 'image'})
            
            self.label_smooothing = conf.LABEL_SMOOTHING
            #print("val ",len(self))
        
        elif(mode=='test'):
            txt_file = open("/data/LEVIR-CD-256/list/test.txt", "r")
            content_list = txt_file.readlines()
            content_list = [c.replace('\n','') for c in content_list]
            self.data = content_list
            self.mode='test'
            self.transform = A.Compose(
            [A.Resize(conf.IMAGE_SIZE, conf.IMAGE_SIZE),
            A.Normalize(mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225)), ToTensorV2()],
            additional_targets={'image0': 'image'})
            
            self.label_smooothing = conf.LABEL_SMOOTHING
            print("test ",len(self))
        
        self.resnet_mean = [0.485, 0.456, 0.406]
        self.resnet_std = [0.229, 0.224, 0.225]

        
        
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
        if self.mode == 'train':
            transformed = self.geometric_transform(image = prev_img, image0 = after_img, mask=label_img)
            prev_img = transformed['image']
            after_img = transformed['image0']
            label_img = transformed['mask']

            transformed = self.transform(image = prev_img)
            prev_img = transformed['image']

            transformed = self.transform(image = after_img)
            after_img = transformed['image']
            label_img = torch.from_numpy(label_img/255.0).to(torch.int64)
        else:
            transformed = self.transform(image = prev_img, image0 = after_img, mask=label_img)
            prev_img = transformed['image']
            after_img = transformed['image0']
            label_img = transformed['mask']
            label_img = (label_img/255.0).to(torch.int64)

        label_img = F.one_hot(label_img,num_classes=2).permute(2,0,1).to(torch.float32)
        label_img = (label_img - 0.5)*(1-self.label_smooothing)+0.5
        
        """
        h w c
        """
        img_dict = {}

        img_dict['prev_img'] = prev_img
        img_dict['after_img'] = after_img
        img_dict['label_img'] = label_img
        img_dict['path'] = img_path
    
        return img_dict

    def __len__(self):
        return len(self.data)