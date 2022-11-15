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
import csv
import os
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
from skimage.exposure import match_histograms

class LEVIR_256(torch.utils.data.Dataset):
    def __init__(self,conf,mode):
        super(LEVIR_256, self).__init__()
        
        if(mode=='train'):
            data = glob('/data/maicon_train/x/*.png')
            dataset = []
            for row in tqdm(data) :
                dataset.append({'path' : row})
            
            self.data = dataset
            self.mode='train'
            self.transform = A.Compose(
                [
                    A.Resize(conf.IMAGE_SIZE, conf.IMAGE_SIZE),
                    A.RandomSizedCrop(min_max_height=(int(0.75*conf.IMAGE_SIZE), conf.IMAGE_SIZE), height=conf.IMAGE_SIZE, width=conf.IMAGE_SIZE, p=0.5),
                    A.Flip(p=0.5),
                    A.OneOf([
                       A.GridDistortion(p=1.0),
                       A.ElasticTransform(p=1.0)
                    ], p=0.5),
                    A.RandomRotate90(p=0.5),
                    A.OneOf([
                       A.GaussNoise(p=1.0),
                       A.ISONoise(p=1.0)
                    ], p=0.5),
                    A.OneOf([
                        A.MedianBlur(blur_limit=3, p=1.0),
                        A.Blur(blur_limit=3, p=1.0),
                    ], p=0.5),
                    A.ColorJitter(0.4,0.4,0.4,0.4),
                    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                    ToTensorV2(),
                    ]
            )

            print(conf.LABEL_SMOOTHING)
            
            self.label_smooothing = conf.LABEL_SMOOTHING
            print(len(self.data))
            print("train ",len(self))

        elif(mode=='val'):
            data = glob('/data/maicon_train/x/*.png')
            dataset = []
            for row in tqdm(data) :
                dataset.append({'path' : row})
            
            self.data = dataset
            self.mode='val'
            self.transform = A.Compose(
            [A.Resize(conf.IMAGE_SIZE, conf.IMAGE_SIZE),
            A.Normalize(mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225)), ToTensorV2()],)
            
            self.label_smooothing = conf.LABEL_SMOOTHING
            #print("val ",len(self))
        
        elif(mode=='test'):
            data = glob('/data/maicon_test/x/*.png')
            dataset = []
            for row in tqdm(data) :
                dataset.append({'path' : row})
            self.data = dataset

            self.mode='test'
            self.transform = A.Compose(
            [A.Resize(conf.IMAGE_SIZE, conf.IMAGE_SIZE),
            A.Normalize(mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225)), ToTensorV2()],)
            
            self.label_smooothing = conf.LABEL_SMOOTHING
            print("test ",len(self))
        
        self.resnet_mean = [0.485, 0.456, 0.406]
        self.resnet_std = [0.229, 0.224, 0.225]

        
        
    def __getitem__(self,index):
        
        img_path = self.data[index]['path']

        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h,w,c = img.shape

        if self.mode!='test':
            label_img_path = img_path.replace('x','y')
            label_img = cv2.imread(label_img_path,0)
            temp = np.zeros((h,w,4))
            for p in range(4):
                temp[:,:,p] = (label_img==p)*1
            label_img = temp
        else:
            label_img = np.zeros((h,w,4))

        if self.mode == 'train':
            transformed = self.transform(image = img, mask=label_img)
            img = transformed['image']
            label_img = transformed['mask']
        else:
            transformed = self.transform(image = img, mask=label_img)
            img = transformed['image']
            label_img = transformed['mask']

        if self.mode != 'test':
            try:
                label_img = label_img.to(dtype=torch.float32).permute(2,0,1)
            except:
                print(label_img_path)
            label_img = (label_img - 0.5)*(1-self.label_smooothing)+0.5
        
        """
        h w c
        """
        img_dict = {}

        img_dict['imgs'] = img
        img_dict['label_img'] = label_img
        img_dict['origin_h'] = h
        img_dict['origin_w'] = w
        img_dict['path'] = img_path
    
        return img_dict

    def __len__(self):
        return len(self.data)
