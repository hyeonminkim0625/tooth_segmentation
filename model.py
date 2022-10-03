import torch
import torch.nn as nn
import torchvision
from collections import OrderedDict
import torch.nn.functional as F
import torchvision.transforms.functional as TF
import timm
from timm.models.layers import Mlp
from einops import rearrange, reduce, repeat
from head import FPNHead


class DiffModule(nn.Module):
    """Some Information about DiffModule"""
    def __init__(self,in_channels,out_channels):
        super(DiffModule, self).__init__()

        self.MLP = Mlp(in_channels,out_features = out_channels)
        self.diff = nn.Sequential(
            nn.Conv2d(out_channels*2, out_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(out_channels),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU()
        )

    def forward(self, prev_img, after_img):

        prev_img = rearrange(prev_img, 'b c h w -> b h w c')
        after_img = rearrange(after_img, 'b c h w -> b h w c')

        prev_img  = self.MLP(prev_img)
        after_img  = self.MLP(after_img)

        prev_img = rearrange(prev_img, 'b h w c -> b c h w')
        after_img = rearrange(after_img, 'b h w c -> b c h w')

        diff = self.diff(torch.cat((prev_img, after_img), dim=1))

        return diff

class Model(nn.Module):
    def __init__(self,conf):
        super(Model, self).__init__()
        self.conf = conf
        self.backbone = None
        if conf.META_ARCHITECTURE == 'vit':
            self.backbone = timm.create_model('vit_base_patch16_224', pretrained=True, num_classes=0)
        elif conf.META_ARCHITECTURE == 'convnext':
            self.backbone = timm.create_model('convnext_tiny', pretrained=True, out_indices=(0,1,2,3), drop_path_rate=0.2,features_only=True)
        elif conf.META_ARCHITECTURE == 'swin':
            self.backbone = timm.create_model('swin_large_patch4_window12_384_in22k', pretrained=True, num_classes=3,drop_path_rate=0.5,drop_rate=0.0)

        """
        1 -> n : low resolution
        96, 192, 384, 768
        """
        
        self.conv_diff1 = DiffModule(96,256)
        self.conv_diff2 = DiffModule(192,256)
        self.conv_diff3 = DiffModule(384,256)
        self.conv_diff4 = DiffModule(768,256)

        self.segmentation_head = FPNHead([256,256,256,256],256,2)
    
    def forward(self, input):

        prev_img, after_img = input
        img = torch.cat([prev_img,after_img],dim=0)
        img1,img2,img3,img4 = self.backbone(img)
        prev_img1, after_img1 = torch.chunk(img1,2)
        prev_img2, after_img2 = torch.chunk(img2,2)
        prev_img3, after_img3 = torch.chunk(img3,2)
        prev_img4, after_img4 = torch.chunk(img4,2)

        diff1 = self.conv_diff1(prev_img1,after_img1)
        diff2 = self.conv_diff2(prev_img2,after_img2)
        diff3 = self.conv_diff3(prev_img3,after_img3)
        diff4 = self.conv_diff4(prev_img4,after_img4)

        output = self.segmentation_head([diff4,diff3,diff2,diff1])

        output = F.interpolate(output, size=prev_img.size()[2:], mode='bilinear', align_corners=True)

        return output