import torch
import torch.nn as nn
import torchvision
from collections import OrderedDict
import torch.nn.functional as F
import torchvision.transforms.functional as TF
import timm
from timm.models.layers import Mlp
from einops import rearrange, reduce, repeat
from head import FPNHead, FaPNHead, LawinHead
from swin import SwinTransformer
from torch.cuda.amp import autocast, GradScaler

def TestTimeAugmentation(model,lossfun,prev_img,after_img,targets,scaler):
    with torch.no_grad():

        losses_list = []
        outputs_list = []
        tta_list = [0.5,0.75,1.0,1.25,1.5,1.75,2.0]
        for scale in tta_list:
            loss = None
            outputs = None
            temp_prev_img = F.interpolate(prev_img,scale_factor=scale, mode='bilinear', align_corners=True)
            temp_after_img = F.interpolate(after_img,scale_factor=scale, mode='bilinear', align_corners=True)

            if scaler is not None:
                with autocast():
                    outputs = model((temp_prev_img,temp_after_img))
                    outputs = F.interpolate(outputs, size=prev_img.size()[2:], mode='bilinear', align_corners=True)
                    loss = lossfun(outputs,targets).mean()
            else:
                outputs = model((temp_prev_img,temp_after_img))   
                outputs = F.interpolate(outputs, size=prev_img.size()[2:], mode='bilinear', align_corners=True)
                loss = lossfun(outputs,targets).mean()

            losses_list.append(loss)
            outputs_list.append(outputs)

        outputs_list = torch.stack(outputs_list,dim=0)
        outputs = torch.mean(outputs_list,dim=0)
        loss = sum(losses_list)/len(losses_list)

    return outputs,loss
    

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
        if 'convnext' in conf.META_ARCHITECTURE:
            self.backbone = timm.create_model(conf.META_ARCHITECTURE, pretrained=True, out_indices=(0,1,2,3), drop_path_rate=0.2,features_only=True)
        elif 'swin' in conf.META_ARCHITECTURE:
            swin_config = None
            if conf.META_ARCHITECTURE == 'swin_tiny':
                checkpoint_file = 'https://download.openmmlab.com/mmsegmentation/v0.5/pretrain/swin/swin_tiny_patch4_window7_224_20220317-1cdeb081.pth'  # noqa
                swin_config = dict(
                init_cfg=dict(type='Pretrained', checkpoint=checkpoint_file),
                embed_dims=96,
                depths=[2, 2, 6, 2],
                num_heads=[3, 6, 12, 24],
                window_size=7,
                use_abs_pos_embed=False,
                drop_path_rate=0.3,
                patch_norm=True)
            elif conf.META_ARCHITECTURE == 'swin_small':
                checkpoint_file = 'https://download.openmmlab.com/mmsegmentation/v0.5/pretrain/swin/swin_small_patch4_window7_224_20220317-7ba6d6dd.pth'  # noqa
                swin_config = dict(
                init_cfg=dict(type='Pretrained', checkpoint=checkpoint_file),
                depths=[2, 2, 18, 2],
                embed_dims=96,
                num_heads=[3, 6, 12, 24],
                window_size=7,
                use_abs_pos_embed=False,
                drop_path_rate=0.3,
                patch_norm=True)
            elif conf.META_ARCHITECTURE == 'swin_base':
                checkpoint_file = 'https://download.openmmlab.com/mmsegmentation/v0.5/pretrain/swin/swin_base_patch4_window7_224_20220317-e9b98025.pth'
                swin_config = dict(
                init_cfg=dict(type='Pretrained', checkpoint=checkpoint_file),
                embed_dims=128,
                depths=[2, 2, 18, 2],
                num_heads=[4, 8, 16, 32],
                window_size=7,
                use_abs_pos_embed=False,
                drop_path_rate=0.3,
                patch_norm=True)
            elif conf.META_ARCHITECTURE == 'swin_base_12':
                checkpoint_file = 'https://download.openmmlab.com/mmsegmentation/v0.5/pretrain/swin/swin_base_patch4_window12_384_20220317-55b0104a.pth'
                swin_config = dict(
                init_cfg=dict(type='Pretrained', checkpoint=checkpoint_file),
                pretrain_img_size=384,
                embed_dims=128,
                depths=[2, 2, 18, 2],
                num_heads=[4, 8, 16, 32],
                window_size=12,
                use_abs_pos_embed=False,
                drop_path_rate=0.3,
                patch_norm=True)
            
            self.backbone = SwinTransformer(**swin_config)

        """
        calculate channels
        """        
        temp = torch.randn(2,3,256,256)
        res = self.backbone(temp)
        channels_list = []
        for r in res:
            channels_list.append(r.shape[1])

        self.conv_diff1 = DiffModule(channels_list[0],256)
        self.conv_diff2 = DiffModule(channels_list[1],256)
        self.conv_diff3 = DiffModule(channels_list[2],256)
        self.conv_diff4 = DiffModule(channels_list[3],256)

        if conf.SEGMENTATION_HEAD == 'lawin':
            self.segmentation_head = LawinHead([256,256,256,256],256,2)
        elif conf.SEGMENTATION_HEAD == 'fpn':
            self.segmentation_head = FPNHead([256,256,256,256],256,2)
        elif conf.SEGMENTATION_HEAD == 'fapn':
            self.segmentation_head = FaPNHead([256,256,256,256],256,2)
    
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

        output = self.segmentation_head([diff1,diff2,diff3,diff4])

        output = F.interpolate(output, size=prev_img.size()[2:], mode='bilinear', align_corners=True)

        return output