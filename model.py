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

class Model(nn.Module):
    def __init__(self,conf):
        super(Model, self).__init__()
        self.conf = conf
        if 'convnext' in conf.META_ARCHITECTURE:
            self.backbone = timm.create_model(conf.META_ARCHITECTURE, pretrained=True, out_indices=(0,1,2,3), drop_path_rate=0.2,features_only=True)
        else:
            self.backbone = timm.create_model(conf.META_ARCHITECTURE, pretrained=True, out_indices=(0,1,2,3), drop_path_rate=0.2,features_only=True)

        """
        calculate channels
        """        
        temp = torch.randn(2,3,256,256)
        res = self.backbone(temp)
        channels_list = []
        for r in res:
            channels_list.append(r.shape[1])

        if conf.SEGMENTATION_HEAD == 'lawin':
            self.segmentation_head = LawinHead([channels_list[0],channels_list[1],channels_list[2],channels_list[3]],256,4)
        elif conf.SEGMENTATION_HEAD == 'fpn':
            self.segmentation_head = FPNHead([channels_list[0],channels_list[1],channels_list[2],channels_list[3]],256,4)
        elif conf.SEGMENTATION_HEAD == 'fapn':
            self.segmentation_head = FaPNHead([channels_list[0],channels_list[1],channels_list[2],channels_list[3]],256,4)
    
    def forward(self, input):

        img1,img2,img3,img4 = self.backbone(input)
        output = self.segmentation_head([img1,img2,img3,img4])

        output = F.interpolate(output, size=input.size()[2:], mode='bilinear', align_corners=True)

        return output