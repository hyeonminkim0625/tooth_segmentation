from tqdm import tqdm
import torchvision
import wandb
import numpy as np
import torch
from shutil import copyfile
import cv2
import pandas as pd
import torchvision.transforms.functional as TF
import torch.nn.functional as F
import random
from sklearn.metrics import accuracy_score,confusion_matrix,roc_auc_score,roc_curve,mean_absolute_error
import torch.distributed as dist
from torch.cuda.amp import autocast, GradScaler
from utils import calculate_iou

def reduce_tensor(tensor):
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    rt /= dist.get_world_size()
    return rt

def train_one_epoch(model,lossfun,data_loader, optimizer, device, conf,model_ema,scaler):
    model.train()
    lossfun.train()
    total_loss = 0
    batch_num = len(data_loader)

    optimizer.zero_grad()
    for i, (samples) in enumerate(tqdm(data_loader)):

        prev_img = samples['prev_img'].to(device)
        after_img = samples['after_img'].to(device)
        targets = samples['label_img'].to(device)
            
        if scaler is not None:
            with autocast():
                outputs = model((prev_img,after_img))
                loss = lossfun(outputs,targets).mean()

            scaler.scale(loss).backward()
            if (i + 1) % conf.SOLVER.ACCUMULATION == 0:
                if conf.SOLVER.GRADIENT_CLIP>0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), conf.SOLVER.GRADIENT_CLIP)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
        else:
            outputs = model((prev_img,after_img))   
            loss = lossfun(outputs,targets).mean()
            loss.backward()
            if (i + 1) % conf.SOLVER.ACCUMULATION == 0:
                if conf.SOLVER.GRADIENT_CLIP>0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), conf.SOLVER.GRADIENT_CLIP)
                optimizer.step()
                optimizer.zero_grad()

        total_loss += float(loss)
        if model_ema is not None:
            model_ema.update(model)
        torch.cuda.synchronize()

    total_loss = total_loss/(batch_num)
    if conf.DISTRIBUTE.DISTRIBUTE:
        total_loss = (torch.ones(1)*total_loss).cuda()
        total_loss = float(reduce_tensor(total_loss))
    
    print(total_loss)

    return {"train average losses" : total_loss}

@torch.no_grad()
def evaluate(model,lossfun ,data_loader, device,conf,scaler):
    model.eval()
    #lossfun.eval()
    total_loss = 0
    batch_num = len(data_loader)
    infos = None
    outputs = None
    
    class0_iou_list =[]
    class1_iou_list =[]
    output_hostipal_list =[]
    loss_list = []

    for i, (samples) in enumerate(tqdm(data_loader)):

        prev_img = samples['prev_img'].to(device)
        after_img = samples['after_img'].to(device)
        targets = samples['label_img'].to(device)
            
        if scaler is not None:
            with autocast():
                outputs = model((prev_img,after_img))
                loss = lossfun(outputs,targets).mean()
        else:
            outputs = model((prev_img,after_img))   
            loss = lossfun(outputs,targets).mean()
        
        if conf.DISTRIBUTE.DISTRIBUTE:
            loss = reduce_tensor(loss)

        total_loss += float(loss)
        torch.cuda.synchronize()

        for j in range(prev_img.shape[0]):
            output_mask = F.one_hot(torch.argmax(outputs[j],dim=0),num_classes=2).permute(2,0,1)    
            target_mask = F.one_hot(targets[j],num_classes=2).permute(2,0,1)

            class0_iou ,class1_iou = calculate_iou(output_mask,target_mask,2)

            class0_iou_list.append(float(class0_iou))
            class1_iou_list.append(float(class1_iou))



    total_loss = total_loss/(batch_num)
    class0_iou_list = sum(class0_iou_list)/len(class0_iou_list)
    class1_iou_list = sum(class1_iou_list)/len(class1_iou_list)

    class0_iou_list = (torch.ones(1)*class0_iou_list).cuda()
    class1_iou_list = (torch.ones(1)*class1_iou_list).cuda()
    if conf.DISTRIBUTE.DISTRIBUTE:
        class0_iou_list = float(reduce_tensor(class0_iou_list))
        class1_iou_list = float(reduce_tensor(class1_iou_list))

    wandb_dict = {}

    wandb_dict['class0_iou'] = class0_iou_list
    wandb_dict['class1_iou'] = class1_iou_list
    wandb_dict['miou'] = (class1_iou_list+class0_iou_list)/2

    wandb_dict["validation average losses"] = total_loss
    print(wandb_dict)

    return wandb_dict