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
from utils import calculate_iou, cutmix_data, halfmix_data
import shutil
import matplotlib.pyplot as plt
from model import TestTimeAugmentation

def train_one_epoch(model,lossfun,data_loader, optimizer, device, conf,model_ema,scaler):
    model.train()
    lossfun.train()
    total_loss = 0
    batch_num = len(data_loader)

    optimizer.zero_grad()
    for i, (samples) in enumerate(tqdm(data_loader)):

        imgs = samples['imgs'].to(device)
        targets = samples['label_img'].to(device)

        if scaler is not None:
            with autocast():
                outputs = model(imgs)
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
            outputs = model(imgs)   
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
    
    return {"train average losses" : total_loss}

@torch.no_grad()
def evaluate(model,lossfun ,data_loader, device,conf,scaler,test=False,exp_name=''):
    model.eval()
    #lossfun.eval()
    total_loss = 0
    batch_num = len(data_loader)
    infos = None
    outputs = None
    loss = None
    
    class0_iou_list =[]
    class1_iou_list =[]
    class2_iou_list =[]
    class3_iou_list =[]
    miou_list =[]
    paths_list =[]
    loss_list = []

    for i, (samples) in enumerate(tqdm(data_loader)):

        imgs = samples['imgs'].to(device)
        targets = samples['label_img'].to(device)
        
        else:
            if scaler is not None:
                with autocast():
                    outputs = model(imgs)
                    if not test:
                        loss = lossfun(outputs,targets).mean()
            else:
                outputs = model(imgs)
                if not test:
                    loss = lossfun(outputs,targets).mean()
        if not test:
            total_loss += float(loss)
        if conf.DISTRIBUTE.DISTRIBUTE and (not test):
            torch.cuda.synchronize()

        if not test:
            for j in range(prev_img.shape[0]):
                output_mask = F.one_hot(torch.argmax(outputs[j],dim=0),num_classes=4).permute(2,0,1)    
                target_mask = F.one_hot(torch.argmax(targets[j],dim=0),num_classes=4).permute(2,0,1)

                class0_iou ,class1_iou,class2_iou,class3_iou = calculate_iou(output_mask,target_mask,4)

                class0_iou_list.append(float(class0_iou))
                class1_iou_list.append(float(class1_iou))
                class2_iou_list.append(float(class2_iou))
                class3_iou_list.append(float(class3_iou))
                miou_list.append(float(np.nanmean(np.array([float(class0_iou),float(class1_iou),float(class2_iou),float(class3_iou)]))))
        
        if test:
            outputs = outputs.argmax(1).cpu().numpy().astype(np.uint8)
            for j in range(prev_img.shape[0]):
                resized_img = cv2.resize(outputs[j], [int(ws[j]//2), int(hs[j])], interpolation=cv2.INTER_NEAREST)
                new_img = np.zeros((int(hs[j]),int(ws[j])))
                new_img[:,:int(ws[j]//2)] = (resized_img==2)*2
                new_img[:,int(ws[j]//2):] = ((resized_img==1)*1 + (resized_img==3)*3)
                new_img = new_img.astype(np.uint8)
                new_path = paths[j].split('/')[-1]
                print("./exp/"+exp_name+'/preds/'+new_path)
                cv2.imwrite("./exp/"+exp_name+'/preds/'+new_path, new_img)

    if test:
        return 0
    total_loss = total_loss/(batch_num)
    class0_iou = np.nanmean(np.array(class0_iou_list))
    class1_iou = np.nanmean(np.array(class1_iou_list))
    class2_iou = np.nanmean(np.array(class2_iou_list))
    class3_iou = np.nanmean(np.array(class3_iou_list))
    miou = np.nanmean(np.array(miou_list))

    if conf.DISTRIBUTE.DISTRIBUTE and (not test):
        class0_iou = (torch.ones(1)*class0_iou).cuda()
        class1_iou = (torch.ones(1)*class1_iou).cuda()
        class0_iou = float(reduce_tensor(class0_iou))
        class1_iou = float(reduce_tensor(class1_iou))
        class2_iou = (torch.ones(1)*class2_iou).cuda()
        class3_iou = (torch.ones(1)*class3_iou).cuda()
        class2_iou = float(reduce_tensor(class2_iou))
        class3_iou = float(reduce_tensor(class3_iou))

        miou = (torch.ones(1)*miou).cuda()
        miou = float(reduce_tensor(miou))
    
    wandb_dict = {}

    wandb_dict['class0_iou'] = class0_iou
    wandb_dict['class1_iou'] = class1_iou
    wandb_dict['class2_iou'] = class2_iou
    wandb_dict['class3_iou'] = class3_iou
    wandb_dict['miou'] = miou

    wandb_dict["validation average losses"] = total_loss
    print(wandb_dict)

    return wandb_dict