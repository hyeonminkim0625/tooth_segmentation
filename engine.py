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
import shutil
import matplotlib.pyplot as plt
from model import TestTimeAugmentation

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
    paths_list =[]
    loss_list = []

    for i, (samples) in enumerate(tqdm(data_loader)):

        prev_img = samples['prev_img'].to(device)
        after_img = samples['after_img'].to(device)
        targets = samples['label_img'].to(device)
        paths = samples['path']

        if conf.ETC.TTA and test:
            outputs, loss = TestTimeAugmentation(model,lossfun,prev_img,after_img,targets,scaler)
        
        else:
            if scaler is not None:
                with autocast():
                    outputs = model((prev_img,after_img))
                    loss = lossfun(outputs,targets).mean()
            else:
                outputs = model((prev_img,after_img))   
                loss = lossfun(outputs,targets).mean()
        
        if conf.DISTRIBUTE.DISTRIBUTE and (not test):
            loss = reduce_tensor(loss)

        total_loss += float(loss)
        if conf.DISTRIBUTE.DISTRIBUTE and (not test):
            torch.cuda.synchronize()

        for j in range(prev_img.shape[0]):
            output_mask = F.one_hot(torch.argmax(outputs[j],dim=0),num_classes=2).permute(2,0,1)    
            target_mask = F.one_hot(targets[j],num_classes=2).permute(2,0,1)

            class0_iou ,class1_iou = calculate_iou(output_mask,target_mask,2)

            class0_iou_list.append(float(class0_iou))
            class1_iou_list.append(float(class1_iou))
        
        if test:
            for j in range(prev_img.shape[0]):
                output_mask = torch.argmax(outputs[j],dim=0)*255
                output_mask = output_mask.cpu().detach().numpy().astype(np.uint8)
                target_mask = (targets[j]*255).cpu().detach().numpy().astype(np.uint8)
                cv2.imwrite('./temp/'+paths[j].replace('.png','')+'_pred.png',output_mask)
                cv2.imwrite('./temp/'+paths[j].replace('.png','')+'_target.png',target_mask)
                paths_list.append(paths[j])

    total_loss = total_loss/(batch_num)
    class0_iou = sum(class0_iou_list)/len(class0_iou_list)
    class1_iou = sum(class1_iou_list)/len(class1_iou_list)

    if conf.DISTRIBUTE.DISTRIBUTE and (not test):
        class0_iou = (torch.ones(1)*class0_iou).cuda()
        class1_iou = (torch.ones(1)*class1_iou).cuda()
        class0_iou = float(reduce_tensor(class0_iou))
        class1_iou = float(reduce_tensor(class1_iou))
    
    if test:
        class1_iou_list = np.array(class1_iou_list)
        """
        filtering low iou img
        """
        low_iou_index_list = np.argsort(class1_iou_list)[:30]
        paths_list = np.array(paths_list)
        for p,iou in zip(paths_list[low_iou_index_list],class1_iou_list[low_iou_index_list]):
            shutil.copy('./temp/'+p.replace('.png','')+'_pred.png', "./exp/"+exp_name+'/imgs/'+str(iou)[:4]+p.replace('.png','')+'_pred.png')
            shutil.copy('./temp/'+p.replace('.png','')+'_target.png', "./exp/"+exp_name+'/imgs/'+str(iou)[:4]+p.replace('.png','')+'_target.png')
            shutil.copy('/data/LEVIR-CD-256/'+'A'+'/'+p, "./exp/"+exp_name+'/imgs/'+str(iou)[:4]+p.replace('.png','')+'_A.png')
            shutil.copy('/data/LEVIR-CD-256/'+'B'+'/'+p, "./exp/"+exp_name+'/imgs/'+str(iou)[:4]+p.replace('.png','')+'_B.png')
        
        plt.hist(class1_iou_list)
        plt.savefig("./exp/"+exp_name+'/iou_hist.png')

    wandb_dict = {}

    wandb_dict['class0_iou'] = class0_iou
    wandb_dict['class1_iou'] = class1_iou
    wandb_dict['miou'] = (class1_iou+class0_iou)/2

    wandb_dict["validation average losses"] = total_loss
    print(wandb_dict)

    return wandb_dict

