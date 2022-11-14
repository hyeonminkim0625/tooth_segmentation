import torch
import torch.nn as nn
from pathlib import Path
from engine import train_one_epoch, evaluate
import wandb
import os
import numpy as np
import warnings
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
import builtins
import math
import datetime
from model import Model
from dataset import LEVIR_256
from omegaconf import OmegaConf
import omegaconf
import torchvision
from timm.optim import create_optimizer_v2
from timm.utils import ModelEmaV2
from torch.cuda.amp import autocast, GradScaler
import shutil
from itertools import product
import copy
import random
from omegaconf.dictconfig import DictConfig
from loss import *

def cfg2dict(cfg: DictConfig):
    """
    Recursively convert OmegaConf to vanilla dict
    :param cfg:
    :return:
    """
    cfg_dict = {}
    for k, v in cfg.items():
        if type(v) == DictConfig:
            cfg_dict[k] = cfg2dict(v)
        else:
            cfg_dict[k] = v
    return cfg_dict

def adjust_learning_rate(optimizer, epoch, conf):
    """Decays the learning rate with half-cycle cosine after warmup"""
    if epoch < conf.SOLVER.WARMUP_EPOCH:
        lr_rate =  (epoch+1) / conf.SOLVER.WARMUP_EPOCH
    else:
        lr_rate =  0.5 * (1. + math.cos(math.pi * (epoch - conf.SOLVER.WARMUP_EPOCH) / (conf.SOLVER.EPOCH - conf.SOLVER.WARMUP_EPOCH)))
    lrs = [conf.SOLVER.BASE_LR, conf.SOLVER.BACKBONE_LR]
    new_lrs = []
    for i,param_group in enumerate(optimizer.param_groups):
        param_group['lr'] = lr_rate*lrs[i]
        new_lrs.append(lr_rate*lrs[i])
    return new_lrs

def main_worker(gpu, conf):

    conf.DISTRIBUTE.GPU = gpu
    # suppress printing if not master
    default_conf = copy.deepcopy(conf)

    sweeps = ['eval']
    if conf.ETC.SWEEP:
        sweep_file = OmegaConf.load('sweep.yaml')
        new_dict = {}
        def recur(dic,parent):
            for k,v in dic.items():
                #print(type(dic))
                if isinstance(dic,omegaconf.dictconfig.DictConfig) and isinstance(dic[k],omegaconf.dictconfig.DictConfig):
                    recur(dic[k],parent+k+'.')
                else:
                    new_dict[parent+k] = dic[k]

        recur(sweep_file,'')
        sweeps = list(product(*[new_dict[k] for k in new_dict.keys()]))
        print(sweeps)

    for sweep in sweeps:
        conf = copy.deepcopy(default_conf)
        if conf.ETC.SWEEP:
            keys =  list(new_dict.keys())
            for i,k in enumerate(keys):
                OmegaConf.update(conf,list(new_dict.keys())[i],sweep[i])
        
        random_seed = 777

        #torch.manual_seed(random_seed)
        #torch.cuda.manual_seed(random_seed)
        #torch.cuda.manual_seed_all(random_seed) # if use multi-GPU
        #torch.backends.cudnn.deterministic = True
        #torch.backends.cudnn.benchmark = False
        #np.random.seed(random_seed)
        #random.seed(random_seed)

        if conf.ETC.WANDB:
            """
            set experiment name
            """
            wandb_conf_dict = {}
            def recur2(dic,parent):
                for k,v in dic.items():
                    #print(type(dic))
                    if isinstance(dic,omegaconf.dictconfig.DictConfig) and isinstance(dic[k],omegaconf.dictconfig.DictConfig):
                        recur2(dic[k],parent+k+'.')
                    else:
                        wandb_conf_dict[parent+k] = dic[k]
            recur2(conf,'')
            
            experiment_name = 'exp_'+str(datetime.datetime.now()).split('.')[0]
            os.mkdir('./exp/'+experiment_name)
            os.mkdir('./exp/'+experiment_name+'/imgs')
            os.mkdir('./exp/'+experiment_name+'/preds')
            wandb.init(project='change_detection',name=experiment_name)
            wandb.config.update(wandb_conf_dict)
            with open('./exp/'+experiment_name+'/config.yaml', "w") as f:
                OmegaConf.save(conf, f)
        model = Model(conf.MODEL)
        
        if conf.ETC.EVAL:
            experiment_name = 'exp_2022-11-09 12:43:59'
            checkpoint = torch.load('exp/'+experiment_name+'/weight_58.pth')['model_state_dict']
            model.load_state_dict(checkpoint)


        model.cuda()

        scaler = None
        if conf.ETC.AMP:
            scaler = GradScaler()
            if conf.DISTRIBUTE.RANK == 0:
                print('Using native Torch AMP. Training in mixed precision.')

        model_ema = None
        if conf.MODEL.EMA > 0:
            # 1 : no update
            # 0 : update everytime
            model_ema = ModelEmaV2(model,decay=conf.MODEL.EMA)
        
        if conf.LOSS.NAME == 'crossentropy':
            criterion = torch.nn.CrossEntropyLoss()
        elif conf.LOSS.NAME == 'DiceBCELoss':
            criterion = DiceBCELoss()
        elif conf.LOSS.NAME == 'DiceFocalLoss':
            criterion = DiceFocalLoss()
        
        base_optimizer = None
        optimizer = None

        conf.SOLVER.BACKBONE_LR = conf.SOLVER.BASE_LR*0.25

        param_dicts = [
        {"params": [p for n, p in model.named_parameters() if "backbone" not in n and p.requires_grad]},
        {
            "params": [p for n, p in model.named_parameters() if "backbone" in n and p.requires_grad],
            "lr": conf.SOLVER.BACKBONE_LR,
        },]
        
        if conf.SOLVER.OPTIMIZER == 'ADAMW':
            optimizer = create_optimizer_v2(param_dicts,'adamw', lr=conf.SOLVER.BASE_LR, weight_decay=conf.SOLVER.WEIGHT_DECAY)#, layer_decay = 0.8)
        
        device = torch.device("cuda:"+str(conf.DISTRIBUTE.GPU))
        
        train_dataset = LEVIR_256(conf.DATASETS,'train')
        val_dataset = LEVIR_256(conf.DATASETS,'val')
        

        train_sampler = None
        val_sampler = None

        train_dataloader = torch.utils.data.DataLoader(
            train_dataset, batch_size=conf.SOLVER.IMS_PER_BATCH, shuffle=(train_sampler is None),
            num_workers=conf.DATALOADER.WORKERS, pin_memory=True, sampler=train_sampler, drop_last=False)

        val_dataloader = torch.utils.data.DataLoader(
            val_dataset, batch_size=conf.SOLVER.IMS_PER_BATCH, shuffle=False,
            num_workers=conf.DATALOADER.WORKERS, pin_memory=True, sampler=val_sampler, drop_last=False)
        
        
        best_score  = [[0,'']]
        if conf.ETC.EVAL:
            #experiment_name = 'exp_2022-11-09 09:13:46'
            test_dataset = LEVIR_256(conf.DATASETS,'test')
            test_dataloader = torch.utils.data.DataLoader(
                test_dataset, batch_size=conf.SOLVER.IMS_PER_BATCH, shuffle=False,
                num_workers=conf.DATALOADER.WORKERS, pin_memory=True, drop_last=False)

            temp_wandb_dict = evaluate(model, criterion, test_dataloader ,device,conf,scaler,True,experiment_name)
            return True
        else:
            for epoch in range(conf.SOLVER.EPOCH):
                if conf.DISTRIBUTE.DISTRIBUTE:
                    train_sampler.set_epoch(epoch)
                
                lrs = adjust_learning_rate(optimizer, epoch, conf)
                wandb_dict_train = train_one_epoch(model,criterion,train_dataloader,optimizer,device,conf,model_ema,scaler)
                print('eval')
                wandb_dict_val = evaluate(model, criterion, val_dataloader ,device,conf,scaler) if conf.MODEL.EMA==0 else evaluate(model_ema.module, criterion, val_dataloader ,device,conf,scaler)
                wandb_dict_train.update(wandb_dict_val)
                wandb_dict_train['lr'] = lrs[0]
                if len(lrs)>1:
                    wandb_dict_train['backbone_lr'] = lrs[1]
                if conf.DISTRIBUTE.GPU==0:
                    if (wandb_dict_train['miou']>best_score[-1][0]):
                        
                        weight_dict = None
                        if conf.MODEL.EMA==0:
                            weight_dict = {
                                'epoch': epoch,
                                'model_state_dict': model.module.state_dict(),}
                        else:
                            weight_dict = {
                                'epoch': epoch,
                                'model_state_dict': model_ema.module.state_dict(),}
                    
                        torch.save(weight_dict,'./exp/'+experiment_name+'/weight_'+str(epoch)+'.pth')
                        best_score.append([wandb_dict_train['miou'],'./exp/'+experiment_name+'/weight_'+str(epoch)+'.pth'])
                        if len(best_score)>20:
                            if os.path.exists(best_score[0][1]):
                                os.remove(best_score[0][1])
                            best_score = best_score[1:]
                    wandb_dict_train['best_miou'] = best_score[-1][0]
                    if conf.ETC.WANDB:
                        wandb.log(wandb_dict_train)
            """
            after learning, inference testset
            """
            if conf.DISTRIBUTE.GPU==0:
                model = Model(conf.MODEL)
                checkpoint = torch.load(best_score[-1][1])['model_state_dict']
                model.load_state_dict(checkpoint)                
                model.cuda()
                test_dataset = LEVIR_256(conf.DATASETS,'test')
                test_dataloader = torch.utils.data.DataLoader(
                    test_dataset, batch_size=conf.SOLVER.IMS_PER_BATCH, shuffle=False,
                    num_workers=conf.DATALOADER.WORKERS, pin_memory=True, drop_last=False)

                temp_wandb_dict = evaluate(model, criterion, test_dataloader ,device,conf,scaler,True,experiment_name)

            
            if conf.ETC.WANDB:
                print(best_score)
                wandb.finish()

    return True

def main():
    conf = OmegaConf.load('config.yaml')
    main_worker(conf.DISTRIBUTE.GPU, conf)

if __name__ == '__main__':
    main()