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

def main_worker(gpu, ngpus_per_node, conf):

    conf.DISTRIBUTE.GPU = gpu
    # suppress printing if not master

    if conf.DISTRIBUTE.MULTIPROCESSING_DISTRIBUTE and conf.DISTRIBUTE.GPU != 0:
        def print_pass(*args):
            pass
        builtins.print = print_pass

    if conf.DISTRIBUTE.GPU is not None:
        print("Use GPU: {} for training".format(conf.DISTRIBUTE.GPU))

    if conf.DISTRIBUTE.DISTRIBUTE:
        if conf.DISTRIBUTE.DIST_URL == "env://" and conf.DISTRIBUTE.RANK == -1:
            conf.DISTRIBUTE.RANK = int(os.environ["RANK"])
        if conf.DISTRIBUTE.MULTIPROCESSING_DISTRIBUTE:
            # For multiprocessing distributed training, rank needs to be the
            # global rank among all the processes
            conf.DISTRIBUTE.RANK = conf.DISTRIBUTE.RANK * ngpus_per_node + gpu
        dist.init_process_group(backend=conf.DISTRIBUTE.DIST_BACKEND, init_method=conf.DISTRIBUTE.DIST_URL,
                                world_size=conf.DISTRIBUTE.WORLD_SIZE, rank=conf.DISTRIBUTE.RANK)
    # create model

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

        torch.manual_seed(random_seed)
        torch.cuda.manual_seed(random_seed)
        torch.cuda.manual_seed_all(random_seed) # if use multi-GPU
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        np.random.seed(random_seed)
        random.seed(random_seed)

        if conf.DISTRIBUTE.GPU==0 and conf.ETC.WANDB:
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
            wandb.init(project='change_detection',name=experiment_name)
            wandb.config.update(wandb_conf_dict)
            with open('./exp/'+experiment_name+'/config.yaml', "w") as f:
                OmegaConf.save(conf, f)
        model = Model(conf.MODEL)
        
        if conf.ETC.EVAL:
            checkpoint = torch.load('weight_27.pth')['model_state_dict']
            model.load_state_dict(checkpoint)

        if conf.DISTRIBUTE.DISTRIBUTE:
            # For multiprocessing distributed, DistributedDataParallel constructor
            # should always set the single device scope, otherwise,
            # DistributedDataParallel will use all available devices.
            if conf.DISTRIBUTE.GPU is not None:
                torch.cuda.set_device(conf.DISTRIBUTE.GPU)
                model.cuda(conf.DISTRIBUTE.GPU)
                model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
                # When using a single GPU per process and per
                # DistributedDataParallel, we need to divide the batch size
                # ourselves based on the total number of GPUs we have
                conf.SOLVER.IMS_PER_BATCH = int(conf.SOLVER.IMS_PER_BATCH / ngpus_per_node)
                conf.DATALOADER.WORKERS = int((conf.DATALOADER.WORKERS + ngpus_per_node - 1) / ngpus_per_node)
                model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[conf.DISTRIBUTE.GPU],find_unused_parameters=True)#,find_unused_parameters=True)
            else:
                model.cuda()
                # DistributedDataParallel will divide and allocate batch_size to all
                # available GPUs if device_ids are not set
                model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
                model = torch.nn.parallel.DistributedDataParallel(model,find_unused_parameters=True)
        else:
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
            
        #torch.nn.CrossEntropyLoss(label_smoothing=conf.DATASETS.LABEL_SMOOTHING)
        #torchvision.ops.sigmoid_focal_loss
        #AsymmetricLossMultiLabel(gamma_neg=4, gamma_pos=0, clip=0.05, disable_torch_grad_focal_loss=True).cuda(conf.DISTRIBUTE.GPU)
        
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
        

        if conf.DISTRIBUTE.DISTRIBUTE:
            train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
            val_sampler = torch.utils.data.distributed.DistributedSampler(val_dataset)
        else:
            train_sampler = None
            val_sampler = None

        train_dataloader = torch.utils.data.DataLoader(
            train_dataset, batch_size=conf.SOLVER.IMS_PER_BATCH, shuffle=(train_sampler is None),
            num_workers=conf.DATALOADER.WORKERS, pin_memory=True, sampler=train_sampler, drop_last=False)

        val_dataloader = torch.utils.data.DataLoader(
            val_dataset, batch_size=conf.SOLVER.IMS_PER_BATCH, shuffle=(val_sampler is None),
            num_workers=conf.DATALOADER.WORKERS, pin_memory=True, sampler=val_sampler, drop_last=False)
        
        

        best_score  = [[0,'']]
        if conf.ETC.EVAL:
            experiment_name = 'exp_2022-10-05 17:31:24'
            test_dataset = LEVIR_256(conf.DATASETS,'test')
            test_dataloader = torch.utils.data.DataLoader(
                test_dataset, batch_size=conf.SOLVER.IMS_PER_BATCH, shuffle=False,
                num_workers=conf.DATALOADER.WORKERS, pin_memory=True, drop_last=False)

            temp_wandb_dict = evaluate(model, criterion, test_dataloader ,device,conf,scaler,True,experiment_name)
            wandb_dict = {}
            wandb_dict['test_class1_iou'] = temp_wandb_dict['class1_iou']
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
                    if (wandb_dict_train['class1_iou']>best_score[-1][0]):
                        
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
                        best_score.append([wandb_dict_train['class1_iou'],'./exp/'+experiment_name+'/weight_'+str(epoch)+'.pth'])
                        if len(best_score)>3:
                            if os.path.exists(best_score[0][1]):
                                os.remove(best_score[0][1])
                            best_score = best_score[1:]
                    wandb_dict_train['best_class1_iou'] = best_score[-1][0]
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
                wandb_dict = {}
                wandb_dict['test_class1_iou'] = temp_wandb_dict['class1_iou']
                if conf.ETC.WANDB:
                    wandb.log(wandb_dict)
            
            if conf.ETC.WANDB:
                print(best_score)
                wandb.finish()

    return True

def main():
    conf = OmegaConf.load('config.yaml')

    if conf.DISTRIBUTE.GPU is not None:
        warnings.warn('You have chosen a specific GPU. This will completely '
                      'disable data parallelism.')

    if conf.DISTRIBUTE.DIST_URL == "env://" and conf.DISTRIBUTE.WORLD_SIZE == -1:
        conf.DISTRIBUTE.WORLD_SIZE= int(os.environ["WORLD_SIZE"])

    conf.DISTRIBUTE.DISTRIBUTE = conf.DISTRIBUTE.WORLD_SIZE > 1 or conf.DISTRIBUTE.MULTIPROCESSING_DISTRIBUTE
    ngpus_per_node = torch.cuda.device_count()
    if conf.DISTRIBUTE.MULTIPROCESSING_DISTRIBUTE:
        # Since we have ngpus_per_node processes per node, the total world_size
        # needs to be adjusted accordingly
        conf.DISTRIBUTE.WORLD_SIZE = ngpus_per_node * conf.DISTRIBUTE.WORLD_SIZE
        # Use torch.multiprocessing.spawn to launch distributed processes: the
        # main_worker process function
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, conf))
    else:
        # Simply call main_worker function
        main_worker(conf.DISTRIBUTE.GPU, ngpus_per_node, conf)

if __name__ == '__main__':
    main()