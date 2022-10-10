import nsml
import torch
import os
from landmark_detection_dataset import LandmarkDetectionDataset
import argparse
import torch.optim as optim
import time
import numpy as np
from tqdm import tqdm
import torchvision
from split_dataset import split_dataset

from data_loader import feed_infer
from evaluation import evaluation_metrics

from model import get_model

from nsml import DATASET_PATH, IS_ON_NSML

if not IS_ON_NSML:
    DATASET_PATH = "dataset/final_dataset"

def collate_fn(batch):
    return tuple(zip(*batch))

def local_eval(model, num_classes, test_loader, test_label_file=None):
    prediction_file = 'pred_train.txt'
    feed_infer(prediction_file, lambda root_path: _infer(model, num_classes, root_path, loader=test_loader))
    if test_label_file == None:
        test_label_file = os.path.join(DATASET_PATH, 'train', "train_label")

    metric_result = evaluation_metrics(
        prediction_file,
        test_label_file
    )
    
    return metric_result

def _infer(model, num_classes, root_path, loader=None, batch_size=4, cuda=torch.cuda.is_available()):
    model.eval()

    if loader == None:
        # run on gpu or get error to avoid timeout while submit
        cuda = True
        dataset = LandmarkDetectionDataset(
            os.path.join(root_path, "test_data"), 
            None, 
            num_classes=num_classes
        )
        loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, num_workers=2, shuffle=False, collate_fn=collate_fn)

    res_preds = None
    res_ids = None
    # for iter, (images, _, image_id) in enumerate(tqdm(loader)):
    for iter, (images, _, image_id) in enumerate(loader):
        if cuda:
            images = list(image.cuda() for image in images)
        
        preds = model(images)

        for i in range(len(preds)):
            pred = preds[i]
            boxes = pred['boxes'].detach().cpu().numpy()
            labels = pred['labels'].detach().cpu().numpy()
            scores = pred['scores'].detach().cpu().numpy()

            preds[i] = {'boxes': boxes, 'labels': labels, 'scores': scores}
        
        if iter == 0:
            res_preds = preds
            res_ids = image_id
        else:
            res_preds = np.concatenate((res_preds, preds), axis=0)
            res_ids = np.concatenate((res_ids, image_id), axis=0)

    return {"ids": res_ids, "preds": res_preds}

def bind_nsml(model, optimizer, scheduler, cuda, num_classes):
    def load(model_dir, **kwargs):
        device = torch.device('cpu')
        if cuda:
            device = torch.device('cuda')
        state = torch.load(os.path.join(model_dir, 'model.pth'), map_location=device)
        model.load_state_dict(state['model'])
        if 'optimizer' in state and not optimizer == None:
            optimizer.load_state_dict(state['optimizer'])
        if 'scheduler' in state and not scheduler == None:
            scheduler.load_state_dict(state['scheduler'])
        print('Model loaded')

    def save(model_dir, **kwargs):
        os.makedirs(model_dir, exist_ok=True)
        state = {
            'model': model.state_dict()
        }
        if not optimizer == None:
            state['optimizer'] = optimizer.state_dict()
        if not scheduler == None:
            state['scheduler'] = scheduler.state_dict()
        torch.save(state, os.path.join(model_dir, 'model.pth'))
        print('Saved')

    def infer(input, top_k=1):
        return _infer(model, num_classes, input, None)

    nsml.bind(save=save, load=load, infer=infer) # 'nsml.bind' function must be called at the end.

def train_epoch(model, optimizer, loader, device, epoch, verbose_period=50):
    model.train()

    scheduler = None
    if epoch == 0:
        warmup_factor = 0.001
        warmup_iters = min(1000, len(loader) - 1)

        def lr_f(x):
            if x >= warmup_iters:
                return 1
            alpha = float(x) / warmup_iters
            return warmup_factor * (1 - alpha) + alpha

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_f)

    tic = time.time()
    len_iter = len(loader)
    epoch_losses = 0
    last_print_loss = 0
    #for iter, (images, targets, sample_id) in (enumerate(tqdm(loader))):
    for iter, (images, targets, sample_id) in (enumerate(loader)):
        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        loss_dict = model(images, targets)

        losses = sum(loss for loss in loss_dict.values())
        epoch_losses += losses

        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

        if scheduler is not None:
            scheduler.step()

        if (iter + 1) % verbose_period == 0:
            toc = time.time()
            elapsed = toc - tic
            tic = toc
            local_loss = epoch_losses - last_print_loss
            last_print_loss = float(epoch_losses)
            print("[Epoch: {}: {} / {} iters] Elapsed time: {}, Local loss: {}".format(epoch + 1, iter + 1, len_iter, elapsed, local_loss))

    return epoch_losses

if __name__ == '__main__':
    args = argparse.ArgumentParser()

    args.add_argument("--num_classes", type=int, default=144)
    args.add_argument("--num_epochs", type=int, default=10)
    args.add_argument("--batch_size", type=int, default=4)
    args.add_argument("--lr", type=float, default=0.005)
    args.add_argument("--weight_decay", type=float, default=0.0005)
    args.add_argument("--step_size", type=float, default=20)
    args.add_argument("--gamma", type=float, default=0.1)
    args.add_argument("--print_iter", type=float, default=200)
    args.add_argument("--num_workers", type=int, default=2)
    args.add_argument("--momentum", type=float, default=0.9)

    args.add_argument("--mode", type=str, default="train")
    args.add_argument("--iteration", type=str, default='0')
    args.add_argument("--pause", type=int, default=0)

    config = args.parse_args()

    os.system("nvidia-smi")
    print("torch version", torch.__version__)
    print("torchvision version", torchvision.__version__)
    cuda = torch.cuda.is_available()
    print('is cuda avaiable', cuda)
    print("device count", torch.cuda.device_count())
    print("device current", torch.cuda.current_device())

    model = get_model(config.num_classes, True)
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=config.lr,momentum=config.momentum, weight_decay=config.weight_decay)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=config.step_size, gamma=config.gamma)

    device = torch.device('cpu')
    if cuda:
        device = torch.device('cuda')

    model = model.to(device)

    if IS_ON_NSML:
        bind_nsml(model, optimizer, scheduler, cuda, config.num_classes)

        if config.pause:
            nsml.paused(scope=locals())

    if config.mode == 'train':
        dataset = LandmarkDetectionDataset(
            os.path.join(DATASET_PATH, "train", "train_data"), 
            os.path.join(DATASET_PATH, "train","train_label"),
            num_classes=config.num_classes
        )

        train_dataset, test_dataset = split_dataset(dataset, ratio=0.7, shuffle=True)

        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=config.batch_size, num_workers=config.num_workers, shuffle=False, collate_fn=collate_fn)
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=config.batch_size, num_workers=config.num_workers, shuffle=False, collate_fn=collate_fn)

        len_iter = len(train_loader)

        for epoch in range(config.num_epochs):
            tic = time.time()

            epoch_losses = train_epoch(model, optimizer, train_loader, device, epoch, config.print_iter)

            scheduler.step()

            if IS_ON_NSML:
                nsml.save(str(epoch + 1))

            metric_result = local_eval(model, config.num_classes, test_loader)

            toc = time.time()
            elapsed = toc - tic
            print('[Epoch {}] elapsed: {} s, loss: {:.4f}, mAP: {:.4f}'.format(epoch + 1, elapsed, epoch_losses, metric_result))


