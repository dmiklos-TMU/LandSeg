import torch
import numpy as np
from IOUEval import SegmentationMetric
import logging
import logging.config
from tqdm import tqdm
import os
import torch.nn as nn
from const import *
# We would like to thank the authors of TwinLiteNet for their repository containing the different classes used in this work.
# Link to adopted work: https://github.com/chequanghuy/TwinLiteNet


LOGGING_NAME="custom"
def set_logging(name=LOGGING_NAME, verbose=True):
    # sets up logging for the given name
    rank = int(os.getenv('RANK', -1))  # rank in world for Multi-GPU trainings
    level = logging.INFO if verbose and rank in {-1, 0} else logging.ERROR
    logging.config.dictConfig({
        'version': 1,
        'disable_existing_loggers': False,
        'formatters': {
            name: {
                'format': '%(message)s'}},
        'handlers': {
            name: {
                'class': 'logging.StreamHandler',
                'formatter': name,
                'level': level,}},
        'loggers': {
            name: {
                'level': level,
                'handlers': [name],
                'propagate': False,}}})
set_logging(LOGGING_NAME)  # run before defining LOGGER
LOGGER = logging.getLogger(LOGGING_NAME)  # define globally (used in train.py, val.py, detect.py, etc.)

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count if self.count != 0 else 0

def poly_lr_scheduler(args, optimizer, epoch, power=2):
    lr = round(args.lr * (1 - epoch / args.max_epochs) ** power, 8)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    return lr



def train(args, train_loader, model, criterion, optimizer, epoch):
    model.train()
    total_batches = len(train_loader)
    pbar = enumerate(train_loader)
    LOGGER.info(('\n' + '%13s' * 4) % ('Epoch','TverskyLoss','FocalLoss' ,'TotalLoss'))
    pbar = tqdm(pbar, total=total_batches, bar_format='{l_bar}{bar:10}{r_bar}')
    for i, (_,inputs, target) in pbar:
        RGB_input, Depth_input = inputs
        
        if args.onGPU == True:
            RGB_input = RGB_input.cuda().float() / 255.0  
            Depth_input = Depth_input.cuda().float() / 65535.0  
            Depth_input=Depth_input.unsqueeze(0)
            Depth_input = Depth_input.permute(1, 0, 2, 3)
        
        output = model(RGB_input,Depth_input)

        optimizer.zero_grad()
        focal_loss,tversky_loss,loss = criterion(output,target)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        pbar.set_description(('%13s' * 1 + '%13.4g' * 3) %
                                     (f'{epoch}/{300 - 1}', tversky_loss/4, focal_loss/4, loss.item()))
        

def train16fp(args, train_loader, model, criterion, optimizer, epoch,scaler):
    model.train()
    print("16fp-------------------")
    total_batches = len(train_loader)
    pbar = enumerate(train_loader)
    LOGGER.info(('\n' + '%13s' * 4) % ('Epoch','TverskyLoss','FocalLoss' ,'TotalLoss'))
    pbar = tqdm(pbar, total=total_batches, bar_format='{l_bar}{bar:10}{r_bar}')
    for i, (_,input, target) in pbar:
        optimizer.zero_grad()
        if args.onGPU == True:
            input = input.cuda().float() / 255.0        
        output = model(input)
        with torch.cuda.amp.autocast():
            focal_loss,tversky_loss,loss = criterion(output,target)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        pbar.set_description(('%13s' * 1 + '%13.4g' * 3) %
                                     (f'{epoch}/{300 - 1}', tversky_loss, focal_loss, loss.item()))

@torch.no_grad()
def val(val_loader, model):

    model.eval()

    LS=SegmentationMetric(2)

    LS_acc_seg = AverageMeter()
    LS_IoU_seg = AverageMeter()
    LS_mIoU_seg = AverageMeter()
    
    total_batches = len(val_loader)
    
    total_batches = len(val_loader)
    pbar = enumerate(val_loader)
    pbar = tqdm(pbar, total=total_batches)
    for i, (_,input, target) in pbar:
        RGB_input, Depth_input = input
        RGB_input = RGB_input.cuda().float() / 255.0  
        Depth_input = Depth_input.cuda().float() / 65535.0  
        Depth_input = Depth_input.unsqueeze(0)
        Depth_input = Depth_input.permute(1, 0, 2, 3)

        input_var = input
        target_var = target

        with torch.no_grad():
            output = model(RGB_input,Depth_input)

        out_LS=output
        target_LS=target

        _,LS_predict=torch.max(out_LS, 1)
        _,LS_gt=torch.max(target_LS, 1)
        
        LS.reset()
        LS.addBatch(LS_predict.cpu(), LS_gt.cpu())
        LS_acc = LS.pixelAccuracy()
        LS_IoU = LS.IntersectionOverUnion()
        LS_mIoU = LS.meanIntersectionOverUnion()
        LS_acc_seg.update(LS_acc,RGB_input.size(0))
        LS_IoU_seg.update(LS_IoU,RGB_input.size(0))
        LS_mIoU_seg.update(LS_mIoU,RGB_input.size(0))

    LS_segment_result = (LS_acc_seg.avg,LS_IoU_seg.avg,LS_mIoU_seg.avg)

    return LS_segment_result





def save_checkpoint(state, filenameCheckpoint='checkpoint.pth.tar'):
    torch.save(state, filenameCheckpoint)

def netParams(model):
    return np.sum([np.prod(parameter.size()) for parameter in model.parameters()])
