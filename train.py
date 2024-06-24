#!/usr/bin/env python
# -*- encoding: utf-8 -*-
import os
import wandb
import shutil
import numpy as np
from tqdm import tqdm
from timm.utils import CheckpointSaver
from timm.models import resume_checkpoint
from easydict import EasyDict
from sklearn.metrics import auc, roc_curve
from scipy.optimize import brentq
from scipy.interpolate import interp1d
import torch
import torch.nn as nn
import torch.distributed as dist
import torch.optim as optim
from tensorboardX import SummaryWriter
from sklearn import metrics
from scipy.optimize import brentq
from scipy.interpolate import interp1d
import copy
from glob import glob
from load_dataset import *

from wandb.proto.wandb_telemetry_pb2 import Feature

import models
from datasets import create_dataset
from utils.logger import Logger
from utils.init import setup
from utils.parameters import get_parameters
from utils.misc import *
import pdb
import random
import cv2
from torch.utils.data import ConcatDataset
import torch.utils.data as data
args = get_parameters()
setup(args)
if args.local_rank == 0:
    if args.wandb.name is None:
        args.wandb.name = args.config.split('/')[-1].replace('.yaml', '')
    wandb.init(**args.wandb)
    allow_val_change = False if args.wandb.resume is None else True
    wandb.config.update(args, allow_val_change)
    wandb.save(args.config)
    if len(wandb.run.dir) > 1:
        args.exam_dir = os.path.dirname(wandb.run.dir)
    else:
        args.exam_dir = 'wandb/debug'
        if os.path.exists(args.exam_dir):
            shutil.rmtree(args.exam_dir)
        os.makedirs(args.exam_dir, exist_ok=True)
    shutil.copytree("configs", f'{args.exam_dir}/Code/configs')
    shutil.copytree("datasets", f'{args.exam_dir}/Code/datasets')
    shutil.copytree("models", f'{args.exam_dir}/Code/models')
    shutil.copytree("utils", f'{args.exam_dir}/Code/utils')
    os.makedirs(f"{args.exam_dir}/seal/real", exist_ok=True)
    os.makedirs(f"{args.exam_dir}/seal/fake", exist_ok=True)



    logger = Logger(name='train', log_path=f'{args.exam_dir}/train.log')
    logger.info(args)
    logger.info(args.exam_dir)
    

def reduce_tensor(tensor: torch.Tensor) -> torch.Tensor:
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    rt /= dist.get_world_size()
    return rt

def save_feature(x,name):
    x = x.permute(2,1,0).transpose(0,1)
    x = (x+1) / 2
    image = x.detach().cpu().numpy()*255
    cv2.imwrite(f"{name}",image)



def model_forward(image, model, post_function=nn.Sigmoid()):
    output = model(image)
    if type(output) is tuple or type(output) is list:
        feature = output[1]
        output = output[0]
        output = post_function(output)
        output = output.squeeze()
        prediction = (output >= 0.5).float()
        return prediction, output,feature
    else:
        output = post_function(output)
        output = output.squeeze()
        prediction = (output >= 0.5).float()
        return prediction, output



def Reserve(model,criterion,dataloader,epoch_size,index):
    model.eval()
    losses = AverageMeter('Loss', ':.5f')
    progress = ProgressMeter(epoch_size, [losses])
    x = torch.rand((1,3,args.transform.image_size,args.transform.image_size)).cuda()
    x = x*0.001

    for epoch in range(args.seal.iter):
        for i, datas in enumerate(dataloader):
            images = datas[0].cuda()
            targets = datas[1].cuda().long()
            if (targets==0.0).float().sum()==0:
                continue
            while True:
                x.requires_grad = True
                model.zero_grad()
                seal_image_real = images[targets==0.0]+x
                seal_image_real = torch.clamp(seal_image_real, min=-1, max=1)
                targets_real = targets[targets==0.0]
                targets_real_reverse = targets_real+1.0
                prediction_hat, hat_output,_ = model_forward(seal_image_real,model)
                attack_rate = (prediction_hat == 1.0).float().sum() / (prediction_hat.shape[0])
                if attack_rate >= args.seal.rate:
                    break
                hat_label_ce = criterion(hat_output,targets_real_reverse)
                loss = hat_label_ce
                loss.backward()
                x = x-0.0001*x.grad.sign()
                x = torch.clamp(x, min=-0.15, max=0.15).detach_()

            losses.update(loss,targets_real_reverse.size(0))

            if i % args.seal.print_interval == 0:
                logger.info(f'Epoch-{epoch}: {progress.display(i)}')
                #save visiualization results
                save_feature(x[0],f"{args.exam_dir}/seal/{args.dataset.task_order[index]}_{i}.jpg")
            
    logger.info("~~~~~~save seal images~~~~~~")
    np.save(f"{args.exam_dir}/seal/{args.dataset.task_order[index]}_{i}.npy",x[0].detach().cpu().numpy())
    model.train()
    return x

def main():
    # Distributed traning
    if args.distributed:
        args.local_rank = int(os.environ.get('LOCAL_RANK', 0))
        dist.init_process_group(backend='nccl', init_method="env://")
        torch.cuda.set_device(args.local_rank)
        args.world_size = dist.get_world_size()

    train_dataloaders, val_dataloaders, test_dataloaders = load_dataset(args)


    # Create model
    device = torch.device("cuda", args.local_rank)
    model = models.__dict__[args.model.name](**args.model.params)
    model = model.to(device)

    start_epoch = 1
    if args.model.ckpt_path is not None:
        checkpoint = torch.load(args.model.ckpt_path)
        state_dict = checkpoint if args.model.ckpt_path.endswith('pth') else checkpoint['state_dict']
        model.load_state_dict(state_dict)
        if args.local_rank == 0:
            logger.info(f'resume model from {args.model.ckpt_path}')



    if args.distributed:
        model = nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank], find_unused_parameters=True)
        if args.sync_bn:
            model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
    else:
        model = nn.parallel.DataParallel(model)

    
    criterion = nn.__dict__[args.loss.name]().to(device)
    # Resume from checkpoint
    

    # Traing misc
    saver = None
    index = 0
    seals = None
    seals_list = []
    centroids = None
    prev_model = None
 
    for train_dataloader,val_dataloader,test_dataloader in zip(train_dataloaders, val_dataloaders, test_dataloaders):
        optimizer = optim.__dict__[args.optimizer.name](model.parameters(), **args.optimizer.params)
        scheduler = optim.lr_scheduler.__dict__[args.scheduler.name](optimizer, **args.scheduler.params)
        epoch_size = len(train_dataloaders[index].dataset) // (args.train.batch_size)
        if args.local_rank == 0:
            wandb.watch(model, log='all')
            os.makedirs(f"{args.exam_dir}/{args.dataset.task_order[index]}", exist_ok=True)
            saver = CheckpointSaver(model, optimizer,
                                    args=args,
                                    checkpoint_dir=f'{args.exam_dir}/{args.dataset.task_order[index]}/',
                                    recovery_dir=f'{args.exam_dir}/{args.dataset.task_order[index]}/',
                                    max_history=2)
        logger.info(f"Training task {index+1}, dataset is {args.dataset.task_order[index]}")
        
        for epoch in range(start_epoch, args.train.epochs + 1):
            if args.distributed and args.debug == False:
                train_dataloader.sampler.set_epoch(epoch)
            validate(val_dataloader, model,criterion,
                    optimizer, scheduler, wandb, saver, device, epoch)
            train(train_dataloader, model, criterion,
                optimizer, wandb, device, epoch, epoch_size,prev_model,seals_list,centroids)
        logger.info(f"~~~~~~generate seal of {args.dataset.task_order[index]}~~~~~~")
        
        seals = Reserve(model,criterion,train_dataloader,epoch_size,index)

        seals_list.append(seals)
    
        
        if args.local_rank == 0:
            for i,td in enumerate(test_dataloaders):
                logger.info(f"testing {args.dataset.task_order[i]}")
                if index ==0:
                    test(td,model,index ,criterion, device,False)
                else:
                    test(td,model,index ,criterion, device,True)
        
        index = index+1
        start_epoch = 1
        prev_model = copy.deepcopy(model)



def train(train_dataloader, model, criterion,
          optimizer, wandb, device, epoch, epoch_size,prev_model=None,seals_list=None,centroids=None):
    acces = AverageMeter('Acc', ':.5f')
    losses = AverageMeter('Loss', ':.5f')
    losses_r = AverageMeter('Loss_Reg', ':.5f')

    progress = ProgressMeter(epoch_size, [acces, losses,losses_r])
    criterion_mse = torch.nn.MSELoss().cuda()
    KL_criterion = torch.nn.KLDivLoss(size_average=True).cuda()

    model.train()
    
    train_loader_len = len(train_dataloader)
    for i, datas in enumerate(train_dataloader):
        images = datas[0].to(device)
        targets = datas[1].to(device).float()
        bs = images.shape[0]
        fake_num = images[targets==1.0].shape[0]
        if seals_list != None and len(seals_list) != 0:
            seals = random.choice(seals_list)
            seal_image_fake = images[targets==0.0]+seals
            seal_image_fake = torch.clamp(seal_image_fake, min=-1, max=1)
            seal_target = targets[targets==0.0]+1.0
            images = torch.cat((images,seal_image_fake))
            targets = torch.cat((targets,seal_target))
        prediction, output,feature = model_forward(images, model)
        loss_ce = criterion(output, targets)
        loss_reg = torch.tensor(0.0)

        if prev_model != None:
            with torch.no_grad():
                prev_model.train()
                _, output_pre,feature_pre = model_forward(images,prev_model)
            loss_mse_real = criterion_mse(feature_pre[targets==0.0],feature[targets==0.0])
            loss_kl_real = KL_criterion(torch.nn.LogSoftmax()(output[targets==0.0]),output_pre[targets==0.0])
            loss_kl_seal = KL_criterion(torch.nn.LogSoftmax()(output[bs:]),output_pre[bs:])
            loss_mse_seal = criterion_mse(feature_pre[bs:],feature[bs:])
            loss_reg = (loss_mse_real+loss_mse_seal)+(loss_kl_seal+loss_kl_real) #kl optional
            loss = loss_ce + loss_reg
        else:
            loss = loss_ce

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        targets[targets > 0.5] = 1.
        targets[targets < 0.5] = 0.
        acc = (prediction == targets).float().mean()

        if args.distributed:
            acces.update(reduce_tensor(acc.data).item(), targets.size(0))
            losses.update(reduce_tensor(loss.data).item(), targets.size(0))
        else:
            acces.update(acc.item(), targets.size(0))
            losses.update(loss.item(), targets.size(0))
            losses_r.update(loss_reg.item(),(targets==0.0).size(0))

        if args.local_rank == 0:
            if i % args.train.print_interval == 0:
                logger.info(f'Epoch-{epoch}: {progress.display(i)}')

    if args.local_rank == 0:
        wandb.log({
            'train_acc': acces.avg,
            'train_loss': losses.avg,
        }, step=epoch)



def validate(val_dataloader, model, criterion,
             optimizer, scheduler, writer, saver, device, epoch):
    acces = AverageMeter('Acc', ':.5f')
    losses = AverageMeter('Loss', ':.5f')

    y_preds = []
    y_trues = []
    model.eval()
    val_loader_len = len(val_dataloader)
    for i, datas in enumerate(tqdm(val_dataloader)):
        with torch.no_grad():
            images = datas[0].to(device)
            targets = datas[1].to(device).float()
            
            prediction, output,_ = model_forward(images, model)
            acc = (targets == prediction).float().mean()
            acces.update(acc.item(),targets.size(0))

            loss = criterion(output, targets)
            losses.update(loss.item(),targets.size(0))

    metrics = EasyDict()
    metrics.acc = acces.avg
    metrics.loss = losses.avg

    if scheduler.__class__.__name__ == 'ReduceLROnPlateau':
        scheduler.step(metrics.acc)
    else:
        scheduler.step()
    if args.local_rank == 0:
        best_acc, best_epoch = saver.save_checkpoint(epoch, metric=acces.avg)

        for k, v in metrics.items():
            logger.info(f'val_{k}: {100 * v:.4f}')
        logger.info(f'val_loss: {losses.avg:.4f}')
        logger.info(f'best_val_auc: {best_acc:.4f} (Epoch-{best_epoch})')

        last_lr = [group['lr'] for group in scheduler.optimizer.param_groups][0]
        wandb.log({
            'val_acc': metrics.acc,
            'val_loss': metrics.loss,
            'lr': last_lr,
        }, step=epoch)


def model_forward_test(args, inputs, model):

    output = model(inputs)
    if type(output) is tuple or type(output) is list:
        output = output[0]

    prob = torch.sigmoid(output).squeeze().cpu().numpy()
    return prob

def test(test_dataloader,model, index,criterion, device,best=False):
    logger.info('##### Test #####')
    args.distributed = 0
    
    if best == True:
        ckpt_path = f'{args.exam_dir}/{args.dataset.task_order[index]}/model_best.pth.tar'
        checkpoint = torch.load(ckpt_path)
        try:
            state_dict = {'module.' + k: w for k, w in checkpoint['state_dict'].items()}
            model.load_state_dict(state_dict)
        except Exception:
            model.load_state_dict(checkpoint['state_dict'])
        logger.info(f'resume model from {ckpt_path}')
    model.eval()

    y_trues = []
    y_preds = []
    acces = []
    img_paths = []
    for i, datas in enumerate(tqdm(test_dataloader)):
        with torch.no_grad():
            images = datas[0].to(device)
            targets = datas[1].float()
            targets = targets.numpy()
            y_trues.extend(targets)
            prob = model_forward_test(args, images, model)
            prediction = (prob >= args.test.threshold).astype(float)
            y_preds.extend(prob)
            acces.extend(targets == prediction)

            if args.test.record_results:
                img_paths.extend(datas[2])

    acc = np.mean(acces)
    fpr, tpr, thresholds = metrics.roc_curve(y_trues, y_preds, pos_label=1)
    AUC = metrics.auc(fpr, tpr)
    eer = brentq(lambda x: 1. - x - interp1d(fpr, tpr)(x), 0., 1.)
    thresh = interp1d(fpr, thresholds)(eer)
    logger.info(f'#Total# ACC:{acc:.5f}  AUC:{AUC:.5f}  EER:{100*eer:.2f}(Thresh:{thresh:.3f})')

    preds = np.array(y_preds) >= args.test.threshold
    pred_fake_nums = np.sum(preds)
    pred_real_nums = len(preds) - pred_fake_nums

    logger.info(f'pred real nums:{pred_real_nums} pred fake nums:{pred_fake_nums}')


if __name__ == '__main__':
    main()
