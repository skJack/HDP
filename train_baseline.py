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
from load_dataset import *


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


def save_feature(x, name):
    x = x.permute(2, 1, 0).transpose(0, 1)
    x = (x + 1) / 2
    image = x.detach().cpu().numpy() * 255
    cv2.imwrite(f"{name}", image)


def model_forward(image, model, post_function=nn.Sigmoid()):
    output = model(image)
    if type(output) is tuple or type(output) is list:
        feature = output[1]
        output = output[0]
        output = post_function(output)
        output = output.squeeze()
        prediction = (output >= 0.5).float()
        return prediction, output, feature
    else:
        output = post_function(output)
        output = output.squeeze()
        prediction = (output >= 0.5).float()
        return prediction, output


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
        state_dict = (
            checkpoint
            if args.model.ckpt_path.endswith('pth')
            else checkpoint['state_dict']
        )
        model.load_state_dict(state_dict)
        if args.local_rank == 0:
            logger.info(f'resume model from {args.model.ckpt_path}')

    if args.distributed:
        model = nn.parallel.DistributedDataParallel(
            model, device_ids=[args.local_rank], find_unused_parameters=True
        )
        if args.sync_bn:
            model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
    else:
        model = nn.parallel.DataParallel(model)

    criterion = nn.__dict__[args.loss.name]().to(device)
    # Resume from checkpoint

    # Traing misc
    saver = None
    index = 0


    for train_dataloader, val_dataloader, test_dataloader in zip(
        train_dataloaders, val_dataloaders, test_dataloaders
    ):
        optimizer = optim.__dict__[args.optimizer.name](
            model.parameters(), **args.optimizer.params
        )
        scheduler = optim.lr_scheduler.__dict__[args.scheduler.name](
            optimizer, **args.scheduler.params
        )
        epoch_size = len(train_dataloaders[index].dataset) // (args.train.batch_size)
        if args.local_rank == 0:
            wandb.watch(model, log='all')
            os.makedirs(
                f"{args.exam_dir}/{args.dataset.task_order[index]}", exist_ok=True
            )
            saver = CheckpointSaver(
                model,
                optimizer,
                args=args,
                checkpoint_dir=f'{args.exam_dir}/{args.dataset.task_order[index]}/',
                recovery_dir=f'{args.exam_dir}/{args.dataset.task_order[index]}/',
                max_history=2,
            )
        logger.info(
            f"Training task {index+1}, dataset is {args.dataset.task_order[index]}"
        )

        for epoch in range(start_epoch, args.train.epochs + 1):
            if args.distributed and args.debug == False:
                train_dataloader.sampler.set_epoch(epoch)
            validate(
                val_dataloader,
                model,
                criterion,
                optimizer,
                scheduler,
                wandb,
                saver,
                device,
                epoch,
            )
            train(
                train_dataloader,
                model,
                criterion,
                optimizer,
                wandb,
                device,
                epoch,
                epoch_size,
            )
        logger.info(f"~~~~~~generate seal of {args.dataset.task_order[index]}~~~~~~")

        if args.local_rank == 0:
            for i, td in enumerate(test_dataloaders):
                logger.info(f"testing {args.dataset.task_order[i]}")
                if index == 0:
                    test(td, model, index, criterion, device, False)
                else:
                    test(td, model, index, criterion, device, True)

        index = index + 1
        start_epoch = 1



def train(
    train_dataloader,
    model,
    criterion,
    optimizer,
    wandb,
    device,
    epoch,
    epoch_size,
):
    acces = AverageMeter('Acc', ':.5f')
    losses = AverageMeter('Loss', ':.5f')
    losses_c = AverageMeter('Loss_center', ':.5f')

    progress = ProgressMeter(epoch_size, [acces, losses, losses_c])
    criterion_mse = torch.nn.MSELoss().cuda()

    model.train()

    train_loader_len = len(train_dataloader)
    for i, datas in enumerate(train_dataloader):
        images = datas[0].to(device)
        targets = datas[1].to(device).float()
        bs = images.shape[0]
        fake_num = images[targets == 1.0].shape[0]
        real_num = bs - fake_num
        prediction, output, feature = model_forward(images, model)
        loss = criterion(output, targets)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        targets[targets > 0.5] = 1.0
        targets[targets < 0.5] = 0.0
        acc = (prediction == targets).float().mean()

        if args.distributed:
            acces.update(reduce_tensor(acc.data).item(), targets.size(0))
            losses.update(reduce_tensor(loss.data).item(), targets.size(0))
        else:
            acces.update(acc.item(), targets.size(0))
            losses.update(loss.item(), targets.size(0))

        if args.local_rank == 0:
            if i % args.train.print_interval == 0:
                logger.info(f'Epoch-{epoch}: {progress.display(i)}')

    if args.local_rank == 0:
        wandb.log(
            {
                'train_acc': acces.avg,
                'train_loss': losses.avg,
            },
            step=epoch,
        )


def validate(
    val_dataloader, model, criterion, optimizer, scheduler, writer, saver, device, epoch
):
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

            prediction, output, _ = model_forward(images, model)
            acc = (targets == prediction).float().mean()
            acces.update(acc.item(), targets.size(0))
            loss = criterion(output, targets)
            losses.update(loss.item(), targets.size(0))

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
        wandb.log(
            {
                'val_acc': metrics.acc,
                'val_loss': metrics.loss,
                'lr': last_lr,
            },
            step=epoch,
        )


def model_forward_test(args, inputs, model):

    output = model(inputs)
    if type(output) is tuple or type(output) is list:
        output = output[0]

    prob = torch.sigmoid(output).squeeze().cpu().numpy()
    return prob


def test(test_dataloader, model, index, criterion, device, best=False):
    logger.info('##### Test #####')
    args.distributed = 0

    if best == True:
        ckpt_path = (
            f'{args.exam_dir}/{args.dataset.task_order[index]}/model_best.pth.tar'
        )
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
    eer = brentq(lambda x: 1.0 - x - interp1d(fpr, tpr)(x), 0.0, 1.0)
    thresh = interp1d(fpr, thresholds)(eer)
    logger.info(
        f'#Total# ACC:{acc:.5f}  AUC:{AUC:.5f}  EER:{100*eer:.2f}(Thresh:{thresh:.3f})'
    )

    preds = np.array(y_preds) >= args.test.threshold
    pred_fake_nums = np.sum(preds)
    pred_real_nums = len(preds) - pred_fake_nums

    logger.info(f'pred real nums:{pred_real_nums} pred fake nums:{pred_fake_nums}')


if __name__ == '__main__':
    main()
