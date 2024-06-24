
import os
import argparse
import numpy as np
from omegaconf import OmegaConf
from tqdm import tqdm
from sklearn import metrics
from scipy.optimize import brentq
from scipy.interpolate import interp1d

import torch

import models
from datasets import create_dataset
from utils.misc import save_test_results
from glob import glob
import pdb
import warnings
import copy
from torch.utils.data import ConcatDataset
import torch.utils.data as data
from load_dataset import *

warnings.filterwarnings("ignore")




def load_model(args):
    if args.ckpt_path is not None:
        print(f'resume model from {args.ckpt_path}')
        checkpoint = torch.load(args.ckpt_path)
        if getattr(args, 'transform', None) is None:
            args.transform = checkpoint['args'].transform
        if getattr(args, 'model', None) is None:
            args.model = checkpoint['args'].model
       

        model = models.__dict__[args.model.name](**args.model.params)

        state_dict = checkpoint if args.ckpt_path.endswith('pth') else checkpoint['state_dict']
        model.load_state_dict(state_dict)
    else:
        assert getattr(args, 'model', None) is not None
        model = models.__dict__[args.model.name](**args.model.params)
    print(args.model)
    return model, args

def load_seal(args):
    seal_path = args.seal_path
    print(f"load seal from {seal_path}")
    seals = torch.from_numpy(np.load(seal_path)).cuda()
    return seals

def model_forward(args, inputs, model):
    
    output = model(inputs)
    if type(output) is tuple or type(output) is list:
        output = output[0]

    prob = torch.sigmoid(output).squeeze().cpu().numpy()
    return prob


def main():
    
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, default='configs/test.yaml')
    parser.add_argument('--distributed', type=int, default=0)
    parser.add_argument('--exam_id', type=str, default='')
    parser.add_argument('--ckpt_path', type=str, default='')
    parser.add_argument('--dataset', type=str, default='')
    parser.add_argument('--seal_path', type=str, default='')

    parser.add_argument('--compress', type=str, default='')
    parser.add_argument('--constract', type=bool, default=False)
    parser.add_argument('--debug', action='store_true', default=False)
    args = parser.parse_args()
    local_config = OmegaConf.load(args.config)
    for k, v in local_config.items():
        if args.dataset != '':
            if k=='dataset':
                v["name"] = args.dataset
                if args.compress != '':
                    v[args.dataset]["compressions"] = args.compress
        setattr(args, k, v)
    # setattr(args,"constract",args.constract)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    os.environ['TORCH_HOME'] = args.torch_home
    if args.exam_id:
        ckpt_path = glob(f'wandb/*{args.exam_id}/model_best.pth.tar')
        if len(ckpt_path) >= 1:
            args.ckpt_path = ckpt_path[0]
    if args.ckpt_path:
        args.output_dir = os.path.dirname(args.ckpt_path)
    model, args = load_model(args)
    model = model.to(device)
    model.eval()

    _,_,test_dataloaders = load_dataset(args)

    seals=None
    for i,test_dataloader in enumerate(test_dataloaders):
        print(f"testing {args.dataset.task_order[i]}")
        test(args,test_dataloader,model, device,seals)

def test(args,test_dataloader,model,device,seals):
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
            prob = model_forward(args, images, model)
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
    print(f'#Total# ACC:{acc:.5f}  AUC:{AUC:.5f}  EER:{100*eer:.2f}(Thresh:{thresh:.3f})')

    preds = np.array(y_preds) >= args.test.threshold
    pred_fake_nums = np.sum(preds)
    pred_real_nums = len(preds) - pred_fake_nums
    print(f'pred real nums:{pred_real_nums} pred fake nums:{pred_fake_nums}')

if __name__ == '__main__':
    main()