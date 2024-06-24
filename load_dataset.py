from torch.utils.data import ConcatDataset
from datasets import create_dataset
import torch.utils.data as data

import pdb
import copy


def load_dataset(args):
    train_dataloaders = []
    val_dataloaders = []
    test_dataloaders = []

    
    new_args = copy.deepcopy(args)
    new_args.dataset.name = "ffpp"
    new_args.dataset.ffpp.methods = ["youtube"]
    ffpp_real_dataloader_train = create_dataset(new_args, split='train')
    ffpp_real_dataloader_val = create_dataset(new_args, split='val')
    ffpp_real_dataloader_test = create_dataset(new_args, split='test')


    for item in args.dataset.task_order:
        print(f"~~~~~~~~load {item} dataset~~~~~~~~")
        new_args.dataset.name = "ffpp"
        new_args.dataset.ffpp.methods = [item]

        fake_train_dataloader = create_dataset(new_args, split='train')
        fake_val_dataloader = create_dataset(new_args, split='val')
        fake_test_dataloader = create_dataset(new_args, split='test')

        train_dataset = ConcatDataset([ffpp_real_dataloader_train.dataset,fake_train_dataloader.dataset])
        val_dataset = ConcatDataset([ffpp_real_dataloader_val.dataset,fake_val_dataloader.dataset])
        test_dataset = ConcatDataset([ffpp_real_dataloader_test.dataset,fake_test_dataloader.dataset])
        train_dataloader = data.DataLoader(train_dataset,
                                    batch_size=args.train.batch_size,
                                    shuffle=True,
                                    sampler=None,
                                    num_workers=6,
                                    pin_memory=True,
                                    drop_last = True)
        val_dataloader = data.DataLoader(val_dataset,
                                    batch_size=args.val.batch_size,
                                    shuffle=True,
                                    sampler=None,
                                    num_workers=6,
                                    pin_memory=True,
                                    drop_last = True)
        test_dataloader = data.DataLoader(test_dataset,
                                    batch_size=args.test.batch_size,
                                    shuffle=True,
                                    sampler=None,
                                    num_workers=6,
                                    pin_memory=True,
                                    drop_last = False)
        train_dataloaders.append(train_dataloader)
        val_dataloaders.append(val_dataloader)
        test_dataloaders.append(test_dataloader)
    return train_dataloaders, val_dataloaders, test_dataloaders