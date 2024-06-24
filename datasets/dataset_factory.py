
from .ffpp import FaceForensics

from .transforms import create_data_transforms
from .transforms import create_data_transforms_alb
import torch.utils.data as data
import pdb


def create_dataset(args, split):
    
    transform = create_data_transforms(args.transform, split)
    base_transform = create_data_transforms_alb(args.transform, split)
    
    kwargs = getattr(args.dataset, args.dataset.name)
    if args.dataset.name == 'ffpp':
        dataset = FaceForensics(split=split, base_transform=base_transform,transform=transform, image_size=args.transform.image_size, **kwargs)
    elif args.dataset.name == 'wild_deepfake':
        pass
    elif args.dataset.name == 'celeb_df':
        pass
    elif args.dataset.name == 'dfdc':
        pass
    
    else:
        raise Exception('Invalid dataset!')

    sampler = None
    if args.distributed:
        sampler = data.distributed.DistributedSampler(dataset)
    shuffle = True if sampler is None and split == 'train' else False
    batch_size = getattr(args, split).batch_size
    if args.debug:
        batch_size = 4
    dataloader = data.DataLoader(dataset,
                                 batch_size=batch_size,
                                 shuffle=shuffle,
                                 sampler=sampler,
                                 num_workers=6,
                                 pin_memory=True,
                                 drop_last = True)
    return dataloader


if __name__ == '__main__':
    import argparse
    from omegaconf import OmegaConf
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, default='../configs/default.yaml')
    parser.add_argument('--distributed', type=int, default=0)
    args = parser.parse_args()

    local_config = OmegaConf.load(args.config)
    for k, v in local_config.items():
        setattr(args, k, v)

    print('Dataset => ' + args.dataset.name)
    dataloader = create_dataset(args, split='train')
    for i, datas in enumerate(dataloader):
        print(i, datas[0].shape, datas[1].shape)
        break

    dataloader = create_dataset(args, split='val')
    for i, datas in enumerate(dataloader):
        print(i, datas[0].shape, datas[1].shape)
        break

    dataloader = create_dataset(args, split='test')
    for i, datas in enumerate(dataloader):
        print(i, datas[0].shape, datas[1].shape)
        break
    