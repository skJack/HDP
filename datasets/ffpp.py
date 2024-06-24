import json
import os
import pdb
import random
from tkinter.messagebox import NO
import cv2
import numpy as np
from PIL import Image, ImageFilter
import torch
from torch.utils.data import Dataset, DataLoader
import pdb
import glob
from .transforms import create_data_transforms_alb
import albumentations as ab
import torch.nn.functional as nnf
import numpy as np
import clip
try:
    from .data_structure import *
except Exception:
    from data_structure import *

class FaceForensics(Dataset):
    def __init__(self, data_root,
                 num_frames, split, transform=None,
                 compressions='c23',methods=None,
                  has_mask=False, balance = True,data_types=""):
        self.data_root = data_root
        self.num_frames = num_frames
        self.split = split

        self.transform = transform
        self.data_types = data_types


        
        self.compressions = compressions
        self.methods = methods
        self.has_mask = has_mask

        self.balabce = balance
        self.fake_id_dict = {}


        if self.methods is None:
            self.methods = ['youtube', 'Deepfakes', 'Face2Face', 'FaceSwap', 'NeuralTextures']
        if "youtube" not in self.methods:
            self.real_items = []
            self.fake_items = self._load_items(self.methods[:])
            
        elif len(self.methods)==1:
            self.real_items = self._load_items([self.methods[0]])
            self.fake_items = []
            if self.split == "train" and self.balabce == True:
                self.real_items = np.random.choice(self.real_items, 11520, replace=False).tolist()
        else:
            self.real_items = self._load_items([self.methods[0]])
            self.fake_items = self._load_items(self.methods[1:])
            pos_len = len(self.real_items)
            neg_len = len(self.fake_items)
            if self.split == 'train' and self.balabce == True:
                np.random.seed(1234)
                if pos_len > neg_len:
                    self.real_items = np.random.choice(self.real_items, neg_len, replace=False).tolist()
                else:
                    self.fake_items = np.random.choice(self.fake_items, pos_len, replace=False).tolist()
                image_len = len(self.real_items)
                print(f'After balance total number of data: {image_len*2,} | pos: {image_len}, neg: {image_len}')



        pos_len = len(self.real_items)
        neg_len = len(self.fake_items)
        print(f'Total number of data: {pos_len+neg_len} | pos: {pos_len}, neg: {neg_len}')

        self.items = self.real_items + self.fake_items
        self.items = sorted(self.items, key=lambda x: x['img_path'])
        
    def _load_items(self, methods):
        subdirs = FaceForensicsDataStructure(root_dir=self.data_root,
                                             methods=methods,
                                             compressions=self.compressions,
                                             data_types=self.data_types).get_subdirs()
        splits_path = os.path.join(self.data_root, 'splits')
        video_ids = get_video_ids(self.split, splits_path)
        video_dirs = []
        for dir_path in subdirs:
            video_paths = listdir_with_full_paths(dir_path)
            videos = [x for x in video_paths if get_file_name(x) in video_ids]
            video_dirs.extend(videos)

        items = []
        for video_dir in video_dirs:
            label = 0. if 'original' in video_dir else 1.
            sub_items = self._load_sub_items(video_dir, label)
            items.extend(sub_items)
        
        return items
    def _load_sub_items(self, video_dir, label):
        if self.split == 'train' and label == 1:
            num_frames = self.num_frames // 3
        else:
            num_frames = self.num_frames
        video_id = get_file_name(video_dir)
        sorted_images_names = np.array(sorted(os.listdir(video_dir), key=lambda x: int(x.split('.')[0])))
        ind = np.linspace(0, len(sorted_images_names) - 1, num_frames, endpoint=True, dtype=np.int)
        sorted_images_names = sorted_images_names[ind]

        sub_items = []
        for image_name in sorted_images_names:
            frame_id = image_name.split('_')[-1].split('.')[0]
            img_path = os.path.join(video_dir, image_name)
            if 'original' in img_path:
                method = "real" 
            else:
                method = img_path.split("/")[4]
            sub_items.append({
                'img_path': img_path,
                'label': label,
                'video_id': video_id,
                'frame_id': frame_id,
                'method': method
            })
        return sub_items

    def __getitem__(self, index):
        item = self.items[index]

        image = cv2.cvtColor(cv2.imread(item['img_path']), cv2.COLOR_BGR2RGB)
        


        if self.transform != None:
            image = self.transform(image=image)['image']
        return image, item['label'], item['img_path']
        

    def __len__(self):
        return len(self.items)


def listdir_with_full_paths(dir_path):
    return [os.path.join(dir_path, x) for x in os.listdir(dir_path)]


def get_file_name(file_path):
    return file_path.split('/')[-1]


def read_json(file_path):
    with open(file_path) as inp:
        return json.load(inp)


def get_sets(data):
    return {x[0] for x in data} | {x[1] for x in data} | {'_'.join(x) for x in data} | {'_'.join(x[::-1]) for x in data}


def get_video_ids(spl, splits_path):
    return get_sets(read_json(os.path.join(splits_path, f'{spl}.json')))

