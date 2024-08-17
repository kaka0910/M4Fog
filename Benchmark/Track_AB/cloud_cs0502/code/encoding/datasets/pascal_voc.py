import os
import numpy as np

import torch

from PIL import Image
from tqdm import tqdm

from .base import BaseDataset

class VOCSegmentation(BaseDataset):
    CLASSES = [
        'background', 'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 
        'bus', 'car', 'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse',
        'motorbike', 'person', 'potted-plant', 'sheep', 'sofa', 'train',
        'tv/monitor', 'ambigious'
    ]
    NUM_CLASS = 21
    BASE_DIR = 'VOCdevkit/VOC2012'
    def __init__(self, root=os.path.expanduser('~/.encoding/data'), split='train',
                 mode=None, transform=None, target_transform=None, **kwargs):
        super(VOCSegmentation, self).__init__(root, split, mode, transform,
                                              target_transform, **kwargs)
        _voc_root = os.path.join(self.root, self.BASE_DIR)
        _mask_dir = os.path.join(_voc_root, 'SegmentationClass')
        _image_dir = os.path.join(_voc_root, 'JPEGImages')
        # train/val/test splits are pre-cut
        _splits_dir = os.path.join(_voc_root, 'ImageSets/Segmentation')
        if self.split == 'train':
            _split_f = os.path.join(_splits_dir, 'trainval.txt')
            # _split_f = os.path.join(_splits_dir, 'train.txt')
        elif self.split == 'val':
            # _split_f = os.path.join(_splits_dir, 'trainval.txt')
            _split_f = os.path.join(_splits_dir, 'val.txt')
        elif self.split == 'test':
            _split_f = os.path.join(_splits_dir, 'test.txt')
        else:
            raise RuntimeError('Unknown dataset split.')
        self.images = []
        self.masks = []
        with open(os.path.join(_split_f), "r") as lines:
            for line in tqdm(lines):
                _image = os.path.join(_image_dir, line.rstrip('\n')+".jpg")
                assert os.path.isfile(_image)
                self.images.append(_image)
                if self.mode != 'test':
                    _mask = os.path.join(_mask_dir, line.rstrip('\n')+".png")
                    assert os.path.isfile(_mask)
                    self.masks.append(_mask)
        if self.mode != 'test':
            assert (len(self.images) == len(self.masks))

    def __getitem__(self, index):
        img = Image.open(self.images[index]).convert('RGB')
        if self.mode == 'test':
            if self.transform is not None:
                img = self.transform(img)
            return img, os.path.basename(self.images[index])
        target = Image.open(self.masks[index])
        # synchrosized transform
        if self.mode == 'train':
            img, target = self._sync_transform( img, target)
        elif self.mode == 'val':
            img, target = self._val_sync_transform( img, target)
        else:
            assert self.mode == 'testval'
            target = self._mask_transform(target)
        # general resize, normalize and toTensor
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return img, target

    def _mask_transform(self, mask):
        target = np.array(mask).astype('int32')
        target[target == 255] = -1
        return torch.from_numpy(target).long()

    def __len__(self):
        return len(self.images)

    @property
    def pred_offset(self):
        return 0

# import os
# import numpy as np

# import torch
# import cv2
# from PIL import Image
# from tqdm import tqdm

# from .base import BaseDataset
# from .transform import *

# class VOCSegmentation(BaseDataset):
#     CLASSES = [
#         'background', 'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 
#         'bus', 'car', 'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse',
#         'motorbike', 'person', 'potted-plant', 'sheep', 'sofa', 'train',
#         'tv/monitor', 'ambigious'
#     ]
#     NUM_CLASS = 21
#     BASE_DIR = 'VOCdevkit/VOC2012'
#     def __init__(self, root=os.path.expanduser('~/.encoding/data'), split='train',
#                  mode=None, transform=None, target_transform=None, **kwargs):
#         super(VOCSegmentation, self).__init__(root, split, mode, transform,
#                                               target_transform, **kwargs)
#         _voc_root = os.path.join(self.root, self.BASE_DIR)
#         _mask_dir = os.path.join(_voc_root, 'SegmentationClass')
#         _image_dir = os.path.join(_voc_root, 'JPEGImages')
#         # train/val/test splits are pre-cut
#         _splits_dir = os.path.join(_voc_root, 'ImageSets/Segmentation')

#         # Mean & Std
#         value_scale = 255
#         mean = [0.485, 0.456, 0.406]
#         self.mean = [item * value_scale for item in mean]
#         std = [0.229, 0.224, 0.225]
#         self.std = [item * value_scale for item in std]

#         if self.split == 'train':
#             _split_f = os.path.join(_splits_dir, 'trainval.txt')
#             # _split_f = os.path.join(_splits_dir, 'train.txt')
#             self.transform = Compose([
#                 RandScale([0.5, 2.0]),
#                 RandRotate([-10, 10], padding=mean, ignore_label=0),
#                 # RandomGaussianBlur(),
#                 RandomHorizontalFlip(),
#                 Crop([self.crop_size, self.crop_size], crop_type='rand', padding=self.mean, ignore_label=0),
#                 ToTensor(),
#                 Normalize(mean=self.mean, std=self.std)])
#         elif self.split == 'val':
#             # _split_f = os.path.join(_splits_dir, 'trainval.txt')
#             _split_f = os.path.join(_splits_dir, 'val.txt')
#             if self.mode == 'val':
#                 self.transform = Compose([
#                     Crop([self.crop_size, self.crop_size], crop_type='center', padding=self.mean, ignore_label=0),
#                     ToTensor(),
#                     Normalize(mean=self.mean, std=self.std)])
#             elif self.mode == 'testval':
#                 self.transform = Compose([
#                     ToTensor(),
#                     Normalize(mean=self.mean, std=self.std)])
#             else: 
#                 raise NotImplementedError("error mode!")
#         elif self.split == 'test':
#             _split_f = os.path.join(_splits_dir, 'test.txt')
#             self.transform = Compose_test([
#                     ToTensor_test(),
#                     Normalize_test(mean=mean, std=std)])
#         else:
#             raise RuntimeError('Unknown dataset split.')
#         self.images = []
#         self.masks = []
#         with open(os.path.join(_split_f), "r") as lines:
#             for line in tqdm(lines):
#                 _image = os.path.join(_image_dir, line.rstrip('\n')+".jpg")
#                 assert os.path.isfile(_image)
#                 self.images.append(_image)
#                 if self.mode != 'test':
#                     _mask = os.path.join(_mask_dir, line.rstrip('\n')+".png")
#                     assert os.path.isfile(_mask)
#                     self.masks.append(_mask)
#         if self.mode != 'test':
#             assert (len(self.images) == len(self.masks))
    
#     def __getitem__(self, index):
#         img = Image.open(self.images[index]).convert('RGB')
#         img = np.array(img, dtype=np.float)
#         if self.mode == 'test':
#             if self.transform is not None:
#                 img = self.transform(img)
#             return img, os.path.basename(self.images[index])
#         target = Image.open(self.masks[index])
#         target = np.array(target)
#         # synchrosized transform
#         if self.mode == 'train':
#             # img, target = self._sync_transform( img, target)
#             img, target = self.transform(img, target)
#         elif self.mode == 'val':
#             # img, target = self._val_sync_transform( img, target)
#             img, target = self.transform(img, target)
#         else:
#             assert self.mode == 'testval'
#             # target = self._mask_transform(target)
#             img, target = self.transform(img, target)
#         # general resize, normalize and toTensor
#         # if self.transform is not None:
#         #     img = self.transform(img)
#         if self.target_transform is not None:
#             target = self.target_transform(target)
#         return img, target

#     # def __getitem__(self, index):
#     #     img = Image.open(self.images[index]).convert('RGB')
#     #     if self.mode == 'test':
#     #         if self.transform is not None:
#     #             img = self.transform(img)
#     #         return img, os.path.basename(self.images[index])
#     #     target = Image.open(self.masks[index])
#     #     # synchrosized transform
#     #     if self.mode == 'train':
#     #         img, target = self._sync_transform( img, target)
#     #     elif self.mode == 'val':
#     #         img, target = self._val_sync_transform( img, target)
#     #     else:
#     #         assert self.mode == 'testval'
#     #         target = self._mask_transform(target)
#     #     # general resize, normalize and toTensor
#     #     if self.transform is not None:
#     #         img = self.transform(img)
#     #     if self.target_transform is not None:
#     #         target = self.target_transform(target)
#     #     return img, target

#     def _mask_transform(self, mask):
#         target = np.array(mask).astype('int32')
#         target[target == 255] = -1
#         return torch.from_numpy(target).long()

#     def __len__(self):
#         return len(self.images)

#     @property
#     def pred_offset(self):
#         return 0
