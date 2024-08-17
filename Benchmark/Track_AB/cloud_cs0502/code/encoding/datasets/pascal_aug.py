import os
import scipy.io
from tqdm import tqdm
from PIL import Image
import numpy as np
import torch
from .base import BaseDataset

class VOCAugSegmentation(BaseDataset):
    voc = [
        'background', 'airplane', 'bicycle', 'bird', 'boat', 'bottle', 
        'bus', 'car', 'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse',
        'motorcycle', 'person', 'potted-plant', 'sheep', 'sofa', 'train',
        'tv'
    ]
    NUM_CLASS = 21
    TRAIN_BASE_DIR = 'VOCaug/dataset/'
    # BASE_DIR = 'VOCdevkit/VOC2012'
    def __init__(self, root=os.path.expanduser('~/.encoding/data'), split='train',
                 mode=None, transform=None, target_transform=None, **kwargs):
        super(VOCAugSegmentation, self).__init__(root, split, mode, transform,
                                                 target_transform, **kwargs)
        # _voc_root = os.path.join(self.root, self.BASE_DIR)
        # _mask_dir = os.path.join(_voc_root, 'SegmentationClassAug')
        # _image_dir = os.path.join(_voc_root, 'JPEGImages')
        # # train/val/test splits are pre-cut
        # _splits_dir = os.path.join(_voc_root, 'ImageSets/Segmentation')
        # if self.split == 'train':
        #     _split_f = os.path.join(_splits_dir, 'trainaug.txt')
        # elif self.split == 'val':
        #     _split_f = os.path.join(_splits_dir, 'val.txt')
        # else:
        #     raise RuntimeError('Unknown dataset split.')
        # self.images = []
        # self.masks = []
        # with open(os.path.join(_split_f), "r") as lines:
        #     for line in tqdm(lines):
        #         _image = os.path.join(_image_dir, line.rstrip('\n')+".jpg")
        #         assert os.path.isfile(_image)
        #         self.images.append(_image)
        #         if self.mode != 'test':
        #             _mask = os.path.join(_mask_dir, line.rstrip('\n')+".png")
        #             assert os.path.isfile(_mask)
        #             self.masks.append(_mask)
        # if self.mode != 'test':
        #     assert (len(self.images) == len(self.masks))

        # train/val/test splits are pre-cut
        _voc_root = os.path.join(root, self.TRAIN_BASE_DIR)
        _mask_dir = os.path.join(_voc_root, 'cls')
        _image_dir = os.path.join(_voc_root, 'img')
        if self.split == 'train':
            _split_f = os.path.join(_voc_root, 'trainval.txt')
        elif self.split == 'val':
            # _split_f = os.path.join(_voc_root, 'trainval.txt')
            _split_f = os.path.join(_voc_root, 'val.txt')
        else:
            raise RuntimeError('Unknown dataset split.')
        self.images = []
        self.masks = []
        with open(os.path.join(_split_f), "r") as lines:
            for line in lines:
                _image = os.path.join(_image_dir, line.rstrip('\n')+".jpg")
                assert os.path.isfile(_image)
                self.images.append(_image)
                if self.mode != 'test':
                    _mask = os.path.join(_mask_dir, line.rstrip('\n')+".mat")
                    assert os.path.isfile(_mask)
                    self.masks.append(_mask)
        print("Total items: {}".format(len(self.images)))

        assert (len(self.images) == len(self.masks))

    def __getitem__(self, index):
        _img = Image.open(self.images[index]).convert('RGB')
        if self.mode == 'test':
            if self.transform is not None:
                _img = self.transform(_img)
            return _img, os.path.basename(self.images[index])
        _target = self._load_mat(self.masks[index])
        # _target = Image.open(self.masks[index])

        # synchrosized transform
        if self.mode == 'train':
            _img, _target = self._sync_transform( _img, _target)
        elif self.mode == 'val':
            _img, _target = self._val_sync_transform( _img, _target)
        else:
            assert self.mode == 'testval'
            _target = self._mask_transform(_target)

        # general resize, normalize and toTensor
        if self.transform is not None:
            _img = self.transform(_img)
        if self.target_transform is not None:
            _target = self.target_transform(_target)
        return _img, _target

    def _mask_transform(self, mask):
        target = np.array(mask).astype('int32')
        target[target == 255] = -1
        return torch.from_numpy(target).long()

    def _load_mat(self, filename):
        mat = scipy.io.loadmat(filename, mat_dtype=True, squeeze_me=True, 
            struct_as_record=False)
        mask = mat['GTcls'].Segmentation
        return Image.fromarray(mask)

    def __len__(self):
        return len(self.images)

# import os
# import scipy.io
# from tqdm import tqdm
# from PIL import Image
# import numpy as np
# import torch
# from .base import BaseDataset
# from .transform import *

# class VOCAugSegmentation(BaseDataset):
#     voc = [
#         'background', 'airplane', 'bicycle', 'bird', 'boat', 'bottle', 
#         'bus', 'car', 'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse',
#         'motorcycle', 'person', 'potted-plant', 'sheep', 'sofa', 'train',
#         'tv'
#     ]
#     NUM_CLASS = 21
#     # TRAIN_BASE_DIR = 'VOCaug/dataset/'
#     BASE_DIR = 'VOCdevkit/VOC2012'
#     def __init__(self, root=os.path.expanduser('~/.encoding/data'), split='train',
#                  mode=None, transform=None, target_transform=None, **kwargs):
#         super(VOCAugSegmentation, self).__init__(root, split, mode, transform,
#                                                  target_transform, **kwargs)
#         _voc_root = os.path.join(self.root, self.BASE_DIR)
#         _mask_dir = os.path.join(_voc_root, 'SegmentationClassAug')
#         _image_dir = os.path.join(_voc_root, 'JPEGImages')
#         # train/val/test splits are pre-cut
#         _splits_dir = os.path.join(_voc_root, 'ImageSets/Segmentation')

#         # Mean & Std
#         value_scale = 255
#         mean = [0.485, 0.456, 0.406]
#         mean = [item * value_scale for item in mean]
#         std = [0.229, 0.224, 0.225]
#         std = [item * value_scale for item in std]

#         if self.split == 'train':
#             _split_f = os.path.join(_splits_dir, 'trainaug.txt')
#             self.transform = Compose([
#                 RandScale([0.5, 2.0]),
#                 RandRotate([-10, 10], padding=mean, ignore_label=0),
#                 # RandomGaussianBlur(),
#                 RandomHorizontalFlip(),
#                 Crop([self.crop_size, self.crop_size], crop_type='rand', padding=mean, ignore_label=0),
#                 ToTensor(),
#                 Normalize(mean=mean, std=std)])
#         elif self.split == 'val':
#             _split_f = os.path.join(_splits_dir, 'val.txt')
#             if self.mode == 'val':
#                 self.transform = Compose([
#                     Crop([self.crop_size, self.crop_size], crop_type='center', padding=mean, ignore_label=0),
#                     ToTensor(),
#                     Normalize(mean=mean, std=std)])
#             elif self.mode == 'testval':
#                 self.transform = Compose([
#                     ToTensor(),
#                     Normalize(mean=mean, std=std)])
#             else: 
#                 raise NotImplementedError("error mode!")
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
#         _img = Image.open(self.images[index]).convert('RGB')
#         _img = np.array(_img, dtype=np.float)
#         if self.mode == 'test':
#             if self.transform is not None:
#                 _img = self.transform(_img)
#             return _img, os.path.basename(self.images[index])

#         # _target = self._load_mat(self.masks[index])
#         _target = Image.open(self.masks[index])
#         _target = np.array(_target)

#         # synchrosized transform
#         if self.mode == 'train':
#             # _img, _target = self._sync_transform( _img, _target)
#             _img, _target = self.transform(_img, _target)
#         elif self.mode == 'val':
#             # _img, _target = self._val_sync_transform( _img, _target)
#             _img, _target = self.transform(_img, _target)
#         else:
#             assert self.mode == 'testval'
#             # _target = self._mask_transform(_target)
#             _img, _target = self.transform(_img, _target)
#         # # general resize, normalize and toTensor
#         # if self.transform is not None:
#         #     _img = self.transform(_img)
#         if self.target_transform is not None:
#             _target = self.target_transform(_target)
#         return _img, _target

#     def _mask_transform(self, mask):
#         target = np.array(mask).astype('int32')
#         target[target == 255] = -1
#         return torch.from_numpy(target).long()

#     def _load_mat(self, filename):
#         mat = scipy.io.loadmat(filename, mat_dtype=True, squeeze_me=True, 
#             struct_as_record=False)
#         mask = mat['GTcls'].Segmentation
#         return Image.fromarray(mask)

#     def __len__(self):
#         return len(self.images)
