import os
import random
import numpy as np
import torch
from torch.utils.data import Dataset

from openstl.datasets.utils import create_loader


class H817xtcDataset_comb4(Dataset):

    def __init__(self, X, Y, use_augment=False, data_name='h8_17x'):
        super(H817xtcDataset_comb4, self).__init__()
        self.X = X # channel is 2
        self.Y = Y
        self.use_augment = use_augment
        self.mean = 0
        self.std = 1
        self.data_name = data_name

    def _augment_seq(self, seqs):
        """Augmentations as a video sequence"""
        # Random Flip
        if random.randint(0, 1):
            seqs = torch.flip(seqs, dims=(3, ))  # horizontal flip
        return seqs

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, index):
        data = torch.tensor(self.X[index]).float()/255.0
        labels = torch.tensor(self.Y[index]).float()/255.0
        if self.use_augment:
            len_data = data.shape[0]  # 4
            seqs = self._augment_seq(torch.cat([data, labels], dim=0))
            data = seqs[:len_data, ...]
            labels = seqs[len_data:, ...]
        return data, labels


def load_data(batch_size, val_batch_size, data_root, num_workers=1,
              pre_seq_length=None, aft_seq_length=None, in_shape=None,
              distributed=False, use_augment=False, use_prefetcher=False, drop_last=False):
    
    # [0:4] x_img, [4:8] y_img, [8:12] x_mask, [-4:] y_mask
    traindata = np.load(os.path.join(data_root, 'H8_17x/train_tc_640.npy'))
    X_train_img = traindata[:,:pre_seq_length,:,:,:]                     # (640, 4, 3, 256, 256)
    X_train_mask = traindata[:,2*pre_seq_length:3*pre_seq_length,:,:,:]  # (640, 4, 3, 256, 256)
    X_train = np.concatenate((X_train_img, X_train_mask),axis=2)          # (640, 4, 6, 256, 256)
    
    Y_train_img = traindata[:,pre_seq_length:2*pre_seq_length,:,:,:]                     
    Y_train_mask = traindata[:,-pre_seq_length:,:,:,:] 
    Y_train = np.concatenate((Y_train_img, Y_train_mask),axis=2)          

    testdata = np.load(os.path.join(data_root, 'H8_17x/val_tc_400.npy'))
    X_test_img = testdata[:,:pre_seq_length,:,:,:]                     # (640, 4, 3, 256, 256)
    X_test_mask = testdata[:,2*pre_seq_length:3*pre_seq_length,:,:,:]  # (640, 4, 3, 256, 256)
    X_test = np.concatenate((X_test_img, X_test_mask),axis=2)          # (640, 4, 6, 256, 256)
    
    Y_test_img = testdata[:,pre_seq_length:2*pre_seq_length,:,:,:]                     
    Y_test_mask = testdata[:,-pre_seq_length:,:,:,:] 
    Y_test = np.concatenate((Y_test_img, Y_test_mask),axis=2)  

    print('XMQ zhe ci yi ding neng pao dui:', X_train.shape, Y_train.shape, X_test.shape, Y_test.shape)
    
    assert X_train.shape[1] == pre_seq_length and Y_train.shape[1] == aft_seq_length
    
    train_set = H817xtcDataset_comb4(X=X_train, Y=Y_train, use_augment=False)
    test_set = H817xtcDataset_comb4(X=X_test, Y=Y_test, use_augment=False)

    dataloader_train = create_loader(train_set,
                                     batch_size=batch_size,
                                     shuffle=True, is_training=True,
                                     pin_memory=True, drop_last=True,
                                     num_workers=num_workers,
                                     distributed=distributed, use_prefetcher=use_prefetcher)
    dataloader_vali = create_loader(test_set,
                                    batch_size=val_batch_size,
                                    shuffle=False, is_training=False,
                                    pin_memory=True, drop_last=drop_last,
                                    num_workers=num_workers,
                                    distributed=distributed, use_prefetcher=use_prefetcher)
    dataloader_test = create_loader(test_set,
                                    batch_size=val_batch_size,
                                    shuffle=False, is_training=False,
                                    pin_memory=True, drop_last=drop_last,
                                    num_workers=num_workers,
                                    distributed=distributed, use_prefetcher=use_prefetcher)

    return dataloader_train, dataloader_vali, dataloader_test


if __name__ == '__main__':
    dataloader_train, _, dataloader_test = \
        load_data(batch_size=16,
                  val_batch_size=4,
                  data_root='/groups/lmm2024/home/share/Sat_Pretrain_xmq/OpenSTL-OpenSTL-Lightning/data/',
                  num_workers=4,
                  pre_seq_length=4, aft_seq_length=4)

    print(len(dataloader_train), len(dataloader_test))
    for item in dataloader_train:
        print(item[0].shape, item[1].shape)
        break
    for item in dataloader_test:
        print(item[0].shape, item[1].shape)
        break
