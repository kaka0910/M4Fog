import numpy as np
import os
from scipy.ndimage import zoom

def cal_dist(npyu,npyv):
    dist = np.zeros(npyu.shape)
    for i in range(npyu.shape[0]):
        for j in range(npyu.shape[1]):
            dist[i][j] = np.sqrt(npyu[i][j]*npyu[i][j] + npyv[i][j] * npyv[i][j])
    return dist

ids = [x for x in os.listdir('/groups/lmm2024/home/share/Sat_Pretrain_xmq/Data/H8_Data/train_all/')+os.listdir('/groups/lmm2024/home/share/Sat_Pretrain_xmq/Data/H8_Data/val_2021/') if x.split('.')[-1]=='npy']
print(len(ids))
ids = [x for x in ids if '0000' not in x]
print(len(ids))
zoom_factor = 1024 / 103

for id in ids:
    motion_path = os.path.join('/groups/lmm2024/home/share/Sat_Pretrain_xmq/Data/H8_Data/motion_tc/results/','{}/').format(id.split('.')[0])
    # motion_path = os.path.join('/groups/lmm2024/home/share/Sat_Pretrain_xmq/Data/H8_Data/motion_tc/results/','{}/').format(id)

    motion_103_U = np.load(os.path.join(motion_path,'U_{}').format(id))
    motion_103_V = np.load(os.path.join(motion_path,'V_{}').format(id))
    
    motion_dist = cal_dist(motion_103_U,motion_103_V)
    np.save('/groups/lmm2024/home/share/Sat_Pretrain_xmq/Data/H8_Data/motion_tc/dist/'+id, motion_dist)