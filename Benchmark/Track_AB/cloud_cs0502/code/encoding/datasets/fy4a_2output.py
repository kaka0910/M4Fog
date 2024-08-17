import torch
import torch.utils.data as data
from torch.autograd import Variable as V
from torchvision import transforms
import cv2
import numpy as np
import os
from PIL import Image
import random

def dropout(img, d=0.1, p=0.5):
    if random.random() > p:
        h, w, c = img.shape
        num_drop = int(h*d)
        y = np.random.randint(0, h, num_drop)
        x = np.random.randint(0, w, num_drop)
        for y_, x_ in zip(y, x):
            drop_c = np.random.randint(0, c, 1)
            img[y_, x_, drop_c] = 0
    return img

def randomHueSaturationValue(image, hue_shift_limit=(-180, 180),
                             sat_shift_limit=(-255, 255),
                             val_shift_limit=(-255, 255), u=0.5):
    if np.random.random() < u:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(image)
        hue_shift = np.random.randint(hue_shift_limit[0], hue_shift_limit[1]+1)
        hue_shift = np.uint8(hue_shift)
        h += hue_shift
        sat_shift = np.random.uniform(sat_shift_limit[0], sat_shift_limit[1])
        s = cv2.add(s, sat_shift)
        val_shift = np.random.uniform(val_shift_limit[0], val_shift_limit[1])
        v = cv2.add(v, val_shift)
        image = cv2.merge((h, s, v))
        #image = cv2.merge((s, v))
        image = cv2.cvtColor(image, cv2.COLOR_HSV2BGR)

    return image

def randomShiftScaleRotate(image, mask,
                           shift_limit=(-0.0, 0.0),
                           scale_limit=(-0.0, 0.0),
                           rotate_limit=(-0.0, 0.0), 
                           aspect_limit=(-0.0, 0.0),
                           borderMode=cv2.BORDER_CONSTANT, u=0.5):
    if np.random.random() < u:
        if len(image.shape)!=4:
            height, width, channel = image.shape

            angle = np.random.uniform(rotate_limit[0], rotate_limit[1])
            scale = np.random.uniform(1 + scale_limit[0], 1 + scale_limit[1])
            aspect = np.random.uniform(1 + aspect_limit[0], 1 + aspect_limit[1])
            sx = scale * aspect / (aspect ** 0.5)
            sy = scale / (aspect ** 0.5)
            dx = round(np.random.uniform(shift_limit[0], shift_limit[1]) * width)
            dy = round(np.random.uniform(shift_limit[0], shift_limit[1]) * height)

            cc = np.math.cos(angle / 180 * np.math.pi) * sx
            ss = np.math.sin(angle / 180 * np.math.pi) * sy
            rotate_matrix = np.array([[cc, -ss], [ss, cc]])

            box0 = np.array([[0, 0], [width, 0], [width, height], [0, height], ])
            box1 = box0 - np.array([width / 2, height / 2])
            box1 = np.dot(box1, rotate_matrix.T) + np.array([width / 2 + dx, height / 2 + dy])

            box0 = box0.astype(np.float32)
            box1 = box1.astype(np.float32)
            mat = cv2.getPerspectiveTransform(box0, box1)
            image = cv2.warpPerspective(image, mat, (width, height), flags=cv2.INTER_LINEAR, borderMode=borderMode,
                                        borderValue=(
                                            0, 0,
                                            0,))
            mask = cv2.warpPerspective(mask, mat, (width, height), flags=cv2.INTER_LINEAR, borderMode=borderMode,
                                    borderValue=(
                                        0, 0,
                                        0,))

    return image, mask

def randomHorizontalFlip(image, mask, u=0.5):
    if np.random.random() < u:
        image = cv2.flip(image, 1)
        mask = cv2.flip(mask, 1)

    return image, mask

def randomVerticleFlip(image, mask, u=0.5):
    if np.random.random() < u:
        image = cv2.flip(image, 0)
        mask = cv2.flip(mask, 0)

    return image, mask

def randomRotate90(image, mask, u=0.5):
    if np.random.random() > u:
        times=int(np.random.random()*4)
        for i in range(times):
            image=np.rot90(image)
            #print("rotimg", image.shape)
            # print("typtofmask",type(mask))

            mask=np.rot90(mask)
            #print("rotmask", mask.shape)
    return image, mask

def rot_90(img, mask, p=0.5):
    if random.random() > p: 
        img = np.rot90(img, k=-1)
        mask = np.rot90(mask, k=-1)
    return img.copy(), mask.copy()

def horizontal_flip(img, mask, p=0.5):
    if random.random() > p: 
        img = img[:, ::-1, :]
        mask = mask[:, ::-1]
    return img.copy(), mask.copy()

def vertical_flip(img, mask, p=0.5):
    if random.random() > p: 
        img = img[::-1, :, :]
        mask = mask[::-1, :]
    return img.copy(), mask.copy()

def heb(image,id):
    # date=int(id.split('_')[-1])
    # month=int(id[4:6])
    # if month==12 or month==1 or month==2:
    #     if (date>=730 and date <= 930) or date==2330 or date<=100:
    #         pass
    #     else:
    #         return image
    # else:
    #     return image

    # channels=[0,1,2,3,4,5]
    channels = range(16)
    for i in channels:
        tmp=image[:,:,i]
        heb_image=cv2.equalizeHist(tmp)
        image[:,:,i]=heb_image
        # tmp=image[:,:,i]
        # clahe = cv2.createCLAHE(clipLimit=5.0, tileGridSize=(8,8))
        # heb_image = clahe.apply(tmp)
        # image[:,:,i]=heb_image
    return image

def mask2onehot(mask, num_classes):
    """
    Converts a segmentation mask (H,W) to (K,H,W) where the last dim is a one
    hot encoding vector

    """
    _mask = [mask == i for i in range(num_classes)]
    return np.array(_mask).astype(np.uint8)

def default_loader(id, root, split, t):
    if t==2:
        id_index_1 = id.split('_')[-1]
        if id_index_1[-2:]=='30':
            id_index_0 = str(int(id_index_1)-30).zfill(4)
        else:
            id_index_0 = str(int(id_index_1)-70).zfill(4)
    
    id_0 = id.split('_')[0]+'_'+str(id_index_0)
    id_0_path = os.path.join(root,'{}.npy').format(id_0)
    # print(id_0)
    
    # print(id_0_path)
    npy_0 = np.load(os.path.join(root,'{}.npy').format(id_0)).astype('uint8')
    # import pdb;
    # pdb.set_trace()
    # npy_0 = np.transpose(npy_0,[2,0,1])
    img_0 = np.array(npy_0, np.float32)
    
    npy = np.load(os.path.join(root,'{}.npy').format(id)).astype('uint8')
    # npy = np.transpose(npy,[2,0,1])
    img_1 = np.array(npy, np.float32)
    
    mask_0 = cv2.imread(os.path.join(root,'conn_{}_300_sp.png').format(id_0), cv2.IMREAD_GRAYSCALE)
    mask_0 = np.array(mask_0, np.float32)/255.0
    mask_0[mask_0>=0.5] = 1
    mask_0[mask_0<=0.5] = 0
    mask_0 = mask_0.reshape(1,1024,1024)

    mask_1 = cv2.imread(os.path.join(root,'conn_{}_300_sp.png').format(id), cv2.IMREAD_GRAYSCALE)
    mask_1 = np.array(mask_1, np.float32)/255.0
    mask_1[mask_1>=0.5] = 1
    mask_1[mask_1<=0.5] = 0
    mask_1 = mask_1.reshape(1,1024,1024)

    return img_0, img_1, mask_0, mask_1

def process_id(trainlists):
    # import pdb;
    # pdb.set_trace()
    trainlist_selected = [x for x in trainlists]
    train_ids = []
    for id in trainlist_selected:
        id_index_1 = id.split('_')[-1]
        if id_index_1[-2:]=='30':
            id_index_0 = str(int(id_index_1)-30).zfill(4)
        else:
            id_index_0 = str(int(id_index_1)-70).zfill(4)
        
        if id[-4:]!='0000' and id[:9]+id_index_0 in trainlist_selected:
            train_ids.append(id)
    
    return train_ids
    
class FY4ASet_2output(data.Dataset):
    NUM_CLASS = 2  # NUM_CLASS = 2
    def __init__(self, root,split,get_name=False):
        self.get_name=get_name
        self.split=split
        root=os.path.join(root,split)
        # 2.Get data list (P.S.Data and Labels are in the same folder)
        imagelist = filter(lambda x: x.find('png') == -1, os.listdir(root))
        trainlist = map(lambda x: x[:-4], imagelist)
        # trainlist_selected = [x for x in trainlist if x[-4:]!='0000']
        self.ids = process_id(trainlist)
        # print(self.ids)
        self.loader = default_loader
        self.root = root
        self.tran = transforms.ToTensor()
        # self.norm = torch.FloatTensor([211, 211, 211, 211, 208, 211, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255]).unsqueeze(1).unsqueeze(1)
        self.mean = torch.FloatTensor([0.228, 0.191, 0.227, 0.042, 0.167, 0.108, 291.984, 291.662, 237.677, 249.9, 274.902, 276.679, 275.362, 249.906]).view(-1, 1, 1)
        self.std = torch.FloatTensor([0.114, 0.126, 0.153, 0.020, 0.095, 0.073, 11.459, 11.570, 5.567, 7.53, 14.273, 15.407, 15.395, 8.250]).view(-1, 1, 1)

    def __getitem__(self, index):
        id = self.ids[index]
        img_0, img_1, mask_0, mask_1 = self.loader(id, self.root, self.split, t=2)
        
        img_0 = torch.Tensor(img_0)
        img_0 = (img_0-self.mean)/self.std
        
        img_1 = torch.Tensor(img_1)
        img_1 = (img_1-self.mean)/self.std
        
        img = torch.concat((img_0,img_1),dim=0)
        
        mask_0 = torch.FloatTensor(mask_0)
        mask_1 = torch.FloatTensor(mask_1)
        mask = torch.concat((mask_0,mask_1),dim=0)
        
        if self.get_name:
            return img,mask,id
        
        return img, mask
    
    def __len__(self):
        return len(self.ids)

# from collections import  Counter
if __name__ == "__main__":
    # a = FY4ASet("/home/ssdk/mmengine/seafog_year_0412/","train")
    pass
