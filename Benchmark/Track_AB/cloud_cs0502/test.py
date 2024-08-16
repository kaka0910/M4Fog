import numpy as np
import torch
import matplotlib.pyplot as plt

if __name__=='__main__':
    filepath='/home/ssdk/cloud/clouddata_v3/train/20200801_0030.npy'
    data=np.load(filepath)
    data=torch.from_numpy(data)
    r=torch.fft.rfft2(data,dim=[0,1])
    _abs=r.abs()
    _angle=r.angle()
    tmp=torch.polar(_abs,_angle)
    rec=torch.fft.irfft2(tmp,dim=[0,1])
    plt.imshow(rec[:,:,0])
    plt.show()
    plt.savefig("./fig.png")
    pass
