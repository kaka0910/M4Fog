import torch
import torch.nn as nn
from torch.autograd import Variable as V


def get_unet_2branchout1(dataset='cloud', **kwargs):
    # infer number of classes
    from ..datasets import datasets
    model = Unet_2branchout1(in_channels=16, num_classes=2, **kwargs)
    return model

class Unet_2branchout1(nn.Module):
    def __init__(
        self, 
        in_channels=14,
        num_classes=2,
        norm_layer=nn.BatchNorm2d,
        criterion_seg=None
    ):
        super(Unet_2branchout1, self).__init__()
        
        self.in_channels = in_channels
        self.nclass = num_classes
        self.criterion_seg = criterion_seg
        self.base_size = 256
        self.crop_size = 256
        self.down1 = self.conv_stage(in_channels, 64)
        self.down2 = self.conv_stage(64, 128)
        self.down3 = self.conv_stage(128, 256)
        self.down4 = self.conv_stage(256, 512)
        
        self.down1_2 = self.conv_stage(in_channels, 64)
        self.down2_2 = self.conv_stage(64, 128)
        self.down3_2 = self.conv_stage(128, 256)
        self.down4_2 = self.conv_stage(256, 512)

        # self.down5 = self.conv_stage(1024, 512)
       
        self.center = self.conv_stage(1024, 1024)
        
        self.up4 = self.conv_stage(1024, 512)
        self.up3 = self.conv_stage(512, 256)
        self.up2 = self.conv_stage(256, 128)
        self.up1 = self.conv_stage(128, 64)
        
        self.trans4 = self.upsample(1024, 512)
        self.trans3 = self.upsample(512, 256)
        self.trans2 = self.upsample(256, 128)
        self.trans1 = self.upsample(128, 64)
        
        self.conv_last = nn.Conv2d(64, num_classes, 3, 1, 1)
        
        self.max_pool = nn.MaxPool2d(2)
        
        self.conv = nn.Conv2d(1024,512,1)
        
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                if m.bias is not None:
                    m.bias.data.zero_()

    def conv_stage(self, dim_in, dim_out, kernel_size=3, stride=1, padding=1, useBN=True):
        if useBN:
            return nn.Sequential(
              nn.Conv2d(dim_in, dim_out, kernel_size=kernel_size, stride=stride, padding=padding),
              nn.BatchNorm2d(dim_out),
              nn.ReLU(inplace=True),
              nn.Conv2d(dim_out, dim_out, kernel_size=kernel_size, stride=stride, padding=padding),
              nn.BatchNorm2d(dim_out),
              nn.ReLU(inplace=True),
            )
        else:
            return nn.Sequential(
              nn.Conv2d(dim_in, dim_out, kernel_size=kernel_size, stride=stride, padding=padding),
              nn.ReLU(inplace=True),
              nn.Conv2d(dim_out, dim_out, kernel_size=kernel_size, stride=stride, padding=padding),
              nn.ReLU(inplace=True)
            )

    def upsample(self, ch_coarse, ch_fine):
        return nn.Sequential(
            nn.ConvTranspose2d(ch_coarse, ch_fine, 4, 2, 1, bias=False),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x, y=None):
        # import pdb;
        # pdb.set_trace()
        x1 = x[:,:16,:,:]
        x2 = x[:,16:,:,:]
        
        conv1_out_1 = self.down1(x1)
        conv2_out_1 = self.down2(self.max_pool(conv1_out_1))
        conv3_out_1 = self.down3(self.max_pool(conv2_out_1))
        conv4_out_1 = self.down4(self.max_pool(conv3_out_1))
        
        conv1_out_2 = self.down1_2(x2)
        conv2_out_2 = self.down2_2(self.max_pool(conv1_out_2))
        conv3_out_2 = self.down3_2(self.max_pool(conv2_out_2))
        conv4_out_2 = self.down4_2(self.max_pool(conv3_out_2))
        # import pdb;
        # pdb.set_trace()
        
        conv4_out = torch.concatenate([conv4_out_1,conv4_out_2], dim=1)
        # conv4_out = self.down5(conv4_out)
        
        out = self.center(self.max_pool(conv4_out))

        out = self.up4(torch.cat((self.trans4(out), conv4_out_2), 1))
        out = self.up3(torch.cat((self.trans3(out), conv3_out_2), 1))
        out = self.up2(torch.cat((self.trans2(out), conv2_out_2), 1))
        out = self.up1(torch.cat((self.trans1(out), conv1_out_2), 1))

        out = self.conv_last(out)

        if self.training:
            loss = self.criterion_seg(out, y.long())
            return out.max(1)[1].detach(), loss
        else:
            return out.detach()
    
    def evaluate(self, x, target=None):
        pred = self.forward(x)
        if isinstance(pred, (tuple, list)):
            pred = pred[0]
        if target is None:
            return pred
        correct, labeled = batch_pix_accuracy(pred.data, target.data)
        inter, union = batch_intersection_union(pred.data, target.data, self.nclass)
        return correct, labeled, inter, union