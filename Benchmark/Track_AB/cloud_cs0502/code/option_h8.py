###########################################################################
# Created by: Hang Zhang 
# Email: zhang.hang@rutgers.edu 
# Copyright (c) 2017
###########################################################################
import argparse

import torch

class Options():
    def __init__(self):
        parser = argparse.ArgumentParser(description='PyTorch \
            Segmentation')
        # model and dataset 
        parser.add_argument('--model', type=str, default='encnet',
                            help='model name (default: encnet)')
        parser.add_argument('--backbone', type=str, default='resnet50',
                            help='backbone name (default: resnet50)')
        parser.add_argument('--jpu', action='store_true', default=
                            False, help='JPU')
        parser.add_argument('--dilated', action='store_true', default=
                            False, help='dilation')
        parser.add_argument('--lateral', action='store_true', default=
                            False, help='employ FPN')
        parser.add_argument('--dataset', type=str, default='ade20k',
                            help='dataset name (default: pascal12)')
        parser.add_argument('--workers', type=int, default=32,
                            metavar='N', help='dataloader threads')
        parser.add_argument('--base-size', type=int, default=1024,
                            help='base image size')
        parser.add_argument('--crop-size', type=int, default=1024,
                            help='crop image size')
        parser.add_argument('--train-split', type=str, default='train',
                            help='dataset train split (default: train)')
        parser.add_argument('--fft',action="store_true",default=False,help="use fft")
        # training hyper params
        parser.add_argument('--dist-url', type=str, default=None,
                            help = 'distribute address')
        parser.add_argument('--world-size', type=int, default=1,
                            help='world size (default:1)')
        parser.add_argument('--rank', type=int, default=0,
                            help='rank (default:0)')
        parser.add_argument('--zoom-factor', type=int, default=8,
                            help='the output stride')
        parser.add_argument('--multiprocessing-distributed', action='store_true',
                            default=True, help='Multiprocessing Distributed Training')
        parser.add_argument('--aux', action='store_true', default= False,
                            help='Auxilary Loss')
        parser.add_argument('--aux-weight', type=float, default=0.2,
                            help='Auxilary loss weight (default: 0.2)')
        parser.add_argument('--se-loss', action='store_true', default= False,
                            help='Semantic Encoding Loss SE-loss')
        parser.add_argument('--se-weight', type=float, default=0.2,
                            help='SE-loss weight (default: 0.2)')
        parser.add_argument('--epochs', type=int, default=None, metavar='N',
                            help='number of epochs to train (default: auto)')
        parser.add_argument('--start_epoch', type=int, default=0,
                            metavar='N', help='start epochs (default:0)')
        parser.add_argument('--batch-size', type=int, default=None,
                            metavar='N', help='input batch size for \
                            training (default: auto)')
        parser.add_argument('--test-batch-size', type=int, default=None,
                            metavar='N', help='input batch size for \
                            testing (default: same as batch size)')
        # optimizer params
        parser.add_argument('--lr', type=float, default=None, metavar='LR',
                            help='learning rate (default: auto)')
        parser.add_argument('--lr-scheduler', type=str, default='poly',
                            help='learning rate scheduler (default: poly)')
        parser.add_argument('--momentum', type=float, default=0.9,
                            metavar='M', help='momentum (default: 0.9)')
        parser.add_argument('--weight-decay', type=float, default=1e-4,
                            metavar='M', help='w-decay (default: 1e-4)')
        # cuda, seed and logging
        parser.add_argument('--no-cuda', action='store_true', default=
                            False, help='disables CUDA training')
        parser.add_argument('--manual-seed', type=int, default=0, metavar='S',
                            help='random seed (default: 1)')
        # checking point
        parser.add_argument('--resume', type=str, default=None,
                            help='put the path to resuming file if needed')
        parser.add_argument('--checkname', type=str, default='default',
                            help='set the checkpoint name')
        parser.add_argument('--model-zoo', type=str, default=None,
                            help='evaluating on model zoo model')
        # finetuning pre-trained models
        parser.add_argument('--ft', action='store_true', default= False,
                            help='finetuning on a different dataset')
        # evaluation option
        parser.add_argument('--split', default='val')
        parser.add_argument('--mode', default='testval')
        parser.add_argument('--ms', action='store_true', default=False,
                            help='multi scale test')
        parser.add_argument('--flip', action='store_true', default=False,
                            help='flip & rot90 test')
        parser.add_argument('--no-val', action='store_true', default= False,
                            help='skip validation during training')
        parser.add_argument('--best-name', type=str, default='best_model_exp1.pth',
                            help = 'The name of the best model')
        parser.add_argument('--save_folder', type=str, default='output',
                            help = 'path to save images')
        # dataset and model directory
        parser.add_argument('--data_root', default='/groups/lmm2024/home/share/Sat_Pretrain_xmq/Data/H8_Data/')
        parser.add_argument('--model_root', default='model')
        parser.add_argument('--vis_root', default='vis')

        # [xmq: model save folder]
        parser.add_argument('--model_savefolder', default='output')
        parser.add_argument('--pretrained', default="/groups/lmm2024/home/share/Sat_Pretrain_xmq/cloud_cs0502/H8_output_cs0502/")

        # the parser
        self.parser = parser
        
        # [xmq: optim]
        parser.add_argument('--model_optimizer', default='SGD')

    def parse(self):
        args = self.parser.parse_args()
        args.cuda = not args.no_cuda and torch.cuda.is_available()
        # default settings for epochs, batch_size and lr
        if args.epochs is None:
            epoches = {
                'coco': 5,
                'citys': 240,
                'pascal_voc': 50, #50,
                'pascal_aug': 80, #80,
                'pcontext': 80,
                'ade20k': 130,
                'lip': 120,
                'cloud': 160
            }
            args.epochs = epoches[args.dataset.lower()]
        if args.batch_size is None:
            args.batch_size = 16
        if args.test_batch_size is None:
            args.test_batch_size = args.batch_size
        if args.lr is None:
            lrs = {
                'coco': 0.004,
                'citys': 0.01,
                'pascal_voc': 0.0001, # 0.0001
                'pascal_aug': 0.001, # 0.001
                'pcontext': 0.001, # 0.001
                'ade20k': 0.004, # 0.005
                'lip': 0.002,
                'cloud': 0.01
            }
            args.lr = lrs[args.dataset.lower()] #/ 32 * args.batch_size
        # print(args)
        for i in vars(args).items():
            print("{} : {}".format(i[0], i[1]))
        return args
