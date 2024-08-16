###########################################################################
# Created by: Hang Zhang 
# Email: zhang.hang@rutgers.edu 
# Copyright (c) 2017
###########################################################################

import os
import time

import encoding.utils as utils
import numpy as np
import torch
import torchvision.transforms as transform
from encoding.datasets import get_segmentation_dataset
from encoding.datasets.lip import LIP
from encoding.models import get_segmentation_model
from encoding.nn import BatchNorm2d, SegmentationLosses, ProbOhemCrossEntropy2d
from encoding.parallel import DataParallelCriterion, DataParallelModel
from tensorboardX import SummaryWriter
from torch.nn.parallel.scatter_gather import gather
from torch.utils import data
from tqdm import tqdm

from option import Options

torch_ver = torch.__version__[:3]
if torch_ver == '0.3':
    from torch.autograd import Variable

class Trainer():
    def __init__(self, args):
        self.args = args
        # data transforms
        input_transform = transform.Compose([
            transform.ToTensor(),
            transform.Normalize([.485, .456, .406], [.229, .224, .225])])
        # dataset
        if args.dataset != 'lip':
            data_kwargs = {'transform': input_transform, 'base_size': args.base_size,
                       'crop_size': args.crop_size}
            trainset = get_segmentation_dataset(args.dataset, root=args.data_root, split=args.train_split, mode='train',
                                           **data_kwargs)
            testset = get_segmentation_dataset(args.dataset, root=args.data_root, split='val', mode ='val',
                                           **data_kwargs)
        else:
            trainset = LIP(list_path='list/lip/trainList.txt')
            testset = LIP(list_path='list/lip/valList.txt', flip=False, multi_scale=False)
        # dataloader
        kwargs = {'num_workers': args.workers, 'pin_memory': True} \
            if args.cuda else {}
        self.trainloader = data.DataLoader(trainset, batch_size=args.batch_size,
                                           drop_last=True, shuffle=True, **kwargs)
        self.valloader = data.DataLoader(testset, batch_size=args.batch_size,
                                         drop_last=False, shuffle=False, **kwargs)
        self.nclass = trainset.num_class
        #vis dor
        visdir=args.vis_root
        if not os.path.exists(visdir):
            os.makedirs(visdir)
        self.writer = SummaryWriter(visdir)

        # model
        model = get_segmentation_model(args.model, dataset = args.dataset,
                                       backbone = args.backbone, dilated = args.dilated,
                                       lateral = args.lateral, jpu = args.jpu, aux = args.aux,
                                       se_loss = args.se_loss, norm_layer = BatchNorm2d,
                                       base_size = args.base_size, crop_size = args.crop_size, root=args.model_root)
        print(model)
        # optimizer using different LR
        if args.dataset == 'ade20k' and args.lr == 0.01:
            print("Using large LR")
            optimizer = torch.optim.SGD(model.parameters(), lr=args.lr,
            momentum=args.momentum, weight_decay=args.weight_decay)
        else:
            print("Using small LR, Head multiply 10")
            params_list = [{'params': model.pretrained.parameters(), 'lr': args.lr},]
            if hasattr(model, 'jpu'):
                params_list.append({'params': model.jpu.parameters(), 'lr': args.lr*10})
            if hasattr(model, 'head'):
                params_list.append({'params': model.head.parameters(), 'lr': args.lr*10})
            if hasattr(model, 'auxlayer'):
                params_list.append({'params': model.auxlayer.parameters(), 'lr': args.lr*10})
            optimizer = torch.optim.SGD(params_list, lr=args.lr,
                momentum=args.momentum, weight_decay=args.weight_decay)
        # criterions
        if args.dataset != 'lip':
            self.criterion = SegmentationLosses(se_loss=args.se_loss, aux=args.aux,
                                            nclass=self.nclass, 
                                            se_weight=args.se_weight,
                                            aux_weight=args.aux_weight)
        else:
            self.criterion = SegmentationLosses(se_loss=args.se_loss, aux=args.aux,
                                            nclass=self.nclass, 
                                            se_weight=args.se_weight,
                                            aux_weight=args.aux_weight,
                                            ignore_index=255)
        self.model, self.optimizer = model, optimizer
        # using cuda
        if args.cuda:
            self.model = DataParallelModel(self.model).cuda()
            self.criterion = DataParallelCriterion(self.criterion).cuda()
        # resuming checkpoint
        if args.resume is not None:
            if not os.path.isfile(args.resume):
                raise RuntimeError("=> no checkpoint found at '{}'" .format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            if args.cuda:
                self.model.module.load_state_dict(checkpoint['state_dict'])
            else:
                self.model.load_state_dict(checkpoint['state_dict'])
            if not args.ft:
                self.optimizer.load_state_dict(checkpoint['optimizer'])
            self.best_pred = checkpoint['best_pred']
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        # clear start epoch if fine-tuning
        if args.ft:
            args.start_epoch = 0
        # lr scheduler
        self.scheduler = utils.LR_Scheduler(args.lr_scheduler, args.lr,
                                            args.epochs, len(self.trainloader), warmup_epochs=3) #0
        self.best_pred = 0.0

    def training(self, epoch):
        self.model.train()
        tbar = tqdm(self.trainloader)
        # begin = time.time()
        # end = time.time()
        train_loss = utils.AverageMeter()
        for i, (image, target) in enumerate(tbar):
            # data_time = time.time() - end
            #print('data_time:{:.2f}'.format(data_time))
            # end = time.time()
            self.scheduler(self.optimizer, i, epoch, self.best_pred)

            self.optimizer.zero_grad()
            if torch_ver == "0.3":
                image = Variable(image)
                target = Variable(target)
            outputs = self.model(image) # seg_pred, sem_pred
            # f_time = time.time() - end
            #print('forward_time:{:.2f}'.format(f_time))
            # end = time.time()

            loss = self.criterion(outputs, target)
            # loss_time = time.time() - end
            #print('loss_time:{:.2f}'.format(loss_time))
            # end = time.time()

            loss.backward()
            # b_time = time.time() - end
            #print('backward_time:{:.2f}'.format(b_time))
            # end = time.time()

            self.optimizer.step()
            # opt_time = time.time() - end
            #print('opt_time:{:.2f}'.format(opt_time))

            train_loss.update(loss.item())
            # total = time.time() - begin
            #print('total_time:{:.2f}'.format(total))
            # begin = time.time()
            # end = time.time()

            tbar.set_description('Train loss: %.3f' % (train_loss.val / (i + 1)))
        print('Epoch: {}     Train loss: {:.3f}'.format(epoch, train_loss.avg))
        self.writer.add_scalar('loss', train_loss.avg, epoch)

        if self.args.no_val:
            # save checkpoint every epoch
            is_best = False
            utils.save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': self.model.module.state_dict(),
                'optimizer': self.optimizer.state_dict(),
                'best_pred': self.best_pred,
            }, self.args, is_best)


    def validation(self, epoch):
        # Fast test during the training
        def eval_batch(model, image, target):
            outputs = model(image)
            outputs = gather(outputs, 0, dim=0)
            pred = outputs[0]
            target = target.cuda()
            correct, labeled = utils.batch_pix_accuracy(pred.data, target)
            inter, union = utils.batch_intersection_union(pred.data, target, self.nclass)
            return correct, labeled, inter, union

        is_best = False
        self.model.eval()
        total_inter, total_union, total_correct, total_label = 0, 0, 0, 0
        tbar = tqdm(self.valloader, desc='\r')
        miou = utils.AverageMeter()
        pixacc = utils.AverageMeter()
        for i, (image, target) in enumerate(tbar):
            if torch_ver == "0.3":
                image = Variable(image, volatile=True)
                correct, labeled, inter, union = eval_batch(self.model, image, target)
            else:
                with torch.no_grad():
                    correct, labeled, inter, union = eval_batch(self.model, image, target)

            total_correct += correct
            total_label += labeled
            total_inter += inter
            total_union += union
            pixAcc = 1.0 * total_correct / (np.spacing(1) + total_label)
            pixacc.update(pixAcc)
            IoU = 1.0 * total_inter / (np.spacing(1) + total_union)
            mIoU = IoU.mean()
            miou.update(mIoU)
            tbar.set_description(
                'pixAcc: %.3f, mIoU: %.3f' % (pixAcc, mIoU))
        print('Epoch: {}   pixAcc: {:.3f}    mIOU: {:.3f}'.format(epoch, pixacc.val, miou.val))
        self.writer.add_scalar('pixAcc', pixacc.avg, epoch)
        self.writer.add_scalar('mIoU', miou.avg, epoch)

        new_pred = mIoU
        if new_pred > self.best_pred:
            is_best = True
            self.best_pred = new_pred
            utils.save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': self.model.module.state_dict(),
                'optimizer': self.optimizer.state_dict(),
                'best_pred': self.best_pred,
            }, self.args, is_best)


if __name__ == "__main__":
    SEED = 0
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    np.random.seed(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    # torch.backends.cudnn.enabled = True
    
    args = Options().parse()
    torch.manual_seed(args.seed)
    trainer = Trainer(args)
    print('Starting Epoch:', trainer.args.start_epoch)
    print('Total Epoches:', trainer.args.epochs)
    print('@@@ Start Training! @@@')
    for epoch in range(trainer.args.start_epoch, trainer.args.epochs):
        trainer.training(epoch)
        if not trainer.args.no_val and epoch >= int(trainer.args.epochs * 0.6):
            trainer.validation(epoch)
    print('-------------Training Done------------')
    print()
    print()
