import argparse
import logging
import os
import random
import time
# import wandb
import cv2
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.parallel
import torch.optim
import torch.utils.data as data
import torchvision.transforms as transform
from tensorboardX import SummaryWriter
from torch.cuda.amp import GradScaler, autocast
from torch.autograd import Variable as V
# from torchstat import stat

from encoding.datasets import get_segmentation_dataset
from encoding.models import get_segmentation_model
from encoding.nn import BootstrappedCELoss, OhemCELoss, SegmentationLosses, dice_ce_loss,focal_loss,ce_loss,bce_loss
from encoding.utils import (AverageMeter, LR_Scheduler,
                            intersectionAndUnionGPU, save_checkpoint)
from option_h8 import Options
from clearml import Task
torch_ver = torch.__version__[:3]
if torch_ver == '0.3':
    from torch.autograd import Variable

# os.environ["NCCL_DEBUG"] = "INFO"
# cv2.ocl.setUseOpenCL(False)
# cv2.setNumThreads(0)

def get_logger():
    logger_name = "main-logger"
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler()
    fmt = "[%(asctime)s %(levelname)s %(filename)s line %(lineno)d %(process)d] %(message)s"
    handler.setFormatter(logging.Formatter(fmt))
    logger.addHandler(handler)
    return logger

def main_process():
    return not args.multiprocessing_distributed or (args.multiprocessing_distributed and args.rank % args.ngpus_per_node == 0)

def main():
    args = Options().parse()
    args.train_gpu = list(range(torch.cuda.device_count()))

    if args.manual_seed is not None:
        random.seed(args.manual_seed)
        np.random.seed(args.manual_seed)
        os.environ['PYTHONHASHSEED'] = str(args.manual_seed)
        torch.manual_seed(args.manual_seed)
        torch.cuda.manual_seed(args.manual_seed)
        torch.cuda.manual_seed_all(args.manual_seed)
        cudnn.benchmark = False
        cudnn.deterministic = True
    if args.dist_url == "env://" and args.world_size == -1:
        args.world_size = int(os.environ["WORLD_SIZE"])
    args.distributed = args.world_size > 1 or args.multiprocessing_distributed
    args.ngpus_per_node = len(args.train_gpu)

    if len(args.train_gpu) == 1:
        args.sync_bn = False
        args.distributed = False
        args.multiprocessing_distributed = False
    
    if args.multiprocessing_distributed:
        args.world_size = args.ngpus_per_node * args.world_size
        mp.spawn(main_worker, nprocs=args.ngpus_per_node, args=(args.ngpus_per_node, args))
    else:
        main_worker(args.train_gpu, args.ngpus_per_node, args)

def main_worker(gpu, ngpus_per_node, argss):
    global args
    args = argss

    if args.distributed:
        if args.dist_url == "env://" and args.rank == -1:
            args.rank = int(os.environ["RANK"])
        if args.multiprocessing_distributed:
            args.rank = args.rank * ngpus_per_node + gpu
        dist.init_process_group(backend='nccl', init_method=args.dist_url, world_size=args.world_size, rank=args.rank)

    criterion_seg = ce_loss()

    model = get_segmentation_model(args.model,criterion_seg=criterion_seg) ##aux
    model_pretrain_path = args.pretrained + 'best_model_exp1.pth'
    model.load_state_dict(torch.load(model_pretrain_path)['state_dict'])


    if main_process():
        global logger, writer
        logger = get_logger()
        writer = SummaryWriter(args.vis_root) ##tensorboard --logdir vis_root
        logger.info(args)
        logger.info("=> creating model ...")
        logger.info("Classes: {}".format(model.nclass))
        logger.info(model)

    # params_list = [{'params': model.pretrained.parameters(), 'lr': args.lr},]
    # if hasattr(model, 'jpu'):
    #     params_list.append({'params': model.jpu.parameters(), 'lr': args.lr*10})
    # if hasattr(model, 'head'):
    #     params_list.append({'params': model.head.parameters(), 'lr': args.lr*10})
    # if hasattr(model, 'auxlayer'):
    #     params_list.append({'params': model.auxlayer.parameters(), 'lr': args.lr*10})
    # optimizer = torch.optim.SGD(params_list, lr=args.lr,
    #     momentum=args.momentum, weight_decay=args.weight_decay)

    # optimizer = torch.optim.SGD(model.parameters(), lr=args.lr,
    #     momentum=args.momentum, weight_decay=args.weight_decay)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr,
        betas=(0.9, 0.999), eps=1e-8, weight_decay=args.weight_decay)

    if args.distributed:
        torch.cuda.set_device(gpu)
        args.batch_size = int(args.batch_size / ngpus_per_node)
        args.workers = int(args.workers / ngpus_per_node)
        # model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
        scaler = GradScaler(enabled=True)
        model = torch.nn.parallel.DistributedDataParallel(model.cuda(), device_ids=[gpu], find_unused_parameters=True)
    else:
        scaler = GradScaler()
        model = torch.nn.DataParallel(model.cuda())

    best_mIoU = 0.0
    # resuming checkpoint
    if args.resume is not None:
        if not os.path.isfile(args.resume):
            raise RuntimeError("=> no checkpoint found at '{}'" .format(args.resume))
        checkpoint = torch.load(args.resume, map_location=lambda storage, loc: storage.cuda())
        args.start_epoch = checkpoint['epoch']
        model.module.load_state_dict(checkpoint['state_dict'])
        if not args.ft:
            optimizer.load_state_dict(checkpoint['optimizer'])
        best_mIoU = checkpoint['best_mIoU'] * 0.01
        print("=> loaded checkpoint '{}' (epoch {})"
              .format(args.resume, checkpoint['epoch']))
    # clear start epoch if fine-tuning
    if args.ft:
        args.start_epoch = 0


    # dataset
    train_data = get_segmentation_dataset(args.dataset, root=args.data_root, split='val_2021')
    val_data = get_segmentation_dataset(args.dataset, root=args.data_root, split='val_2021',get_name=True)

    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_data)
        val_sampler = torch.utils.data.distributed.DistributedSampler(val_data)
    else:
        train_sampler = None
        val_sampler = None

    train_loader = torch.utils.data.DataLoader(train_data, batch_size=args.batch_size, 
                                               shuffle=(train_sampler is None), num_workers=args.workers, 
                                               pin_memory=True, sampler=train_sampler, drop_last=True)
    val_loader = torch.utils.data.DataLoader(val_data, batch_size=args.batch_size, 
                                             shuffle=False, num_workers=args.workers, 
                                             pin_memory=True, sampler=val_sampler)
    print(len(train_loader),len(val_loader))

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max = args.epochs, eta_min = 1e-6)
    # scheduler = LR_Scheduler(args.lr_scheduler, args.lr, args.epochs, len(train_loader), warmup_epochs=10)
    currentSteps=[0]
    for epoch in range(0,1):
        epoch_log = epoch + 1
        if args.distributed:
            train_sampler.set_epoch(epoch)
        if main_process():
            logger.info('>>>>>>>>>>>>>>>> Start Validation >>>>>>>>>>>>>>>>')

        loss_val, mIoU_val, mAcc_val, allAcc_val = validate(val_loader, model, criterion_seg, train_data.NUM_CLASS,epoch=epoch)
        if main_process():
            writer.add_scalar('loss_val', loss_val, epoch_log)
            writer.add_scalar('mIoU_val', mIoU_val, epoch_log)
            writer.add_scalar('mAcc_val', mAcc_val, epoch_log)
            writer.add_scalar('allAcc_val', allAcc_val, epoch_log)




def validate(val_loader, model, criterion, nclass,epoch):
    if main_process():
        logger.info('>>>>>>>>>>>>>>>> Start Evaluation >>>>>>>>>>>>>>>>')
    batch_time = AverageMeter()
    data_time = AverageMeter()
    loss_meter = AverageMeter()
    intersection_meter = AverageMeter()
    union_meter = AverageMeter()
    target_meter = AverageMeter()

    model.eval()
    end = time.time()
    for i, (input, target, ids) in enumerate(val_loader):
        data_time.update(time.time() - end)
        input = input.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)
        # target = (target.cuda(non_blocking=True),False)
        with torch.no_grad():
            output = model(input)
        # if args.zoom_factor != 8:
        #     output = F.interpolate(output, size=target.size()[1:], mode='bilinear', align_corners=True)
        loss = criterion(output,target)

        n = input.size(0)
        if args.multiprocessing_distributed:
            loss = loss * n  # not considering ignore pixels
            count = target.new_tensor([n], dtype=torch.long)
            dist.all_reduce(loss), dist.all_reduce(count)
            n = count.item()
            loss = loss / n
        else:
            loss = torch.mean(loss)

        output = output.max(1)[1]

        B,H,W = output.shape
        outputs = output.cpu().numpy().astype(np.uint8)*255
                
        # for index,id in enumerate(ids):
        #     savepath = '/groups/lmm2024/home/share/Sat_Pretrain_xmq/cloud_cs0502/H8_output_cs0502/test/06abcnet/pred_01unetpp_'+id[:13]+'.png'
        #     cv2.imwrite(savepath, outputs[index,:,:])
        
        #     image=np.zeros((H,W*3,3))
        #     norm = np.array([211,211,255]).reshape(3,1,1)
        #     image[:,:W,:]=(images[index,[0,1,2]]*norm).transpose([1,2,0])[:,:,[2,1,0]]
        #     image[:,W:2*W,:]=(targets[index]*255)[:,:,np.newaxis].repeat(3,2)
        #     image[:,W*2:W*3,:]=(outputs[index]*255)[:,:,np.newaxis].repeat(3,2)
        #     saveimg = image[:,W*2:W*3,:]
        #     savepath = '/groups/lmm2024/home/share/Sat_Pretrain_xmq/cloud_cs0502/H8_output_cs0502/test/'+args.model+'_'+id[:13]+'_pred.png'
        #     cv2.imwrite(savepath, saveimg)
        #     Task.current_task().get_logger().report_image("train", id, iteration=epoch, image=image.astype(np.uint8))


        intersection, union, target = intersectionAndUnionGPU(output, target, nclass, 10)

        if args.multiprocessing_distributed:
            dist.all_reduce(intersection), dist.all_reduce(union), dist.all_reduce(target)
        intersection, union, target = intersection.cpu().numpy(), union.cpu().numpy(), target.cpu().numpy()
        intersection_meter.update(intersection), union_meter.update(union), target_meter.update(target)

        accuracy = sum(intersection_meter.val) / (sum(target_meter.val) + 1e-10)
        loss_meter.update(loss.item(), input.size(0))
        batch_time.update(time.time() - end)
        end = time.time()
        if ((i + 1) % 100 == 0) and main_process():
            logger.info('Evaluation: [{}/{}] '
                        'Data {data_time.val:.3f} ({data_time.avg:.3f}) '
                        'Batch {batch_time.val:.3f} ({batch_time.avg:.3f}) '
                        'Loss {loss_meter.val:.3f} ({loss_meter.avg:.3f}) '
                        'Accuracy {accuracy:.2f}.'.format(i + 1, len(val_loader),
                                                        data_time=data_time,
                                                        batch_time=batch_time,
                                                        loss_meter=loss_meter,
                                                        accuracy=accuracy * 100))

    iou_class = intersection_meter.sum / (union_meter.sum + 1e-10)
    accuracy_class = intersection_meter.sum / (target_meter.sum + 1e-10)
    mIoU = np.mean(iou_class)
    mAcc = np.mean(accuracy_class)
    allAcc = sum(intersection_meter.sum) / (sum(target_meter.sum) + 1e-10)
    if main_process():
        logger.info('Val result: Loss {:.3f} | mIoU {:.2f} | mAcc {:.2f} | allAcc {:.2f}.'.format(loss_meter.avg, mIoU*100, mAcc*100, allAcc*100))
        for i in range(nclass):
            logger.info('Class_{} Result: iou {:.2f} | accuracy {:.2f}.'.format(i, iou_class[i]*100, accuracy_class[i]*100))
        logger.info('<<<<<<<<<<<<<<<<< End Evaluation <<<<<<<<<<<<<<<<<')
        logger.info('\n')
    return loss_meter.avg, mIoU, mAcc, allAcc


if __name__ == '__main__':
    args = Options().parse()
    # task = Task.init(project_name='seafog', task_name='test_all_unetformer_h8', auto_connect_arg_parser={'rank': False},reuse_last_task_id=True)  # noqa: F841

    # task.get_logger().set_default_upload_destination("s3://10.3.240.41:19000/clearml")
    # task.upload_artifact(name='data file', artifact_object=os.path.join('/home/ssdk/cloud/','*.py') )        #os.path.join('/data/xmq/cloud/code/encoding'))
    torch.multiprocessing.set_start_method('spawn',force=True)

    main()
