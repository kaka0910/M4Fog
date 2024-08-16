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
    # start a new wandb run to track this script
    # wandb.init(
    #     # set the wandb project where this run will be logged
    #     project="test", name='unet_lr2_0307'
    # )
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
    # for s2unet
    model = get_segmentation_model(args.model,criterion_seg=criterion_seg) ##aux
    # # for unet
    # model = get_segmentation_model(args.model, dataset = args.dataset, 
    #                                norm_layer = nn.BatchNorm2d,
    #                                criterion_seg=criterion_seg) ##aux

    if main_process():
        global logger, writer
        logger = get_logger()
        writer = SummaryWriter(args.vis_root) ##tensorboard --logdir vis_root
        logger.info(args)
        logger.info("=> creating model ...")
        logger.info("Classes: {}".format(model.nclass))
        logger.info(model)


    if args.model_optimizer =='SGD':
        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr,
            momentum=args.momentum, weight_decay=args.weight_decay)
    elif args.model_optimizer =='AdamW':
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
    train_data = get_segmentation_dataset(args.dataset, root=args.data_root, split='train_all')
    val_data = get_segmentation_dataset(args.dataset, root=args.data_root, split='val_2021')
    print(len(train_data),len(val_data))

    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_data)
        val_sampler = torch.utils.data.distributed.DistributedSampler(val_data)
    else:
        train_sampler = None
        val_sampler = None

    train_loader = torch.utils.data.DataLoader(train_data, batch_size=args.batch_size, 
                                               shuffle=(train_sampler is None), num_workers=0, 
                                               pin_memory=True, sampler=train_sampler, drop_last=True)
    val_loader = torch.utils.data.DataLoader(val_data, batch_size=args.batch_size, 
                                             shuffle=False, num_workers=0, 
                                             pin_memory=True, sampler=val_sampler)
    print(len(train_loader),len(val_loader))

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max = args.epochs, eta_min = 1e-6)
    # scheduler = LR_Scheduler(args.lr_scheduler, args.lr, args.epochs, len(train_loader), warmup_epochs=10)
    currentSteps=[0]
    for epoch in range(args.start_epoch, args.epochs):
        epoch_log = epoch + 1
        if args.distributed:
            train_sampler.set_epoch(epoch)
        if main_process():
            logger.info('>>>>>>>>>>>>>>>> Start One Epoch Training >>>>>>>>>>>>>>>>')
        loss_train_0, loss_train_1, mIoU_train, mAcc_train, allAcc_train = train(train_loader, model, scaler, optimizer, epoch, scheduler, train_data.NUM_CLASS,currentSteps,wandbHandle=None)
        
        loss_val_0, loss_val_1, mIoU_val, mAcc_val, allAcc_val = validate(val_loader, model, criterion_seg, train_data.NUM_CLASS,epoch=epoch)
        
        if main_process():
            writer.add_scalar('loss_val', (loss_val_0+loss_val_1)/2, epoch_log)
            writer.add_scalar('mIoU_val', mIoU_val, epoch_log)
            writer.add_scalar('mAcc_val', mAcc_val, epoch_log)
            writer.add_scalar('allAcc_val', allAcc_val, epoch_log)
        
        if main_process() and mIoU_val >= best_mIoU:
            best_mIoU = mIoU_val
            filename = args.model_savefolder + args.best_name
            logger.info('Saving checkpoint to: ' + filename)
            logger.info('\n')
            torch.save({'epoch': epoch_log, 
                    'best_mIoU': best_mIoU * 100,
                    'state_dict': model.module.state_dict(), 
                    'optimizer': optimizer.state_dict()}, 
                    filename)
    
    if main_process():
        filename = args.model_savefolder + 'last_model_0502.pth'
        torch.save({'epoch': epoch_log, 
                    'best_mIoU': mIoU_val * 100,
                    'state_dict': model.module.state_dict(), 
                    'optimizer': optimizer.state_dict()}, 
                    filename)
        # wandb.finish()

def train(train_loader, model, scaler, optimizer, epoch, scheduler, nclass,currentSteps,wandbHandle=None):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    main_loss_meter = AverageMeter()  ##
    aux_loss_meter = [AverageMeter(),AverageMeter()] ##
    loss_meter = [AverageMeter(),AverageMeter()]
    intersection_meter = [AverageMeter(),AverageMeter()]
    union_meter = [AverageMeter(),AverageMeter()]
    target_meter = [AverageMeter(),AverageMeter()]

    model.train()
    end = time.time()
    max_iter = args.epochs * len(train_loader)
        
    # 每一个batch里面
    for i, (input, target) in enumerate(train_loader):
        currentSteps[0]+=1
        data_time.update(time.time() - end)
        
        # scheduler(optimizer, i, epoch)
        input = input.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)
        # target =torch.unsqueeze(target, dim=1) 
        optimizer.zero_grad()
        # with autocast():
        with autocast():
            if args.aux: ##false
                output1, output2, main_loss, jpu_loss, aux_loss = model(input, target)
                loss = main_loss + 0.1 * jpu_loss + args.aux_weight * aux_loss
            else: 
                output1, output2, main_loss = model(input, target) ##jpu
                loss = main_loss #+ 0.25 * 0.4 * jpu_loss

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        # scheduler.step()  # batch内更新lr
        scaler.update()

        # loss.backward()
        # optimizer.step()

        n = input.size(0)

        if args.multiprocessing_distributed:
            if args.aux: ##false
                main_loss, aux_loss, jpu_loss, loss = main_loss.detach() * n, jpu_loss.detach() * n, aux_loss.detach() * n, loss.detach() * n  # not considering ignore pixels
                count = target.new_tensor([n], dtype=torch.long)
                dist.all_reduce(main_loss), dist.all_reduce(aux_loss), dist.all_reduce(jpu_loss), dist.all_reduce(loss), dist.all_reduce(count)
                n = count.item()
                main_loss, aux_loss, jpu_loss, loss = main_loss / n, aux_loss / n, jpu_loss / n, loss / n
            else:
                loss = loss.detach() * n  # not considering ignore pixels ## loss
                count = target.new_tensor([n], dtype=torch.long)
                dist.all_reduce(loss), dist.all_reduce(count)
                n = count.item()
                loss = loss / n

        for number in range(2):
            # import pdb;
            # pdb.set_trace()
            if number==0:
                output = output1
            else:
                output = output2

            intersection, union, target_y = intersectionAndUnionGPU(output, target[:,number,:,:].clone(), nclass, 10)
        
            if args.multiprocessing_distributed:
                dist.all_reduce(intersection), dist.all_reduce(union), dist.all_reduce(target_y)
            
            intersection, union, target_y = intersection.cpu().numpy(), union.cpu().numpy(), target_y.cpu().numpy()
            intersection_meter[number].update(intersection), union_meter[number].update(union), target_meter[number].update(target_y)

            accuracy = sum(intersection_meter[number].val) / (sum(target_meter[number].val) + 1e-10)

            if args.aux:
                aux_loss_meter.update(aux_loss.item(), n)
            loss_meter[number].update(loss.item(), n) ##
            batch_time.update(time.time() - end)
            end = time.time()

            current_iter = epoch * len(train_loader) + i + 1
            lr = optimizer.param_groups[0]['lr']
            remain_iter = max_iter - current_iter
            remain_time = remain_iter * batch_time.avg
            t_m, t_s = divmod(remain_time, 60)
            t_h, t_m = divmod(t_m, 60)
            remain_time = '{:02d}:{:02d}:{:02d}'.format(int(t_h), int(t_m), int(t_s))

            if (i + 1) % 10 == 0 and main_process():
                logger.info('Epoch:[{}/{}][{}/{}] '
                            'Data {data_time.val:.3f} ({data_time.avg:.3f}) '
                            'Batch {batch_time.val:.3f} ({batch_time.avg:.3f}) '
                            'LR:{lr:.6f} '
                            'Remain:{remain_time} '
                            'Loss:{loss_meter.val:.3f}({loss_meter.avg:.3f}) '
                            'Accuracy:{accuracy:.2f}'.format(epoch+1, args.epochs,
                                                            i + 1, len(train_loader),
                                                            batch_time=batch_time,
                                                            data_time=data_time,
                                                            lr=lr,
                                                            remain_time=remain_time,
                                                            loss_meter=loss_meter[number],
                                                            accuracy=accuracy * 100))
        # if main_process():
        #     writer.add_scalar('loss_train_batch', loss_meter.val, currentSteps[0]) ##
        #     writer.add_scalar('mIoU_train_batch', np.mean(intersection / (union + 1e-10)), currentSteps[0])
        #     writer.add_scalar('mAcc_train_batch', np.mean(intersection / (target_y + 1e-10)), currentSteps[0])
        #     writer.add_scalar('allAcc_train_batch', accuracy, currentSteps[0])
        #     writer.add_scalar('Lr', lr, currentSteps[0])

    scheduler.step()  # epoch内不更新lr，epoch之间更新lr

    # Image 0 
    iou_class_0 = intersection_meter[0].sum / (union_meter[0].sum + 1e-10)
    accuracy_class_0 = intersection_meter[0].sum / (target_meter[0].sum + 1e-10)
    mIoU_0 = np.mean(iou_class_0)
    mAcc_0 = np.mean(accuracy_class_0)
    allAcc_0 = sum(intersection_meter[0].sum) / (sum(target_meter[0].sum) + 1e-10)
    
    # Image 1
    iou_class_1 = intersection_meter[1].sum / (union_meter[1].sum + 1e-10)
    accuracy_class_1 = intersection_meter[1].sum / (target_meter[1].sum + 1e-10)
    mIoU_1 = np.mean(iou_class_1)
    mAcc_1 = np.mean(accuracy_class_1)
    allAcc_1 = sum(intersection_meter[1].sum) / (sum(target_meter[1].sum) + 1e-10)
    
    
    if main_process():
        logger.info('>>>>>>>>>>>>>>>> One Training Epoch Done Image 0 >>>>>>>>>>>>>>>>')
        logger.info('Train epoch [{}/{}]: Loss {:.3f} | mIoU {:.2f} | mAcc {:.2f} | allAcc {:.2f} '.format(epoch+1, args.epochs, loss_meter[0].avg, mIoU_0 * 100, mAcc_0 * 100, allAcc_0 * 100))
        for i in range(nclass):
            logger.info('Class_{} Result: iou {:.2f} | accuracy {:.2f}.'.format(i, iou_class_0[i]*100, accuracy_class_0[i]*100))
        logger.info('>>>>>>>>>>>>>>>> One Training Epoch Done Image 1 >>>>>>>>>>>>>>>>')
        logger.info('Train epoch [{}/{}]: Loss {:.3f} | mIoU {:.2f} | mAcc {:.2f} | allAcc {:.2f} '.format(epoch+1, args.epochs, loss_meter[1].avg, mIoU_1 * 100, mAcc_1 * 100, allAcc_1 * 100))
        for i in range(nclass):
            logger.info('Class_{} Result: iou {:.2f} | accuracy {:.2f}.'.format(i, iou_class_1[i]*100, accuracy_class_1[i]*100))
        logger.info('>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>')
        logger.info('\n')
    
    return loss_meter[0].avg, loss_meter[1].avg, (mIoU_0+mIoU_1)/2, (mAcc_0+mAcc_1)/2, (allAcc_0+allAcc_1)/2

def validate(val_loader, model, criterion, nclass,epoch):
    if main_process():
        logger.info('>>>>>>>>>>>>>>>> Start Evaluation >>>>>>>>>>>>>>>>')
    batch_time = AverageMeter()
    data_time = AverageMeter()
    loss_meter = [AverageMeter(),AverageMeter()]
    intersection_meter = [AverageMeter(),AverageMeter()]
    union_meter = [AverageMeter(),AverageMeter()]
    target_meter = [AverageMeter(),AverageMeter()]

    model.eval()
    end = time.time()
    for i, (input, target) in enumerate(val_loader):
        data_time.update(time.time() - end)
        input = input.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)
        # target = (target.cuda(non_blocking=True),False)
        with torch.no_grad():
            output1, output2 = model(input,target)
        
        # import pdb;
        # pdb.set_trace()
        
        for number in range(2):
            if number==0:
                output=output1
            else:
                output=output2
            
            loss = criterion(output, target[:,number,:,:].clone())

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

            intersection, union, target_y = intersectionAndUnionGPU(output, target[:,number,:,:].clone(), nclass, 10)

            if args.multiprocessing_distributed:
                dist.all_reduce(intersection), dist.all_reduce(union), dist.all_reduce(target_y)
            intersection, union, target_y = intersection.cpu().numpy(), union.cpu().numpy(), target_y.cpu().numpy()
            intersection_meter[number].update(intersection), union_meter[number].update(union), target_meter[number].update(target_y)

            accuracy = sum(intersection_meter[number].val) / (sum(target_meter[number].val) + 1e-10)
            loss_meter[number].update(loss.item(), input.size(0))
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
                                                            loss_meter=loss_meter[number],
                                                            accuracy=accuracy * 100))

    # Image 0 
    iou_class_0 = intersection_meter[0].sum / (union_meter[0].sum + 1e-10)
    accuracy_class_0 = intersection_meter[0].sum / (target_meter[0].sum + 1e-10)
    mIoU_0 = np.mean(iou_class_0)
    mAcc_0 = np.mean(accuracy_class_0)
    allAcc_0 = sum(intersection_meter[0].sum) / (sum(target_meter[0].sum) + 1e-10)
    
    # Image 1
    iou_class_1 = intersection_meter[1].sum / (union_meter[1].sum + 1e-10)
    accuracy_class_1 = intersection_meter[1].sum / (target_meter[1].sum + 1e-10)
    mIoU_1 = np.mean(iou_class_1)
    mAcc_1 = np.mean(accuracy_class_1)
    allAcc_1 = sum(intersection_meter[1].sum) / (sum(target_meter[1].sum) + 1e-10)
    
    if main_process():
        # Image 0 
        logger.info('Val result: Loss {:.3f} | mIoU {:.2f} | mAcc {:.2f} | allAcc {:.2f}.'.format(loss_meter[0].avg, mIoU_0*100, mAcc_0*100, allAcc_0*100))
        for i in range(nclass):
            logger.info('Class_{} Result: iou {:.2f} | accuracy {:.2f}.'.format(i, iou_class_0[i]*100, accuracy_class_0[i]*100))
        # Image 1
        logger.info('Val result: Loss {:.3f} | mIoU {:.2f} | mAcc {:.2f} | allAcc {:.2f}.'.format(loss_meter[1].avg, mIoU_1*100, mAcc_1*100, allAcc_1*100))
        for i in range(nclass):
            logger.info('Class_{} Result: iou {:.2f} | accuracy {:.2f}.'.format(i, iou_class_1[i]*100, accuracy_class_1[i]*100))
        logger.info('<<<<<<<<<<<<<<<<< End Evaluation <<<<<<<<<<<<<<<<<')
        logger.info('\n')
    
    return loss_meter[0].avg, loss_meter[1].avg, (mIoU_0+mIoU_1)/2, (mAcc_0+mAcc_1)/2, (allAcc_0+allAcc_1)/2


if __name__ == '__main__':
    args = Options().parse()
    task = Task.init(project_name='seafog', task_name='08unetformer-aug04-heb-allchan', auto_connect_arg_parser={'rank': False})  # noqa: F841

    task.get_logger().set_default_upload_destination("s3://10.3.240.41:19000/clearml")
    task.upload_artifact(name='data file', artifact_object=os.path.join('/home/ssdk/cloud/','*.py') )        #os.path.join('/data/xmq/cloud/code/encoding'))
    torch.multiprocessing.set_start_method('spawn',force=True)

    main()
