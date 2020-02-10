
import argparse
import os
import time
import shutil
import pickle
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn

from dataset import DatasetTrain, DatasetVal

from utils import Logger, AverageMeter, save_checkpoint ,draw_curve


from utils import Logger, AverageMeter, save_checkpoint
from model import UNet3D
from losses import DiceLoss
from predict import main_test
import random

parser = argparse.ArgumentParser()





# arguments for training
parser.add_argument('--trn-root', default='/data1/JM/sk_project/data2th_trainvalid_3d_patches_48_48_48_st_16_bg_0.1_nonzero_0.1')
parser.add_argument('--tst-root', default='/data1/JM/sk_project/data2th_test_3d_patches_48_48_48_st_16_bg_1_nonzero_0.1')
parser.add_argument('--work-dir', default='/data1/JM/sk_project/Segmentation-3D')
parser.add_argument('--exp', type=str)



parser.add_argument('--batch-size', default=64, type=int)
parser.add_argument('--input-size', default=[48,48,48], nargs='+', type=int)
parser.add_argument('--lr-schedule', default=[20,30,35], nargs='+', type=int)
parser.add_argument('--weight-decay', default=0.0005, type=float)


parser.add_argument('--loss-function', default='bce', type=str) # 'bce', 'dice' ,'weight_bce'
parser.add_argument('--bce-weight', default=1, type=int)
parser.add_argument('--lr', default=0.1, type=float)
parser.add_argument('--momentum', default=0.8, type=float)
parser.add_argument('--gamma', default=0.1, type=float)
parser.add_argument('--num-workers', default=8, type=int)

parser.add_argument('--optim', default='sgd', type=str)


# arguments for model
parser.add_argument('--model', default='unet', type=str)
parser.add_argument('--f-maps', default=[32, 64, 128, 256], nargs='+', type=int)

parser.add_argument('--conv-layer-order', default='cbr', type=str)
parser.add_argument('--num-groups', default=1, type=int)
parser.add_argument('--depth-stride', default=[2, 2, 2, 2], nargs='+', type=int)


# arguments for test mode
parser.add_argument('--test-root', default=['/data2/woans0104/sk_hemorrhage_dataset/data_1rd',
                                            '/data2/woans0104/sk_hemorrhage_dataset/data_2rd',
                                            ], nargs='+', type=str)
parser.add_argument('--stride-test', default=None, nargs='+', type=int) #default=[1,16,16])
parser.add_argument('--target-depth-for-padding', default=None, type=int)
parser.add_argument('--inplace-test', default=1, type=int)
parser.add_argument('--file-name', default='result_all', type=str)

# arguments for slack
parser.add_argument('--token',type=str)


args = parser.parse_args()

def main():


    work_dir = os.path.join(args.work_dir, args.exp)
    if not os.path.exists(work_dir):
        os.makedirs(work_dir)

    # copy this file to work dir to keep training configuration
    shutil.copy(__file__, os.path.join(work_dir, 'main.py'))
    with open(os.path.join(work_dir, 'args.pkl'), 'wb') as f:
        pickle.dump(args, f)


    #train 0.8 val 0.2
    image_root = os.path.join(args.trn_root, 'images')
    exam_ids = os.listdir(image_root)
    random.shuffle(exam_ids)


    train_exam_ids = exam_ids[:int(len(exam_ids)*0.8)]
    val_exam_ids = exam_ids[int(len(exam_ids) * 0.8):]


    # train_dataset
    trn_dataset = DatasetTrain(args.trn_root,train_exam_ids, options=args, input_stats=[0.5, 0.5])
    trn_loader = torch.utils.data.DataLoader(trn_dataset,
                                             batch_size=args.batch_size,
                                             shuffle=True,
                                             num_workers=args.num_workers)

    # save input stats for later use
    np.save(os.path.join(work_dir, 'input_stats.npy'), trn_dataset.input_stats)

    # val_dataset
    val_dataset = DatasetVal(args.tst_root,val_exam_ids, options=args, input_stats=trn_dataset.input_stats)
    val_loader = torch.utils.data.DataLoader(val_dataset,
                                            batch_size=args.batch_size,
                                             shuffle=True,
                                             num_workers=args.num_workers)


    # make logger
    trn_logger = Logger(os.path.join(work_dir, 'train.log'))
    trn_raw_logger = Logger(os.path.join(work_dir, 'train_raw.log'))
    val_logger = Logger(os.path.join(work_dir, 'validation.log'))


    # model_select
    if args.model == 'unet':
        net = UNet3D(1, 1, f_maps=args.f_maps, depth_stride=args.depth_stride,
                    conv_layer_order=args.conv_layer_order,
                    num_groups=args.num_groups)

    else:
        raise ValueError('Not supported network.')




    # loss_select
    if args.loss_function == 'bce':
        criterion = nn.BCEWithLogitsLoss(pos_weight=torch.Tensor([args.bce_weight])).cuda()
    elif args.loss_function == 'dice':
        criterion = DiceLoss().cuda()
    elif args.loss_function =='weight_bce':
        criterion = nn.BCEWithLogitsLoss(pos_weight=torch.FloatTensor([5])).cuda()
    else:
        raise ValueError('{} loss is not supported yet.'.format(args.loss_function))

    # optim_select
    if args.optim == 'sgd':
        optimizer = optim.SGD(net.parameters(),
                          lr=args.lr,
                          momentum=args.momentum,
                          weight_decay=args.weight_decay,nesterov=False)

    elif args.optim == 'adam':
        optimizer = optim.Adam(net.parameters(),
                           lr=args.lr,
                           weight_decay=args.weight_decay)
    else:
        raise ValueError('{} optim is not supported yet.'.format(args.optim))


    net = nn.DataParallel(net).cuda()
    cudnn.benchmark = True


    lr_schedule = args.lr_schedule
    lr_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=lr_schedule[:-1], gamma=0.1)

    best_iou = 0
    for epoch in range(lr_schedule[-1]):

        train(trn_loader, net, criterion, optimizer, epoch, trn_logger, trn_raw_logger)
        iou = validate(val_loader, net, criterion, epoch, val_logger)


        lr_scheduler.step()

        # save model parameter
        is_best = iou > best_iou
        best_iou = max(iou, best_iou)
        checkpoint_filename = 'model_checkpoint_{:0>3}.pth'.format(epoch+1)
        save_checkpoint({'epoch': epoch+1,
                         'state_dict': net.state_dict(),
                         'optimizer': optimizer.state_dict()},
                        is_best,
                        work_dir,
                        checkpoint_filename)


    # visualize curve
    draw_curve(work_dir, trn_logger, val_logger)


    if args.inplace_test:
        # calc overall performance and save figures
        print('Test mode ...')
        main_test(model=net, args=args)


def train(trn_loader, model, criterion, optimizer, epoch, logger, sublogger):

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    ious = AverageMeter()
    dices = AverageMeter()

    model.train()

    end = time.time()
    for i, (input, target) in enumerate(trn_loader):

        data_time.update(time.time() - end)

        input = input.cuda()
        target = target.cuda()

        output = model(input)
        loss = criterion(output, target).cuda()

        iou, dice = performance(output, target)

        losses.update(loss.item(), input.size(0))
        ious.update(iou, input.size(0))
        dices.update(dice, input.size(0))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        batch_time.update(time.time() - end)
        end = time.time()

        print('Epoch: [{0}][{1}/{2}]\t'
              'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
              'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
              'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
              'IoU {iou.val:.4f} ({iou.avg:.4f})\t'
              'Dice {dice.val:.4f} ({dice.avg:.4f})\t'.format(
               epoch, i, len(trn_loader), batch_time=batch_time,
               data_time=data_time, loss=losses,
               iou=ious, dice=dices))

        if i % 10 == 0:
            sublogger.write([epoch, i, loss.item(), iou, dice])

    logger.write([epoch, losses.avg, ious.avg, dices.avg])


def validate(val_loader, model, criterion, epoch, logger):

    batch_time = AverageMeter()
    losses = AverageMeter()
    ious = AverageMeter()
    dices = AverageMeter()

    model.eval()

    with torch.no_grad():
        end = time.time()
        for i, (input, target) in enumerate(val_loader):
            input = input.cuda()
            target = target.cuda()

            output = model(input)
            loss = criterion(output, target).cuda()

            iou, dice = performance(output, target)

            losses.update(loss.item(), input.size(0))
            ious.update(iou, input.size(0))
            dices.update(dice, input.size(0))

            batch_time.update(time.time() - end)
            end = time.time()

    print(' * IoU {ious.avg:.3f}({ious.std:.3f}) Dice {dices.avg:.3f}({dices.std:.3f})'.format(
           ious=ious, dices=dices))

    logger.write([epoch, losses.avg, ious.avg, dices.avg])

    return ious.avg

def performance(output, target):

    pos_probs = torch.sigmoid(output)
    pos_preds = (pos_probs > 0.5).float()

    pos_preds = pos_preds.cpu().numpy().squeeze()
    target = target.cpu().numpy().squeeze()

    if target.sum() == 0: # background patch
        return None, None

    # IoU
    union = ((pos_preds + target) != 0).sum()
    intersection = (pos_preds * target).sum()
    iou = intersection / union

    # dice
    dice = (2 * intersection) / (pos_preds.sum() + target.sum())

    return iou, dice

if __name__=='__main__':
    main()
