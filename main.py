import argparse
import os
import time
import shutil
import pickle
import numpy as np
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torch.utils.data as data
from dataloader import Segmentation_data, Classification_Data, check_epoch, load_kaggle_data, MTL_data
from utils import Logger, AverageMeter, save_checkpoint ,draw_curve ,str2bool, send_slack_message, History, Performance
from model import *
from losses import DiceLoss,tversky_loss, NLL_OHEM
from optimizers import RAdam
from torchvision import models
from multiprocessing import Process
from torch.utils.tensorboard import SummaryWriter
from efficientnet_pytorch import EfficientNet
from sampler import ImbalancedDatasetSampler
import pandas as pd

# Multi-Task-Learning Baseline
# Based On * A Feature Transfer Enabled Multi-Task Deep Learning Model on Medical Imaging *
# Fei Gao, Hyunsoo Yoon, Teresa Wu, Xianghua Chu

parser = argparse.ArgumentParser()

parser.add_argument('--trn-root', default=['/data2/sk_data/data_1rd/trainvalid_3d',
                                           '/data2/sk_data/data_2rd/trainvalid_3d',
                                           '/data2/sk_data/data_3rd/trainvalid_3d',
                                           '/data2/sk_data/data_4rd/trainvalid_3d',
                                           '/data2/sk_data/data_5rd/trainvalid_3d',
                                          ],nargs='+', type=str)

parser.add_argument('--work-dir', default='/data1/workspace/geongyu')
parser.add_argument('--exp',default="test4", type=str)
parser.add_argument('--data-mode', default='all', type=str) # label : use label data  / all : use all data

parser.add_argument('--batch-size', default=16, type=int)
parser.add_argument('--lr-schedule', default=[20,40], nargs='+', type=int)
parser.add_argument('--initial-lr', default=0.1, type=float)
parser.add_argument('--weight-decay', default=0.0001, type=float)
parser.add_argument('--loss-function', default='bce', type=str) # 'bce', 'dice'
parser.add_argument('--optim-function', default='sgd', type=str)
parser.add_argument('--momentum',default=0.9, type=float)
parser.add_argument('--bce-weight', default=1, type=int)
parser.add_argument('--num-workers', default=12, type=int)
parser.add_argument('--padding-size', default=1, type=int)
parser.add_argument('--batchnorm-momentum', default=0.1, type=float)
parser.add_argument('--arch', default='unet', type=str)

# arguments for test mode
parser.add_argument('--test-root', default=['/data2/sk_data/data_1rd/test_3d',
                                            '/data2/sk_data/data_2rd/test_3d',
                                            '/data2/sk_data/data_3rd/test_3d',
                                            '/data2/sk_data/data_4rd/test_3d',
                                            '/data2/sk_data/data_5rd/test_3d'],
                    nargs='+', type=str)
parser.add_argument('--file-name', default='result_train_s_test_s', type=str)

# arguments for slack
parser.add_argument("--kaggle", default=False, type=bool, help="If TRUE == load_kaggle_data_with_balanced, Else FALSE == Classification_Data with WeightedRandomSampler")
parser.add_argument('--tenosrboardwriter', default="Test", type=str)
parser.add_argument('--number', default="0", type=str)
parser.add_argument('--aug', default="False", type=str)
parser.add_argument('--smooth', default="False", type=str)

args = parser.parse_args()

writer_train = SummaryWriter(log_dir='tensor_log/' + args.tenosrboardwriter + '/train', comment="SK")
writer_valid = SummaryWriter(log_dir='tensor_log/' + args.tenosrboardwriter + '/vaild', comment="SK")
writer_image = SummaryWriter(log_dir='tensor_log/' + args.tenosrboardwriter + '/image', comment='SK')


def main():
    print(args.work_dir, args.exp)
    work_dir = os.path.join(args.work_dir, args.exp)

    kaggle_path = "/data2/sk_data/kaggle_data/stage_1_train_images_png"
    kaggle_csv_path = "/data2/sk_data/kaggle_data/bin_dataframe.csv"
    label_data = pd.read_csv(kaggle_csv_path)

    if not os.path.exists(work_dir):
        os.makedirs(work_dir)

    # copy this file to work dir to keep training configuration
    shutil.copy(__file__, os.path.join(work_dir, 'main.py'))
    with open(os.path.join(work_dir, 'args.pkl'), 'wb') as f:
        pickle.dump(args, f)

    # 1.dataset
    train_filename = args.trn_root
    test_filename = args.test_root

    trainset = MTL_data(train_filename)
    valiset = MTL_data(test_filename)

    train_loader = data.DataLoader(trainset, batch_size=args.batch_size, num_workers=args.num_workers)
    valid_loader = data.DataLoader(valiset, batch_size=args.batch_size, num_workers=args.num_workers)

    trn_logger = Logger(os.path.join(work_dir, 'train.log'))
    trn_raw_logger = Logger(os.path.join(work_dir, 'train_raw.log'))
    val_logger = Logger(os.path.join(work_dir, 'validation.log'))

    if args.arch == 'unet':
        net = Unet2D(in_shape=(args.multi_input, 512, 512), padding=args.padding_size, momentum=args.batchnorm_momentum)
    elif args.arch == 'unetcoord':
        print('radious', args.radious, type(args.radious))
        net = Unetcoordconv(in_shape=(args.multi_input, 512, 512), padding=args.padding_size,
                            momentum=args.batchnorm_momentum, coordnumber=args.coordconv_no, radius=args.radious)
    elif args.arch == 'UnetSkipConnection':
        net = UnetSkipConnection(in_shape=(args.multi_input, 512, 512), padding=args.padding_size,
                                 momentum=args.batchnorm_momentum, calculate=args.cal_mode)
    elif args.arch == 'deeplab':
        net = DeepLabv3_plus(nInputChannels=args.multi_input, n_classes=1, os=16, pretrained=True, _print=True,
                             momentum=args.batchnorm_momentum)
    elif args.arch == 'unetmultiinput':
        net = Unet2D_multiinput(in_shape=(args.multi_input, 512, 512), padding=args.padding_size,
                                momentum=args.batchnorm_momentum)
    elif args.arch == 'scse_block':
        net = Unet_sae(in_shape=(args.multi_input, 512, 512), padding=args.padding_size,
                       momentum=args.batchnorm_momentum, coordconv=True)
    else:
        raise ValueError('Not supported network.')

    # loss
    if args.loss_function == 'bce':
        criterion = nn.BCEWithLogitsLoss(pos_weight=torch.Tensor([args.bce_weight])).cuda()
    elif args.loss_function == 'dice':
        criterion = DiceLoss().cuda()
    elif args.loss_function == 'tversky':
        criterion = tversky_loss(alpha=args.tversky_alpha, beta=args.tversky_beta).cuda()
    elif args.loss_function == 'NLL_OHEM':
        criterion = NLL_OHEM(ratio=3, loss='bce').cuda()
    else:
        raise ValueError('{} loss is not supported yet.'.format(args.loss_function))

    # optim
    if args.optim_function == 'sgd':
        optimizer = torch.optim.SGD(net.parameters(), lr=args.initial_lr, momentum=args.momentum,
                                weight_decay=args.weight_decay)
    elif args.optim_function == 'adam':
        optimizer = torch.optim.Adam(net.parameters(), lr=args.initial_lr, weight_decay=args.weight_decay)
    elif args.optim_function == 'radam':
        optimizer = RAdam(net.parameters(), lr=args.initial_lr, weight_decay = args.weight_decay)
    else:
        raise ValueError('{} loss is not supported yet.'.format(args.optim_function))

    net = nn.DataParallel(net).cuda()

    cudnn.benchmark = True

    lr_schedule = args.lr_schedule
    lr_scheduler = optim.lr_scheduler.MultiStepLR(optimizer,
                                                  milestones=lr_schedule[:-1],
                                                  gamma=0.1)
    best_iou = 0

    try:
        for epoch in range(lr_schedule[-1]):
            main_train(train_loader, net, criterion, optimizer, epoch, trn_logger, trn_raw_logger)

            iou = validate(valid_loader, net, criterion_cls, criterion_seg, epoch, val_logger)
            lr_scheduler.step()

            is_best = iou > best_iou
            best_iou = max(iou, best_iou)
            checkpoint_filename = 'model_checkpoint_{:0>3}.pth'.format(epoch + 1)
            save_checkpoint({'epoch': epoch + 1,
                             'state_dict': net.state_dict(),
                             'optimizer': optimizer.state_dict()},
                            is_best,
                            work_dir,
                            checkpoint_filename)
    except RuntimeError as e:
        print(e)
        import ipdb
        ipdb.set_trace()

    draw_curve(work_dir, trn_logger, val_logger)

def check_logger(output, target):
    correct = ((output - target) == 0).float()
    return correct


def segmentation_train(trn_loader, model, criterion, optimizer, epoch, logger, sublogger,
                   work_dir=os.path.join(args.work_dir, args.exp)):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    ious = AverageMeter()
    dices = AverageMeter()

    model.train()

    end = time.time()
    slice_level_acc = 0
    total_data_counts = 0
    lr_schedule = args.lr_schedule
    train_history = History(len(trn_loader.dataset))
    length_data = 0

    for i, (input, target) in enumerate(trn_loader):
        data_time.update(time.time() - end)

        input = input.cuda()
        target = target.cuda()

        output = model(input)
        loss = criterion(output_seg, target)

        # Segmentation Measure
        pos_probs = torch.sigmoid(output_seg)
        pos_preds = (pos_probs > 0.5).float()

        p = 0
        k = 0
        correct_per_pixel = []
        for dap, predict in zip(target, pos_preds):
            if dap.max() == 1 and predict.max() == 1:
                p += 1
                k += 1
            elif dap.max() == 0 and predict.max() == 0:
                p += 1
                k += 1
            else:
                k += 1
        slice_level_acc += p
        total_data_counts += k

        iou, dice = performance(output, target)
        losses.update(loss.item(), input.size(0))
        ious.update(iou, input.size(0))
        dices.update(dice, input.size(0))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        batch_time.update(time.time() - end)
        end = time.time()

        pos_probs = torch.sigmoid(output)
        pos_preds = (pos_probs > 0.5).float()

        print('Training Epoch: [{0}][{1}/{2}]\t'
              'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
              'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
              'Loss {loss:.4f}({loss.avg:.4f})\t'
              'IoU {iou.val:.4f} ({iou.avg:.4f})\t'
              'Dice {dice.val:.4f} ({dice.avg:.4f})\t'
              'Slice Level ACC {slice:4f}\t'.format(
            epoch, i, len(trn_loader), batch_time=batch_time, data_time=data_time, loss=losses.item(),
            iou=ious, dice=dices, slice=slice_level_acc / total_data_counts))

        if i % 10 == 0:
            sublogger.write([epoch, i, loss.item(), iou, dice])

    slice_acc = slice_level_acc / total_data_counts

    writer_train.add_scalar('Loss', losses.avg, epoch)
    writer_train.add_scalar('IoU', ious.avg, epoch)
    writer_train.add_scalar('Dice Score', dices.avg, epoch)
    writer_train.add_scalar('Slice-Level-Accuracy', slice_level_acc / total_data_counts, epoch)

    logger.write([epoch, losses.avg, ious.avg, dices.avg, slice_acc])

def performance(output, target):
    pos_probs = torch.sigmoid(output)
    pos_preds = (pos_probs > 0.5).float()
    pos_preds = pos_preds.cpu().numpy().squeeze()
    target = target.cpu().numpy().squeeze()
    if target.sum() == 0:  # background patch
        return 0, 0
    # IoU
    union = ((pos_preds + target) != 0).sum()
    intersection = (pos_preds * target).sum()
    iou = intersection / union
    # dice
    dice = (2 * intersection) / (pos_preds.sum() + target.sum())
    return iou, dice

def validate(val_loader, model, criterion, epoch, logger):
    batch_time = AverageMeter()
    losses = AverageMeter()
    ious = AverageMeter()
    dices = AverageMeter()

    model.eval()
    length_data = 0

    with torch.no_grad():
        end = time.time()
        slice_level_acc = 0
        total_data_counts = 0
        for i, (input, target) in enumerate(val_loader):
            input = input.cuda()
            target = target.cuda()

            output  = model(input)
            loss = criterion(output, target)
            iou, dice = performance(output_seg, target)

            losses.update(loss.item(), input.size(0))
            ious.update(iou, input.size(0))
            dices.update(dice, input.size(0))

            batch_time.update(time.time() - end)
            end = time.time()

            pos_probs = torch.sigmoid(output)
            pos_preds = (pos_probs > 0.5).float()

            p = 0
            k = 0
            correct_per_pixel = []
            for dap, predict in zip(target, pos_preds):
                if dap.max() == 1 and predict.max() == 1:
                    p += 1
                    k += 1
                elif dap.max() == 0 and predict.max() == 0:
                    p += 1
                    k += 1
                else:
                    k += 1
            slice_level_acc += p
            total_data_counts += k
            print('Validation Epoch: [{0}][{1}/{2}]\t'
              'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
              'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
              'IoU {iou.val:.4f} ({iou.avg:.4f})\t'
              'Dice {dice.val:.4f} ({dice.avg:.4f})\t'
              'Slice Level ACC {slice:4f}\t'.format(
            epoch, i, len(val_loader), batch_time=batch_time, loss=losses, iou=ious, dice=dices, slice=slice_level_acc / total_data_counts))

    print(' * IoU {ious.avg:.3f}({ious.std:.3f}) Dice {dices.avg:.3f}({dices.std:.3f} )'.format(
        ious=ious, dices=dices))
    slice_acc = slice_level_acc / total_data_counts
    logger.write([epoch, losses.avg, ious.avg, dices.avg, slice_acc])

    writer_valid.add_scalar('Loss', losses.avg, epoch)
    writer_valid.add_scalar('IoU', ious.avg, epoch)
    writer_valid.add_scalar('Dice Score', dices.avg, epoch)
    writer_valid.add_scalar('Slice-Level-Accuracy', slice_level_acc / total_data_counts, epoch)

    return ious.avg

if __name__=='__main__':
    main()