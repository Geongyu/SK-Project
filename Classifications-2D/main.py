import argparse
import os
import time
import shutil
import pickle
import numpy as np
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torch.utils.data as data
from dataloader import Classification_Data, load_kaggle_data, load_kaggle_data_with_balanced
from utils import Logger, AverageMeter, save_checkpoint ,draw_curve ,str2bool, History, Performance
from model import *
from optimizers import RAdam
from torchvision import models
from multiprocessing import Process
from torch.utils.tensorboard import SummaryWriter
import pandas as pd 

parser = argparse.ArgumentParser()

parser.add_argument('--trn-root', default=['/data2/sk_data/data_1rd/trainvalid_3d',
                                           '/data2/sk_data/data_2rd/trainvalid_3d',
                                           '/data2/sk_data/data_3rd/trainvalid_3d',
                                           '/data2/sk_data/data_4rd/trainvalid_3d',
                                           '/data2/sk_data/data_5rd/trainvalid_3d',
                                          ],nargs='+', type=str)

parser.add_argument('--work-dir', default='/data1/workspace/geongyu/cls_problem/result_exp')
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

parser.add_argument('--arch', default='unet', type=str)

# arguments for test mode
parser.add_argument('--test-root', default=['/data2/sk_data/data_1rd/test_3d',
                                            '/data2/sk_data/data_2rd/test_3d',
                                            '/data2/sk_data/data_3rd/test_3d',
                                            '/data2/sk_data/data_4rd/test_3d',
                                            '/data2/sk_data/data_5rd/test_3d'],
                    nargs='+', type=str)
parser.add_argument('--inplace-test', default=1, type=int)
parser.add_argument('--file-name', default='result_train_s_test_s', type=str)
parser.add_argument("--kaggle", default=False, type=bool, help="If TRUE == load_kaggle_data_with_balanced, Else FALSE == Classification_Data with WeightedRandomSampler")
parser.add_argument('--tenosrboardwriter', default="Test", type=str)
parser.add_argument('--number', default="0", type=str)

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

    if args.arch == "efficientnet" :
        if args.kaggle == True :
            trainset = load_kaggle_data_with_balanced(kaggle_path, kaggle_csv_path)
            class_sample_count = np.array([len(np.where(label_data["any"]==t)[0]) for t in np.unique(label_data["any"])])
            weight = 1. / class_sample_count
            train_weights = np.array([weight[t] for t in label_data["any"]])
            train_sampler = torch.utils.data.WeightedRandomSampler(weights=train_weights,
                                 num_samples=len(train_weights))
        else : 
            trainset = Classification_Data(train_filename)
            class_sample_count = np.array([len(np.where(label_data["any"]==t)[0]) for t in np.unique(label_data["any"])])
            weight = 1. / class_sample_count
            train_weights = np.array([weight[t] for t in label_data["any"]])
            train_sampler = torch.utils.data.WeightedRandomSampler(weights=train_weights,
                                 num_samples=len(train_weights))
        valiset = Classification_Data(test_filename)

    elif args.arch == "resnet" :
        trainset = Classification_Data(train_filename)
        valiset = Classification_Data(test_filename)
    else :
        raise ValueError('Not supported network.')

    # train_history = History(len(trainset))
    if args.kaggle == False :
        train_loader = data.DataLoader(trainset, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=False, sampler = train_sampler)
    else : 
        train_loader = data.DataLoader(trainset, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=True)

    valid_loader = data.DataLoader(valiset, batch_size=args.batch_size, num_workers=args.num_workers)

    # save input stats for later use
    trn_logger = Logger(os.path.join(work_dir, 'train.log'))
    trn_raw_logger = Logger(os.path.join(work_dir, 'train_raw.log'))
    val_logger = Logger(os.path.join(work_dir, 'validation.log'))
    print(len(trainset))

    # model

    if args.arch == 'unet':
        net = Unet2D(in_shape=(args.multi_input, 512, 512), padding=args.padding_size, momentum=args.batchnorm_momentum)
    elif args.arch == 'efficientnet' :
        net = EfficientNet.from_pretrained('efficientnet-' + args.number, num_classes=1)
    elif args.arch == 'resnet' :
        net = models.resnet50(pretrained=True)
        num_ftrs = net.fc.in_fetures
        net.fc = nn.Linear(num_ftrs, 1)
        print("Load Resnet-50")
    else:
        raise ValueError('Not supported network.')

    # loss
    if args.loss_function == 'bce':
        criterion = nn.BCEWithLogitsLoss(pos_weight=torch.Tensor([args.bce_weight])).cuda()
    elif args.loss_function == "cross_entropy" :
        criterion = torch.nn.CrossEntropyLoss()
    else:
        raise ValueError('{} loss is not supported yet.'.format(args.loss_function))

    # optim
    if args.optim_function == 'sgd':
        optimizer = torch.optim.SGD(net.parameters(), lr=args.initial_lr, momentum=args.momentum,
                                    weight_decay=args.weight_decay)
    elif args.optim_function == 'adam':
        optimizer = torch.optim.Adam(net.parameters(), lr=args.initial_lr)
    elif args.optim_function == 'radam':
        optimizer = RAdam(net.parameters(), lr=args.initial_lr, weight_decay=args.weight_decay)
    else:
        raise ValueError('{} loss is not supported yet.'.format(args.optim_function))

    net = nn.DataParallel(net).cuda()
    cudnn.benchmark = True

    lr_schedule = args.lr_schedule
    lr_scheduler = optim.lr_scheduler.MultiStepLR(optimizer,
                                                  milestones=lr_schedule[:-1],
                                                  gamma=0.1)

    best_acc = 0
    for epoch in range(lr_schedule[-1]):
        train_all_data(train_loader, net, criterion, optimizer, epoch, trn_logger, sublogger=trn_raw_logger, trainset = trainset, val_loader= valid_loader, val_logger = val_logger, val_mode=True)

        print("Done")
        loss, acc = validate(valid_loader, net, criterion, epoch, val_logger)

        lr_scheduler.step()
        if best_acc == 0 :
            best_acc = acc
        else :
            best_acc = max(acc, best_acc)
        is_best = True 

        if is_best == True :
            checkpoint_filename = 'model_checkpoint_{:0>3}.pth'.format(epoch + 1)
            save_checkpoint({'epoch': epoch + 1,
                                'state_dict': net.state_dict(),
                                'optimizer': optimizer.state_dict()},
                            is_best,
                            work_dir,
                            checkpoint_filename)

    draw_curve(work_dir, trn_logger, val_logger)

def train_all_data(trn_loader, model, criterion, optimizer, epoch, logger, sublogger, trainset, history=None, work_dir= os.path.join(args.work_dir, args.exp), val_loader=False, val_logger=False, val_mode=False):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()

    model.train()

    end = time.time()
    accuracy = 0
    length_data = 0
    #train_history = History(len(trn_loader.dataset))
    if args.kaggle == False :
        trainset.give_the_epoch(epoch)
    iterate = 0
    for i, (input, target) in enumerate(trn_loader):
        data_time.update(time.time() - end)
        input = input.cuda()
        target = target.cuda()
        #re_labe = re_labe.cuda()

        output = model(input)
        loss = criterion(output, target)

        pos_probs = torch.sigmoid(output)
        pos_preds = (pos_probs > 0.5).float()
        #sh = re_labe.shape[0]
        #re_labe = re_labe.reshape(sh, 1)
        counts = pos_preds - target
        counts = sum(np.abs(counts.cpu()))
        acc = (len(input) - counts) / len(input)
        accuracy += (len(input) - counts)

        losses.update(loss.item(), input.size(0))
        optimizer.zero_grad()
        loss.backward()

        optimizer.step()

        batch_time.update(time.time() - end)
        end = time.time()
        length_data += len(input)
        real_acc = accuracy / length_data

        print('Epoch: [{0}][{1}/{2}]\t'
              'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
              'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
              'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
              'ACC {Acc}\t'.format(
               epoch, i, len(trn_loader), batch_time=batch_time,
               data_time=data_time, loss=losses, Acc = acc.item()))
        if i == 0 : 
            pass
        else : 
            if (i % 1000 == 0) and (val_mode == True):
                print("Start A Validation -- *")
            #import ipdb; ipdb.set_trace()
                writer_train.add_scalar('loss', losses.avg, iterate)
                writer_train.add_scalar("Acc", real_acc.item(), iterate)
                logger.write([epoch, losses.avg, real_acc.item()])
                loss_val, acc = validate(val_loader, model, criterion, iterate, val_logger)
                iterate += 1
                print(loss_val, acc)

        if i % 10 == 0:
            print(type(loss), loss)
            print(type(acc), acc)
            try :
                sublogger.write([epoch, i, loss.item(), acc.item()])
            except :
                sublogger.write([epoch, i, loss, acc.item()])


    print("loss", losses.avg, epoch)

def validate(val_loader, model, criterion, epoch, logger):

    batch_time = AverageMeter()
    losses = AverageMeter()

    model.eval()
    from torch.utils.tensorboard import SummaryWriter
    accuracy = 0
    length_data = 0
    confusion = Performance()
    with torch.no_grad():
        end = time.time()
        from tqdm import tqdm
        for i, (input, target, _) in tqdm(enumerate(val_loader)):
            #import ipdb; ipdb.set_trace()
            input = input.cuda()
            target = target.cuda()

            output = model(input)
            loss = criterion(output, target)
            losses.update(loss.item(), input.size(0))

            batch_time.update(time.time() - end)
            end = time.time()

            pos_probs = torch.sigmoid(output)
            pos_preds = (pos_probs > 0.5).float()
            counts = pos_preds - target
            counts = sum(np.abs(counts.cpu()))
            accuracy += (len(input) - counts)
            length_data += len(input)

            #if epoch % 3 == 0 :
            for prediction, real_label in zip(pos_preds, target) :
                confusion.cal_confusion(prediction, real_label)
    #if epoch == args.lr_schedule[-1] :
    confusion_matrix = confusion.return_matrix()
    torch.save(confusion_matrix, args.work_dir + "/" + args.exp + "/" + "confusion{0}.pth".format(epoch))
    real_acc = accuracy / length_data
    print(' * Loss {loss:.3f}, * ACC {ACC}'.format(loss= losses.avg, ACC=accuracy/length_data))
    logger.write([epoch, losses.avg, real_acc.item()])


    writer_valid.add_scalar('loss', losses.avg, epoch)
    writer_valid.add_scalar("Acc", real_acc.item(), epoch)
    print("Writer.close")
    return losses.avg, real_acc

def check_logger(output, target):
    correct = ((output - target) == 0).float()
    return correct

if __name__=='__main__':
    main()
