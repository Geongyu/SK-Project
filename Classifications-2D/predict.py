import os
import argparse
import json

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import shutil
import pickle

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import time
from model import *
from dataloader import Classification_Data, load_kaggle_data, load_kaggle_full_label
from utils import Logger,str2bool, Performance, AverageMeter
import ipdb
from torch.utils.tensorboard import SummaryWriter
from efficientnet_pytorch import EfficientNet
import cv2


parser = argparse.ArgumentParser()

parser.add_argument('--image-dir', default="/data2/sk_data/kaggle_data/stage_1_train_images_png", type=str)
parser.add_argument('--label', default="/data2/sk_data/kaggle_data/bin_dataframe.csv", type=str)
parser.add_argument('--exp', default="test", type=str)
parser.add_argument('--mainfolder', default="/data1/workspace/geongyu/evalution/exp_result")
parser.add_argument('--batch-size', default=16, type=int)
parser.add_argument('--loss-function', default='bce', type=str)
parser.add_argument('--arch', default='efficientnet', type=str)
parser.add_argument('--tensorboardwriter', default="Test", type=str)
parser.add_argument("--model-selection", default="/data_hdd/home/geongyu/cls_problem/model_best.pth", type=str)
parser.add_argument("--select_columns", default="all", type=str)
parser.add_argument("--mode", default="kaggle", type=str)
parser.add_argument("--sktest", default=['/data2/sk_data/data_1rd/test_3d',
                                        '/data2/sk_data/data_2rd/test_3d',
                                        '/data2/sk_data/data_3rd/test_3d',
                                        '/data2/sk_data/data_4rd/test_3d',
                                        '/data2/sk_data/data_5rd/test_3d'], type=str)
parser.add_argument("--effinet", default="1", type=str)
parser.add_argument("--cam", default="Deactivate", type=str)
parser.add_argument("--savedir", default="saved", type=str)
args = parser.parse_args()

tensor_writer = SummaryWriter(log_dir="tensor_log/" + args.tensorboardwriter + "/predict_per_iter", comment="SK CLS PROBLEMS")

def main() :
    print("-" * 40)
    print("  Start Prediction - * Folder Information")
    print("    Load Image Path  : ", args.image_dir)
    print("    Load Label Path  : ", args.label)
    print("    Save Base Path   : ", args.mainfolder)
    print("    Save Sub Path    : ", args.exp)
    print("    Model State Dict : ", args.model_selection)
    print("    CAM Mode         : ", args.cam)
    print("-" * 40)

    work_dir = os.path.join(args.mainfolder, args.exp)

    if not os.path.exists(work_dir) :
        os.makedirs(work_dir)

    shutil.copy(__file__, os.path.join(work_dir, 'main.py'))
    with open(os.path.join(work_dir, 'args.pkl'), 'wb') as f :
        pickle.dump(args, f)

    image_data_folder = args.image_dir
    label_data_file = args.label
    sk_image_data = args.sktest

    # Load Non-Pre Trained model
    if args.effinet == "7" :
        net = EfficientNet.from_pretrained('efficientnet-b7', num_classes=1)
    elif args.effinet =="1" :
        net = EfficientNet.from_pretrained('efficientnet-b1', num_classes=1)
    elif args.effinet =="4" :
        net = EfficientNet.from_pretrained('efficientnet-b4', num_classes=1)
    net = nn.DataParallel(net).cuda()

    checkpoint_path = os.path.join(args.model_selection)
    state = torch.load(checkpoint_path)
    net.load_state_dict(state['state_dict'])
    cudnn.benchmark = True

    #loss
    if args.loss_function == "bce" :
        criterion = nn.BCEWithLogitsLoss()
    elif args.loss_function == "cross_entropy" :
        criterion = nn.CrossEntropyLoss()

    test_logger = Logger(os.path.join(work_dir, "test.log"))
    if args.mode == "kaggle" :
        predict(net, criterion, image_data_folder, label_data_file, test_logger, args.mode)
    elif args.mode == "sk" :
        predict(net, criterion, sk_image_data, label_data_file, test_logger, args.mode)

    print("-" * 40)
    print("  End    Prediction - * Folder Information")
    print("    Load Image Path  : ", args.image_dir)
    print("    Load Label Path  : ", args.label)
    print("    Save Base Path   : ", args.mainfolder)
    print("    Save Sub Path    : ", args.exp)
    print("    Model State Dict : ", args.model_selection)
    print("    CAM Mode         : ", args.cam)
    print("-" * 40)

def predict(model, criterion, image_path, label_path, logger, mode) :
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    if mode == "kaggle" :
        tst_datasets = load_kaggle_full_label(path=image_path, label=label_path,  mode="annotation", selective="{0}".format(args.select_columns),)
        tst_loader = torch.utils.data.DataLoader(tst_datasets, batch_size=args.batch_size, shuffle=False)
    elif mode == "sk" :
        tst_datasets = Classification_Data(image_path)
        tst_loader = torch.utils.data.DataLoader(tst_datasets, batch_size=args.batch_size, shuffle=False)
    make_confusion_matrix = Performance()
    work_dir = os.path.join(args.mainfolder, args.exp)

    model.eval()
    accuracy = 0
    length_data = 0
    end = time.time()
    other = []
    with torch.no_grad() :
        for i, (images, labels, other) in enumerate(tst_loader) :
            data_time.update(time.time() - end)
            images, labels = images.cuda(), labels.cuda()
            output = model(images)
            #import ipdb;ipdb.set_trace()
            loss = criterion(output, labels)

            pos_probs = torch.sigmoid(output)
            pos_preds = (pos_probs > 0.5).float()
            counts = pos_preds - labels
            counts = sum(np.abs(counts.cpu()))
            acc = (len(images) - counts) / len(images)
            accuracy += (len(images) - counts)

            losses.update(loss.item(), images.size(0))
            batch_time.update(time.time() - end)
            end = time.time()
            length_data += len(images)

            other.append([output, other])

            print('Iteration : [{0}/{1}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'ACC {Acc}\t'.format(
                    i, len(tst_loader), batch_time=batch_time,
                   data_time=data_time, loss=losses, Acc = acc.item()))
            logger.write([i, loss.item(), acc.item()])
            tensor_writer.add_scalar("loss", loss.item(), i)
            tensor_writer.add_scalar("acc", acc.item())
            for prediction, real_label in zip(pos_preds, labels) :
                make_confusion_matrix.cal_confusion(prediction, real_label)

            confusion = make_confusion_matrix.return_matrix()
            torch.save(confusion, "confusion_{1}_{0}.pth".format(args.select_columns, args.effinet, work_dir))
    torch.save(accuracy, "{2}/Accuracy_{1}_{0}_AllDATA.pth".format(args.select_columns, args.effinet, work_dir))
    torch.save(other, "{1}/Check_sub_acc_{0}.pth".format(args.effinet, work_dir))

if __name__ == "__main__" :
    main()
