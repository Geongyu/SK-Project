
import os
import glob

import numpy as np
from skimage import io

try:
    import torch
except:
    pass

import shutil
from collections import Iterable
import matplotlib.pyplot as plt
from slacker import Slacker
import argparse

class AverageMeter(object):

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.sum_2 = 0 # sum of squares
        self.count = 0
        self.std = 0

    def update(self, val=None, n=1):
        if val!=None: # update if val is not None
            self.val = val
            self.sum += val * n
            self.sum_2 += val**2 * n
            self.count += n
            self.avg = self.sum / self.count
            self.std = np.sqrt(self.sum_2/self.count - self.avg**2)
        else:
            pass

class Logger(object):
    def __init__(self, path, int_form=':03d', float_form=':.4f'):
        self.path = path
        self.int_form = int_form
        self.float_form = float_form
        self.width = 0

    def __len__(self):
        try: return len(self.read())
        except: return 0

    def write(self, values):
        if not isinstance(values, Iterable):
            values = [values]
        if self.width == 0:
            self.width = len(values)
        assert self.width == len(values), 'Inconsistent number of items.'
        line = ''
        for v in values:
            if isinstance(v, int):
                line += '{{{}}} '.format(self.int_form).format(v)
            elif isinstance(v, float):
                line += '{{{}}} '.format(self.float_form).format(v)
            elif isinstance(v, str):
                line += '{} '.format(v)
            else:
                raise Exception('Not supported type.',v)
        with open(self.path, 'a') as f:
            f.write(line[:-1] + '\n')

    def read(self):
        with open(self.path, 'r') as f:
            log = []
            for line in f:
                values = []
                for v in line.split(' '):
                    try:
                        v = float(v)
                    except:
                        pass
                    values.append(v)
                log.append(values)
        return log

def save_checkpoint(state, is_best, work_dir, filename='checkpoint.pth'):
    checkpoint_path = os.path.join(work_dir, filename)
    if is_best:
        torch.save(state, checkpoint_path)
        shutil.copyfile(checkpoint_path,
                        os.path.join(work_dir, 'model_best.pth'))

def load_exam(exam_dir, ftype='png'):

    file_extension = '.'.join(['*', ftype])
    data_paths = glob.glob(os.path.join(exam_dir, file_extension))
    data_paths = sorted(data_paths, key=lambda x: x.split('/')[-1]) # sort by filename

    slices = []
    for data_path in data_paths:
        arr = io.imread(data_path)
        slices.append(arr)

    data_3d = np.stack(slices)

    return data_3d

def pad_3d(data_3d, target_length, padding_value=0):

    d, h, w = data_3d.shape # assume single channel
    margin = target_length - d # assume that h and w are sufficiently larger than target_length
    padding_size = margin // 2
    upper_padding_size = padding_size
    lower_padding_size = margin - upper_padding_size

    padded = np.pad(data_3d, ((upper_padding_size, lower_padding_size),
                              (0,0), (0,0)),
                    'constant', constant_values=(padding_value,padding_value))

    return padded, (upper_padding_size, lower_padding_size)

def calc_stats(data_root):

    data_ids = os.listdir(os.path.join(data_root, 'images'))

    mean_meter = AverageMeter()
    std_meter = AverageMeter()

    for data_id in data_ids:
        image_dir = os.path.join(data_root, 'images', data_id)
        image_3d = load_exam(image_dir, ftype='png')
        pixel_mean = image_3d.mean()
        pixel_std = image_3d.std()

        mean_meter.update(pixel_mean, image_3d.size)
        std_meter.update(pixel_std, image_3d.size)

    total_mean = mean_meter.avg
    total_std = np.sqrt(std_meter.sum_2/std_meter.count)

    return {'mean': total_mean, 'std': total_std}

def draw_curve(work_dir,logger1,logger2):
    # Logger 2개를 입력으로 받아와 각각의 계산된 값들을 시각화하여 처리한다
    logger1 = logger1.read()
    logger2 = logger2.read()

    epoch, loss1, iou1, dice1, sacc1 = zip(*logger1)
    epoch, loss2, iou2, dice2, sacc2 = zip(*logger2)

    plt.figure(1)
    plt.plot(epoch, loss1, 'navy', label='Train Loss')
    plt.plot(epoch, loss1, 'darkorange', label='Validation Loss')
    plt.grid()

    plt.xlabel('Epoch')
    plt.legend()
    plt.title('Compare Loss')
    plt.ylim(0.0, 1.0)
    plt.savefig(os.path.join(work_dir, 'Loss.png'))

    plt.figure(2)
    plt.plot(epoch, iou1, 'navy', label='Train IoU')
    plt.plot(epoch, iou2, 'darkorange', label='Validation IoU')
    plt.grid()

    plt.xlabel('Epoch')
    plt.legend()
    plt.title('Compare IoU')
    plt.ylim(0.0, 1.0)
    plt.savefig(os.path.join(work_dir, 'IoU.png'))

    plt.figure(3)
    plt.plot(epoch, dice1, 'navy', label='Train DICE Score')
    plt.plot(epoch, dice2, 'darkorange', label='Validation DICE Score')
    plt.grid()

    plt.xlabel('Epoch')
    plt.legend()
    plt.title('Compare DICE Score')
    plt.ylim(0.0, 1.0)
    plt.savefig(os.path.join(work_dir, 'DICE_Score.png'))

    plt.figure(4)
    plt.plot(epoch, sacc1, 'navy', label='Train Slice Level Accuracy')
    plt.plot(epoch, sacc2, 'darkorange', label='Validation Slice Level Accuracy')
    plt.grid()

    plt.xlabel('Epoch')
    plt.legend()
    plt.title('Compare Slice Level Accuracy')
    plt.ylim(0.5, 1.0)
    plt.savefig(os.path.join(work_dir, 'Slice_Level_Accuracy.png'))


def check_correct_forgget(output, target, ephoch,
size, correct, before_correct) :
    if ephoch == 0 :
        correct = ((output-target) == 0).float() # 맞추면 1 틀리면 0
        forget = None
        added = None
    else :
        added = ((output-target) == 0).float()
        forget = np.zeros(size) - (((before_correct - added) == 1).float())
        correct += added
    return correct, forget, added

def send_slack_message(token,channel,messge):
    token = token
    slack = Slacker(token)
    slack.chat.post_message(channel, messge)


def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

class History(object):
    def __init__(self, n_data, train_mode=False, total_epoch=None):
        self.n_data = n_data
        self.counter = np.zeros(n_data)
        if train_mode:
            self.correctness = np.zeros((n_data, total_epoch))
        else:
            self.correctness = np.zeros((n_data, 1, 512, 512))

    def update(self, data_idx, correctness, epoch=None):
        if type(data_idx) is list :
            for i in data_idx :
                self.correctness[i] += correctness.cpu().numpy()
        else :
            self.correctness[data_idx] += correctness.cpu().numpy()

    def get_sum_correctness(self):
        sum_correctness = self.correctness[:, :].sum(axis=1)
        return sum_correctness

    def return_correct(self) :
        re_cor = self.correctness
        tmp = []
        for i in range(len(re_cor)) :
            print(i, " / ", len(re_cor))
            tmp.append(np.where(re_cor[i].flatten() == 0))
        return tmp

    def multi_io(self) :
        tmp = TMP()
        tmp.sum_correct(self.correctness)

class Performance(object) :
    def __init__(self) :
        self.confusionmatrix = [0, 0, 0, 0]
        # TP, TN, FP, FN

    def cal_confusion(self, pred, real) :
        if real == 1 and pred == 1 :
            self.confusionmatrix[0] += 1
        elif real == 1 and pred == 0 :
            self.confusionmatrix[3] += 1
        elif real == 0 and pred == 0 :
            self.confusionmatrix[1] += 1
        else :
            self.confusionmatrix[2] += 1

    def return_matrix(self) :
        return self.confusionmatrix

if __name__ == "__main__" :
    import os
    work_dir = "/data1/workspace/geongyu/MTL/MTL/REALBASELINE"

    trn_logger = Logger(os.path.join(work_dir, 'train.log'))
    trn_raw_logger = Logger(os.path.join(work_dir, 'train_raw.log'))
    val_logger = Logger(os.path.join(work_dir, 'validation.log'))

    import ipdb; ipdb.set_trace()
