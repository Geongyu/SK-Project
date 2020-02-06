
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

    epoch, total_loss1, iou1, dice1  = zip(*logger1)
    epoch, total_loss2, iou2, dice2 = zip(*logger2)

    plt.figure(1)
    plt.plot(epoch, total_loss1, 'navy', label='Train Total Loss')
    plt.plot(epoch, total_loss2, 'darkorange', label='Validation Total Loss')
    plt.grid()

    plt.xlabel('Epoch')
    plt.legend()
    plt.title('Compare Total Loss')
    plt.ylim(0.0, 1.0)
    plt.savefig(os.path.join(work_dir, 'Total_Loss.png'))


    plt.figure(2)
    plt.plot(epoch, iou1, 'navy', label='Train IoU')
    plt.plot(epoch, iou2, 'darkorange', label='Validation IoU')
    plt.grid()

    plt.xlabel('Epoch')
    plt.legend()
    plt.title('Compare IoU')
    plt.ylim(0.0, 1.0)
    plt.savefig(os.path.join(work_dir, 'IoU.png'))



def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')



