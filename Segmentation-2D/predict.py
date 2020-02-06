import os
import argparse
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from torch import nn 

from model import *
from dataloader import Segmentation_data, Test_datasets
from utils import Logger,str2bool
import ipdb
from torch.utils.tensorboard import SummaryWriter
from medpy import metric
from losses import DiceLoss

'''
# Data Level
top_1th = ['401', '403', '426', '432', '450', '490', '514']
top_2th = data2th  # test2 all top level
top_3th = ['59_KKO', '562', '564', '566', '583', '584', '599']
top_4th = ['120', '132', '146', '157', '158', '169', '199', '217', '222', '234', '609', '617', '623', '634', '652', '662', '671', '673', '676', '686', '697']
top_5th = ['261', '262', '263', '305', '340', '374', '375', '392']

middle_1th = ['410', '425', '433', '441', '443', '513', '518']
middle_3th = ['61_CDJ', '66_YYB', '70_PJH', '575', '576', '590', '595']
middle_4th = ['106', '113', '114', '140', '152', '159', '164', '180', '224', '233', '235', '238', '242', '244', '601', '611', '624', '635', '645', '648', '654', '663']
middle_5th = ['251', '264', '273', '289', '293', '324', '328', '341', '347', '388']

low_1th = ['405', '411', '430', '459', '464', '480']
low_3th = ['56_KMK', '63_JJW', '72_TKH', '561', '567', '585']
low_4th = ['102', '130', '148', '151', '156', '160', '161', '174', '185', '189', '205', '211', '213', '215', '221', '226', '608', '615', '616', '632', '658', '665']
low_5th = ['253', '274', '296', '301', '303', '326', '334', '338', '351', '377']
'''

def find_level() :
    top = ['401', '403', '426', '432', '450', '490', '514', '59_KKO', '562', '564', '566', '583', '584', '599', '120', '132', '146', 
        '157', '158', '169', '199', '217', '222', '234', '609', '617', '623', '634', '652', '662', '671', '673', '676', '686', '697'
        '261', '262', '263', '305', '340', '374', '375', '392', '8_KYK', '29_MOY', '40_LSH', '46_YMS', '52_KMK', '483', '500', '534', '535', '536']
    
    middle = ['410', '425', '433', '441', '443', '513', '518', '61_CDJ', '66_YYB', '70_PJH', '575', '576', '590', '595',
        '106', '113', '114', '140', '152', '159', '164', '180', '224', '233', '235', '238', '242', '244', '601', '611', '624', '635', '645', '648', '654', '663',
        '251', '264', '273', '289', '293', '324', '328', '341', '347', '388']
    
    low = ['405', '411', '430', '459', '464', '480', '56_KMK', '63_JJW', '72_TKH', '561', '567', '585', '253', '274', '296', '301', '303', '326', '334', '338', '351', '377',
        '102', '130', '148', '151', '156', '160', '161', '174', '185', '189', '205', '211', '213', '215', '221', '226', '608', '615', '616', '632', '658', '665']

    return top, middle, low

class compute_overall_performance() :
    def __init__(self) :
        self.confusion_matrix = np.zeros((4, ))
        self.ious = 0
        self.dices = 0
        self.slice_level_acc = 0 
    
    def update(self, measure) :
        if measure == "dice" :
            self.dices += measure
        elif measure == "ious" :
            self.ious += measure
        elif measure == "slice_level_acc" :
            self.slice_level_acc += measure     

def predict(data_path, model, state_dict, work_dir) : 
    model = MTL_Unet(in_shape=(1, 512, 512), padding=1, momentum=0.1)
    model = nn.DataParallel(model).cuda().cpu()
    model.load_state_dict(torch.load(state_dict, map_location=lambda storage, loc: storage)['state_dict'])
    overall_performacnce = compute_overall_performance()
    
    model.eval()

    if not os.path.exists(work_dir):
        os.makedirs(work_dir)

    test_datasets = Test_datasets(data_path)
    test_loader = torch.utils.data.DataLoader(test_datasets, batch_size=1, shuffle=False, num_workers=8)

    criterion = DiceLoss()
    top, middle, low = find_level()

    for i, (image, target, img_file_path, label_file_path) in test_loader :
        image = image.cpu()
        target = target.cpu()

        output_seg = model(image)
        segmentation_loss = criterion(output_seg, target)

        import ipdb; ipdb.set_trace()

        dice_score = metric.dc(output_seg, target)
        jaccard_score = metric.jc(output_seg, target)

        if max(target) == 0 :
            pass
        else : 
            



        tmp = img_file_path.split("/")
        save_dir = tmp[3] + str(tmp[6]) 
        file_name = tmp[-1]

        if str(tmp[6]) in top :
            level = 'top'
        elif str(tmp[6]) in middle :
            level = 'middle'
        else :
            level = 'low'

        full_dir = os.path.join(work_dir, level, save_dir)

        if not os.path.exists(full_dir):
            os.makedirs(full_dir)
        


if __name__ == "__main__" : 
    data_path = ['/data2/sk_data/data_1rd/test_3d', 
                '/data2/sk_data/data_2rd/test_3d',
                '/data2/sk_data/data_3rd/test_3d',
                '/data2/sk_data/data_4rd/test_3d',
                '/data2/sk_data/data_5rd/test_3d']
    
    state_dict = '/data1/workspace/geongyu/MTL/MTL/Segmentation_SGD/model_best.pth'

    work_dir = '/data1/workspace/geongyu/MTL/MTL/Segmentation_SGD/Predcition'

    draw_plot(data_path, model=None, state_dict=state_dict, work_dir = work_dir)