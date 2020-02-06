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
from dataloader import Segmentation_data, Segmentation_test_data
from utils import Logger,str2bool
import ipdb
from torch.utils.tensorboard import SummaryWriter
from medpy import metric
from losses import DiceLoss

def dict_for_datas() :
    allData_dic = {
        'data1th': {
            'top_1th': ['401', '403', '426', '432', '450', '490', '514'],
            'mid_1th': ['410', '425', '433', '441', '443', '513', '518'],
            'low_1th': ['405', '411', '430', '459', '464', '480']
        },
        'data2th': {
            'top_2th': ['52_KMK', '483', '29_MOY', '46_YMS', '40_LSH', '534', '8_KYK', '535', '536', '500'],
            'mid_2th': "",
            'low_2th': ""
        },
        'data3th': {
            'top_3th': ['59_KKO', '562', '564', '566', '583', '584', '599'],
            'mid_3th': ['61_CDJ', '66_YYB', '70_PJH', '575', '576', '590', '595'],
            'low_3th': ['56_KMK', '63_JJW', '72_TKH', '561', '567', '585']
        },
        'overal': {
            'overal_1th': ['403', '401', '411', '490', '518', '426', '432', '480', '430', '464', '425', '443',
                            '441', '433', '405', '459', '450', '514', '513', '410'],
            'overal_2th': ['403', '401', '411', '490', '518', '426', '432', '480', '430', '464', '425', '443',
                            '441', '433', '405', '459', '450', '514', '513', '410']
                            + ['52_KMK', '483', '29_MOY', '46_YMS', '40_LSH', '534', '8_KYK', '535', '536', '500'],
            'overal_3th': ['403', '401', '411', '490', '518', '426', '432', '480', '430', '464', '425', '443',
                            '441', '433', '405', '459', '450', '514', '513', '410']
                            + ['52_KMK', '483', '29_MOY', '46_YMS', '40_LSH', '534', '8_KYK', '535', '536', '500']
                            + ['575', '567', '70_PJH', '561', '583', '72_TKH', '564', '56_KMK', '599', '61_CDJ',
                                '66_YYB', '584', '562', '59_KKO', '585', '590', '566', '595', '576', '63_JJW'],
        }
    }
    return allData_dic
    
class compute_overall_performance() :
    def __init__(self) :
        self.confusion_matrix = np.zeros((4, ))
        self.ious = 0
        self.dices = 0
        self.slice_level_acc = 0 
    
    def update_overall(self, measure, value) :
        if measure == "dice" :
            self.dices += value
        elif measure == "ious" :
            self.ious += value
        elif measure == "slice_level_acc" :
            self.slice_level_acc += value 

def draw_photo(preds, img, label, performance, save_dir, file_name) :
    def _overlay_mask(img, mask, color='red'):
        # convert gray to color
        color_img = np.dstack([img, img, img])
        mask_idx = np.where(mask == 1)
        if color == 'red':
            color_img[mask_idx[0], mask_idx[1], :] = np.array([255, 0, 0])
        elif color == 'blue':
            color_img[mask_idx[0], mask_idx[1], :] = np.array([0, 0, 255])

        return color_img

    
    ori_image = np.asarray(plt.imread(img).convert("L"))
    label_img = np.asarray(plt.imread(label).convert("L"))

    fig = plt.figure(figsize=(15, 5))
    ax = []
    ax.append(fig.add_subplot(1, 3, 1))
    plt.imshow(ori_image, 'gray')

    ax.append(fig.add_subplot(1, 3, 2))
    plt.imshow(_overlay_mask(ori_image, label_img, color='red'))

    ax.append(fig.add_subplot(1, 3, 3)) 
    plt.imshow(_overlay_mask(ori_image, preds, color='blue'))
    ax[-1].set_title('IoU = {0:.4f}, DICE = {1:.4f}'.format(performance[1], performance[0]))

    iou = performance[1]

    for i in ax:
            i.axes.get_xaxis().set_visible(False)
            i.axes.get_yaxis().set_visible(False)

    if iou == -1:
        res_img_path = os.path.join(save_dir,
                                    'FILE{slice_id:0>4}_{iou}.png'.format(slice_id=file_name, iou='NA'))
    else:
        res_img_path = os.path.join(save_dir,
                                    'FILE{slice_id:0>4}_{iou:.4f}.png'.format(slice_id=file_name, iou=iou))
    plt.savefig(res_img_path, bbox_inches='tight')
    plt.close()
    


def predict(data_path, model, state_dict, work_dir) : 
    model = Unet_sae(in_shape=(1, 512, 512), padding=1, momentum=0.1)
    model = nn.DataParallel(model).cuda()
    model.load_state_dict(torch.load(state_dict, map_location=lambda storage, loc: storage)['state_dict'])
    overall_performacnce = compute_overall_performance()
    data_dict = dict_for_datas()
    
    model.eval()

    if not os.path.exists(work_dir):
        os.makedirs(work_dir)

    test_datasets = Segmentation_test_data(data_path)
    test_loader = torch.utils.data.DataLoader(test_datasets, batch_size=1, shuffle=False, num_workers=8)

    criterion = DiceLoss()

    for i, (image, target, img_file_path, label_file_path) in test_loader :
        image = image.cuda()
        target = target.cuda()

        output_seg = model(image)
        loss = criterion(output_seg, target)

        dice_score = metric.dc(output_seg, target)
        jaccard_score = metric.jc(output_seg, target)

        pos_preds = (torch.nn.Sigmoid(target) > 0.5) * 1.0 

        if max(target) == 0 :
            pass
        else : 
            overall_performacnce.update_overall("dice", dice_score)
            overall_performacnce.update_overall("ious", jaccard_score)
            if max(target) <= 1 and max(pos_preds) <= 1 :
                overall_performacnce.update_overall("slice_level_acc", 1)
            else : 
                overall_performacnce.update_overall("slice_level_acc", 0)
        

        tmp = img_file_path.split("/")
        save_dir = tmp[3] + str(tmp[6]) 
        file_name = tmp[-1]

        full_dir = os.path.join(work_dir, save_dir)

        if not os.path.exists(full_dir):
            os.makedirs(full_dir)
        
        draw_photo(pos_preds, img_file_path, label_file_path, [dice_score, jaccard_score], full_dir, file_name)

def test(exam_path, model, input_stats, work_dir) :
    collated_performance = {}
    for i in range(len(args.test_root)):
        exam_ids = os.listdir(os.path.join(args.test_root[i], 'images'))
        for exam_id in exam_ids:
            print('Processing {}'.format(exam_id))
            exam_path = os.path.join(args.test_root[i], 'images', exam_id)  # '/data2/test_3d/images/403'
            prediction_list, org_input_list, org_target_list = predict(exam_path, model, input_stats, work_dir)

            # measure performance
            performance = performance_by_slice(prediction_list, org_target_list)

            # find folder
            find_folder = ''
            count = 0
            for data_no, level_no in data_dict.items():
                for level_key, level_val in level_no.items():
                    if exam_id in level_val:
                        if 'overal' in level_key.split('_'):  # prevent duplicate data save
                            continue
                        find_folder = level_key
                        count += 1
            assert count == 1, 'duplicate folder'

            result_dir_sep = os.path.join(result_dir, find_folder)
            # save_fig(exam_id, org_input_list, org_target_list, prediction_list, performance, result_dir_sep)

            collated_performance[exam_id] = performance

    for data_no, level_no in data_dict.items():
        for level_key, level_val in level_no.items():
            sep_dict = seperate_dict(collated_performance, level_val)
            if len(sep_dict) == 0:
                continue
            sep_performance = compute_overall_performance(sep_dict)

            with open(os.path.join(result_dir, '{}_performance.json'.fomat(level_key)), 'w') as f:
                json.dump(sep_performance, f)
    


if __name__ == "__main__" : 
    data_path = ['/data2/sk_data/data_1rd/test_3d', 
                '/data2/sk_data/data_2rd/test_3d',
                '/data2/sk_data/data_3rd/test_3d',
                '/data2/sk_data/data_4rd/test_3d',
                '/data2/sk_data/data_5rd/test_3d']
    
    state_dict = '/data1/workspace/geongyu/MTL/MTL/Segmentation_SGD/model_best.pth'

    work_dir = '/data1/workspace/geongyu/MTL/MTL/Segmentation_SGD/Predcition'

    draw_plot(data_path, model=None, state_dict=state_dict, work_dir = work_dir)