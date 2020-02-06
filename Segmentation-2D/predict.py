import os
import argparse
import json

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.cm as cm

from model import *
from dataloader import *
from utils import Logger,str2bool
import ipdb
from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter()

def main_test(model=None, args=None, val_mode=False):
    #import ipdb; ipdb.set_trace()
    work_dir = os.path.join(args.work_dir, args.exp)
    file_name = args.file_name
    if not val_mode:
        result_dir = os.path.join(work_dir, file_name)
        if not os.path.exists(result_dir):
            os.makedirs(result_dir)

        # load model and input stats
        # Note: here, the model should be given manually
        # TODO: try to import model configuration later
        if model is None:
            model = load_model(args.arch)
            model = nn.DataParallel(model).cuda()

        checkpoint_path = os.path.join(work_dir, 'model_best.pth')
        state = torch.load(checkpoint_path)
        model.load_state_dict(state['state_dict'])
        cudnn.benchmark = True
    
    input_stats = np.load(os.path.join(work_dir, 'input_stats.npy')).tolist()        
    if not val_mode:
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
                                '66_YYB', '584', '562', '59_KKO', '585', '590', '566', '595', '576', '63_JJW'], } }

    # compute seperate performace
    overall_data3th, overall_data2th, data1th,top_1th,top_2th,top_3th, middle_1th,middle_3th, low_1th,low_3th = make_level_list(args.test_root)
    # filepath


    # list exam ids
    collated_performance = {}
    for i in range(len(args.test_root)):
        exam_ids = os.listdir(os.path.join(args.test_root[i], 'images'))
        for exam_id in exam_ids:
            print('Processing {}'.format(exam_id))
            exam_path = os.path.join(args.test_root[i], 'images', exam_id)  # '/data2/test_3d/images/403'
            #import ipdb; ipdb.set_trace()
            prediction_list, org_input_list, org_target_list = predict(model, exam_path, input_stats, args=args)

            # measure performance
            performance = performance_by_slice(prediction_list, org_target_list)

            #find folder
            find_folder = ''
            count = 0
            for data_no, level_no in allData_dic.items():
                for level_key, level_val in level_no.items():
                    if exam_id in level_val:
                        if 'overal' in level_key.split('_'):  # prevent duplicate data save
                            continue
                        find_folder = level_key
                        count += 1
            assert count == 1, 'duplicate folder'

            #save_fig(exam_id, org_input_list, org_target_list, prediction_list, performance, result_dir_sep)
            collated_performance[exam_id] = performance


    level_id_list = [overall_data3th, overall_data2th, data1th, top_1th, top_2th, top_3th,
                     middle_1th, middle_3th, low_1th, low_3th]
    level_name_list = ['overrall_data5th', 'overrall_data4th', 'overrall_data3th', 'overall_data2th', 'only_data1th', 'top_1th', 'top_2th', 'top_3th',
                       'top_4th', 'top_5th', 'middle_1th', 'middle_3th', 'middle_4th', 'middle_5th', 'low_1th', 'low_3th', 'low_4th', 'low_5th']
    dir_path_list = [result_dir, result_dir, result_dir, result_dir, result_dir,
                     os.path.join(result_dir, 'top_1th'), os.path.join(result_dir, 'top_2th'),
                     os.path.join(result_dir, 'top_3th'),os.path.join(result_dir, 'top_4th'), os.path.join(result_dir, 'top_5th'), os.path.join(result_dir, 'middle_1th'), os.path.join(result_dir, 'middle_3th'),  os.path.join(result_dir, 'middle_4th'),  os.path.join(result_dir, 'middle_5th'), os.path.join(result_dir, 'low_1th'),
                     os.path.join(result_dir, 'low_3th'), os.path.join(result_dir, 'low_4th'), os.path.join(result_dir, 'low_5th')]


    assert len(level_id_list) == len(dir_path_list), 'not same level_id and dir_path length'

    overrall_data3th_dic = ''
    for h in range(len(level_id_list)):
        make_save_performance(collated_performance, level_id_list[h], dir_path_list[h], level_name_list[h],save_mode=True)
        if h==0:
            overrall_data3th_dic=make_save_performance(collated_performance, level_id_list[h], dir_path_list[h], level_name_list[h],save_mode=True)

    count_pixel_dic = overrall_data3th_dic
    total_img = 1679
    threshold = 100



def predict(model, exam_root, input_stats, args=None):
    tst_dataset = Segmentation_2d_data(exam_root)
    tst_loader = torch.utils.data.DataLoader(tst_dataset,
                                             batch_size=1,
                                             shuffle=False,
                                             num_workers=0)

    print('exam_root',exam_root)
    print(len(tst_loader))
    prob_img_list = []
    input_img_list = []
    target_img_list = []
    model.eval()
    with torch.no_grad():
        for i, (input, target,ori_img, idx) in enumerate(tst_loader):

            input = input.cuda()
            #import ipdb; ipdb.set_trace()
            output, _ = model(input)

            # convert to prob

            pos_probs = torch.sigmoid(output)
            pos_probs = pos_probs.squeeze().cpu().numpy()
            input_ = ori_img.squeeze().cpu().numpy()
            target_ = target.squeeze().cpu().numpy()

            prob_img_list.append(pos_probs)
            input_img_list.append(input_)
            target_img_list.append(target_)
        print('end---------')
        return prob_img_list, input_img_list, target_img_list


def performance_by_slice(output_list, target_list):

    assert len(output_list) == len(target_list), 'not same list lenths'

    performance = {}
    for i in range(len(output_list)):
        preds =  output_list[i]
        slice_pred = (preds > 0.5).astype('float')
        slice_target = target_list[i]
        gt_pixel =int( slice_target.sum())
        pred_pixel = int(slice_pred.sum())

        # slice-level classification performance
        tp = fp = tn = fn = 0
        is_gt_positive = slice_target.max()
        is_pred_positive = slice_pred.max()
        if is_gt_positive:
            if is_pred_positive:
                tp = 1
            else:
                fn = 1
        else:
            if is_pred_positive:
                fp = 1
            else:
                tn = 1

        # slice-level segmentation performance
        iou = dice = -1
        if is_gt_positive:
            union = ((slice_pred + slice_target) != 0).sum()
            intersection = (slice_pred * slice_target).sum()

            iou = intersection / union
            dice = (2 * intersection) / (slice_pred.sum() + slice_target.sum())

        # TODO: not need to store gt and pred
        performance[str(i)] = {'cls': [tp, fp, tn, fn],
                                  'seg': [iou, dice],
                                  'gt': slice_target,
                                  'pred': slice_pred,
                                'pixel_num' : [gt_pixel,pred_pixel]
                               }
        #'pixel': [gt_pixel, pred_pixel],

    return performance


def compute_overall_performance(collated_performance):

    confusion_matrix = np.zeros((4,))
    iou_sum = dice_sum = n_valid_slices = 0

    for res_exam in collated_performance.values():
        for res_slice in res_exam.values():
            confusion_matrix += np.array(res_slice['cls'])
            if res_slice['gt'].sum() != 0: # consider only annotated slices
                iou_sum += res_slice['seg'][0]
                dice_sum += res_slice['seg'][1]
                n_valid_slices += 1

    iou_mean = iou_sum / n_valid_slices
    dice_mean = dice_sum / n_valid_slices

    return {'confusion_matrix': list(confusion_matrix),
            'slice_level_accuracy': (confusion_matrix[0] + confusion_matrix[2]) / confusion_matrix.sum(),
            'segmentation_performance': [iou_mean, dice_mean]}


def make_level_list(test_root):
    overall_data2th = []
    overall_data3th = []
    overall_data4th = []
    overall_data5th = []

    data1th = os.listdir(os.path.join(test_root[0], 'images'))
    data2th = os.listdir(os.path.join(test_root[1], 'images'))
    data3th = os.listdir(os.path.join(test_root[2], 'images'))
    data4th = os.listdir(os.path.join(test_root[3], 'images'))
    data5th = os.listdir(os.path.join(test_root[4], 'images'))

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

    overall_data2th.extend(data1th)
    overall_data2th.extend(data2th)

    overall_data3th.extend(overall_data2th)
    overall_data3th.extend(data3th)

    overall_data4th.extend(overall_data3th)
    overall_data4th.extend(data4th)

    overall_data5th.extend(overall_data4th)
    overall_data5th.extend(data5th)

    return overall_data5th, overall_data4th, overall_data3th, overall_data2th, data1th,top_1th,top_2th,top_3th, top_4th, top_5th, middle_1th,middle_3th, middle_4th, middle_5th, low_1th,low_3th,low_4th, low_5th



def compute_overall_pixel(collated_performance):

    confusion_matrix = np.zeros((4,))
    iou_sum = dice_sum = n_valid_slices = 0

    gt_pixel=[]
    pred_pixel=[]

    tp_pixel_gt = []
    tp_pixel_pred = []
    fp_pixel_gt = []
    fp_pixel_pred = []
    tn_pixel_gt = []
    tn_pixel_pred = []
    fn_pixel_gt = []
    fn_pixel_pred = []

    for res_exam in collated_performance.values():
        for res_slice in res_exam.values():

            gt_pixel.extend([res_slice['pixel_num'][0]])
            pred_pixel.extend([res_slice['pixel_num'][1]])

            #cls: [tp, fp, tn, fn]

            if res_slice['cls'][0] == 1:
                tp_pixel_gt.extend([res_slice['pixel_num'][0]])
                tp_pixel_pred.extend([res_slice['pixel_num'][1]])
            elif res_slice['cls'][1] == 1:
                fp_pixel_gt.extend([res_slice['pixel_num'][0]])
                fp_pixel_pred.extend([res_slice['pixel_num'][1]])
            elif res_slice['cls'][2] == 1:
                tn_pixel_gt.extend([res_slice['pixel_num'][0]])
                tn_pixel_pred.extend([res_slice['pixel_num'][1]])
            elif res_slice['cls'][3] == 1:
                fn_pixel_gt.extend([res_slice['pixel_num'][0]])
                fn_pixel_pred.extend([res_slice['pixel_num'][1]])

            confusion_matrix += np.array(res_slice['cls'])
            if res_slice['gt'].sum() != 0: # consider only annotated slices
                iou_sum += res_slice['seg'][0]
                dice_sum += res_slice['seg'][1]
                n_valid_slices += 1


    iou_mean = iou_sum / n_valid_slices
    dice_mean = dice_sum / n_valid_slices


    return {'confusion_matrix': list(confusion_matrix),
            'slice_level_accuracy': (confusion_matrix[0] + confusion_matrix[2]) / confusion_matrix.sum(),
            'segmentation_performance': [iou_mean, dice_mean],
            'gt_pixel' : gt_pixel, 'pre_pixel' : pred_pixel,'tp_pixel_gt' : tp_pixel_gt, 'tp_pixel_pred' : tp_pixel_pred,
            'fp_pixel_gt' : fp_pixel_gt, 'fp_pixel_pred' : fp_pixel_pred, 'tn_pixel_gt' : tn_pixel_gt, 'tn_pixel_pred' : tn_pixel_pred,
            'fn_pixel_gt'  :fn_pixel_gt, 'fn_pixel_pred' : fn_pixel_pred }




def load_model(model_name):
    if model_name == 'unet':
        model = Unet2D(in_shape=(args.multi_input, 512, 512), padding=args.padding_size,
                       momentum=args.batchnorm_momentum)
    elif model_name == 'unetcoord':
        print('radious', args.radious, type(args.radious))
        model = Unet2D_coordconv(in_shape=(args.multi_input, 512, 512), padding=args.padding_size,
                              momentum=args.batchnorm_momentum, coordnumber=args.coordconv_no,
                              radius=args.radious)
    elif model_name == 'unetmultiinput':
        model = Unet2D_multiinput(in_shape=(args.multi_input, 512, 512), padding=args.padding_size,
                                  momentum=args.batchnorm_momentum)
    elif model_name == 'scse_block':
        model = Unet_sae(in_shape=(args.multi_input, 512, 512), padding=args.padding_size,
                         momentum=args.batchnorm_momentum)
    else:
        raise ValueError('Not supported network.')

    return model


def save_fig(exam_id, org_input, org_target, prediction,
             slice_level_performance, result_dir,save_mode=None):

    def _overlay_mask(img, mask, color='red'):

        # convert gray to color
        color_img = np.dstack([img, img, img])
        mask_idx = np.where(mask == 1)
        if color == 'red':
            color_img[mask_idx[0], mask_idx[1], :] = np.array([255,0,0])
        elif color == 'blue':
            color_img[mask_idx[0], mask_idx[1], :] = np.array([0,0,255])

        return color_img

    result_exam_dir = os.path.join(result_dir, exam_id)
    if not os.path.exists(result_exam_dir):
        os.makedirs(result_exam_dir)



    assert (len(org_target) == len(prediction) \
                     == len(slice_level_performance)), '# of results not matched.'


    # convert prob to pred

    prediction = np.array(prediction)
    prediction = (prediction > 0.5).astype('float')


    for slice_id in slice_level_performance:
        #'cls': [tp, fp, tn, fn]
        # save fp
        if save_mode :
            if save_mode == 'tp':
                if slice_level_performance['0']['cls'][0] != 1:
                    continue
            elif save_mode == 'fp':
                if slice_level_performance['0']['cls'][1] != 1:
                    continue
            elif save_mode == 'tn':
                if slice_level_performance['0']['cls'][2] != 1:
                    continue
            elif save_mode == 'fn':
                if slice_level_performance['0']['cls'][3] != 1:
                    continue


        iou, dice = slice_level_performance[slice_id]['seg']
        input_slice = org_input[int(slice_id)]
        target_slice = org_target[int(slice_id)]
        pred_slice = prediction[int(slice_id)]

        target_slice_pos_pixel =  target_slice.sum() /(512*512)
        target_slice_pos_pixel_rate = np.round(target_slice_pos_pixel*100,2)

        pred_slice_pos_pixel = pred_slice.sum() / (512 * 512)
        pred_slice_pos_pixel_rate = np.round(pred_slice_pos_pixel * 100, 2)


        fig = plt.figure(figsize=(15,5))
        ax = []
        # show original img
        ax.append(fig.add_subplot(1,3,1))
        plt.imshow(input_slice, 'gray')
        # show img with gt
        ax.append(fig.add_subplot(1,3,2))
        plt.imshow(_overlay_mask(input_slice, target_slice, color='red'))
        ax[1].set_title('GT_pos_pixel = {0:.4f}({1}%)'.format(target_slice_pos_pixel,target_slice_pos_pixel_rate))
        # show img with pred
        ax.append(fig.add_subplot(1,3,3))
        plt.imshow(_overlay_mask(input_slice, pred_slice, color='blue'))
        ax[-1].set_title('IoU = {0:.4f} \n pred_pos_pixel = {1:.4f}({2}%)'.format(iou, pred_slice_pos_pixel,
                                                                                   pred_slice_pos_pixel_rate))

        # remove axis
        for i in ax:
            i.axes.get_xaxis().set_visible(False)
            i.axes.get_yaxis().set_visible(False)

        if iou == -1:
            res_img_path = os.path.join(result_exam_dir,
                                        'FILE{slice_id:0>4}_{iou}.png'.format(slice_id=slice_id, iou='NA'))
        else:
            res_img_path = os.path.join(result_exam_dir,
                                        'FILE{slice_id:0>4}_{iou:.4f}.png'.format(slice_id=slice_id, iou=iou))
        plt.savefig(res_img_path, bbox_inches='tight')
        plt.close()


def seperate_dict(ori_dict, serch_list):
    new_dict = {}
    for i in serch_list:
        if i in ori_dict:  
            new_dict[i] = ori_dict[i]
    return new_dict

def make_save_performance(collated_performance,level_id,dir_path,file_name,save_mode=False):
    sep_dict = seperate_dict(collated_performance, level_id)
    #overall_performance = compute_overall_performance(sep_dict)
    overall_performance = compute_overall_pixel(sep_dict)

    serch_list = ['confusion_matrix', 'slice_level_accuracy', 'segmentation_performance']
    new_overall_performance = seperate_dict(overall_performance,serch_list)

    if save_mode:
        with open(os.path.join(dir_path, '{}_performance.json'.format(str(file_name))), 'w') as f:
            json.dump(new_overall_performance, f)

    return overall_performance


def count_pixel(sep_dict):
    #sep_dict = seperate_dict(collated_performance, level_id)
    overall_performance = compute_overall_pixel(sep_dict)
    return overall_performance

def count_ratio(data_min, data_max, total_img):
    data_ratio_min = np.round((len(data_min) / total_img) * 100, 2)
    data_ratio_max = np.round((len(data_max) / total_img) * 100, 2)

    return data_ratio_min, data_ratio_max


if __name__=='__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--test-root', default=['/data2/sk_data/data_1rd/test_3d',
                                                '/data2/sk_data/data_2rd/test_3d',
                                                '/data2/sk_data/data_3rd/test_3d'],
                        nargs='+', type=str)

    #parser.add_argument('--test-root', default=['/data2/test_3d'], nargs='+', type=str)
    parser.add_argument('--batch-size', default=1, type=int)
    parser.add_argument('--work-dir', default='/data1/workspace/geongyu/sk_proj/exp_data3th')
    parser.add_argument('--exp', type=str)
    parser.add_argument('--arch', default='unet', type=str)
    parser.add_argument('--coordconv-no', default=[0,1], nargs='+', type=int)
    parser.add_argument('--radious', default=False, type=str2bool)

    parser.add_argument('--skip-mode', default='sum', type=str)
    parser.add_argument('--batchnorm-momentum', default=0.1, type=float)


    parser.add_argument('--padding-size', default=1, type=int)
    parser.add_argument('--multi-input', default=1, type=int)
    parser.add_argument('--file-name', default='fffffffffffffff', type=str)

    args = parser.parse_args()

    main_test(args=args)
    # test24_diceloss
    # /Users/s