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
from utils import Logger



def predict(model, exam_root, args=None):
    tst_dataset = Segmentation_test_data(exam_root)
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
        for i, (input, target, ori_img, idx) in enumerate(tst_loader):

            input = input.cuda()
            output = model(input)

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


def seperate_dict(ori_dict, serch_list):
    new_dict = {}
    for i in serch_list:
        if i in ori_dict:
            new_dict[i] = ori_dict[i]
    return new_dict

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
            if res_slice['gt'].sum() != 0:  # consider only annotated slices
                iou_sum += res_slice['seg'][0]
                dice_sum += res_slice['seg'][1]
                n_valid_slices += 1
    

    iou_mean = iou_sum / n_valid_slices
    dice_mean = dice_sum / n_valid_slices

    return {'confusion_matrix': list(confusion_matrix),
            'slice_level_accuracy': (confusion_matrix[0] + confusion_matrix[2]) / confusion_matrix.sum(),
            'segmentation_performance': [iou_mean, dice_mean]}

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
        #ipdb.set_trace()
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
        ax[1].set_title("Ground Turth Image")
        # show img with pred
        ax.append(fig.add_subplot(1,3,3))
        plt.imshow(_overlay_mask(input_slice, pred_slice, color='blue'))
        ax[-1].set_title('Predict Image \n IoU = {0:.4f} \ Dice = {1:.4f}'.format(iou, dice))

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

def main_test(model=None, args=None, val_mode=False):
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
            model = Unet2D((1, 512, 512))
            model = nn.DataParallel(model).cuda()

        # load model
        checkpoint_path = os.path.join(work_dir, 'model_best.pth')
        state = torch.load(checkpoint_path)

        model.load_state_dict(state['state_dict'])
        cudnn.benchmark = True

    # return overall_performance

    if not val_mode:
        # filepath

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
                'overal_2th': ['52_KMK', '483', '29_MOY', '46_YMS', '40_LSH', '534', '8_KYK', '535', '536', '500'],
                'overal_3th': ['575', '567', '70_PJH', '561', '583', '72_TKH', '564', '56_KMK', '599', '61_CDJ',
                                 '66_YYB', '584', '562', '59_KKO', '585', '590', '566', '595', '576', '63_JJW'],

            }

        }

        # list exam ids
        collated_performance = {}
        for i in range(len(args.test_root)):
            exam_ids = os.listdir(os.path.join(args.test_root[i], 'images'))
            for exam_id in exam_ids:
                print('Processing {}'.format(exam_id))
                exam_path = os.path.join(args.test_root[i], 'images', exam_id)  # '/data2/test_3d/images/403'
                prediction_list, org_input_list, org_target_list = predict(model, exam_path, args=args)

                # measure performance
                performance = performance_by_slice(prediction_list, org_target_list)

                # find folder
                find_folder = ''
                count = 0
                for data_no, level_no in allData_dic.items():
                    for level_key, level_val in level_no.items():
                        if exam_id in level_val:
                            if 'overal' in level_key.split('_'):  # prevent duplicate data save
                                continue
                            find_folder = level_key
                            count += 1
                #import ipdb; ipdb.set_trace()
                assert count == 1, 'duplicate folder'

                result_dir_sep = os.path.join(result_dir, find_folder)
                save_fig(exam_id, org_input_list, org_target_list, prediction_list, performance, result_dir_sep)

                collated_performance[exam_id] = performance
    
        for data_no, level_no in allData_dic.items():
            for level_key, level_val in level_no.items():
                sep_dict = seperate_dict(collated_performance, level_val)
                if len(sep_dict) == 0:
                    continue
                sep_performance = compute_overall_performance(sep_dict)

                with open(os.path.join(result_dir, '{}_performance.json'.format(level_key)), 'w') as f:
                    json.dump(sep_performance, f)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--test-root', default=['/daintlab/data/sk_data/data_1rd/test_3d',
                                                '/daintlab/data/sk_data/data_2rd/test_3d',
                                                '/daintlab/data/sk_data/data_3rd/test_3d'
                                                ], nargs='+', type=str)
    parser.add_argument('--augment', default=None, nargs='+', type=str)
    parser.add_argument('--batch-size', default=1, type=int)
    parser.add_argument('--work-dir', default='/daintlab/workspace/geongyu/sk-test')
    parser.add_argument('--exp', type=str)
    parser.add_argument('--file-name', default='test_delete_ok', type=str)

    args = parser.parse_args()

    main_test(args=args)



