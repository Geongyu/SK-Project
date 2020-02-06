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

from model import UNet3D
from dataset import DatasetTest
from utils import Logger, send_slack_message



def predict(model, exam_root, input_stats, args=None):
    tst_dataset = DatasetTest(exam_root, options=args, input_stats=input_stats)
    tst_loader = torch.utils.data.DataLoader(tst_dataset,
                                             batch_size=args.batch_size,
                                             shuffle=False,
                                             num_workers=4)

    # original input and target
    org_input_3d = tst_dataset.input_3d  # [0,255]
    org_target_3d = tst_dataset.target_3d  # [0,1]

    # padding info
    upper_padding_size = tst_dataset.upper_padding_size
    lower_padding_size = tst_dataset.lower_padding_size

    # create 1D tensor to store probs and counts
    input_shape = tst_dataset.padded_input_3d.shape
    prediction_size = tst_dataset.padded_input_3d.size
    prob_tensor = np.zeros((prediction_size,))
    count_tensor = np.zeros((prediction_size,))

    model.eval()

    with torch.no_grad():
        for i, (input_patch, index_patch) in enumerate(tst_loader):
            input_patch = input_patch.cuda()
            output_patch = model(input_patch)

            # convert to prob
            pos_probs = torch.sigmoid(output_patch)
            pos_probs = pos_probs.cpu().numpy()

            prob_tensor[index_patch.flatten()] += pos_probs.flatten()
            count_tensor[index_patch.flatten()] += 1

    output_prob = prob_tensor / count_tensor
    output_prob = output_prob.reshape(input_shape)

    # remove padded area
    output_prob = output_prob[upper_padding_size:-lower_padding_size, :, :]

    return output_prob, org_input_3d, org_target_3d


def seperate_dict(ori_dict, serch_list):
    new_dict = {}
    for i in serch_list:
        if i in ori_dict:
            new_dict[i] = ori_dict[i]
    return new_dict


def performance_by_slice(output_3d, target_3d):
    preds = (output_3d > 0.5).astype('float')

    performance = {}
    for slice_idx, slice_pred in enumerate(preds):
        slice_target = target_3d[slice_idx, :, :]

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
        performance[str(slice_idx)] = {'cls': [tp, fp, tn, fn],
                                       'seg': [iou, dice],
                                       'gt': slice_target,
                                       'pred': slice_pred}
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
             slice_level_performance, result_dir):
    def _overlay_mask(img, mask, color='red'):

        # convert gray to color
        color_img = np.dstack([img, img, img])
        mask_idx = np.where(mask == 1)
        if color == 'red':
            color_img[mask_idx[0], mask_idx[1], :] = np.array([255, 0, 0])
        elif color == 'blue':
            color_img[mask_idx[0], mask_idx[1], :] = np.array([0, 0, 255])

        return color_img

    result_exam_dir = os.path.join(result_dir, exam_id)
    if not os.path.exists(result_exam_dir):
        os.makedirs(result_exam_dir)

    n_slices = org_input.shape[0]
    assert (n_slices == org_target.shape[0] \
            == prediction.shape[0] \
            == len(slice_level_performance)), '# of results not matched.'

    # convert prob to pred
    prediction = (prediction > 0.5).astype('float')

    for slice_id in slice_level_performance:
        iou, dice = slice_level_performance[slice_id]['seg']
        input_slice = org_input[int(slice_id), :, :]
        target_slice = org_target[int(slice_id), :, :]
        pred_slice = prediction[int(slice_id), :, :]

        fig = plt.figure(figsize=(15, 5))
        ax = []
        # show original img
        ax.append(fig.add_subplot(1, 3, 1))
        plt.imshow(input_slice, 'gray')
        # show img with gt
        ax.append(fig.add_subplot(1, 3, 2))
        plt.imshow(_overlay_mask(input_slice, target_slice, color='red'))
        # show img with pred
        ax.append(fig.add_subplot(1, 3, 3))
        plt.imshow(_overlay_mask(input_slice, pred_slice, color='blue'))
        ax[-1].set_title('IoU = {0:.4f}'.format(iou))

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
            model = UNet3D(1, 1, f_maps=args.f_maps, depth_stride=args.depth_stride, conv_layer_order='cbr',
                           num_groups=1)
            model = nn.DataParallel(model).cuda()

        # load model
        checkpoint_path = os.path.join(work_dir, 'model_best.pth')
        state = torch.load(checkpoint_path)

        model.load_state_dict(state['state_dict'])
        cudnn.benchmark = True

    input_stats = np.load(os.path.join(work_dir, 'input_stats.npy'), allow_pickle=True).tolist()

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

        # list exam ids
        collated_performance = {}
        for i in range(len(args.test_root)):
            exam_ids = os.listdir(os.path.join(args.test_root[i], 'images'))
            for exam_id in exam_ids:
                print('Processing {}'.format(exam_id))
                exam_path = os.path.join(args.test_root[i], 'images', exam_id)  # '/data2/test_3d/images/403'
                prediction_list, org_input_list, org_target_list = predict(model, exam_path, input_stats, args=args)

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
                assert count == 1, 'duplicate folder'

                result_dir_sep = os.path.join(result_dir, find_folder)
                # save_fig(exam_id, org_input_list, org_target_list, prediction_list, performance, result_dir_sep)

                collated_performance[exam_id] = performance

        for data_no, level_no in allData_dic.items():
            for level_key, level_val in level_no.items():
                sep_dict = seperate_dict(collated_performance, level_val)
                if len(sep_dict) == 0:
                    continue
                sep_performance = compute_overall_performance(sep_dict)

                with open(os.path.join(result_dir, '{}_performance.json'.fomat(level_key)), 'w') as f:
                    json.dump(sep_performance, f)


#
#
# def main_test(model=None, args=None, val_mode=False):
#     work_dir = os.path.join(args.work_dir, args.exp)
#     file_name = args.file_name
#     if not val_mode:
#         result_dir = os.path.join(work_dir, file_name)
#         if not os.path.exists(result_dir):
#             os.makedirs(result_dir)
#
#         # load model and input stats
#         # Note: here, the model should be given manually
#         # TODO: try to import model configuration later
#         if model is None:
#             model = UNet3D(1, 1, f_maps=args.f_maps, depth_stride=args.depth_stride, conv_layer_order='cbr',
#                            num_groups=1)
#             model = nn.DataParallel(model).cuda()
#
#         # load model
#         checkpoint_path = os.path.join(work_dir, 'model_best.pth')
#         state = torch.load(checkpoint_path)
#
#         model.load_state_dict(state['state_dict'])
#         cudnn.benchmark = True
#
#     input_stats = np.load(os.path.join(work_dir, 'input_stats.npy'), allow_pickle=True).tolist()
#
#     # return overall_performance
#
#     if not val_mode:
#         # filepath
#         # import ipdb; ipdb.set_trace()
#
#         overall_data2th = []
#         overall_data3th = []
#
#         data1th = os.listdir(os.path.join(args.test_root[0], 'images'))
#         data2th = os.listdir(os.path.join(args.test_root[1], 'images'))
#         data3th = os.listdir(os.path.join(test_root[2], 'images'))
#
#         top_1th = ['401', '403', '426', '432', '450', '490', '514']
#         top_2th = data2th  # test2 all top level
#         top_3th = ['59_KKO', '562', '564', '566', '583', '584', '599']
#
#         middle_1th = ['410', '425', '433', '441', '443', '513', '518']
#         middle_3th = ['61_CDJ', '66_YYB', '70_PJH', '575', '576', '590', '595']
#
#         low_1th = ['405', '411', '430', '459', '464', '480']
#         low_3th = ['56_KMK', '63_JJW', '72_TKH', '561', '567', '585']
#
#         overall_data2th.extend(data1th)
#         overall_data2th.extend(data2th)
#
#         overall_data3th.extend(overall_data2th)
#         overall_data3th.extend(data3th)
#
#         top_1th_result_dir = os.path.join(result_dir, 'top_1th')
#         top_2th_result_dir = os.path.join(result_dir, 'top_2th')
#         middle_result_dir = os.path.join(result_dir, 'middle')
#         low_result_dir = os.path.join(result_dir, 'low')
#         import ipdb;
#         ipdb.set_trace()
#         # list exam ids
#         collated_performance = {}
#         for i in range(len(args.test_root)):
#             exam_ids = os.listdir(os.path.join(args.test_root[i], 'images'))
#             for exam_id in exam_ids:
#                 print('Processing {}'.format(exam_id))
#                 exam_path = os.path.join(args.test_root[i], 'images', exam_id)  # '/data2/test_3d/images/403'
#                 prediction_list, org_input_list, org_target_list = predict(model, exam_path, input_stats, args=args)
#
#                 # measure performance
#                 performance = performance_by_slice(prediction_list, org_target_list)
#
#                 # save results as figure
#                 if not val_mode:
#                     # exam_id serch top or middle or low list
#                     if exam_id in top_1th:
#                         result_dir_sep = top_1th_result_dir
#                     elif exam_id in top_2th:
#                         result_dir_sep = top_2th_result_dir
#                     elif exam_id in middle:
#                         result_dir_sep = middle_result_dir
#                     elif exam_id in low:
#                         result_dir_sep = low_result_dir
#                     else:
#                         raise ValueError('Not directory.')
#
#                     if not os.path.exists(result_dir_sep):
#                         os.makedirs(result_dir_sep)
#
#                     save_fig(exam_id, org_input_list, org_target_list, prediction_list, performance, result_dir_sep)
#
#                 collated_performance[exam_id] = performance
#
#         # seperate dict
#         top_1th_sep_dict = seperate_dict(collated_performance, top_1th)
#         top_2th_sep_dict = seperate_dict(collated_performance, top_2th)
#         low_sep_dict = seperate_dict(collated_performance, low)
#         middle_sep_dict = seperate_dict(collated_performance, middle)
#
#         # overall dict
#         data1th_sep_dict = seperate_dict(collated_performance, data1th)
#         data2th_sep_dict = seperate_dict(collated_performance, data2th)
#         all_dict = collated_performance
#
#         # compute performance and save
#         top_1th_overall_performance = compute_overall_performance(top_1th_sep_dict)
#         top_2th_overall_performance = compute_overall_performance(top_2th_sep_dict)
#         low_overall_performance = compute_overall_performance(low_sep_dict)
#         middle_overall_performance = compute_overall_performance(middle_sep_dict)
#
#         data1th_overall_performance = compute_overall_performance(data1th_sep_dict)
#         data2th_overall_performance = compute_overall_performance(data2th_sep_dict)
#         all_overall_performance = compute_overall_performance(all_dict)
#
#         with open(os.path.join(top_1th_result_dir, 'top_1th_performance.json'), 'w') as f:
#             json.dump(top_1th_overall_performance, f)
#         with open(os.path.join(top_2th_result_dir, 'top_2th_performance.json'), 'w') as f:
#             json.dump(top_2th_overall_performance, f)
#         with open(os.path.join(low_result_dir, 'low_performance.json'), 'w') as f:
#             json.dump(low_overall_performance, f)
#         with open(os.path.join(middle_result_dir, 'middle_performance.json'), 'w') as f:
#             json.dump(middle_overall_performance, f)
#
#         with open(os.path.join(result_dir, 'data1th_only_performance.json'), 'w') as f:
#             json.dump(data1th_overall_performance, f)
#         with open(os.path.join(result_dir, 'data2th_only_performance.json'), 'w') as f:
#             json.dump(data2th_overall_performance, f)
#         with open(os.path.join(result_dir, 'all_performance.json'), 'w') as f:
#             json.dump(all_overall_performance, f)
#
#     """
#    # list exam ids
#     exam_ids = os.listdir(os.path.join(args.test_root, 'images'))
#
#     collated_performance = {}
#     for exam_id in exam_ids:
#         print('Processing {}'.format(exam_id))
#         exam_path = os.path.join(args.test_root, 'images', exam_id)
#         prediction, org_input, org_target = predict(model, exam_path, input_stats, args=args)
#
#         # measure performance
#         performance = performance_by_slice(prediction, org_target)
#
#         if not val_mode:
#             # save results as figure
#             save_fig(exam_id, org_input, org_target, prediction, performance, result_dir)
#
#         collated_performance[exam_id] = performance
#
#     # compute overall performance and save
#     overall_performance = compute_overall_performance(collated_performance)
#
#     with open(os.path.join(result_dir, 'performance.pkl'), 'wb') as f:
#         pickle.dump(overall_performance, f)
#
#     if not val_mode:
#         with open(os.path.join(result_dir, 'performance.json'), 'w') as f:
#             json.dump(overall_performance, f)
#     """
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--test-root', default=['/data2/woans0104/sk_hemorrhage_dataset/data_2rd/test_3d',
                                                # '/data2/woans0104/sk_hemorrhage_dataset/data_2rd/test_3d',
                                                # '/data2/woans0104/sk_hemorrhage_dataset/data_3rd/test_3d'
                                                ], nargs='+', type=str)
    parser.add_argument('--input-size', default=[48, 96, 96], nargs='+', type=int)
    parser.add_argument('--stride-test', default=[1, 16, 16], nargs='+', type=int)

    parser.add_argument('--f-maps', default=[32, 64, 128, 256], nargs='+', type=int)
    parser.add_argument('--depth-stride', default=[2, 2, 2, 2], nargs='+', type=int)
    parser.add_argument('--augment', default=None, nargs='+', type=str)
    parser.add_argument('--target-depth-for-padding', default=None, type=int)
    parser.add_argument('--batch-size', default=1, type=int)
    parser.add_argument('--work-dir', default='/data1/JM/segmentation_3d')
    parser.add_argument('--exp', type=str)
    parser.add_argument('--file-name', default='test_delete_ok', type=str)

    args = parser.parse_args()

    main_test(args=args)

