
import os
import glob
import itertools

import numpy as np

from skimage import io


def generate_patches(data_root,
                     data_id,
                     dst_dir,
                     patch_size=(48,48,48),
                     stride=(1,16,16),
                     target_depth_for_padding=45,
                     bg_sampling_prob=0.1):

    """
    generate_patches : make 3d patch

    Args:
        data_root : load path
        data_id : load folder
        dst_dir : save folder name
        patch_size : c x w x h
        stride : channel is 1 and only applies to w, h
        target_depth_for_padding : depth of maxium patch
        bg_sampling_prob : threshold of background

    """



    def _load_exam(data_dir, ftype='png'):

        """
        _load_exam : concat 2d images in a folder

        """

        file_extension = '.'.join(['*', ftype])
        data_paths = glob.glob(os.path.join(data_dir, file_extension))
        data_paths = sorted(data_paths, key=lambda x: x.split('/')[-1]) # sort by filename

        slices = []
        for data_path in data_paths:
            arr = io.imread(data_path)
            slices.append(arr)

        data_3d = np.stack(slices) # data.shape = (depth, height, width)

        return data_3d

    def _padding_3d(data_3d, target_length, padding_value=0):

        """
        _padding_3d : pad up and down by target_length

        """

        d, h, w = data_3d.shape
        margin = target_length - d
        padding_size = margin // 2
        upper_padding_size = padding_size
        lower_padding_size = margin - upper_padding_size

        padded = np.pad(data_3d, ((upper_padding_size, lower_padding_size), (0,0), (0,0)),
                        'constant', constant_values=(padding_value,padding_value))

        return padded




    image_dir = os.path.join(data_root, 'images', data_id)
    mask_dir = os.path.join(data_root, 'masks', data_id)

    import ipdb; ipdb.set_trace()
    image_3d = _load_exam(image_dir, ftype='png')
    mask_3d = _load_exam(mask_dir, ftype='gif')

    input_d, input_h, input_w = image_3d.shape
    patch_d, patch_h, patch_w = patch_size

    padded_image_3d = _padding_3d(image_3d, target_length=target_depth_for_padding)
    padded_mask_3d = _padding_3d(mask_3d, target_length=target_depth_for_padding)

    assert padded_image_3d.shape == padded_mask_3d.shape, \
        'The shapes of 3D image and mask are not the same.'

    # generate patches
    padded_d, padded_h, padded_w = padded_image_3d.shape
    stride_d, stride_h, stride_w = stride
    if ((padded_h - patch_h) % stride_h != 0) or \
       ((padded_d - patch_d) % stride_d != 0):
        import ipdb; ipdb.set_trace()
        print('Some boundary voxels might be lost!')

    patch_image_dir = os.path.join(dst_dir, 'images', data_id)
    patch_mask_dir = os.path.join(dst_dir, 'masks', data_id)
    if not os.path.exists(dst_dir): os.makedirs(dst_dir)
    if not os.path.exists(patch_image_dir): os.makedirs(patch_image_dir)
    if not os.path.exists(patch_mask_dir): os.makedirs(patch_mask_dir)

    i_list = np.arange(start=patch_w, stop=padded_w+1, step=stride_w)
    j_list = np.arange(start=patch_h, stop=padded_h+1, step=stride_h)
    k_list = np.arange(start=patch_d, stop=padded_d+1, step=stride_d)

    k_idxs, j_idxs, i_idxs = np.meshgrid(k_list, j_list, i_list, indexing='ij')
    kji_list = list(zip(k_idxs.flatten(), j_idxs.flatten(), i_idxs.flatten()))

    count = 0
    count_bg_patch = 0
    count_roi_patch = 0

    for kji_end in kji_list:
        k_idx, j_idx, i_idx = kji_end
        patch_image_3d = padded_image_3d[k_idx-patch_d:k_idx, j_idx-patch_h:j_idx, i_idx-patch_w:i_idx]
        patch_mask_3d = padded_mask_3d[k_idx-patch_d:k_idx, j_idx-patch_h:j_idx, i_idx-patch_w:i_idx]

        if patch_image_3d.shape != patch_size:
            import ipdb; ipdb.set_trace()

        # sample valid background and
        # discard input patch that has many zero pixels
        if patch_mask_3d.sum() == 0:
            nonzero_ratio = np.count_nonzero(patch_image_3d) / patch_image_3d.size
            if nonzero_ratio > 0.1: # if zero pixel ratio < 0.9
                if np.random.random() <= bg_sampling_prob:
                    count_bg_patch += 1
                else:
                    continue
            else:
                continue
        else:
            count_roi_patch += 1

        patch_image_path = os.path.join(patch_image_dir, 'patch_{:0>5}.npy'.format(count))
        patch_mask_path = os.path.join(patch_mask_dir, 'patch_{:0>5}_mask.npy'.format(count))

        # save a patch
        np.save(patch_image_path, patch_image_3d)
        np.save(patch_mask_path, patch_mask_3d)

        count += 1

    return (data_id, {'n_bg_patches': count_bg_patch, 'n_roi_patchs': count_roi_patch})

if __name__=='__main__':


    def generate_patch(mode,patchsize,stride,bg_prob,nonzero):

        # training
        if mode == 'train':
            data_root = ['/data2/woans0104/sk_hemorrhage_dataset/data_1rd/trainvalid_3d',
                         '/data2/woans0104/sk_hemorrhage_dataset/data_2rd/trainvalid_3d',
                         '/data2/woans0104/sk_hemorrhage_dataset/data_3rd/trainvalid_3d',
                         '/data2/woans0104/sk_hemorrhage_dataset/data_4rd/trainvalid_3d'
                         ]
            dst_dir = '/data1/JM/segmentation_3d/data2th_trainvalid_3d_patches_48_{}_{}_st_{}_bg_{}_nonzero_{}'.format(patchsize,patchsize,stride,bg_prob,nonzero)

        # test
        elif mode == 'test':
            data_root = ['/data2/woans0104/sk_hemorrhage_dataset/data_1rd/test_3d',
                         '/data2/woans0104/sk_hemorrhage_dataset/data_2rd/test_3d',
                         '/data2/woans0104/sk_hemorrhage_dataset/data_3rd/test_3d',
                         '/data2/woans0104/sk_hemorrhage_dataset/data_4rd/test_3d'
                         ]
            dst_dir = '/data1/JM/segmentation_3d/data2th_test_3d_patches_48_{}_{}_st_{}_bg_{}_nonzero_{}'.format(patchsize,patchsize,stride,bg_prob,nonzero)


        data_ids_list = []
        for i in range(len(data_root)):
            data_ids = os.listdir(os.path.join(data_root[i], 'images'))
            data_ids_list.append(data_ids)


        generation_results = []
        for j in range(len(data_root)):
            for data_id in data_ids_list[j]:
                res = generate_patches(data_root[j], data_id=data_id, dst_dir=dst_dir,
                                       patch_size=(48,patchsize,patchsize),
                                       stride=(1,stride,stride),
                                       target_depth_for_padding=48,
                                       bg_sampling_prob=bg_prob)

                print('Patches for {} are generated.'.format(data_id))
                generation_results.append(res)


    
    mode_list =['test']
    for k in range(len(mode_list)):
        generate_patch(mode_list[k],patchsize=96,stride=52,bg_prob=0.05,nonzero=0.1)