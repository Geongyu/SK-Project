
import os
import glob
import itertools
import importlib

import numpy as np

import torch
from torch.utils.data import Dataset
from torchvision.transforms import Compose

from utils import load_exam, pad_3d


def make_data_list(data_root,data_dir):

    if type(data_dir) is not list:

        image_root = os.path.join(data_root, 'images')
        mask_root = os.path.join(data_root, 'masks')
        exam_ids = os.listdir(image_root)
        assert set(exam_ids) == set(os.listdir(mask_root)), \
            'Exam ids are not matched.'

        image_paths = [f for f in glob.glob(os.path.join(image_root, "**/*.npy"), recursive=True)]
        mask_paths = [f for f in glob.glob(os.path.join(mask_root, "**/*.npy"), recursive=True)]

        assert len(image_paths) == len(mask_paths), '# of patches are not matched.'

        # make mask paths by replacing some strings
        mask_paths = []
        for image_path in image_paths:
            mask_path = image_path.replace('images', 'masks').replace('.npy', '_mask.npy')
            mask_paths.append(mask_path)

        print('Making data list completed.')

        return list(zip(image_paths, mask_paths))

    else:
        image_root = os.path.join(data_root, 'images')
        mask_root = os.path.join(data_root, 'masks')
        exam_ids_set = data_dir
        exam_ids = os.listdir(image_root)

        image_paths=[]
        mask_paths=[]
        for i in range(len(exam_ids)):
            if exam_ids[i] in exam_ids_set:
                image_paths.extend(glob.glob(os.path.join(os.path.join(image_root,exam_ids[i]), "*.npy")))
                mask_paths.extend(glob.glob(os.path.join(os.path.join(mask_root,exam_ids[i]), "*.npy")))

        assert len(image_paths) == len(mask_paths), '# of patches are not matched.'

        # make mask paths by replacing some strings
        mask_paths = []
        for image_path in image_paths:
            mask_path = image_path.replace('images', 'masks').replace('.npy', '_mask.npy')
            mask_paths.append(mask_path)
        print('Making data list completed.')


        return list(zip(image_paths, mask_paths))




class DatasetTrain(Dataset):

    def __init__(self, train_root,train_dir, options=None, input_stats=None):

        self.trn_data = make_data_list(train_root,train_dir)
        self.options = options
        self.patch_d, self.patch_h, self.patch_w = options.input_size

        # augmentation
        if self.options.augment is not None:
            transforms = importlib.import_module('transforms')
            transforms_list = [getattr(transforms, augment) for augment in self.options.augment]

            rs_input = np.random.RandomState(seed=75)
            rs_mask = np.random.RandomState(seed=75)

            self.augment_input = Compose([transform(rs_input, self.options) for transform in transforms_list])
            self.augment_mask = Compose([transform(rs_mask, self.options) for transform in transforms_list])

        # TODO: clean below lines
        if input_stats is None:
            self.input_stats = self.estimate_input_stats()
        elif type(input_stats) is list:
            mean, std = input_stats
            if (self.options.augment is not None) and ('RandomCrop' in self.options.augment):
                crop_d, crop_h, crop_w = self.options.crop_size
                mean = np.full((crop_d, crop_h, crop_w), mean)
                std = np.full((crop_d, crop_h, crop_w), std)
            else:
                mean = np.full((self.patch_d, self.patch_h, self.patch_w), mean)
                std = np.full((self.patch_d, self.patch_h, self.patch_w), std)
            self.input_stats = {'mean': mean, 'std': std}
        else:
            self.input_stats = input_stats

    def __len__(self):
        return len(self.trn_data)

    def __getitem__(self, idx):

        img_path, mask_path = self.trn_data[idx]

        # {0,255} -> {0,1}
        img =  np.load(img_path).astype(np.float32) / 255
        mask = np.load(mask_path).astype(np.float32) / 255

        # augmentation
        if self.options.augment is not None:
            img = self.augment_input(img).copy()
            mask = self.augment_mask(mask).copy()

        # normalize
        img = (img - self.input_stats['mean']) / (self.input_stats['std'])

        img = np.expand_dims(img, 0) # add channel
        mask = np.expand_dims(mask, 0)

        img = torch.from_numpy(img).float()
        mask = torch.from_numpy(mask).float()


        return (img, mask)

    def estimate_input_stats(self, max_n=1e5):
        """
        A function that calculates the mean and variance to normalize.

        """
        if len(self.trn_data) > max_n:
            sample_paths = np.random.choice(self.trn_data, max_n, replace=False)
        else:
            sample_paths = self.trn_data

        sample_shape = np.load(sample_paths[0][0]).shape
        samples = np.zeros((len(sample_paths), sample_shape[0], sample_shape[1], sample_shape[2]))

        for idx, sample in enumerate(sample_paths):
            sample_arr = np.load(sample[0]).astype(np.float32) / 255
            samples[idx, :, :, :] = sample_arr


        mean = np.mean(samples, dtype=np.float32)
        std = np.std(samples, dtype=np.float32)

        mean = np.full((self.options.input_size, self.options.input_size, self.options.input_size), mean)
        std = np.full((self.options.input_size, self.options.input_size, self.options.input_size), std)
        print('Data statistics are computed.')

        return {'mean': mean, 'std': std}

class DatasetVal(Dataset):

    def __init__(self, val_root,val_dir, options=None, input_stats=None):

        self.val_data = make_data_list(val_root,val_dir)
        self.options = options

        assert (input_stats is not None), 'Normalizing statistics should be given.'
        self.input_stats = input_stats

    def __len__(self):
        return len(self.val_data)

    def __getitem__(self, idx):

        img_path, mask_path = self.val_data[idx]

        img = np.load(img_path).astype(np.float32) / 255
        mask = np.load(mask_path).astype(np.float32) / 255

        # normalize
        img = (img - self.input_stats['mean']) / (self.input_stats['std'])
        img = np.expand_dims(img, 0)
        mask = np.expand_dims(mask, 0)

        img = torch.from_numpy(img).float()
        mask = torch.from_numpy(mask).float()

        return (img, mask)

class DatasetTest(Dataset):

    def __init__(self, exam_root, options=None, input_stats=None):

        self.options = options
        assert (input_stats is not None), 'Normalizing statistics should be given.'
        self.input_stats = input_stats

        # if 'RandomCrop', patch size should be changed.
        if (self.options.augment is not None) and ('RandomCrop' in self.options.augment):
            self.patch_d, self.patch_h, self.patch_w = self.options.crop_size
        else:
            self.patch_d, self.patch_h, self.patch_w = self.options.input_size

        self.stride_d, self.stride_h, self.stride_w = options.stride_test

        self.input_3d = load_exam(exam_root, ftype='png')
        target_root = exam_root.replace('images','masks')
        self.target_3d = load_exam(target_root, ftype='gif') / 255 # [0,255] -> [0,1]

        # padding
        if self.options.target_depth_for_padding is None:
            self.padding_depth = self.patch_d
        else:
            self.padding_depth = self.options.target_depth_for_padding
        self.padded_input_3d, \
        (self.upper_padding_size, self.lower_padding_size) = pad_3d(self.input_3d, self.padding_depth)
        self.padded_target_3d, _ = pad_3d(self.target_3d, self.padding_depth)

        assert self.padded_input_3d.size == self.padded_target_3d.size, \
               'Sizes of input and target are differ.'

        # make index tensor whose shape is equal to that of input_3d
        self.index_3d = np.arange(self.padded_input_3d.size).reshape(self.padded_input_3d.shape)

        # generate patches
        self.patch_list = self._generate_patch(self.padded_input_3d,
                                               self.padded_target_3d,
                                               self.index_3d)

    def _generate_patch(self, input_3d, target_3d, index_3d):

        input_d, input_h, input_w = input_3d.shape

        i_list = np.arange(start=self.patch_w, stop=input_w+1, step=self.stride_w)
        j_list = np.arange(start=self.patch_h, stop=input_h+1, step=self.stride_h)
        k_list = np.arange(start=self.patch_d, stop=input_d+1, step=self.stride_d)

        k_idxs, j_idxs, i_idxs = np.meshgrid(k_list, j_list, i_list, indexing='ij')
        kji_list = list(zip(k_idxs.flatten(), j_idxs.flatten(), i_idxs.flatten()))

        patch_list = []
        for kji_end in kji_list:
            k_idx, j_idx, i_idx = kji_end
            patch_input_3d = input_3d[k_idx-self.patch_d:k_idx, j_idx-self.patch_h:j_idx, i_idx-self.patch_w:i_idx]
            patch_target_3d = target_3d[k_idx-self.patch_d:k_idx, j_idx-self.patch_h:j_idx, i_idx-self.patch_w:i_idx]
            patch_index_3d = index_3d[k_idx-self.patch_d:k_idx, j_idx-self.patch_h:j_idx, i_idx-self.patch_w:i_idx]
            patch_list.append((patch_input_3d, patch_target_3d, patch_index_3d))

        return patch_list

    def __len__(self):
        return len(self.patch_list)

    def __getitem__(self, idx):

        patch_img, patch_target, patch_idx = self.patch_list[idx]

        patch_img = patch_img.astype(np.float32) / 255

        # normalize
        patch_img = (patch_img - self.input_stats['mean']) / (self.input_stats['std'])
        patch_img = np.expand_dims(patch_img, 0)

        patch_img = torch.from_numpy(patch_img).float()

        return (patch_img, patch_idx) # patch_img.shape = (N, 1, D, H, W), patch_idx.shape = (N, D, H, W)



