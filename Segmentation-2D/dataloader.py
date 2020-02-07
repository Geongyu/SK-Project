import numpy as np
import torchvision.transforms as transforms
import torch
from torch.utils.data import Dataset
import ast
import os
import glob
from PIL import Image
import pandas as pd
import random
import torch.utils.data as data

def find_file(file_root):
    # Find All interesting Files
    root_folder = file_root
    sub_folder_image = []
    
    for folders in root_folder :
        print(root_folder)
        check = os.listdir(folders + "/images")
        for folder in check :
            sub_folder_image.append(folders + "/images/" + str(folder))
    image_file_path = []
    target_file_path = []
    for images in sub_folder_image :
        image_file_list = glob.glob(images + "/*")
        file_list_img = [file for file in image_file_list if file.endswith(".png") or file.endswith(".gif")]
        file_list_mask = [file.replace("images", "masks").replace(".png", "_mask.gif") for file in image_file_list if file.endswith(".png") or file.endswith(".gif")]
        image_file_path += file_list_img
        target_file_path += file_list_mask

    image_file_path.sort()
    target_file_path.sort()

    return image_file_path, target_file_path

def find_file2(file_root):
    image_file_path = []
    target_file_path = []
    image_file_list = glob.glob(file_root + "/*")
    file_list_img = [file for file in image_file_list if file.endswith(".png") or file.endswith(".gif")]
    file_list_mask = [file.replace("images", "masks").replace(".png", "_mask.gif") for file in image_file_list if file.endswith(".png") or file.endswith(".gif")]
    image_file_path += file_list_img
    target_file_path += file_list_mask

    image_file_path.sort()
    target_file_path.sort()

    return image_file_path, target_file_path

class load_kaggle_data(Dataset) :
    # Use Kaggle RSNA Datasets on Efficient Net
    # For Classification Dataset
    # Baseline Datasets
    def __init__(self, path, label) :
        # Load File List
        self.file_path = path
        file_list = os.listdir(path)
        self.label_file = label
        self.file_list = file_list
        file_list.sort()
        label = pd.read_csv(label)
        label = label.sort_values(by=["ID"], axis=0)
        label = label.reset_index(drop=True)
        # Transform
        self.transforms1 = transforms.ToTensor()
        self.transforms2 = transforms.Normalize([0.5], [0.5])

    def __len__(self) :
        return len(self.file_list)

    def __getitem__(self, idx) :
        # Labelfile type = Pandas Dataframe
        labels = self.label_file.iloc[idx][1]
        labels = torch.tensor([labels]).float()
        images = Image.open(self.file_path + "/" + self.label_file.iloc[idx][0][:-4] + ".png")

        images = self.transforms1(images)
        images = self.transforms2(images)
        # Because Efficient Net need 3 channels
        images = images.repeat(3, 1, 1)

        return images, labels

class load_kaggle_data_with_balanced(Dataset) :
    # Kaggle Datasets Loader
    # Balanced Datasets, Use Upsampling
    def __init__(self, path, label) :
        self.file_path = path
        file_list = os.listdir(path)
        label = pd.read_csv(label)
        label = label.sort_values(by=["ID"], axis=0)
        label = label.reset_index(drop=True)
        self.label_file = label
        self.any_file = self.label_file.loc[self.label_file["any"] == 1].reset_index(drop=True)
        file_list.sort()
        self.file_list = file_list
        self.transforms1 = transforms.ToTensor()
        self.transforms2 = transforms.Normalize([0.5], [0.5])

    def __len__(self) :
        # If idx is over a length of original file list, then choose hemorrhage images
        return int(len(self.file_list) * 1.5)

    def __getitem__(self, idx) :
        if idx >= len(self.file_list) :
            rdx = np.random.randint(97102, size=1)
            rdx = rdx[0]
            labels = self.any_file.iloc[rdx][1]
            labels = torch.tensor([labels]).float()
            images = Image.open(self.file_path + "/" + self.any_file.iloc[rdx].values[0][:-4] + ".png")

            images = self.transforms1(images)
            images = self.transforms2(images)
            # Because Efficient Net need 3 channels
            images = images.repeat(3, 1, 1)
        else :
            labels = self.label_file.iloc[idx][1]
            labels = torch.tensor([labels]).float()
            images = Image.open(self.file_path + "/" + self.label_file.iloc[idx][0][:-4] + ".png")

            images = self.transforms1(images)
            images = self.transforms2(images)
            # Because Efficient Net need 3 channels
            images = images.repeat(3, 1, 1)

        return images, labels

class Classification_Data(Dataset) :
    # Classification Data Use SK datasets (Azu Univ. Hospital Dataset)
    def __init__(self, exam_root) :
        self.transforms1 = transforms.ToTensor()
        self.transforms2 = transforms.Normalize([0.5], [0.5])
        self.file_root = exam_root
        self.image_path, self.target_path = find_file(self.file_root)

    def __len__(self) :
        return len(self.image_path)

    def __getitem__(self, idx):
        image = Image.open(self.image_path[idx])
        label = Image.open(self.target_path[idx])
        image = self.transforms1(image)
        image = self.transforms2(image)
        image = image.repeat(3, 1, 1)
        label = np.array(label)
        if sum(sum(label)) > 10 :
            label = torch.tensor([1]).float()
        else :
            label = torch.tensor([0]).float()
        return image, label

class Segmentation_2d_data(Dataset) :
    def __init__(self, exam_root) :
        self.transforms1 = transforms.ToTensor()
        self.transforms2 = transforms.Normalize([0.5], [0.5])
        self.file_root = exam_root
        self.image_path, self.target_path = find_file(self.file_root)

    def __len__(self) :
        return len(self.image_path)

    def give_the_epoch(self, epoch=0) :
        self.epoch = epoch

    def __getitem__(self, idx):

        image = Image.open(self.image_path[idx])
        label = Image.open(self.target_path[idx])
        label = np.array(label, dtype=np.uint8)
        label = (label != 0) * 1.0
        image = self.transforms1(image)
        image = self.transforms2(image)
        label = self.transforms1(label)

        return image, label, idx

class Segmentation_test_data(Dataset) :
    def __init__(self, exam_root) :
        self.transforms1 = transforms.ToTensor()
        self.transforms2 = transforms.Normalize([0.5], [0.5])
        self.file_root = exam_root
        self.image_path, self.target_path = find_file2(self.file_root)

    def __len__(self) :
        return len(self.image_path)

    def give_the_epoch(self, epoch=0) :
        self.epoch = epoch

    def __getitem__(self, idx):

        image = Image.open(self.image_path[idx])
        ori_image = self.transforms1(image)
        label = Image.open(self.target_path[idx])
        label = np.array(label, dtype=np.uint8)
        label = (label != 0) * 1.0
        image = self.transforms1(image)
        image = self.transforms2(image)
        label = self.transforms1(label)

        return image, label, ori_image, idx



if __name__=='__main__':

    exam_root = ['/data2/sk_data/data_1rd/trainvalid_3d',
                 '/data2/sk_data/data_2rd/trainvalid_3d',
                 '/data2/sk_data/data_3rd/trainvalid_3d',
                 '/data2/sk_data/data_4rd/trainvalid_3d',
                 '/data2/sk_data/data_5rd/trainvalid_3d']

    test_root = ['/data2/sk_data/data_1rd/test_3d',
                '/data2/sk_data/data_2rd/test_3d',
                '/data2/sk_data/data_3rd/test_3d',
                '/data2/sk_data/data_4rd/test_3d',
                '/data2/sk_data/data_5rd/test_3d']

    trainset = Segmentation_test_data(exam_root)

    trainloader = torch.utils.data.DataLoader(trainset,
                                             batch_size=36,
                                             shuffle=False,
                                             num_workers=0)

    #import ipdb; ipdb.set_trace()

    for i, (image, label, idx) in enumerate(trainloader) :
        import ipdb; ipdb.set_trace()
