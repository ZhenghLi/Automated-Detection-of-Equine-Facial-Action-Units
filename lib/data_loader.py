import torch
import os
from torch.utils import data
import random
from torchvision import transforms
from PIL import Image
import pandas as pd


class DataSetDefine(data.Dataset):
    """
    base_tensor, angle_tensor, diff_tensor
    """
    def __init__(self, img_list, label_tensor_list, transforms, config):
        self.img_list = img_list
        self.label_tensor_list = label_tensor_list
        self.transforms = transforms
        self.cfg = config

    def __getitem__(self, index):
        img_th = self.get_img_th(self.img_list[index])
        label_th = self.label_tensor_list[index]

        return img_th, label_th

    def get_img_th(self, img_path):
        """
        :param img_path: .jpg
        :return: tensor (channel, height, width)
        """

        with Image.open(img_path) as img:
            img = img.convert('RGB')

        if img is None:
            print(img_path)
            return torch.zeros(3, 200, 200)

        # img = cv2.resize(img, dsize=(self.cfg.width, self.cfg.height))
        img_th = self.transforms(img)

        return img_th

    def __len__(self):
        len_0 = len(self.img_list)
        len_1 = len(self.label_tensor_list)

        assert len_0 == len_1
        return len_0


class DataSet(object):
    def __init__(self, config):
        """
        :param config: config parameters
        :param parse: parse origin images to torch tensor and save to .pkl
        :param load: load torch tensor from .pkl
        """
        self.cfg = config

        self.train_image_list = []
        self.train_label_tensor_list = []

        self.val_image_list = []
        self.val_label_tensor_list = []

        self.test_image_list = []
        self.test_label_tensor_list = []

        self.train_transforms = transforms.Compose([
            transforms.Resize(size=(self.cfg.height, self.cfg.width)),
            transforms.ToTensor(),
            transforms.Normalize(mean = (0.199, 0.174, 0.172), std = (0.120, 0.114, 0.118)),
        ])

        self.val_transforms = transforms.Compose([
            transforms.Resize(size=(self.cfg.height, self.cfg.width)),
            transforms.ToTensor(),
            transforms.Normalize(mean = (0.199, 0.174, 0.172), std = (0.120, 0.114, 0.118)),
        ])

        self.test_transforms = transforms.Compose([
            transforms.Resize(size=(self.cfg.height, self.cfg.width)),
            transforms.ToTensor(),
            transforms.Normalize(mean = (0.199, 0.174, 0.172), std = (0.120, 0.114, 0.118)),
        ])

        self.load_list(data_type='train')
        self.load_list(data_type='val')
        self.load_list(data_type='test')

        self.train_dataset = DataSetDefine(self.train_image_list,
                                           self.train_label_tensor_list,
                                           self.train_transforms,
                                           self.cfg)

        self.val_dataset = DataSetDefine(self.val_image_list,
                                          self.val_label_tensor_list,
                                          self.val_transforms,
                                          self.cfg)

        self.test_dataset = DataSetDefine(self.test_image_list,
                                          self.test_label_tensor_list,
                                          self.test_transforms,
                                          self.cfg)

        self.train_loader = data.DataLoader(dataset=self.train_dataset,
                                            batch_size=self.cfg.train_batch_size,
                                            shuffle=True,
                                            num_workers=0)

        self.val_loader = data.DataLoader(dataset=self.val_dataset,
                                           batch_size=self.cfg.val_batch_size,
                                           shuffle=False,
                                           num_workers=0)

        self.test_loader = data.DataLoader(dataset=self.test_dataset,
                                           batch_size=self.cfg.test_batch_size,
                                           shuffle=False,
                                           num_workers=0)

    def load_list(self, data_type='train'):
        assert data_type in ['train', 'val', 'test']

        if data_type == 'train':
            info_path = self.cfg.train_info
        elif data_type == 'val':
            info_path = self.cfg.val_info
        else:
            info_path = self.cfg.test_info

        line_list = pd.read_excel(info_path)

        for i in range(line_list.shape[0]):
            line = line_list.iloc[i,:]
            name = line[0]

            label = [int(line[1])]

            if data_type == 'train':
                self.train_image_list.append(os.path.join(self.cfg.image_dir, name + ".png"))
                self.train_label_tensor_list.append(torch.LongTensor(label))
            elif data_type == 'val':
                self.val_image_list.append(os.path.join(self.cfg.image_dir, name + ".png"))
                self.val_label_tensor_list.append(torch.LongTensor(label))
            else:
                self.test_image_list.append(os.path.join(self.cfg.image_dir, name + ".png"))
                self.test_label_tensor_list.append(torch.LongTensor(label))


if __name__ == '__main__':
    import config
    obj = DataSet(config)

    for img, label in obj.train_loader:
        print(img.size())
        print(label.size())
