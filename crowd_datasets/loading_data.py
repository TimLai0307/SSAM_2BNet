import os
import random
import torch
import numpy as np
from torch.utils.data import Dataset
from PIL import Image
import cv2
import glob
import scipy.io as io
import h5py
import torchvision.transforms as standard_transforms


# DeNormalize used to get original images
class DeNormalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        for t, m, s in zip(tensor, self.mean, self.std):
            t.mul_(s).add_(m)
        return tensor


def get_data(data_root):
    # the pre-proccessing transform
    transform = standard_transforms.Compose([
        standard_transforms.ToTensor(),
        standard_transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                      std=[0.229, 0.224, 0.225]),
    ])
    # create the training dataset
    train_set = get_data_input_gt(data_root, train=True, transform=transform, patch=True, flip=True)
    # create the validation dataset
    val_set = get_data_input_gt(data_root, train=False, transform=transform)

    return train_set, val_set


class get_data_input_gt(Dataset):
    def __init__(self, data_root, transform=None, train=False, patch=False, flip=False):
        self.root_path = data_root
        self.train_lists = "train.txt"
        self.eval_list = "test.txt"
        # there may exist multiple list files
        self.img_list_file = self.train_lists.split(',')
        if train:
            self.img_list_file = self.train_lists.split(',')
        else:
            self.img_list_file = self.eval_list.split(',')

        self.img_map = {}
        self.img_list = []
        # loads the image/gt pairs
        for _, train_list in enumerate(self.img_list_file):
            train_list = train_list.strip()
            with open(os.path.join(self.root_path, train_list)) as fin:
                for line in fin:
                    if len(line) < 2:
                        continue
                    line = line.strip().split()
                    self.img_map[os.path.join(self.root_path, line[0].strip())] = \
                        os.path.join(self.root_path, line[1].strip())
        self.img_list = sorted(list(self.img_map.keys()))
        # number of samples
        self.nSamples = len(self.img_list)

        self.transform = transform
        self.train = train
        self.patch = patch
        self.flip = flip

    def __len__(self):
        return self.nSamples

    def __getitem__(self, index):
        assert index <= len(self), 'index range error'

        img_path = self.img_list[index]
        gtp_path = self.img_map[img_path]
        gtd_path = gtp_path.replace('txt', 'h5')

        # load image and ground truth
        img, point, den = load_data((img_path, gtp_path, gtd_path), self.train)
        # applu augumentation
        if self.transform is not None:
            img = self.transform(img)

        # random crop augumentaiton
        if self.train and self.patch:
            img, point, den = random_crop(img, point, den)
            for i, _ in enumerate(point):
                point[i] = torch.Tensor(point[i])

        # random flipping
        if random.random() > 0.5 and self.train and self.flip:
            img = torch.Tensor(img[:, :, :, ::-1].copy())
            den = torch.Tensor(den[:, :, ::-1].copy())
            for i, _ in enumerate(point):
                if len(point[i]) == 0:
                    point = point
                else:
                    point[i][:, 0] = 128 - point[i][:, 0]

        if not self.train:
            point = [point]
            den = den.unsqueeze(0)

        img = torch.Tensor(img)
        # pack up related infos
        target = [{} for i in range(len(point))]
        for i, _ in enumerate(point):
            target[i]['point'] = torch.Tensor(point[i])
            image_id = int(img_path.split('/')[-1].split('.')[0].split('_')[-1])
            image_id = torch.Tensor([image_id]).long()
            target[i]['image_id'] = image_id
            target[i]['labels'] = torch.ones([point[i].shape[0]]).long()
            target[i]['den_map'] = torch.Tensor(den[i])

        return img, target


def load_data(img_gt_path, train):
    img_path, gtp_path, gtd_path = img_gt_path
    # load the images
    img = cv2.imread(img_path)
    # img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    # load ground truth points
    points = []
    with open(gtp_path) as f_label:
        for line in f_label:
            x = float(line.strip().split(' ')[0])
            y = float(line.strip().split(' ')[1])
            points.append([x, y])

    gtd_file = h5py.File(gtd_path)
    den = np.asarray(gtd_file['density'])
    den = torch.Tensor(den)

    return img, np.array(points), den


# random crop augumentation
def random_crop(img, point, den, num_patch=4):
    half_h = 128
    half_w = 128
    result_img = np.zeros([num_patch, img.shape[0], half_h, half_w])
    result_point = []
    result_den = np.zeros([num_patch, half_h, half_w])
    # crop num_patch for each image
    for i in range(num_patch):
        start_h = random.randint(0, img.size(1) - half_h)
        start_w = random.randint(0, img.size(2) - half_w)
        end_h = start_h + half_h
        end_w = start_w + half_w
        # copy the cropped rect
        result_img[i] = img[:, start_h:end_h, start_w:end_w]
        # copy the cropped points
        if len(point) == 0:
            record_point = []
        else:
            idx = (point[:, 0] >= start_w) & (point[:, 0] <= end_w) & (point[:, 1] >= start_h) & (
                    point[:, 1] <= end_h)
            # shift the corrdinates
            record_point = point[idx]
            record_point[:, 0] -= start_w
            record_point[:, 1] -= start_h

        result_point.append(record_point)

        den_crop = den[start_h:end_h, start_w:end_w]
        result_den[i] = den_crop

    return result_img, result_point, result_den