import numpy
import numpy as np
import shutil
import os
import csv
import pandas as pd
import scipy.io as io
import cv2
import glob

path = r'D:\Lai\counting_dataset\test\fix_kernel\jhu++\test\images'
target_path = r'D:\Lai\counting_dataset\test\adaptive_kernel\UCF_QNRF'
cls = 'jhu++'

# SHH
if cls == 'SHH':

    img_list = os.listdir(path + '/' + 'train_data/images')
    file_list = os.listdir(path + '/' + 'train_data/ground-truth')

    for i in range(len(file_list)):

        name = img_list[i].split('.')
        f2 = open(target_path + '/' + name[0] + '.txt', 'w')
        f = io.loadmat(path + '/train_data/ground-truth/' + file_list[i])
        f1 = f['image_info']
        num = len(f1[0][0][0][0][0])

        for j in range(num):
            x = str(f1[0][0][0][0][0][j][0])
            y = str(f1[0][0][0][0][0][j][1])
            f2.write(x + ' ' + y + '\n')

        f2.close()

# NWPU
if cls == 'NWPU':
    img_path = path + 'image'
    img_file = os.listdir(img_path)

    for i in range(len(img_file)):

        mat = io.loadmat((path + 'image/' + img_file[i]).replace('image', 'mats').replace('jpg', 'mat'))
        gt = mat['annPoints']
        num = len(gt)

        if i < 3109:
            target = target_path + 'train/'
            if not os.path.isdir(target):
                os.mkdir(target)
            img = cv2.imread(path + 'image/' + img_file[i])
            cv2.imwrite(target + img_file[i], img)
            with open(target + img_file[i].replace('jpg', 'txt'), 'w') as f:
                for j in range(num):
                    x = str(gt[j][0])
                    y = str(gt[j][1])
                    f.write(x + ' ' + y + '\n')

        else:
            target = target_path + 'test/'
            if not os.path.isdir(target):
                os.mkdir(target)
            img = cv2.imread(path + 'image/' + img_file[i])
            cv2.imwrite(target + img_file[i], img)
            with open(target + img_file[i].replace('jpg', 'txt'), 'w') as f:
                for j in range(num):
                    x = str(gt[j][0])
                    y = str(gt[j][1])
                    f.write(x + ' ' + y + '\n')


if cls == 'QNRF':
    train = os.path.join(path, 'Train')
    test = os.path.join(path, 'Test')
    path_sets = [train, test]
    # path_sets = [test]

    img_paths = []
    for path in path_sets:
        for img_path in glob.glob(os.path.join(path, '*.jpg')):
            img_paths.append(img_path)

    j = 0

    img_paths = img_paths[j:]

    for img_path in img_paths:
        print(img_path, j)
        if j < 1201:
            target_root = target_path + '/train'
            img_name = img_path.split('\\')[-1]
            name = img_name.split('.')
            mat = io.loadmat(img_path.replace('.jpg', '_ann.mat'))
            img = cv2.imread(img_path)
            max_size = max(img.shape)

            gt = mat['annPoints']
            if max_size > 1920:
                scale = 1920 / max_size
                # scale the image and points
                if len(gt) == 0:
                    gt = gt
                else:
                    gt *= scale

            f2 = open(target_path + '/train_dot/' + name[0] + '.txt', 'w')
            for i in range(len(gt)):
                if int(gt[i][1]) < img.shape[0] and int(gt[i][0]) < img.shape[1]:
                    x = str(gt[i][0])
                    y = str(gt[i][1])
                    f2.write(x + ' ' + y + '\n')
            f2.close()

        else:
            target_root = target_path + '/test'
            img_name = img_path.split('\\')[-1]
            name = img_name.split('.')
            mat = io.loadmat(img_path.replace('.jpg', '_ann.mat'))
            img = cv2.imread(img_path)
            max_size = max(img.shape)

            gt = mat['annPoints']
            if max_size > 1920:
                scale = 1920 / max_size
                # scale the image and points
                img = cv2.resize(img, None, fx=scale, fy=scale)
                if len(gt) == 0:
                    gt = gt
                else:
                    gt *= scale
            f2 = open(target_path + '/test_dot/' + name[0] + '.txt', 'w')
            for i in range(len(gt)):
                if int(gt[i][1]) < img.shape[0] and int(gt[i][0]) < img.shape[1]:
                    x = str(gt[i][0])
                    y = str(gt[i][1])
                    f2.write(x + ' ' + y + '\n')
            f2.close()

        j += 1


if cls == 'CC_50':


    img_paths = []
    for img_path in glob.glob(os.path.join(path, '*.jpg')):
        img_paths.append(img_path)

    j = 0

    img_paths = img_paths[j:]

    for img_path in img_paths:
        print(img_path, j)

        target_root = target_path
        img_name = img_path.split('\\')[-1]
        name = img_name.split('.')
        mat = io.loadmat(img_path.replace('.jpg', '_ann.mat'))
        img = cv2.imread(img_path)
        max_size = max(img.shape)

        gt = mat['annPoints']
        if max_size > 1920:
            scale = 1920 / max_size
            # scale the image and points
            if len(gt) == 0:
                gt = gt
            else:
                gt *= scale

        f2 = open(target_path + '/' + name[0] + '.txt', 'w')
        for i in range(len(gt)):
            x = str(gt[i][0])
            y = str(gt[i][1])
            f2.write(x + ' ' + y + '\n')
        f2.close()



        j += 1


# jhu++
if cls == 'jhu++':


    img_paths = []

    for img_file in glob.glob(os.path.join(path, '*.jpg')):
        img_paths.append(img_file)

    for img_path in img_paths:


        point_path = img_path.replace('images', 'gt').replace('.jpg','.txt')
        img = cv2.imread(img_path)
        max_size = max(img.shape)
        min_size = min(img.shape[0], img.shape[1])

        with open(point_path, 'r') as files:
            lines = files.readlines()
            gt = np.zeros((len(lines), 2))
            i = 0
            for line in lines:
                points = line.split(' ')
                gt[i][0] = points[0]
                gt[i][1] = points[1]
                i += 1
        if max_size > 1920:
            scale = 1920 / max_size
            # scale the image and points
            if len(gt) == 0:
                gt = gt
            else:
                gt *= scale


        if min_size < 128:
            scale = 128 / min_size
            # scale the image and points
            img = cv2.resize(img, None, fx=scale, fy=scale)
            if len(gt) == 0:
                gt = gt
            else:
                gt *= scale


        with open(img_path.replace('jpg', 'txt'), 'w') as f:

            for j in range(len(gt)):
                x = str(gt[j][0])
                y = str(gt[j][1])
                f.write(x + ' ' + y + '\n')







