import h5py
import scipy.io as io
import PIL.Image as Image
import numpy as np
import os
import glob
from matplotlib import pyplot as plt
from scipy.ndimage.filters import gaussian_filter
import scipy
import json
from matplotlib import cm as CM
import torch
import cv2

def gaussian_filter_density_fix(gt):
    print(gt.shape)
    density = np.zeros(gt.shape, dtype=np.float32)
    gt_count = np.count_nonzero(gt)

    if gt_count == 0:
        return density

    print('generate density...')
    sigma = 4
    density += scipy.ndimage.filters.gaussian_filter(gt, sigma, mode='constant')
    print('done.')

    return density

# def gaussian_filter_density(gt):
#     print(gt.shape)
#     density = np.zeros(gt.shape, dtype=np.float32)
#     gt_count = np.count_nonzero(gt)
#     print('gt_count = ' + str(gt_count))
#     if gt_count == 0:
#         return density
#
#     pts = np.array(list(zip(np.nonzero(gt)[1], np.nonzero(gt)[0])))
#     leafsize = 2048
#     # build kdtree
#     tree = scipy.spatial.KDTree(pts.copy(), leafsize=leafsize)
#     # query kdtree
#     if gt_count<4:
#         distances, locations = tree.query(pts, k=gt_count)
#     else:
#         distances, locations = tree.query(pts, k=4)
#
#     print('generate density...')
#     for i, pt in enumerate(pts):
#         pt2d = np.zeros(gt.shape, dtype=np.float32)
#         pt2d[pt[1],pt[0]] = 1.
#         if gt_count == 1:
#             sigma = np.average(np.array(gt.shape)) / 2. / 2.  # case: 1 point
#
#         elif gt_count == 2:
#             sigma = (distances[i][1]) * 0.1
#
#         elif gt_count == 3:
#             sigma = (distances[i][1] + distances[i][2]) * 0.1
#
#         else:
#             sigma = (distances[i][1] + distances[i][2] + distances[i][3]) * 0.1
#
#         density += scipy.ndimage.filters.gaussian_filter(pt2d, sigma, mode='constant')
#     print('done.')
#     return density



root = r'D:\Lai\counting_dataset\ShanghaiTech\part_B'
target_path = r'D:\Lai\counting_dataset\test\fix_kernel\SHHB\train\images'
cls = 'SHH'

if cls == 'SHH':
    train = os.path.join(root,'train_data','images')
    test = os.path.join(root,'test_data','images')
    path_sets = [train, test]

    img_paths = []
    for path in path_sets:
        for img_path in glob.glob(os.path.join(path, '*.jpg')):
            img_paths.append(img_path)

    for img_path in img_paths:
        print(img_path)
        mat = io.loadmat(img_path.replace('.jpg','.mat').replace('images','ground-truth').replace('IMG_','GT_IMG_'))
        img= cv2.imread(img_path)
        img_name = img_path.split('\\')[-1]
        gt = mat["image_info"][0,0][0,0][0]

        if img_path.split('\\')[-3] == 'train_data':
            target_root = target_path + '/train'
        if img_path.split('\\')[-3] == 'test_data':
            target_root = target_path + '/test'

        if not os.path.exists(target_root):
            os.mkdir(target_root)

        # save image
        cv2.imwrite(target_root + '/' + img_name, img)

        # generate density map
        k = np.zeros((img.shape[0], img.shape[1]))
        for i in range(0,len(gt)):
            if int(gt[i][1])<img.shape[0] and int(gt[i][0])<img.shape[1]:
                k[int(gt[i][1]),int(gt[i][0])]=1
        k = gaussian_filter_density_fix(k)
        # print(k.sum())
        with h5py.File((target_root + '/' + img_name).replace('.jpg','.h5'), 'w') as hf:
                hf['density'] = k

        # plt.imshow(k, cmap=CM.jet)
        # plt.axis('off')
        # count = k.sum()
        # plt.savefig('D:/Lai/counting_dataset/test/result/input/ground-truth/' + str(count) + '.jpg', bbox_inches='tight', pad_inches=0)

        # generate point txt
        f2 = open(target_root + '/' + img_name.replace('jpg', 'txt'), 'w')
        for j in range(len(gt)):
            if int(gt[j][1]) < img.shape[0] and int(gt[j][0]) < img.shape[1]:
                x = str(gt[j][0])
                y = str(gt[j][1])
                f2.write(x + ' ' + y + '\n')
        f2.close()



elif cls == 'NWPU':
    path = root + '/image'

    img_paths = []

    for img_path in glob.glob(os.path.join(path, '*.jpg')):
        img_paths.append(img_path)

    j = 0

    img_paths = img_paths[j:]

    for img_path in img_paths:
        print(img_path, j)

        img_name = img_path.split('\\')[-1]
        mat = io.loadmat(img_path.replace('.jpg', '.mat').replace('image', 'mats'))
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

        if j < 3109:
            target_root = target_path + '/train'
        else:
            target_root = target_path + '/test'

        if not os.path.exists(target_root):
            os.mkdir(target_root)

        # save image
        cv2.imwrite(target_root + '/' + img_name, img)

        # generate density map
        k = np.zeros((img.shape[0], img.shape[1]))
        for i in range(0, len(gt)):
            if int(gt[i][1]) < img.shape[0] and int(gt[i][0]) < img.shape[1]:
                k[int(gt[i][1]), int(gt[i][0])] = 1
        k = gaussian_filter_density_fix(k)
        # print(k.sum())
        with h5py.File((target_root + '/' + img_name).replace('.jpg', '.h5'), 'w') as hf:
            hf['density'] = k

        # generate point txt
        f2 = open(target_root + '/' + img_name.replace('jpg', 'txt'), 'w')
        for j in range(len(gt)):
            if int(gt[j][1]) < img.shape[0] and int(gt[j][0]) < img.shape[1]:
                x = str(gt[j][0])
                y = str(gt[j][1])
                f2.write(x + ' ' + y + '\n')
        f2.close()

elif cls == 'UCF_QNRF':
    train = os.path.join(root, 'Train')
    test = os.path.join(root, 'Test')
    path_sets = [train, test]

    img_paths = []
    for path in path_sets:
        for img_path in glob.glob(os.path.join(path, '*.jpg')):
            img_paths.append(img_path)

    for img_path in img_paths:
        print(img_path)

        img_name = img_path.split('\\')[-1]
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

        if img_path.split('\\')[-2] == 'Train':
            target_root = target_path + '/train'
        if img_path.split('\\')[-2] == 'Test':
            target_root = target_path + '/test'

        if not os.path.exists(target_root):
            os.mkdir(target_root)

        # save image
        cv2.imwrite(target_root + '/' + img_name, img)

        # generate density map
        k = np.zeros((img.shape[0], img.shape[1]))
        for i in range(0, len(gt)):
            if int(gt[i][1]) < img.shape[0] and int(gt[i][0]) < img.shape[1]:
                k[int(gt[i][1]), int(gt[i][0])] = 1
        k = gaussian_filter_density_fix(k)
        # print(k.sum())
        with h5py.File((target_root + '/' + img_name).replace('.jpg', '.h5'), 'w') as hf:
            hf['density'] = k

        # generate point txt
        f2 = open(target_root + '/' + img_name.replace('jpg', 'txt'), 'w')
        for j in range(len(gt)):
            if int(gt[j][1]) < img.shape[0] and int(gt[j][0]) < img.shape[1]:
                x = str(gt[j][0])
                y = str(gt[j][1])
                f2.write(x + ' ' + y + '\n')
        f2.close()
        # print('den_count = ' + str(k.sum()))

elif cls == 'UCF_CC_50':

    img_paths = []

    for img_path in glob.glob(os.path.join(root, '*.jpg')):
        img_paths.append(img_path)



    for img_path in img_paths:
        print(img_path)
        target_root = target_path
        img_name = img_path.split('\\')[-1]
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

        # save image
        cv2.imwrite(target_root + '/' + img_name, img)

        # generate density map
        k = np.zeros((img.shape[0], img.shape[1]))
        for i in range(0, len(gt)):
            if int(gt[i][1]) < img.shape[0] and int(gt[i][0]) < img.shape[1]:
                k[int(gt[i][1]), int(gt[i][0])] = 1
        k = gaussian_filter_density_fix(k)
        # print(k.sum())
        with h5py.File((target_root + '/' + img_name).replace('.jpg', '.h5'), 'w') as hf:
            hf['density'] = k

        # generate point txt
        f2 = open(target_root + '/' + img_name.replace('jpg', 'txt'), 'w')
        for j in range(len(gt)):
            if int(gt[j][1]) < img.shape[0] and int(gt[j][0]) < img.shape[1]:
                x = str(gt[j][0])
                y = str(gt[j][1])
                f2.write(x + ' ' + y + '\n')
        f2.close()
        # print('den_count = ' + str(k.sum()))


# jhu++
elif cls == 'jhu++':

    train = os.path.join(root, 'train', 'images')
    test = os.path.join(root, 'test', 'images')
    val = os.path.join(root, 'val', 'images')
    path_sets = [train, test, val]
    img_paths = []

    for img_path in glob.glob(os.path.join(root, '*.jpg')):
        img_paths.append(img_path)

    for img_path in img_paths:
        print(img_path)

        point_path = img_path.replace('images', 'gt').replace('.jpg','.txt')
        with open(point_path, 'r') as files:
            lines = files.readlines()
            gt = np.zeros((len(lines), 2))
            j = 0
            for line in lines:
                points = line.split(' ')
                gt[j][0] = points[0]
                gt[j][1] = points[1]
                j += 1

        img_name = img_path.split('\\')[-1]
        img = cv2.imread(img_path)
        max_size = max(img.shape)
        min_size = min(img.shape[0], img.shape[1])

        if min_size < 128:
            scale = 128 / min_size
            # scale the image and points
            img = cv2.resize(img, None, fx=scale, fy=scale)
            if len(gt) == 0:
                gt = gt
            else:
                gt *= scale
        if max_size > 1920:
            scale = 1920 / max_size
            # scale the image and points
            img = cv2.resize(img, None, fx=scale, fy=scale)
            if len(gt) == 0:
                gt = gt
            else:
                gt *= scale

        # save image
        cv2.imwrite(target_root + '/' + img_name, img)

        # generate density map
        k = np.zeros((img.shape[0], img.shape[1]))
        for i in range(0, len(gt)):
            if int(gt[i][1]) < img.shape[0] and int(gt[i][0]) < img.shape[1]:
                k[int(gt[i][1]), int(gt[i][0])] = 1
        k = gaussian_filter_density_fix(k)
        # print(k.sum())
        with h5py.File((target_root + '/' + img_name).replace('.jpg', '.h5'), 'w') as hf:
            hf['density'] = k

        # generate point txt
        f2 = open(target_root + '/' + img_name.replace('jpg', 'txt'), 'w')
        for j in range(len(gt)):
            if int(gt[j][1]) < img.shape[0] and int(gt[j][0]) < img.shape[1]:
                x = str(gt[j][0])
                y = str(gt[j][1])
                f2.write(x + ' ' + y + '\n')
        f2.close()
        # print('den_count = ' + str(k.sum()))