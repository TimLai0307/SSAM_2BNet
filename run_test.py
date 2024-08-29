import argparse
import datetime
import random
import time
from pathlib import Path
from time import sleep
from tqdm import tqdm, trange
import torch
import torchvision.transforms as standard_transforms
import numpy as np
import scipy.io as io
from PIL import Image
import cv2
from crowd_datasets import build_dataset
from engine import *
from models import build_model_can
import os
import warnings
warnings.filterwarnings('ignore')

def get_args_parser():
    parser = argparse.ArgumentParser('Set parameters for P2PNet evaluation', add_help=False)
    
    # * Backbone
    parser.add_argument('--backbone', default='vgg16_bn', type=str,
                        help="name of the convolutional backbone to use")

    parser.add_argument('--row', default=2, type=int,
                        help="row number of anchor points")
    parser.add_argument('--line', default=2, type=int,
                        help="line number of anchor points")

    parser.add_argument('--output_dir', default='./result/test',
                        help='path where to save')
    parser.add_argument('--weight_path', default='D:/Lai/counting/Crowdcounting_model/ckpt/best_mae.pth',
                        help='path where the trained weights saved')

    parser.add_argument('--gpu_id', default=0, type=int, help='the gpu used for evaluation')

    return parser

def main(args, debug=False):

    os.environ["CUDA_VISIBLE_DEVICES"] = '{}'.format(args.gpu_id)

    print(args)
    device = torch.device('cuda')
    # get the P2PNet
    model = build_model_can(args)
    # move to GPU
    model.to(device)
    # load trained model
    if args.weight_path is not None:
        checkpoint = torch.load(args.weight_path, map_location='cpu')
        model.load_state_dict(checkpoint['model'])
    # convert to eval mode
    model.eval()
    with torch.no_grad():
        # create the pre-processing transform
        transform = standard_transforms.Compose([
            standard_transforms.ToTensor(),
            standard_transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        root = r'D:\Lai\counting_dataset\ShanghaiTech\part_A\test_data\images'
        img_list = os.listdir(root)

        maes = []
        mses = []

        for img_name in img_list:
            # load the images
            img_path = os.path.join(root, img_name)
            img = cv2.imread(img_path)
            # img_raw = Image.open(img_path).convert('RGB')
            # round the size
            w = img.shape[1]
            h = img.shape[0]
            c = img.shape[2]

            # if max(img_raw.size) > 1920:
            #     max_size = max(img_raw.size)
            #     scale = 1920 / max_size
            #     width = int(width * scale)
            #     height = int(height * scale)

            # new_width = width // 128 * 128
            # new_height = height // 128 * 128
            #
            # img_raw = img_raw.resize((new_width, new_height), Image.ANTIALIAS)
            # pre-proccessing
            # img = transform(img_raw)
            img = transform(img)
            new_h = ((h - 1) // 128 + 1) * 128
            new_w = ((w - 1) // 128 + 1) * 128
            input = torch.zeros((c, new_h, new_w))
            input[:, : h, : w] = img

            mat = io.loadmat(img_path.replace('.jpg', '.mat').replace('images', 'ground-truth').replace('IMG_', 'GT_IMG_'))
            gt_cnt = mat["image_info"][0, 0][0, 0][1][0, 0]
            # mat = io.loadmat(img_path.replace('.jpg', '.mat').replace('image', 'mats'))
            # gt = mat['annPoints']
            # gt_cnt = len(gt)

            samples = torch.Tensor(input).unsqueeze(0)
            samples = samples.to(device)
            # run inference
            outputs = model(samples)
            outputs_scores = torch.nn.functional.softmax(outputs['pred_logits'], -1)[:, :, 1][0]

            outputs_points = outputs['pred_points'][0]

            threshold = 0.5
            # filter the predictions
            points = outputs_points[outputs_scores > threshold].detach().cpu().numpy().tolist()
            predict_cnt = int((outputs_scores > threshold).sum())
            #print(predict_cnt)

            mae = abs(predict_cnt - gt_cnt)
            mse = (predict_cnt - gt_cnt) * (predict_cnt - gt_cnt)
            maes.append(float(mae))
            mses.append(float(mse))
            # calc MAE, MSE


            outputs_scores = torch.nn.functional.softmax(outputs['pred_logits'], -1)[:, :, 1][0]

            outputs_points = outputs['pred_points'][0]
            # draw the predictions
            # size = 2
            # img_to_draw = cv2.cvtColor(np.array(img_raw), cv2.COLOR_RGB2BGR)
            # for p in points:
            #     img_to_draw = cv2.circle(img_to_draw, (int(p[0]), int(p[1])), size, (0, 0, 255), -1)
            # save the visualized image
            # cv2.imwrite(os.path.join(args.output_dir, img_name + '_pred{}.jpg'.format(predict_cnt)), img_to_draw)

        mae = np.mean(maes)
        mse = np.sqrt(np.mean(mses))
        print("mae:", mae, "  ",  "mse:", mse)

if __name__ == '__main__':
    parser = argparse.ArgumentParser('P2PNet evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()
    main(args)