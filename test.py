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
from models import build_model_P_D
import os
import warnings
import h5py
from matplotlib import cm as CM
import glob
from matplotlib import pyplot as plt
warnings.filterwarnings('ignore')


mngr = plt.get_current_fig_manager()
mngr.window.wm_geometry("+300+300")

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
    parser.add_argument('--weight_path', default='D:/Lai/counting/counting_PD/ckpt/final_ver/SHHA/best_mae_mean.pth',
                        help='path where the trained weights saved')

    parser.add_argument('--gpu_id', default=0, type=int, help='the gpu used for evaluation')

    return parser

def main(args, debug=False):

    os.environ["CUDA_VISIBLE_DEVICES"] = '{}'.format(args.gpu_id)

    print(args)
    device = torch.device('cuda')
    # get the P2PNet
    model = build_model_P_D(args)
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

        root = r'D:\Lai\counting_dataset\test\vis_image\CC_50'
        img_list = os.listdir(root)
        # img_list = glob.glob(os.path.join(root, '*.jpg'))

        progress = tqdm(total=len(img_list))

        maes = []
        mses = []

        for img_name in img_list:
            # load the images
            progress.update(1)
            img_path = os.path.join(root, img_name)
            img = cv2.imread(img_path)
            # show_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            # # show_img = img
            #
            # # visual test
            # array = np.asarray(show_img)
            #
            # lutSize = 8
            # reductuibFactor = float(256) / float(lutSize)
            #
            # rows = array.shape[0]
            # colums = array.shape[1]
            #
            # fre_image = np.zeros((rows, colums))
            # # print(fre_image.shape)
            #
            # gamut3D = np.zeros((lutSize, lutSize, lutSize))
            # # print(gamut3D.shape)
            #
            # for i in range(rows):
            #     for j in range(colums):
            #         red = math.floor(float(array[i][j][0] / reductuibFactor))
            #         green = math.floor(float(array[i][j][1] / reductuibFactor))
            #         blue = math.floor(float(array[i][j][2] / reductuibFactor))
            #         gamut3D[red][green][blue] = gamut3D[red][green][blue] + 1
            #
            # for i in range(rows):
            #     for j in range(colums):
            #         red = math.floor(float(array[i][j][0] / reductuibFactor))
            #         green = math.floor(float(array[i][j][1] / reductuibFactor))
            #         blue = math.floor(float(array[i][j][2] / reductuibFactor))
            #         freq = gamut3D[red][green][blue]
            #         fre_image[i][j] = freq





            # gt_path = img_name.replace('jpg','h5').replace('image','gt_h5')
            # gt_file = h5py.File(gt_path)
            # target = np.asarray(gt_file['density'])
            # gt_cnt = target.sum()

            # img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            # round the size
            # width, height = img_raw.size

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
            img = transform(img)
            #
            # trans_img = img.permute(1, 2, 0).numpy()
            # trans_img = cv2.cvtColor(trans_img, cv2.COLOR_BGR2RGB)
            # array = np.asarray(trans_img)
            #
            # fre_image1 = np.zeros((rows, colums))
            # # print(fre_image.shape)
            #
            # gamut3D1 = np.zeros((lutSize, lutSize, lutSize))
            # # print(gamut3D.shape)
            #
            # for i in range(rows):
            #     for j in range(colums):
            #         red = math.floor(float(array[i][j][0] / reductuibFactor))
            #         green = math.floor(float(array[i][j][1] / reductuibFactor))
            #         blue = math.floor(float(array[i][j][2] / reductuibFactor))
            #         gamut3D1[red][green][blue] = gamut3D1[red][green][blue] + 1
            #
            # for i in range(rows):
            #     for j in range(colums):
            #         red = math.floor(float(array[i][j][0] / reductuibFactor))
            #         green = math.floor(float(array[i][j][1] / reductuibFactor))
            #         blue = math.floor(float(array[i][j][2] / reductuibFactor))
            #         freq = gamut3D1[red][green][blue]
            #         fre_image1[i][j] = freq

            # #  crop
            # crop_imgs, crop_masks = [], []
            # c, h, w = img.size()
            # rh, rw = 128, 128
            # for i in range(0, h, rh):
            #     gis, gie = max(min(h - rh, i), 0), min(h, i + rh)
            #     for j in range(0, w, rw):
            #         gjs, gje = max(min(w - rw, j), 0), min(w, j + rw)
            #         crop_imgs.append(img[:, gis:gie, gjs:gje].unsqueeze(0))
            #         mask = torch.zeros([1, 1, h, w]).cuda()
            #         mask[:, :, gis:gie, gjs:gje].fill_(1.0)
            #         crop_masks.append(mask)
            # crop_imgs, crop_masks = map(lambda x: torch.cat(x, dim=0), (crop_imgs, crop_masks))
            # crop_imgs = torch.Tensor(crop_imgs)
            # crop_imgs = crop_imgs.to(device)
            #
            # crop_preds = []
            # nz, bz = crop_imgs.size(0), 16
            # for i in range(0, nz, bz):
            #     gs, gt = i, min(nz, i + bz)
            #     crop_pred = model(crop_imgs[gs:gt])
            #
            #     _, _, h1, w1 = crop_pred.size()
            #
            #     # crop_pred = F.interpolate(crop_pred, size=(h1 * 8, w1 * 8), mode='bilinear', align_corners=True) / 64
            #
            #     crop_preds.append(crop_pred)
            # crop_preds = torch.cat(crop_preds, dim=0)
            #
            # # splice them to the original size
            # idx = 0
            # pred_map = torch.zeros([1, 1, h, w]).to(device)
            # for i in range(0, h, rh):
            #     gis, gie = max(min(h - rh, i), 0), min(h, i + rh)
            #     for j in range(0, w, rw):
            #         gjs, gje = max(min(w - rw, j), 0), min(w, j + rw)
            #         pred_map[:, :, gis:gie, gjs:gje] += crop_preds[idx]
            #         idx += 1
            # # for the overlapping area, compute average value
            # mask = crop_masks.sum(dim=0).unsqueeze(0)
            # outputs = pred_map / mask
            # count = outputs.detach().cpu().sum().numpy()

            # img_err = count[0].item() - torch.sum(outputs).item()
            # print("Img name: ", name, "Error: ", img_err, "GT count: ", count[0].item(), "Model out: ",
            #       torch.sum(outputs).item())
            # image_errs.append(img_err)
            # result.append([name, count[0].item(), torch.sum(outputs).item(), img_err])

            # padding
            c,h,w = img.size()
            new_h = ((h - 1) // 128 + 1) * 128
            new_w = ((w - 1) // 128 + 1) * 128

            input = np.zeros((c, new_h, new_w))
            input[:, : h, : w] = img


            # samples = torch.Tensor(input).unsqueeze(0)
            samples = torch.Tensor(input).unsqueeze(0)
            samples = samples.to(device)

            # run inference
            outputs = model(samples)
            out1 = outputs[1][:, :, : h, : w]
            out2 = outputs[2][:, :, : h, : w]

            out = (out1 + out2)/2
            # out = torch.max(out1, out2)
            # out = torch.min(out1, out2)

            count = out.detach().cpu().sum().numpy()

            # filter the predictions
            #print(predict_cnt)

            # mae = abs(count - gt_cnt)
            # print("pic", img_name, " ", "mae", mae)
            # mse = (count - gt_cnt) * (count - gt_cnt)
            # maes.append(float(mae))
            # mses.append(float(mse))
            # calc MAE, MSE


            # plt.subplot(221)
            # plt.imshow(show_img)
            plt.axis('off')
            # plt.subplot(222)
            plt.imshow(out.squeeze().detach().cpu().numpy(), cmap=CM.jet)
            # plt.title("predict: %f" % count)
            # plt.axis('off')
            # plt.subplot(223)
            # plt.imshow(target, cmap=CM.jet)
            # plt.title("gt: %f" % gt_cnt)
            # plt.axis('off')
            # plt.suptitle("MAE: %f" % mae)
            # plt.show()
            plt.savefig(r'D:\Lai\counting\counting_PD\results\vis\CC_50' + '/' + img_name + '_' + str(count) + '.jpg',
                        bbox_inches='tight',
                        pad_inches=0)
            # plt.imsave('D:/Lai/counting/counting_PD/results/CC50_f1/' + img_name + '_' + str(count) + '.jpg', out.squeeze().detach().cpu().numpy(),
            #            cmap=CM.jet)
            # plt.figure()
            # plt.subplot(231)
            # plt.imshow(show_img)
            # plt.title('origin')
            # plt.axis('off')
            # plt.subplot(232)
            # plt.imshow(trans_img)
            # plt.title('transpose')
            # plt.axis('off')
            # plt.subplot(234)
            # plt.imshow(fre_image, cmap='gray')
            # plt.title('cfreq_origin')
            # plt.axis('off')
            # plt.subplot(235)
            # plt.imshow(fre_image1, cmap='gray')
            # plt.title('cfreq_transpose')
            # plt.axis('off')
            # plt.colorbar()
            # plt.subplot(233)
            # plt.imshow(outputs.squeeze().detach().cpu().numpy(), cmap=CM.jet)
            # plt.title('predict')
            # plt.axis('off')
            # plt.colorbar()
            # plt.show()
            # plt.savefig(
            #     'D:/Lai/counting/Crowdcounting_model/result/visual_test/' + img_name +  '.jpg',
            #     bbox_inches='tight', pad_inches=0)

            # sleep(0.01)

        # mae = np.mean(maes)
        # mse = np.sqrt(np.mean(mses))
        # print("maes:", mae, "  ",  "mse:", mse)
        # print("min:", min(maes))

if __name__ == '__main__':
    parser = argparse.ArgumentParser('P2PNet evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()
    main(args)