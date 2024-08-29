import argparse
import datetime
import random
import time
import thop
import torch
from torch.utils.data import DataLoader, DistributedSampler

# from crowd_datasets import build_dataset
from crowd_datasets.loading_data import get_data
from engine import *
from models import build_model
import os
from tensorboardX import SummaryWriter
import warnings
warnings.filterwarnings('ignore')

def get_args_parser(): #引數定義
    parser = argparse.ArgumentParser('Set parameters for training', add_help=False)
    parser.add_argument('--lr', default=1e-4, type=float)
    parser.add_argument('--lr_backbone', default=1e-5, type=float)
    parser.add_argument('--batch_size', default=12, type=int)
    # parser.add_argument('--weight_decay', default=1e-4, type=float)
    parser.add_argument('--epochs', default=6000, type=int)
    parser.add_argument('--lr_drop', default=20000, type=int)
    parser.add_argument('--clip_max_norm', default=0.1, type=float,
                        help='gradient clipping max norm')

    # Model parameters
    parser.add_argument('--frozen_weights', type=str, default=None,
                        help="Path to the pretrained model. If set, only the mask head will be trained")

    # * Backbone
    parser.add_argument('--backbone', default='vgg16_bn', type=str,
                        help="Name of the convolutional backbone to use")

    # * Matcher
    parser.add_argument('--set_cost_class', default=1, type=float,
                        help="Class coefficient in the matching cost")
    parser.add_argument('--set_cost_point', default=0.05, type=float,
                        help="L1 point coefficient in the matching cost")

    # * Loss coefficients
    parser.add_argument('--point_loss_coef', default=0.001, type=float)
    parser.add_argument('--den1_loss_coef', default=0.5, type=float)
    parser.add_argument('--den2_loss_coef', default=1, type=float)

    parser.add_argument('--eos_coef', default=0.5, type=float,
                        help="Relative classification weight of the no-object class")
    parser.add_argument('--row', default=2, type=int,
                        help="row number of anchor points")
    parser.add_argument('--line', default=2, type=int,
                        help="line number of anchor points")

    # dataset parameters

    parser.add_argument('--data_root', default='D:/Lai/counting_dataset/test/dotden/SHHA',
                        help='path where the dataset is')
    
    parser.add_argument('--output_dir', default='./log',
                        help='path where to save, empty for no saving')
    parser.add_argument('--checkpoints_dir', default='./ckpt',
                        help='path where to save checkpoints, empty for no saving')
    parser.add_argument('--tensorboard_dir', default='./tensorboard',
                        help='path where to save, empty for no saving')

    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--start_epoch', default=1, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--num_workers', default=8, type=int)
    parser.add_argument('--eval_freq', default=5, type=int,
                        help='frequency of evaluation, default setting is evaluating in every 5 epoch')
    parser.add_argument('--gpu_id', default=0, type=int, help='the gpu used for training')

    return parser

def main(args): #主程式
    os.environ["CUDA_VISIBLE_DEVICES"] = '{}'.format(args.gpu_id)
    # create the logging file
    run_log_name = os.path.join(args.output_dir, 'run_log.txt')
    with open(run_log_name, "w") as log_file:
        log_file.write('Eval Log %s\n' % time.strftime("%c"))

    if args.frozen_weights is not None:
        assert args.masks, "Frozen training is meant for segmentation only"

    # backup the arguments
    print(args)
    with open(run_log_name, "a") as log_file:
        log_file.write("{}".format(args))
    device = torch.device('cuda')
    # fix the seed for reproducibility
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    # get the counting model
    model, criterion = build_model(args, training=True)
    # move to GPU
    model.to(device)
    criterion.to(device)

    model_without_ddp = model

    # # cal flops, parameters
    # input = torch.randn(48, 3, 128, 128).cuda()
    # Flops, params = thop.profile(model, inputs=(input,))
    # print('Flops: % .4fG' % (Flops / 1000000000))
    # print('params: % .4fM' % (params / 1000000))

    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('number of params:', n_parameters)
    # use different optimation params for different parts of the model
    param_dicts = [
        {"params": [p for n, p in model_without_ddp.named_parameters() if "backbone" not in n and p.requires_grad]},
        {
            "params": [p for n, p in model_without_ddp.named_parameters() if "backbone" in n and p.requires_grad],
            "lr": args.lr_backbone,
        },
    ]

    if args.resume:
        lr = args.lr * (0.5 ** (args.start_epoch//args.lr_drop))
    else:
        lr = args.lr

    # Optimizer is using Adam by default
    optimizer = torch.optim.Adam(param_dicts, lr=lr)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, args.lr_drop, gamma=0.95)

    # create the dataset
    # loading_data = build_dataset(args=args)
    # create the training and valiation set
    train_set, val_set = get_data(args.data_root)
    # create the sampler used during training
    sampler_train = torch.utils.data.RandomSampler(train_set)
    sampler_val = torch.utils.data.SequentialSampler(val_set)

    batch_sampler_train = torch.utils.data.BatchSampler(
        sampler_train, args.batch_size, drop_last=True)
    # the dataloader for training
    data_loader_train = DataLoader(train_set, batch_sampler=batch_sampler_train,
                                   collate_fn=utils.collate_fn_crowd, num_workers=args.num_workers)

    data_loader_val = DataLoader(val_set, batch_size=1, sampler=sampler_val,
                                    drop_last=False, collate_fn=utils.collate_fn_crowd, num_workers=args.num_workers)

    if args.frozen_weights is not None:
        checkpoint = torch.load(args.frozen_weights, map_location='cpu')
        model_without_ddp.detr.load_state_dict(checkpoint['model'])
    # resume the weights and training state if exists
    if args.resume:
        checkpoint = torch.load(args.resume, map_location='cpu')
        model_without_ddp.load_state_dict(checkpoint['model'])
        args.start_epoch = checkpoint['epoch'] + 1
        if not args.eval and 'optimizer' in checkpoint and 'lr_scheduler' in checkpoint and 'epoch' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer'])
            lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
            args.start_epoch = checkpoint['epoch'] + 1


    # Start training
    print("Start training")
    start_time = time.time()
    # save the performance during the training
    mae_mean = []
    mse_mean = []
    mae_max = []
    mse_max = []
    mae_min = []
    mse_min = []

    mae1 = []
    mae2 = []

    # the logger writer
    writer = SummaryWriter(args.tensorboard_dir)
    
    step = args.start_epoch // args.eval_freq + 1
    # training starts here
    for epoch in range(args.start_epoch, args.epochs+1):
        t1 = time.time()
        stat = train_one_epoch(
            model, criterion, data_loader_train, optimizer, device, epoch,
            args.clip_max_norm)


        # record the training states after every epoch
        if writer is not None:
            with open(run_log_name, "a") as log_file:
                log_file.write("loss/loss@{}: {}".format(epoch, stat['loss']))
                log_file.write("loss/loss_den@{}: {}".format(epoch, stat['loss_den']))
                log_file.write("loss/loss_den2@{}: {}".format(epoch, stat['loss_den2']))
                log_file.write("loss/loss_points@{}: {}".format(epoch, stat['loss_points']))
                log_file.write("loss/loss_ce@{}: {}".format(epoch, stat['loss_ce']))

            writer.add_scalar('loss/loss', stat['loss'], epoch)
            writer.add_scalar('loss/loss_den', stat['loss_den'], epoch)
            writer.add_scalar('loss/loss_den2', stat['loss_den2'], epoch)
            writer.add_scalar('loss/loss_points', stat['loss_points'], epoch)
            writer.add_scalar('loss/loss_ce', stat['loss_ce'], epoch)

        t2 = time.time()
        print('[ep %d][lr %.7f][%.2fs]' % \
              (epoch, optimizer.param_groups[0]['lr'], t2 - t1))
        with open(run_log_name, "a") as log_file:
            log_file.write('[ep %d][lr %.7f][%.2fs]' % (epoch, optimizer.param_groups[0]['lr'], t2 - t1))
        # change lr according to the scheduler
        lr_scheduler.step()
        # save latest weights every epoch
        checkpoint_latest_path = os.path.join(args.checkpoints_dir, 'latest.pth')
        torch.save({
            'model': model_without_ddp.state_dict(),
            'epoch': epoch,
        }, checkpoint_latest_path)
        # run evaluation
        if epoch % args.eval_freq == 0 and epoch != 0:
            t1 = time.time()
            result = evaluate_crowd_no_overlap(model, data_loader_val, device)
            t2 = time.time()

            mae_mean.append(result[0])
            mse_mean.append(result[1])
            mae_max.append(result[2])
            mse_max.append(result[3])
            mae_min.append(result[4])
            mse_min.append(result[5])

            mae1.append(result[6])
            mae2.append(result[7])

            print('=======================================test=======================================')

            print("mae_mean:", result[0], "mse_mean:", result[1], "mae_max:", result[2], "mse_max:", result[3], "mae_min:", result[4], "mse_min:", result[5], 'mae1:', result[6], 'mae2:', result[7], "time:", t2 - t1, "best mae_mean:", np.min(mae_mean), "best mae_max:", np.min(mae_max), "best mae_min:", np.min(mae_min), "best mae1:", np.min(mae1), "best mae2:", np.min(mae2),)
            with open(run_log_name, "a") as log_file:
                log_file.write("mae_mean:{}, mse_mean:{}, mae_max:{}, mse_max:{}, mae_min:{}, mse_min:{}, time:{}, best mae_mean:{},  best mae_max:{},  best mae_min:{}".format(result[0],
                                result[1], result[2], result[3], result[4], result[5], t2 - t1, np.min(mae_mean), np.min(mae_max), np.min(mae_min)))

            print('=======================================test=======================================')
            # recored the evaluation results
            if writer is not None:
                with open(run_log_name, "a") as log_file:
                    log_file.write("metric/mae_mean@{}: {}".format(step, result[0]))
                    log_file.write("metric/mse_mean@{}: {}".format(step, result[1]))
                    log_file.write("metric/mae_max@{}: {}".format(step, result[2]))
                    log_file.write("metric/mse_max@{}: {}".format(step, result[3]))
                    log_file.write("metric/mae_min@{}: {}".format(step, result[4]))
                    log_file.write("metric/mse_min@{}: {}".format(step, result[5]))

                writer.add_scalar('metric/mae_mean', result[0], step)
                writer.add_scalar('metric/mse_mean', result[1], step)
                writer.add_scalar('metric/mae_max', result[2], step)
                writer.add_scalar('metric/mse_max', result[3], step)
                writer.add_scalar('metric/mae_min', result[4], step)
                writer.add_scalar('metric/mse_min', result[5], step)

                step += 1

            # save the best model since begining
            if abs(np.min(mae_mean) - result[0]) < 0.01:
                checkpoint_best_path = os.path.join(args.checkpoints_dir, 'best_mae_mean.pth')
                torch.save({
                    'model': model_without_ddp.state_dict(),
                    'epoch': epoch
                }, checkpoint_best_path)
            if abs(np.min(mae_max) - result[2]) < 0.01:
                checkpoint_best_path = os.path.join(args.checkpoints_dir, 'best_mae_max.pth')
                torch.save({
                    'model': model_without_ddp.state_dict(),
                    'epoch': epoch
                }, checkpoint_best_path)
            if abs(np.min(mae_min) - result[4]) < 0.01:
                checkpoint_best_path = os.path.join(args.checkpoints_dir, 'best_mae_min.pth')
                torch.save({
                    'model': model_without_ddp.state_dict(),
                    'epoch': epoch
                }, checkpoint_best_path)

    # total time for training
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))

if __name__ == '__main__':
    parser = argparse.ArgumentParser('Training and evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()
    main(args)