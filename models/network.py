import torch
import torch.nn.functional as F
from torch import nn

from util.misc import (NestedTensor, nested_tensor_from_tensor_list,
                       accuracy, get_world_size, interpolate,
                       is_dist_avail_and_initialized)

from .backbone import build_backbone
from .matcher import build_matcher_crowd

import numpy as np
import time

# the network frmawork of the regression branch
class RegressionModel(nn.Module):
    def __init__(self, num_features_in, num_anchor_points=4, feature_size=256):
        super(RegressionModel, self).__init__()

        self.conv1 = nn.Conv2d(num_features_in, feature_size, kernel_size=3, padding=1)
        self.act1 = nn.ReLU()

        self.conv2 = nn.Conv2d(feature_size, feature_size, kernel_size=3, padding=1)
        self.act2 = nn.ReLU()

        self.conv3 = nn.Conv2d(feature_size, feature_size, kernel_size=3, padding=1)
        self.act3 = nn.ReLU()

        self.conv4 = nn.Conv2d(feature_size, feature_size, kernel_size=3, padding=1)
        self.act4 = nn.ReLU()

        self.output = nn.Conv2d(feature_size, num_anchor_points * 2, kernel_size=3, padding=1)
    # sub-branch forward
    def forward(self, x):
        out = self.conv1(x)
        out = self.act1(out)

        out = self.conv2(out)
        out = self.act2(out)

        out = self.output(out)

        out = out.permute(0, 2, 3, 1)

        return out.contiguous().view(out.shape[0], -1, 2)

# the network frmawork of the classification branch
class ClassificationModel(nn.Module):
    def __init__(self, num_features_in, num_anchor_points=4, num_classes=80, prior=0.01, feature_size=256):
        super(ClassificationModel, self).__init__()

        self.num_classes = num_classes
        self.num_anchor_points = num_anchor_points

        self.conv1 = nn.Conv2d(num_features_in, feature_size, kernel_size=3, padding=1)
        self.act1 = nn.ReLU()

        self.conv2 = nn.Conv2d(feature_size, feature_size, kernel_size=3, padding=1)
        self.act2 = nn.ReLU()

        self.conv3 = nn.Conv2d(feature_size, feature_size, kernel_size=3, padding=1)
        self.act3 = nn.ReLU()

        self.conv4 = nn.Conv2d(feature_size, feature_size, kernel_size=3, padding=1)
        self.act4 = nn.ReLU()

        self.output = nn.Conv2d(feature_size, num_anchor_points * num_classes, kernel_size=3, padding=1)
        self.output_act = nn.Sigmoid()
    # sub-branch forward
    def forward(self, x):
        out = self.conv1(x)
        out = self.act1(out)

        out = self.conv2(out)
        out = self.act2(out)

        out = self.output(out)

        out1 = out.permute(0, 2, 3, 1)

        batch_size, width, height, _ = out1.shape

        out2 = out1.view(batch_size, width, height, self.num_anchor_points, self.num_classes)

        return out2.contiguous().view(x.shape[0], -1, self.num_classes)

# generate the reference points in grid layout
def generate_anchor_points(stride=16, row=3, line=3):
    row_step = stride / row
    line_step = stride / line

    shift_x = (np.arange(1, line + 1) - 0.5) * line_step - stride / 2
    shift_y = (np.arange(1, row + 1) - 0.5) * row_step - stride / 2

    shift_x, shift_y = np.meshgrid(shift_x, shift_y)

    anchor_points = np.vstack((
        shift_x.ravel(), shift_y.ravel()
    )).transpose()

    return anchor_points
# shift the meta-anchor to get an acnhor points
def shift(shape, stride, anchor_points):
    shift_x = (np.arange(0, shape[1]) + 0.5) * stride
    shift_y = (np.arange(0, shape[0]) + 0.5) * stride

    shift_x, shift_y = np.meshgrid(shift_x, shift_y)

    shifts = np.vstack((
        shift_x.ravel(), shift_y.ravel()
    )).transpose()

    A = anchor_points.shape[0]
    K = shifts.shape[0]
    all_anchor_points = (anchor_points.reshape((1, A, 2)) + shifts.reshape((1, K, 2)).transpose((1, 0, 2)))
    all_anchor_points = all_anchor_points.reshape((K * A, 2))  # 像是展開(256,4)攤開 -> 1024 接續著

    return all_anchor_points

# this class generate all reference points on all pyramid levels
class AnchorPoints(nn.Module):
    def __init__(self, pyramid_levels=None, strides=None, row=3, line=3):
        super(AnchorPoints, self).__init__()

        if pyramid_levels is None:
            self.pyramid_levels = [3, 4, 5, 6, 7]
        else:
            self.pyramid_levels = pyramid_levels

        if strides is None:
            self.strides = [2 ** x for x in self.pyramid_levels]

        self.row = row
        self.line = line

    def forward(self, image):
        image_shape = image.shape[2:]
        image_shape = np.array(image_shape)
        image_shapes = [(image_shape + 2 ** x - 1) // (2 ** x) for x in self.pyramid_levels]  # 下降3次圖片大小除2^3

        all_anchor_points = np.zeros((0, 2)).astype(np.float32)
        # get reference points for each level
        for idx, p in enumerate(self.pyramid_levels):
            anchor_points = generate_anchor_points(2**p, row=self.row, line=self.line)
            shifted_anchor_points = shift(image_shapes[idx], self.strides[idx], anchor_points)
            all_anchor_points = np.append(all_anchor_points, shifted_anchor_points, axis=0)

        all_anchor_points = np.expand_dims(all_anchor_points, axis=0)
        # send reference points to device
        if torch.cuda.is_available():
            return torch.from_numpy(all_anchor_points.astype(np.float32)).cuda()
        else:
            return torch.from_numpy(all_anchor_points.astype(np.float32))


class Shiftwindow_attention(nn.Module):
    def __init__(self, kernel_size=3):
        super(Shiftwindow_attention, self).__init__()
        self.conv_satt = nn.Conv2d(512, 1, kernel_size=1, padding='same', bias=False)
        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax(dim=-1)

    def shift_window(self, x):
        B, C, H, W = x.size()
        swin_att = torch.zeros((B, C, H, W)).cuda()
        for i in range(4):

            mask1 = (torch.ones((B, C, H, W)) * float('-inf')).cuda()
            mask1[:, :, 0 + int(H * (i / 8)):int((H / 2) + H * (i / 8)),
            0 + int(W * (i / 8)):int((W / 2) + W * (i / 8))] = 0

            mask2 = (torch.ones((B, C, H, W)) * float('-inf')).cuda()
            mask2[:, :, 0 + int(H * (i / 8)):int((H / 2) + H * (i / 8)),
            int(W / 2) + int(W * (i / 8)):int(W + W * (i / 8))] = 0
            mask2[:, :, 0 + int(H * (i / 8)):int((H / 2) + H * (i / 8)),
            0:int(W * (i / 8))] = 0

            mask3 = (torch.ones((B, C, H, W)) * float('-inf')).cuda()
            mask3[:, :, int(H / 2) + int(H * (i / 8)):int(H + H * (i / 8)),
            0 + int(W * (i / 8)):int((W / 2) + W * (i / 8))] = 0
            mask3[:, :, 0:int(H * (i / 8)),
            0 + int(W * (i / 8)):int((W / 2) + W * (i / 8))] = 0

            mask4 = (torch.ones((B, C, H, W)) * float('-inf')).cuda()
            mask4[:, :, int(H / 2) + int(H * (i / 8)):int(H + H * (i / 8)),
            int(W / 2) + int(W * (i / 8)):int(W + W * (i / 8))] = 0
            mask4[:, :, 0:int(H * (i / 8)), 0:int(W * (i / 8))] = 0
            mask4[:, :, int(H / 2) + int(H * (i / 8)):int(H + H * (i / 8)), 0:int(W * (i / 8))] = 0
            mask4[:, :, 0:int(H * (i / 8)), int(W / 2) + int(W * (i / 8)):int(W + W * (i / 8))] = 0

            x1 = x + mask1
            x2 = x + mask2
            x3 = x + mask3
            x4 = x + mask4
            x1 = (self.softmax(x1.view(B, C, -1))).view(B, C, H, W)
            x2 = (self.softmax(x2.view(B, C, -1))).view(B, C, H, W)
            x3 = (self.softmax(x3.view(B, C, -1))).view(B, C, H, W)
            x4 = (self.softmax(x4.view(B, C, -1))).view(B, C, H, W)

            x_out = x1 + x2 + x3 + x4

            swin_att += x_out
        return swin_att


    def forward(self, input):
        x = self.conv_satt(input)
        swin_att = self.shift_window(x)
        att_out = swin_att * input
        att_out += input
        return att_out


def make_layers(cfg, in_channels = 3,batch_norm=False,dilation = False):
    if dilation:
        d_rate = 2
    else:
        d_rate = 1
    layers = []
    for v in cfg:
        if v == 'U':
            layers += [nn.Upsample(scale_factor=2, mode='nearest')]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=d_rate,dilation = d_rate)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)

class ContextualModule(nn.Module):
    def __init__(self, features, out_features=512, sizes=(1, 2, 3)):
        super(ContextualModule, self).__init__()
        self.scales = []
        self.scales = nn.ModuleList([self._make_scale(features, size) for size in sizes])
        self.bottleneck = nn.Conv2d(features * 4, out_features, kernel_size=1)
        self.relu = nn.ReLU()
        self.softmax = torch.nn.Softmax(dim=1)
        self.weight_net = nn.Conv2d(features,features,kernel_size=1)


    def __make_weight(self,feature,scale_feature):
        weight_feature = feature * scale_feature
        return weight_feature

    def _make_scale(self, features, size):
        conv = nn.Conv2d(features, features, kernel_size=3, padding=size, bias=False, dilation=size)
        return conv

    def forward(self, feats):
        multi_scales = [stage(feats) for stage in self.scales]
        weights = [self.__make_weight(feats,scale_feature) for scale_feature in multi_scales]
        overall_features = torch.cat((multi_scales[0], multi_scales[1], multi_scales[2]), 1)
        overall_weight = self.softmax(torch.cat((weights[0], weights[1], weights[2]), 1))
        output_features = overall_features * overall_weight
        bottles = self.bottleneck(torch.cat((output_features, feats), 1))
        return self.relu(bottles)


class Network(nn.Module):
    def __init__(self, backbone, row=2, line=2):
        super().__init__()
        self.backbone = backbone
        self.num_classes = 2
        # the number of all anchor points
        num_anchor_points = row * line
        self.conv1x1 = nn.Conv2d(512, 256, kernel_size=1)

        self.regression = RegressionModel(num_features_in=256, num_anchor_points=num_anchor_points)
        self.classification = ClassificationModel(num_features_in=256, \
                                            num_classes=self.num_classes, \
                                            num_anchor_points=num_anchor_points)

        self.anchor_points = AnchorPoints(pyramid_levels=[3,], row=row, line=line)

        self.upsampled = nn.Upsample(scale_factor=2, mode='nearest')
        self.context = ContextualModule(512, 512)
        self.att = Shiftwindow_attention()

        # first branch density decoder
        self.backend_feat = ['U', 256, 'U', 128, 'U', 64]
        self.backend = make_layers(self.backend_feat, in_channels=256, dilation=True)
        self.output_layer = nn.Conv2d(64, 1, kernel_size=1)

        # second branch density decoder
        self.backend_feat2 = ['U', 512, 512, 'U', 256, 'U', 128, 'U', 64]
        self.backend2 = make_layers(self.backend_feat2, in_channels=1024, dilation=True)
        self.output_layer2 = nn.Conv2d(64, 1, kernel_size=1)


    def forward(self, samples: NestedTensor):
        # get the backbone features
        features = self.backbone(samples)
        # forward the feature pyramid
        features_fpn = self.upsampled(features[3])
        features_fpn = self.conv1x1(features_fpn)

        # first_branch
        # point
        batch_size = features[0].shape[0]
        # run the regression and classification branch
        regression = self.regression(features_fpn) * 100
        classification = self.classification(features_fpn)
        anchor_points = self.anchor_points(samples).repeat(batch_size, 1, 1)
        # decode the points as prediction
        output_coord = regression + anchor_points
        output_class = classification
        out = {'pred_logits': output_class, 'pred_points': output_coord}
        # density
        features_de = self.backend(features_fpn)
        out_den = self.output_layer(features_de)

        # second_branch
        features_can = self.context(features[3])
        features_att = self.att(features[3])
        cnn_out = torch.cat((features_can, features_att), dim=1)
        features_de = self.backend2(cnn_out)
        out_den2 = self.output_layer2(features_de)

        return out, out_den, out_den2

class SetCriterion_Crowd(nn.Module):

    def __init__(self, num_classes, matcher, weight_dict, eos_coef, losses):
        """ Create the criterion.
        Parameters:
            num_classes: number of object categories, omitting the special no-object category
            matcher: module able to compute a matching between targets and proposals
            weight_dict: dict containing as key the names of the losses and as values their relative weight.
            eos_coef: relative classification weight applied to the no-object category
            losses: list of all the losses to be applied. See get_loss for list of available losses.
        """
        super().__init__()
        self.num_classes = num_classes
        self.matcher = matcher
        self.weight_dict = weight_dict
        self.eos_coef = eos_coef
        self.losses = losses
        empty_weight = torch.ones(self.num_classes + 1)
        empty_weight[0] = self.eos_coef
        self.register_buffer('empty_weight', empty_weight)

        self.den_loss = nn.MSELoss(size_average=False)
        self.sigmoid = nn.Sigmoid()

    def loss_density(self, outputs,  out, targets, indices, num_points):
        loss_den = 0
        out_den = out[1]
        output = out_den.squeeze(1)
        batch_size = out_den.size(0)
        for i in range(batch_size):
            loss = self.den_loss(output[i], targets[i]['den_map'])
            loss_den += loss
        losses = {'loss_den': loss_den / (2 * batch_size)}

        return losses

    def loss_density2(self, outputs,  out, targets, indices, num_points):
        loss_den = 0
        out_den2 = out[2]
        output = out_den2.squeeze(1)
        batch_size = out_den2.size(0)
        for i in range(batch_size):
            loss = self.den_loss(output[i], targets[i]['den_map'])
            loss_den += loss
        losses = {'loss_den2': loss_den / (2 * batch_size)}

        return losses


    def loss_labels(self, outputs,  out, targets, indices, num_points):
        """Classification loss (NLL)
        targets dicts must contain the key "labels" containing a tensor of dim [nb_target_boxes]
        """
        assert 'pred_logits' in outputs
        src_logits = outputs['pred_logits']

        idx = self._get_src_permutation_idx(indices)
        target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)])
        target_classes = torch.full(src_logits.shape[:2], 0,
                                    dtype=torch.int64, device=src_logits.device)
        target_classes[idx] = target_classes_o

        loss_ce = F.cross_entropy(src_logits.transpose(1, 2), target_classes, self.empty_weight)
        losses = {'loss_ce': loss_ce}

        return losses

    def loss_points(self, outputs,  out, targets, indices, num_points):

        assert 'pred_points' in outputs
        idx = self._get_src_permutation_idx(indices)
        src_points = outputs['pred_points'][idx]
        target_points = torch.cat([t['point'][i] for t, (_, i) in zip(targets, indices)], dim=0)

        loss_bbox = F.mse_loss(src_points, target_points, reduction='none')

        losses = {}
        losses['loss_points'] = loss_bbox.sum() / num_points

        return losses

    def _get_src_permutation_idx(self, indices):
        # permute predictions following indices
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def _get_tgt_permutation_idx(self, indices):
        # permute targets following indices
        batch_idx = torch.cat([torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)])
        tgt_idx = torch.cat([tgt for (_, tgt) in indices])
        return batch_idx, tgt_idx

    def get_loss(self, loss, outputs, out, targets, indices, num_points, **kwargs):
        loss_map = {
            'labels': self.loss_labels,
            'points': self.loss_points,
            'count': self.loss_density,
            'count2': self.loss_density2,
        }
        assert loss in loss_map, f'do you really want to compute {loss} loss?'
        return loss_map[loss](outputs, out, targets, indices, num_points, **kwargs)

    def forward(self, outputs, targets):
        """ This performs the loss computation.
        Parameters:
             outputs: dict of tensors, see the output specification of the model for the format
             targets: list of dicts, such that len(targets) == batch_size.
                      The expected keys in each dict depends on the losses applied, see each loss' doc
        """
        output1 = {'pred_logits': outputs[0]['pred_logits'], 'pred_points': outputs[0]['pred_points']}


        indices1 = self.matcher(output1, targets)

        num_points = sum(len(t["labels"]) for t in targets)
        num_points = torch.as_tensor([num_points], dtype=torch.float, device=next(iter(output1.values())).device)
        if is_dist_avail_and_initialized():
            torch.distributed.all_reduce(num_points)
        num_boxes = torch.clamp(num_points / get_world_size(), min=1).item()

        losses = {}
        for loss in self.losses:
            losses.update(self.get_loss(loss, output1, outputs, targets, indices1, num_boxes))

        return losses

# create the model
def build_network(args, training):
    # treats persons as a single class
    num_classes = 1

    backbone = build_backbone(args)
    model = Network(backbone, args.row, args.line)
    if not training: 
        return model

    weight_dict = {'loss_ce': 1, 'loss_points': args.point_loss_coef, 'loss_den': args.den1_loss_coef, 'loss_den2': args.den2_loss_coef}
    losses = ['labels', 'points', 'count', 'count2']
    matcher = build_matcher_crowd(args) #1v1 match
    criterion = SetCriterion_Crowd(num_classes, \
                                matcher=matcher, weight_dict=weight_dict, \
                                eos_coef=args.eos_coef, losses=losses)

    return model, criterion