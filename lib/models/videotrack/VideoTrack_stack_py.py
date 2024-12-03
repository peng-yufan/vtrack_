# -*-coding:utf-8-*-
from functools import partial
import torch.nn.functional as F
import torch
import torch.nn as nn
import os
from ..mixformer_vit.mixformer_online import build_mixformer_vit_online_score
from lib.models.videotrack.score_decoder_mul import MulScoreDecoder
from lib.models.mixformer_cvt.utils import FrozenBatchNorm2d
from lib.models.component.mlp import MultiLayerMlp
from matplotlib import pyplot as plt
import cv2
import numpy as np
import random
from timm.models.layers import Mlp

class conv(nn.Module):
    def __init__(self, in_embedding=768, out_embedding=768):
        super().__init__()
        # self.net0 = nn.Sequential(
        #     nn.Linear(in_embedding, out_embedding),
        #     nn.BatchNorm1d(out_embedding),
        #     nn.ReLU(inplace=True),
        # )
        self.net0 = nn.Sequential(
            nn.Conv2d(in_channels=in_embedding, out_channels=out_embedding
                      , kernel_size=1, padding=0, stride=1),
            nn.BatchNorm2d(out_embedding),
            # FrozenBatchNorm2d(out_embedding),
            nn.ReLU(inplace=True),
        )
        self.net1 = nn.Sequential(
            nn.Conv2d(in_channels=out_embedding, out_channels=out_embedding
                      , kernel_size=3, padding=1, stride=1, groups=out_embedding),
            nn.BatchNorm2d(out_embedding),
            # FrozenBatchNorm2d(out_embedding),
            nn.ReLU(inplace=True),
        )
    def forward(self, x):
        # x = x.permute([0, 2, 3, 1])
        # shape = x.shape
        # x = x.reshape(-1, shape[-1])
        x = self.net0(x)
        # shape = [i for i in shape[:3]]
        # shape.append(x.shape[-1])
        # x = x.view(shape)
        # x = x.permute([0, 3, 1, 2])

        x = self.net1(x)
        return x
class VideoTrack(nn.Module):
    def __init__(self, embedding=768, backbone=None, head_bbox=None, head_score=None):
        super().__init__()
        self.embedding = embedding
        self.backbone = backbone
        self.head_bbox = head_bbox
        self.head_score = head_score
    def forward(self, template, online_template, search):
        # search: (b, n, c, h, w)
        if template.dim() == 5:
            template = template.squeeze(1)
        if online_template.dim() == 5:
            online_template = online_template.squeeze(1)
        if search.dim() == 5:
            search = search.squeeze(1)
        template, search = self.backbone(template, online_template, search)
        # search shape: (b, 384, 20, 20)
        # Forward the corner head and score head
        out, outputs_coord_new = self.head(search, template)
        return out, outputs_coord_new

class Pyramid_MulSearch_Predictor(nn.Module):
    """ Corner Predictor module"""

    def __init__(self, inplanes=64, channel=256, feat_sz=20, stride=16, freeze_bn=False, num_search=1):
        super(Pyramid_MulSearch_Predictor, self).__init__()
        self.feat_sz = feat_sz
        self.stride = stride
        self.img_sz = self.feat_sz * self.stride
        self.num_search = num_search
        inplanes *= num_search
        channel *= num_search
        '''top-left corner'''
        self.conv1_tl = conv(inplanes, channel)
        self.conv2_tl = conv(channel, channel // 2)
        self.conv3_tl = conv(channel // 2, channel // 4)
        self.conv4_tl = conv(channel // 4, channel // 8)
        self.conv5_tl = nn.Conv2d(channel // 8, num_search, kernel_size=1)

        self.adjust1_tl = conv(inplanes, channel // 2)
        self.adjust2_tl = conv(inplanes, channel // 4)

        self.adjust3_tl = nn.Sequential(conv(channel // 2, channel // 4),
                                        conv(channel // 4, channel // 8),
                                        conv(channel // 8, num_search))
        self.adjust4_tl = nn.Sequential(conv(channel // 4, channel // 8),
                                        conv(channel // 8, num_search))

        '''bottom-right corner'''
        self.conv1_br = conv(inplanes, channel)
        self.conv2_br = conv(channel, channel // 2)
        self.conv3_br = conv(channel // 2, channel // 4)
        self.conv4_br = conv(channel // 4, channel // 8)
        self.conv5_br = nn.Conv2d(channel // 8, num_search, kernel_size=1)

        self.adjust1_br = conv(inplanes, channel // 2)
        self.adjust2_br = conv(inplanes, channel // 4)

        self.adjust3_br = nn.Sequential(conv(channel // 2, channel // 4),
                                        conv(channel // 4, channel // 8),
                                        conv(channel // 8, num_search))
        self.adjust4_br = nn.Sequential(conv(channel // 4, channel // 8),
                                        conv(channel // 8, num_search))

        '''about coordinates and indexs'''
        with torch.no_grad():
            self.indice = torch.arange(0, 4*self.feat_sz).view(-1, 1) * int(self.stride/4)
            # generate mesh-grid
            self.coord_x = self.indice.repeat((4*self.feat_sz, 1)) \
                .view((4*self.feat_sz * 4*self.feat_sz,)).float().cuda()
            self.coord_y = self.indice.repeat((1, 4*self.feat_sz)) \
                .view((4*self.feat_sz * 4*self.feat_sz,)).float().cuda()

    def forward(self, x, return_dist=False, softmax=True):
        """ Forward pass with input x. """
        score_map_tl, score_map_br = self.get_score_map(x)
        if return_dist:
            coorx_tl, coory_tl, prob_vec_tl = self.soft_argmax(score_map_tl, return_dist=True, softmax=softmax)
            coorx_br, coory_br, prob_vec_br = self.soft_argmax(score_map_br, return_dist=True, softmax=softmax)
            return torch.stack((coorx_tl, coory_tl, coorx_br, coory_br), dim=1) / self.img_sz, prob_vec_tl, prob_vec_br
        else:
            coorx_tl, coory_tl = self.soft_argmax(score_map_tl)
            coorx_br, coory_br = self.soft_argmax(score_map_br)
            return torch.stack((coorx_tl, coory_tl, coorx_br, coory_br), dim=1) / self.img_sz

    def get_score_map(self, x):
        x_init = x
        # top-left branch
        x_tl1 = self.conv1_tl(x)
        x_tl2 = self.conv2_tl(x_tl1)

        #up-1
        x_init_up1 = F.interpolate(self.adjust1_tl(x_init), scale_factor=2)
        x_up1 = F.interpolate(x_tl2, scale_factor=2)
        x_up1 = x_init_up1 + x_up1

        x_tl3 = self.conv3_tl(x_up1)

        #up-2
        x_init_up2 = F.interpolate(self.adjust2_tl(x_init), scale_factor=4)
        x_up2 = F.interpolate(x_tl3, scale_factor=2)
        x_up2 = x_init_up2 + x_up2

        x_tl4 = self.conv4_tl(x_up2)
        score_map_tl = self.conv5_tl(x_tl4) + F.interpolate(self.adjust3_tl(x_tl2), scale_factor=4) + F.interpolate(self.adjust4_tl(x_tl3), scale_factor=2)

        # bottom-right branch
        x_br1 = self.conv1_br(x)
        x_br2 = self.conv2_br(x_br1)

        # up-1
        x_init_up1 = F.interpolate(self.adjust1_br(x_init), scale_factor=2)
        x_up1 = F.interpolate(x_br2, scale_factor=2)
        x_up1 = x_init_up1 + x_up1

        x_br3 = self.conv3_br(x_up1)

        # up-2
        x_init_up2 = F.interpolate(self.adjust2_br(x_init), scale_factor=4)
        x_up2 = F.interpolate(x_br3, scale_factor=2)
        x_up2 = x_init_up2 + x_up2

        x_br4 = self.conv4_br(x_up2)
        score_map_br = self.conv5_br(x_br4) + F.interpolate(self.adjust3_br(x_br2), scale_factor=4) + F.interpolate(self.adjust4_br(x_br3), scale_factor=2)
        return score_map_tl, score_map_br

    def soft_argmax(self, score_map, return_dist=False, softmax=True):
        """ get soft-argmax coordinate for a given heatmap """
        score_vec = score_map.view((-1, self.num_search, 4*self.feat_sz * 4*self.feat_sz))  # (batch, T,feat_sz * feat_sz)
        prob_vec = nn.functional.softmax(score_vec, dim=-1)
        exp_x = torch.sum((self.coord_x * prob_vec), dim=-1)
        exp_y = torch.sum((self.coord_y * prob_vec), dim=-1)
        if return_dist:
            if softmax:
                return exp_x, exp_y, prob_vec
            else:
                return exp_x, exp_y, score_vec
        else:
            return exp_x, exp_y


def build_videonet(cfg, settings=None, training=True):
    current_dir = os.path.dirname(os.path.abspath(__file__))  # This is your Project Root
    pretrained_path = os.path.join(current_dir, 'pretrained_models')
    if settings!=None:
        settings.stage1_model = pretrained_path + '/mixformer_vit_base_online.pth.tar'
    backbone = build_mixformer_vit_online_score(cfg, settings=settings, train=training)
    head_bbox = Pyramid_MulSearch_Predictor()
    head_score = MulScoreDecoder()
    model = VideoTrack(backbone=backbone, head_bbox=head_bbox, head_score=head_score)
    # for name, param in model.backforward_net.z_proj.named_parameters():
    #    nn.init.zeros_(param)
    # forward_net_checkpoint = torch.load(pretrained_path + '/CTTrack-B.pth.tar',
    #                                     map_location=torch.device('cpu'))
    # model.forward_net.load_state_dict(forward_net_checkpoint['net'], strict=False)
    # forward_net_checkpoint = torch.load(pretrained_path + '/CTTrack_online.pth.tar',
    #                                     map_location=torch.device('cpu'))
    # model.forward_net.load_state_dict(forward_net_checkpoint['net'], strict=False)
    # forward_net_checkpoint = torch.load(pretrained_path + '/CTTrack_train_ep0020.pth.tar',
    #                                     map_location=torch.device('cpu'))
    # model.forward_net.load_state_dict(forward_net_checkpoint['net'], strict=False)

    # del model.forward_net.cross_decoder
    # del model.forward_net.decoder
    return model

