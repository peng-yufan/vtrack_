# -*-coding:utf-8-*-
from functools import partial

import torch
import torch.nn as nn
import os
from ..mixformer_vit.mixformer_online import build_mixformer_vit_online_score
from lib.models.videotrack.score_decoder_mul import MulScoreDecoder
from lib.utils.box_ops import box_xyxy_to_cxcywh, box_cxcywh_to_xyxy
from lib.models.component.mlp import MultiLayerMlp
from matplotlib import pyplot as plt
import cv2
import numpy as np
import random
from timm.models.layers import Mlp
from lib.models.mixformer_cvt.utils import FrozenBatchNorm2d
from lib.utils.box_ops import box_cxcywh_to_xyxy, box_xywh_to_xyxy
import torch.nn.functional as F
class VideoTrack(nn.Module):
    def __init__(self, embedding=768, backbone=None, head_bbox=None, head_score=None):
        super().__init__()
        self.embedding = embedding
        self.backbone = backbone
        self.head_bbox = head_bbox
        self.head_score = head_score
    def forward(self, data):
        searches = []
        templates = []
        # template0 = data['template_images'][-1]
        # with torch.no_grad():
        #     # self.net.module.backbone.forward(data['template_images'][0], data['template_images'][1])
        #     for template, search in zip(data['template_images'][:-1], data['search_images']):
        #         out, _ = self.backbone(template0, template, search)
        #         templates.append(out['template_feature'].detach())
        #         searches.append(out['search_feature'].detach())
        #     templates = torch.stack(templates, dim=0)
        #     searches = torch.stack(searches, dim=0)
        #     N, B, C, Hs, Ws = searches.shape
        with torch.no_grad():
            # self.net.module.backbone.forward(data['template_images'][0], data['template_images'][1])
            template0 = data['template_images'][-1]
            b, c, ht, wt = template0.shape
            b, c, hs, ws = data['search_images'][0].shape

            for template, search in zip(data['template_images'][:-1], data['search_images']):
                out, _ = self.backbone(template0, template, search)
                templates.append(out['template_feature'].detach())
                searches.append(out['search_feature'].detach())
            templates = torch.stack(templates, dim=0)
            searches = torch.stack(searches, dim=0)

            # template = data['template_images'][:-1].view(-1, c, ht, wt)
            # search = data['search_images'].view(-1, c, hs, ws)
            # out, _ = self.backbone(template0.repeat(int(data['search_images'].shape[0]), 1, 1, 1), template, search)
            # _, c, hs, ws = out['search_feature'].shape
            # _, c, ht, wt = out['template_feature'].shape
            # templates = out['template_feature'].reshape(-1, b, c, ht, wt).detach()
            # searches = out['search_feature'].reshape(-1, b, c, hs, ws).detach()

        # search shape: (b, 384, 20, 20)
        # Forward the corner head and score head
        out_dict = self.forward_pass(templates, searches)
        # out_dict = self.net(template.clone().detach(), search.clone().detach(), gts=gt.clone().detach())
        return out_dict
    def forward_pass(self, template_tokens, search_tokens, gts=None):
        N, B, C, Hs, Ws = search_tokens.shape
        out_bb = self.head_bbox(search_tokens)
        # search_tokens = search_tokens.reshape(B, self.num_search, int(NC/self.num_search), Hs, Ws)\
        #     .permute([1, 0, 2, 3, 4]).contiguous()
        if gts is None:
            gts = box_xywh_to_xyxy(out_bb.permute([2, 0, 1]))
        out_score = self.head_score(search_tokens, template_tokens, gts)

        out_dict = {'pred_boxes': box_xyxy_to_cxcywh(out_bb.permute([2, 0, 1])), #N, B, 4
                    'pred_scores': out_score.view(-1, B)
                    }
        # out_dict = {'pred_boxes': out_bb, #N, B, 4
        #             'pred_scores': out_score.view(-1, B)
        #             }
        return out_dict

def Conv2d(in_planes, out_planes, kernel_size=3, stride=1, padding=1, dilation=1,
         freeze_bn=False):
    if freeze_bn:
        return nn.Sequential(
            nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride,
                      padding=padding, dilation=dilation, bias=True),
            FrozenBatchNorm2d(out_planes),
            nn.ReLU(inplace=True))
    else:
        return nn.Sequential(
            nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride,
                      padding=padding, dilation=dilation, bias=True),
            nn.BatchNorm2d(out_planes),
            nn.ReLU(inplace=True))

class MulSearchHead(nn.Module):
    def __init__(self, inplanes=768, channel=256,  num_search=10, feat_sz=18, stride=16):
        super().__init__()
        self.embedding = inplanes
        self.num_search = num_search
        self.feat_sz = feat_sz
        self.stride = stride
        self.img_sz = self.feat_sz * self.stride
        self.conv1_tl = Conv2d(inplanes, channel)
        self.conv2_tl = Conv2d(channel, channel // 2)
        self.conv3_tl = Conv2d(channel//2, channel // 4)
        self.conv4_tl = Conv2d(channel//4, channel // 8)
        self.conv5_tl_mul = nn.Conv2d(num_search * channel//8, num_search, kernel_size=1)

        self.conv2d_t_tl1 = Conv2d(num_search, num_search)
        self.conv2d_t_tl2 = Conv2d(num_search, num_search)
        self.conv2d_t_tl3 = Conv2d(num_search, num_search)
        self.conv2d_t_tl4 = Conv2d(num_search, num_search)

        self.conv1_br = Conv2d(inplanes, channel)
        self.conv2_br = Conv2d(channel, channel // 2)
        self.conv3_br = Conv2d(channel//2, channel // 4)
        self.conv4_br = Conv2d(channel//4, channel // 8)
        self.conv5_br_mul = nn.Conv2d(num_search * channel//8, num_search, kernel_size=1)

        self.conv2d_t_br1 = Conv2d(num_search, num_search)
        self.conv2d_t_br2 = Conv2d(num_search, num_search)
        self.conv2d_t_br3 = Conv2d(num_search, num_search)
        self.conv2d_t_br4 = Conv2d(num_search, num_search)

        '''about coordinates and indexs'''
        with torch.no_grad():
            self.indice = torch.arange(0, self.feat_sz).view(-1, 1) * self.stride
            # generate mesh-grid
            self.coord_x = self.indice.repeat((self.feat_sz, 1)) \
                .view((self.feat_sz * self.feat_sz,)).float().cuda()
            self.coord_y = self.indice.repeat((1, self.feat_sz)) \
                .view((self.feat_sz * self.feat_sz,)).float().cuda()

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
        x = x.permute([1, 0, 2, 3, 4]) # B, N, C, H, W
        B, N, C, H, W = x.shape
        # top-left branch
        x_tl1 = (self.conv1_tl(x.reshape(-1, C, H, W))).view(B, N, -1, H, W)
        xt_tl1 = self.conv2d_t_tl1(x_tl1.permute([0, 2, 1, 3, 4]).reshape(-1, N, H, W)).view(B, -1, N, H, W).permute(
            [0, 2, 1, 3, 4])
        x_tl2 = (self.conv2_tl(xt_tl1.reshape(B*N, -1, H, W))).view(B, N, -1, H, W)
        xt_tl2 = self.conv2d_t_tl2(x_tl2.permute([0, 2, 1, 3, 4]).reshape(-1, N, H, W)).view(B, -1, N, H, W).permute(
            [0, 2, 1, 3, 4])

        x_tl3 = (self.conv3_tl(xt_tl2.reshape(B*N, -1, H, W))).view(B, N, -1, H, W)
        xt_tl3 = self.conv2d_t_tl3(x_tl3.permute([0, 2, 1, 3, 4]).reshape(-1, N, H, W)).view(B, -1, N, H, W).permute(
            [0, 2, 1, 3, 4])

        x_tl4 = (self.conv4_tl(xt_tl3.reshape(B*N, -1, H, W))).view(B, N, -1, H, W)
        xt_tl4 = self.conv2d_t_tl4(x_tl4.permute([0, 2, 1, 3, 4]).reshape(-1, N, H, W)).view(B, -1, N, H, W).permute(
            [0, 2, 1, 3, 4])

        score_map_tl = self.conv5_tl_mul(xt_tl4.reshape(B, -1, H, W))

        # bottom-right branch
        x_br1 = (self.conv1_br(x.reshape(-1, C, H, W))).view(B, N, -1, H, W)
        xt_br1 = self.conv2d_t_br1(x_br1.permute([0, 2, 1, 3, 4]).reshape(-1, N, H, W)).view(B, -1, N, H, W).permute(
            [0, 2, 1, 3, 4])
        x_br2 = (self.conv2_br(xt_br1.reshape(B * N, -1, H, W))).view(B, N, -1, H, W)
        xt_br2 = self.conv2d_t_br2(x_br2.permute([0, 2, 1, 3, 4]).reshape(-1, N, H, W)).view(B, -1, N, H, W).permute(
            [0, 2, 1, 3, 4])

        x_br3 = (self.conv3_br(xt_br2.reshape(B * N, -1, H, W))).view(B, N, -1, H, W)
        xt_br3 = self.conv2d_t_br3(x_br3.permute([0, 2, 1, 3, 4]).reshape(-1, N, H, W)).view(B, -1, N, H, W).permute(
            [0, 2, 1, 3, 4])

        x_br4 = (self.conv4_br(xt_br3.reshape(B * N, -1, H, W))).view(B, N, -1, H, W)
        xt_br4 = self.conv2d_t_br4(x_br4.permute([0, 2, 1, 3, 4]).reshape(-1, N, H, W)).view(B, -1, N, H, W).permute(
            [0, 2, 1, 3, 4])

        score_map_br = self.conv5_br_mul(xt_br4.reshape(B, -1, H, W))
        return score_map_tl, score_map_br
    def soft_argmax(self, score_map, return_dist=False, softmax=True):
        """ get soft-argmax coordinate for a given heatmap """
        score_vec = score_map.view((-1, self.num_search, self.feat_sz * self.feat_sz))  # (batch, T,feat_sz * feat_sz)
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



class Mul_Pyramid_Corner_Predictor(nn.Module):
    """ Corner Predictor module"""

    def __init__(self, inplanes=64, channel=256, feat_sz=20, stride=16, freeze_bn=False, number_search=1):
        super(Mul_Pyramid_Corner_Predictor, self).__init__()
        self.feat_sz = feat_sz
        self.stride = stride
        self.img_sz = self.feat_sz * self.stride
        self.number_search = number_search
        '''top-left corner'''
        self.conv1_tl = Conv2d(inplanes, channel, freeze_bn=freeze_bn)
        self.conv2_tl = Conv2d(channel, channel // 2, freeze_bn=freeze_bn)
        self.conv3_tl = Conv2d(channel // 2, channel // 4, freeze_bn=freeze_bn)
        self.conv4_tl = Conv2d(channel // 4, channel // 8, freeze_bn=freeze_bn)
        self.conv5_tl = nn.Conv2d(channel // 8, 1, kernel_size=1)
        # self.time_conv5_tl = nn.Conv2d(number_search * channel // 8, number_search, kernel_size=1)

        self.adjust1_tl = Conv2d(inplanes, channel // 2, freeze_bn=freeze_bn)
        self.adjust2_tl = Conv2d(inplanes, channel // 4, freeze_bn=freeze_bn)
        self.adjust3_tl = nn.Sequential(Conv2d(channel // 2, channel // 4, freeze_bn=freeze_bn),
                                        Conv2d(channel // 4, channel // 8, freeze_bn=freeze_bn),
                                        Conv2d(channel // 8, 1, freeze_bn=freeze_bn))
        self.adjust4_tl = nn.Sequential(Conv2d(channel // 4, channel // 8, freeze_bn=freeze_bn),
                                        Conv2d(channel // 8, 1, freeze_bn=freeze_bn))
        # self.time_adjust3_tl = nn.Sequential(Conv2d(number_search * channel // 2, number_search * channel // 4, freeze_bn=freeze_bn),
        #                                 Conv2d(number_search * channel // 4, number_search * channel // 8, freeze_bn=freeze_bn),
        #                                 Conv2d(number_search * channel // 8, number_search, freeze_bn=freeze_bn))
        # self.time_adjust4_tl = nn.Sequential(Conv2d(number_search * channel // 4, number_search * channel // 8, freeze_bn=freeze_bn),
        #                                 Conv2d(number_search * channel // 8, number_search, freeze_bn=freeze_bn))

        '''bottom-right corner'''
        self.conv1_br = Conv2d(inplanes, channel, freeze_bn=freeze_bn)
        self.conv2_br = Conv2d(channel, channel // 2, freeze_bn=freeze_bn)
        self.conv3_br = Conv2d(channel // 2, channel // 4, freeze_bn=freeze_bn)
        self.conv4_br = Conv2d(channel // 4, channel // 8, freeze_bn=freeze_bn)
        self.conv5_br = nn.Conv2d(channel // 8, 1, kernel_size=1)
        # self.time_conv5_br = nn.Conv2d(number_search * channel // 8, number_search, kernel_size=1)

        self.adjust1_br = Conv2d(inplanes, channel // 2, freeze_bn=freeze_bn)
        self.adjust2_br = Conv2d(inplanes, channel // 4, freeze_bn=freeze_bn)

        self.adjust3_br = nn.Sequential(Conv2d(channel // 2, channel // 4, freeze_bn=freeze_bn),
                                        Conv2d(channel // 4, channel // 8, freeze_bn=freeze_bn),
                                        Conv2d(channel // 8, 1, freeze_bn=freeze_bn))
        self.adjust4_br = nn.Sequential(Conv2d(channel // 4, channel // 8, freeze_bn=freeze_bn),
                                        Conv2d(channel // 8, 1, freeze_bn=freeze_bn))

        # self.time_adjust3_br = nn.Sequential(Conv2d(number_search * channel // 2, number_search * channel // 4, freeze_bn=freeze_bn),
        #                                 Conv2d(number_search * channel // 4, number_search * channel // 8, freeze_bn=freeze_bn),
        #                                 Conv2d(number_search * channel // 8, number_search, freeze_bn=freeze_bn))
        # self.time_adjust4_br = nn.Sequential(Conv2d(number_search * channel // 4, number_search * channel // 8, freeze_bn=freeze_bn),
        #                                 Conv2d(number_search * channel // 8, number_search, freeze_bn=freeze_bn))

        self.time_tl_conv1 = Conv2d(number_search, number_search)
        self.time_tl_conv2 = Conv2d(number_search, number_search)
        self.time_tl_conv3 = Conv2d(number_search, number_search)

        self.time_br_conv1 = Conv2d(number_search, number_search)
        self.time_br_conv2 = Conv2d(number_search, number_search)
        self.time_br_conv3 = Conv2d(number_search, number_search)
        '''about coordinates and indexs'''
        with torch.no_grad():
            self.indice = torch.arange(0, self.feat_sz).view(-1, 1) * self.stride
            # generate mesh-grid
            self.coord_x = self.indice.repeat((self.feat_sz, 1)) \
                .view((self.feat_sz * self.feat_sz,)).float().cuda()
            self.coord_y = self.indice.repeat((1, self.feat_sz)) \
                .view((self.feat_sz * self.feat_sz,)).float().cuda()

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
        x = x.permute([1, 0, 2, 3, 4]) # B, N, C, H, W
        B, N, C, H, W = x.shape
        x = x.reshape(-1, C, H, W)

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

        # time tl
        x = self.time_tl_conv1(x_tl2.view(B, N, -1, H, W).permute([0, 2, 1, 3, 4]).reshape(-1, N, H, W))
        x_tl2 = x_tl2 + x.view(B, -1, N, H, W).permute(0, 2, 1, 3, 4).reshape(B * N, -1, H, W)

        x = self.time_tl_conv2(x_tl3.view(B, N, -1, 2 * H, 2 * W).permute([0, 2, 1, 3, 4]).reshape(-1, N, 2 * H, 2 * W))
        x_tl3 = x_tl3 + x.view(B, -1, N, 2 * H, 2 * W).permute(0, 2, 1, 3, 4).reshape(B * N, -1, 2 * H, 2 * W)

        x = self.time_tl_conv3(x_tl4.view(B, N, -1, 4 * H, 4 * W).permute([0, 2, 1, 3, 4]).reshape(-1, N, 4 * H, 4 * W))
        x_tl4 = x_tl4 + x.view(B, -1, N, 4 * H, 4 * W).permute(0, 2, 1, 3, 4).reshape(B * N, -1, 4 * H, 4 * W)
        # time br
        x = self.time_br_conv1(x_br2.view(B, N, -1, H, W).permute([0, 2, 1, 3, 4]).reshape(-1, N, H, W))
        x_br2 = x_br2 + x.view(B, -1, N, H, W).permute(0, 2, 1, 3, 4).reshape(B * N, -1, H, W)

        x = self.time_br_conv2(x_br3.view(B, N, -1, 2 * H, 2 * W).permute([0, 2, 1, 3, 4]).reshape(-1, N, 2 * H, 2 * W))
        x_br3 = x_br3 + x.view(B, -1, N, 2 * H, 2 * W).permute(0, 2, 1, 3, 4).reshape(B * N, -1, 2 * H, 2 * W)

        x = self.time_br_conv3(x_br4.view(B, N, -1, 4 * H, 4 * W).permute([0, 2, 1, 3, 4]).reshape(-1, N, 4 * H, 4 * W))
        x_br4 = x_br4 + x.view(B, -1, N, 4 * H, 4 * W).permute(0, 2, 1, 3, 4).reshape(B * N, -1, 4 * H, 4 * W)

        # # score_map
        # x_tl2 = x_tl2.view(B, N, -1, H, W).view(B, -1, H, W)
        # x_tl3 = x_tl3.view(B, N, -1, 2 * H, 2 * W).view(B, -1, 2 * H, 2 * W)
        # x_tl4 = x_tl4.view(B, N, -1, 4 * H, 4 * W).view(B, -1, 4 * H, 4 * W)
        #
        # x_br2 = x_br2.view(B, N, -1, H, W).view(B, -1, H, W)
        # x_br3 = x_br3.view(B, N, -1, 2 * H, 2 * W).view(B, -1, 2 * H, 2 * W)
        # x_br4 = x_br4.view(B, N, -1, 4 * H, 4 * W).view(B, -1, 4 * H, 4 * W)

        score_map_tl = self.conv5_tl(x_tl4) + F.interpolate(self.adjust3_tl(x_tl2), scale_factor=4) + F.interpolate(
            self.adjust4_tl(x_tl3), scale_factor=2)
        score_map_br = self.conv5_br(x_br4) + F.interpolate(self.adjust3_br(x_br2), scale_factor=4) + F.interpolate(
            self.adjust4_br(x_br3), scale_factor=2)
        return score_map_tl.view(B, N, self.feat_sz, self.feat_sz), score_map_br.view(B, N, self.feat_sz, self.feat_sz)

    def soft_argmax(self, score_map, return_dist=False, softmax=True):
        """ get soft-argmax coordinate for a given heatmap """
        score_vec = score_map.view((-1, self.number_search, self.feat_sz * self.feat_sz))  # (batch, T,feat_sz * feat_sz)
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
    # stride = 16
    stride = 4
    feat_sz = int(cfg.DATA.SEARCH.SIZE / stride)
    channel = getattr(cfg.MODEL, "HEAD_DIM", 384)
    # head_bbox = MulSearchHead(inplanes=cfg.MODEL.HIDDEN_DIM, channel=channel,
    #                                                    feat_sz=feat_sz, stride=stride, num_search=cfg.DATA.SEARCH.NUMBER)
    head_bbox = Mul_Pyramid_Corner_Predictor(inplanes=cfg.MODEL.HIDDEN_DIM, channel=channel,
                                                       feat_sz=feat_sz, stride=stride, number_search=cfg.DATA.SEARCH.NUMBER)
    head_score = MulScoreDecoder(num_search=cfg.DATA.SEARCH.NUMBER)
    checkpoint = backbone.box_head.state_dict()
    head_bbox.load_state_dict(checkpoint, strict= False)
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

