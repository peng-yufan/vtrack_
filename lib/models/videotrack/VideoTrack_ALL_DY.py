# -*-coding:utf-8-*-
from functools import partial

import torch
import torch.nn as nn
import os
from ..mixformer_vit.mixformer_online import build_mixformer_vit_online_score
from ..mixformer_convmae.mixformer_online import build_mixformer_convmae_online_score
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
            gts = out_bb.permute([2, 0, 1])
        out_score = self.head_score(search_tokens, template_tokens, gts) if self.head_score!=None else None

        out_dict = {'pred_boxes': box_xyxy_to_cxcywh(out_bb.permute([2, 0, 1])), #N, B, 4
                    'pred_scores': out_score.view(-1, B) if out_score!=None else None
                    }
        # out_dict = {'pred_boxes': out_bb, #N, B, 4
        #             'pred_scores': out_score.view(-1, B)
        #             }
        return out_dict


class Conv2d(nn.Module):
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
        x = self.net0(x)
        x = self.net1(x)
        return x
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
        x = self.net0(x)
        x = self.net1(x)
        return x
class conv_3d(nn.Module):
    def __init__(self, in_embedding=768, out_embedding=768):
        super().__init__()
        # self.net0 = nn.Sequential(
        #     nn.Linear(in_embedding, out_embedding),
        #     nn.BatchNorm1d(out_embedding),
        #     nn.ReLU(inplace=True),
        # )
        self.net0 = nn.Sequential(
            nn.Conv3d(in_channels=in_embedding, out_channels=out_embedding
                      , kernel_size=3, padding=1, stride=1),
            nn.BatchNorm3d(out_embedding),
            # FrozenBatchNorm2d(out_embedding),
            nn.GELU(),
        )
    def forward(self, x):
        x = self.net0(x)
        return x

class conv_dy_3d(nn.Module):
    def __init__(self, in_embedding=768, out_embedding=768, num_search=1, kz=3, time=2):
        super().__init__()
        self.num_search = num_search
        self.net_3d = nn.Sequential(
            nn.Conv3d(in_channels=in_embedding, out_channels=out_embedding
                      , kernel_size=[time,kz,kz], padding=[int((time-1)/2+1),int((kz-1)/2),int((kz-1)/2)], stride=1),
            nn.BatchNorm3d(out_embedding),
            nn.GELU(),
        )
        self.net_2d = nn.Sequential(
            nn.Conv2d(in_channels=in_embedding, out_channels=out_embedding
                      , kernel_size=[kz,kz], padding=int((kz-1)/2), stride=1),
            nn.BatchNorm2d(out_embedding),
            nn.GELU(),
        )
        self.net_alpha = nn.Sequential(
            nn.Conv2d(in_channels=num_search * out_embedding, out_channels=num_search * out_embedding
                      , kernel_size=[kz,kz], padding=int((kz-1)/2), stride=1, groups=num_search * out_embedding),
            nn.Conv2d(in_channels=num_search * out_embedding, out_channels=1
                      , kernel_size=1, padding=0, stride=1, groups=1),
            # nn.BatchNorm2d(out_embedding),
            # nn.GELU(),
        )
    def forward(self, x):
        b,c,t,h,w=x.shape
        x_3d = self.net_3d(x)
        x_3d = x_3d[:, :, :self.num_search]
        x_2d = (self.net_2d(x.permute([0,2,1,3,4]).reshape(-1,c,h,w))).reshape(b,t,-1,h,w).permute([0,2,1,3,4])
        alpha = (torch.sigmoid(self.net_alpha(x_3d.reshape(b,-1, h, w)))).unsqueeze(1)
        out = alpha*x_3d+(1-alpha)*x_2d
        return out
class MulSearchHead(nn.Module):
    def __init__(self, embedding=768, channel=256,  num_search=10, feat_sz=18, stride=16):
        super().__init__()
        self.embedding = embedding
        self.num_search = num_search
        self.feat_sz = feat_sz
        self.stride = stride
        self.img_sz = self.feat_sz * self.stride
        self.conv2d_tl1 = Conv2d(num_search * embedding, num_search * channel)
        self.conv2d_tl2 = Conv2d(num_search * channel, num_search * channel // 2)
        self.conv2d_tl3 = Conv2d(num_search * channel//2, num_search * channel // 4)
        self.conv2d_tl4 = Conv2d(num_search * channel//4, num_search * channel // 8)
        self.conv2d_tl5 = nn.Conv2d(num_search * channel//8, num_search, kernel_size=1)

        self.conv2d_br1 = Conv2d(num_search * embedding, num_search * channel)
        self.conv2d_br2 = Conv2d(num_search * channel, num_search * channel // 2)
        self.conv2d_br3 = Conv2d(num_search * channel//2, num_search * channel // 4)
        self.conv2d_br4 = Conv2d(num_search * channel//4, num_search * channel // 8)
        self.conv2d_br5 = nn.Conv2d(num_search * channel//8, num_search, kernel_size=1)

        '''about coordinates and indexs'''
        with torch.no_grad():
            self.indice = torch.arange(0, self.feat_sz).view(-1, 1) * self.stride
            # generate mesh-grid
            self.coord_x = self.indice.repeat((self.feat_sz, 1)) \
                .view((self.feat_sz * self.feat_sz,)).float().cuda()
            self.coord_y = self.indice.repeat((1, self.feat_sz)) \
                .view((self.feat_sz * self.feat_sz,)).float().cuda()

    def forward(self, x, return_dist=False, softmax=True):
        if x.dim() == 5:
            n, b, c, h, w = x.shape
            x = x.permute([1, 0, 2, 3, 4]).reshape([b, -1, h, w])
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
        # top-left branch
        x_tl1 = self.conv2d_tl1(x)
        x_tl2 = self.conv2d_tl2(x_tl1)
        x_tl3 = self.conv2d_tl3(x_tl2)
        x_tl4 = self.conv2d_tl4(x_tl3)
        score_map_tl = self.conv2d_tl5(x_tl4)

        # bottom-right branch
        x_br1 = self.conv2d_br1(x)
        x_br2 = self.conv2d_br2(x_br1)
        x_br3 = self.conv2d_br3(x_br2)
        x_br4 = self.conv2d_br4(x_br3)
        score_map_br = self.conv2d_br5(x_br4)
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
            self.indice = torch.arange(0, self.feat_sz).view(-1, 1) * int(self.stride)
            # generate mesh-grid
            self.coord_x = self.indice.repeat((self.feat_sz, 1)) \
                .view((self.feat_sz * self.feat_sz,)).float().cuda()
            self.coord_y = self.indice.repeat((1, self.feat_sz)) \
                .view((self.feat_sz * self.feat_sz,)).float().cuda()

    def forward(self, x, return_dist=False, softmax=True):
        """ Forward pass with input x. """
        if x.dim() == 5:
            n, b, c, h, w = x.shape
            x = x.permute([1, 0, 2, 3, 4]).reshape([b, -1, h, w])
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
class Pyramid_3D_Search_Predictor(nn.Module):
    """ Corner Predictor module"""

    def __init__(self, inplanes=64, channel=256, feat_sz=20, stride=16, freeze_bn=False, num_search=1):
        super(Pyramid_3D_Search_Predictor, self).__init__()
        self.feat_sz = feat_sz
        self.stride = stride
        self.img_sz = self.feat_sz * self.stride
        self.num_search = num_search
        '''top-left corner'''
        self.conv1_tl = conv_3d(inplanes, channel)
        self.conv2_tl = conv_3d(channel, channel // 2)
        self.conv3_tl = conv_3d(channel // 2, channel // 4)
        self.conv4_tl = conv_3d(channel // 4, channel // 8)
        self.conv5_tl = nn.Conv3d(channel // 8, 1, kernel_size=1)

        self.adjust1_tl = conv_3d(inplanes, channel // 2)
        self.adjust2_tl = conv_3d(inplanes, channel // 4)

        self.adjust3_tl = nn.Sequential(conv_3d(channel // 2, channel // 4),
                                        conv_3d(channel // 4, channel // 8),
                                        conv_3d(channel // 8, 1))
        self.adjust4_tl = nn.Sequential(conv_3d(channel // 4, channel // 8),
                                        conv_3d(channel // 8, 1))

        '''bottom-right corner'''
        self.conv1_br = conv_3d(inplanes, channel)
        self.conv2_br = conv_3d(channel, channel // 2)
        self.conv3_br = conv_3d(channel // 2, channel // 4)
        self.conv4_br = conv_3d(channel // 4, channel // 8)
        self.conv5_br = nn.Conv3d(channel // 8, 1, kernel_size=1)

        self.adjust1_br = conv_3d(inplanes, channel // 2)
        self.adjust2_br = conv_3d(inplanes, channel // 4)

        self.adjust3_br = nn.Sequential(conv_3d(channel // 2, channel // 4),
                                        conv_3d(channel // 4, channel // 8),
                                        conv_3d(channel // 8, 1))
        self.adjust4_br = nn.Sequential(conv_3d(channel // 4, channel // 8),
                                        conv_3d(channel // 8, 1))

        '''about coordinates and indexs'''
        with torch.no_grad():
            self.indice = torch.arange(0, self.feat_sz).view(-1, 1) * int(self.stride)
            # generate mesh-grid
            self.coord_x = self.indice.repeat((self.feat_sz, 1)) \
                .view((self.feat_sz * self.feat_sz,)).float().cuda()
            self.coord_y = self.indice.repeat((1, self.feat_sz)) \
                .view((self.feat_sz * self.feat_sz,)).float().cuda()

    def forward(self, x, return_dist=False, softmax=True):
        """ Forward pass with input x. """
        if x.dim() == 5:
            n, b, c, h, w = x.shape
            x = x.permute([1, 2, 0, 3, 4])
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
        b, c, n, h, w = x.shape
        x_init = x
        # top-left branch
        x_tl1 = self.conv1_tl(x)
        x_tl2 = self.conv2_tl(x_tl1)

        #up-1
        x_init_up1 = (F.interpolate((self.adjust1_tl(x_init)).view(b, -1, h, w), scale_factor=2)).view(b, -1, n, 2*h, 2*w)
        x_up1 = (F.interpolate(x_tl2.view(b, -1, h, w), scale_factor=2)).view(b, -1, n, 2*h, 2*w)
        x_up1 = x_init_up1 + x_up1

        x_tl3 = self.conv3_tl(x_up1)

        #up-2


        x_init_up2 = (F.interpolate((self.adjust2_tl(x_init)).view(b, -1, h, w), scale_factor=4)).view(b, -1, n, 4*h, 4*w)
        x_up2 = (F.interpolate(x_tl3.view(b, -1, 2*h, 2*w), scale_factor=2)).view(b, -1, n, 4*h, 4*w)
        x_up2 = x_init_up2 + x_up2

        x_tl4 = self.conv4_tl(x_up2)
        score_map_tl = (self.conv5_tl(x_tl4)).view(b,-1,4*h,4*w) + (F.interpolate((self.adjust3_tl(x_tl2)).view(b,-1,h,w), scale_factor=4)).view(b,-1,4*h,4*w) + (F.interpolate((self.adjust4_tl(x_tl3)).view(b,-1,2*h,2*w), scale_factor=2)).view(b,-1,4*h,4*w)

        # bottom-right branch
        x_br1 = self.conv1_br(x)
        x_br2 = self.conv2_br(x_br1)

        # up-1
        x_init_up1 = (F.interpolate((self.adjust1_br(x_init)).view(b, -1, h, w), scale_factor=2)).view(b, -1, n, 2 * h,
                                                                                                       2 * w)
        x_up1 = (F.interpolate(x_br2.view(b, -1, h, w), scale_factor=2)).view(b, -1, n, 2 * h, 2 * w)
        x_up1 = x_init_up1 + x_up1

        x_br3 = self.conv3_br(x_up1)

        # up-2

        x_init_up2 = (F.interpolate((self.adjust2_br(x_init)).view(b, -1, h, w), scale_factor=4)).view(b, -1, n, 4 * h,
                                                                                                       4 * w)
        x_up2 = (F.interpolate(x_br3.view(b, -1, 2 * h, 2 * w), scale_factor=2)).view(b, -1, n, 4 * h, 4 * w)
        x_up2 = x_init_up2 + x_up2

        x_br4 = self.conv4_br(x_up2)
        score_map_br = (self.conv5_br(x_br4)).view(b, -1, 4 * h, 4 * w) + (
            F.interpolate((self.adjust3_br(x_br2)).view(b, -1, h, w), scale_factor=4)).view(b, -1, 4 * h, 4 * w) + (
                           F.interpolate((self.adjust4_br(x_br3)).view(b, -1, 2 * h, 2 * w), scale_factor=2)).view(b,
                                                                                                                   -1,
                                                                                                                   4 * h,
                                                                                                                   4 * w)
        return score_map_tl, score_map_br

    def soft_argmax(self, score_map, return_dist=False, softmax=True):
        """ get soft-argmax coordinate for a given heatmap """
        try:
            score_vec = score_map.view((-1, self.num_search, self.feat_sz * self.feat_sz))  # (batch, T,feat_sz * feat_sz)
        except:
            score_vec = score_map.view((score_map.shape[0], -1, self.feat_sz * self.feat_sz))
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

class Pyramid_3D_Dy_Search_Predictor(nn.Module):
    """ Corner Predictor module"""

    def __init__(self, inplanes=64, channel=256, feat_sz=20, stride=16, freeze_bn=False, num_search=1):
        super(Pyramid_3D_Dy_Search_Predictor, self).__init__()
        self.feat_sz = feat_sz
        self.stride = stride
        self.img_sz = self.feat_sz * self.stride
        self.num_search = num_search
        '''top-left corner'''
        self.conv1_tl = conv_dy_3d(inplanes, channel, num_search)
        self.conv2_tl = conv_dy_3d(channel, channel // 2, num_search)
        self.conv3_tl = conv_dy_3d(channel // 2, channel // 4, num_search)
        self.conv4_tl = conv_dy_3d(channel // 4, channel // 8, num_search)
        # self.conv5_tl = nn.Conv3d(channel // 8, 1, kernel_size=1)
        self.conv5_tl = conv_dy_3d(channel // 8, 1, num_search, 1, 3)
        self.adjust1_tl = conv_dy_3d(inplanes, channel // 2, num_search)
        self.adjust2_tl = conv_dy_3d(inplanes, channel // 4, num_search)

        self.adjust3_tl = nn.Sequential(conv_dy_3d(channel // 2, channel // 4, num_search),
                                        conv_dy_3d(channel // 4, channel // 8, num_search),
                                        conv_dy_3d(channel // 8, 1, num_search))
        self.adjust4_tl = nn.Sequential(conv_dy_3d(channel // 4, channel // 8, num_search),
                                        conv_dy_3d(channel // 8, 1, num_search))

        '''bottom-right corner'''
        self.conv1_br = conv_dy_3d(inplanes, channel, num_search)
        self.conv2_br = conv_dy_3d(channel, channel // 2, num_search)
        self.conv3_br = conv_dy_3d(channel // 2, channel // 4, num_search)
        self.conv4_br = conv_dy_3d(channel // 4, channel // 8, num_search)
        # self.conv5_br = nn.Conv3d(channel // 8, 1, kernel_size=1)
        self.conv5_br = conv_dy_3d(channel // 8, 1, num_search, 1, 3)

        self.adjust1_br = conv_dy_3d(inplanes, channel // 2, num_search)
        self.adjust2_br = conv_dy_3d(inplanes, channel // 4, num_search)

        self.adjust3_br = nn.Sequential(conv_dy_3d(channel // 2, channel // 4, num_search),
                                        conv_dy_3d(channel // 4, channel // 8, num_search),
                                        conv_dy_3d(channel // 8, 1, num_search))
        self.adjust4_br = nn.Sequential(conv_dy_3d(channel // 4, channel // 8, num_search),
                                        conv_dy_3d(channel // 8, 1, num_search))

        '''about coordinates and indexs'''
        with torch.no_grad():
            self.indice = torch.arange(0, self.feat_sz).view(-1, 1) * int(self.stride)
            # generate mesh-grid
            self.coord_x = self.indice.repeat((self.feat_sz, 1)) \
                .view((self.feat_sz * self.feat_sz,)).float().cuda()
            self.coord_y = self.indice.repeat((1, self.feat_sz)) \
                .view((self.feat_sz * self.feat_sz,)).float().cuda()

    def forward(self, x, return_dist=False, softmax=True):
        """ Forward pass with input x. """
        if x.dim() == 5:
            n, b, c, h, w = x.shape
            x = x.permute([1, 2, 0, 3, 4])
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
        b, c, n, h, w = x.shape
        x_init = x
        # top-left branch
        x_tl1 = self.conv1_tl(x)
        x_tl2 = self.conv2_tl(x_tl1)

        #up-1
        x_init_up1 = (F.interpolate((self.adjust1_tl(x_init)).view(b, -1, h, w), scale_factor=2)).view(b, -1, n, 2*h, 2*w)
        x_up1 = (F.interpolate(x_tl2.view(b, -1, h, w), scale_factor=2)).view(b, -1, n, 2*h, 2*w)
        x_up1 = x_init_up1 + x_up1

        x_tl3 = self.conv3_tl(x_up1)

        #up-2


        x_init_up2 = (F.interpolate((self.adjust2_tl(x_init)).view(b, -1, h, w), scale_factor=4)).view(b, -1, n, 4*h, 4*w)
        x_up2 = (F.interpolate(x_tl3.view(b, -1, 2*h, 2*w), scale_factor=2)).view(b, -1, n, 4*h, 4*w)
        x_up2 = x_init_up2 + x_up2

        x_tl4 = self.conv4_tl(x_up2)
        score_map_tl = (self.conv5_tl(x_tl4)).view(b,-1,4*h,4*w) + (F.interpolate((self.adjust3_tl(x_tl2)).view(b,-1,h,w), scale_factor=4)).view(b,-1,4*h,4*w) + (F.interpolate((self.adjust4_tl(x_tl3)).view(b,-1,2*h,2*w), scale_factor=2)).view(b,-1,4*h,4*w)

        # bottom-right branch
        x_br1 = self.conv1_br(x)
        x_br2 = self.conv2_br(x_br1)

        # up-1
        x_init_up1 = (F.interpolate((self.adjust1_br(x_init)).view(b, -1, h, w), scale_factor=2)).view(b, -1, n, 2 * h,
                                                                                                       2 * w)
        x_up1 = (F.interpolate(x_br2.view(b, -1, h, w), scale_factor=2)).view(b, -1, n, 2 * h, 2 * w)
        x_up1 = x_init_up1 + x_up1

        x_br3 = self.conv3_br(x_up1)

        # up-2

        x_init_up2 = (F.interpolate((self.adjust2_br(x_init)).view(b, -1, h, w), scale_factor=4)).view(b, -1, n, 4 * h,
                                                                                                       4 * w)
        x_up2 = (F.interpolate(x_br3.view(b, -1, 2 * h, 2 * w), scale_factor=2)).view(b, -1, n, 4 * h, 4 * w)
        x_up2 = x_init_up2 + x_up2

        x_br4 = self.conv4_br(x_up2)
        score_map_br = (self.conv5_br(x_br4)).view(b, -1, 4 * h, 4 * w) + (
            F.interpolate((self.adjust3_br(x_br2)).view(b, -1, h, w), scale_factor=4)).view(b, -1, 4 * h, 4 * w) + (
                           F.interpolate((self.adjust4_br(x_br3)).view(b, -1, 2 * h, 2 * w), scale_factor=2)).view(b,
                                                                                                                   -1,
                                                                                                                   4 * h,
                                                                                                                   4 * w)
        return score_map_tl, score_map_br

    def soft_argmax(self, score_map, return_dist=False, softmax=True):
        """ get soft-argmax coordinate for a given heatmap """
        try:
            score_vec = score_map.view((-1, self.num_search, self.feat_sz * self.feat_sz))  # (batch, T,feat_sz * feat_sz)
        except:
            score_vec = score_map.view((score_map.shape[0], -1, self.feat_sz * self.feat_sz))
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
    if cfg.MODEL.VIT_TYPE == 'base_patch16':
        if settings != None:
            settings.stage1_model = pretrained_path + '/mixformer_vit_base_online.pth.tar'
        backbone = build_mixformer_vit_online_score(cfg, settings=settings, train=training)
    elif cfg.MODEL.VIT_TYPE == 'convmae_large':
        if settings != None:
            settings.stage1_model = pretrained_path + '/mixformer_convmae_large_online.pth.tar'
        backbone = build_mixformer_convmae_online_score(cfg, settings=settings, train=training)

    # stride = 16
    stride = 4
    feat_sz = int(cfg.DATA.SEARCH.SIZE / stride)
    channel = getattr(cfg.MODEL, "HEAD_DIM", 384)
    # head_bbox = Pyramid_MulSearch_Predictor(inplanes=cfg.MODEL.HIDDEN_DIM, channel=channel,
    #                                                    feat_sz=feat_sz, stride=stride, num_search=cfg.DATA.SEARCH.NUMBER)
    head_bbox = Pyramid_3D_Dy_Search_Predictor(inplanes=cfg.MODEL.HIDDEN_DIM, channel=channel,
                                                       feat_sz=feat_sz, stride=stride, num_search=cfg.DATA.SEARCH.NUMBER)
    # stride = 16
    # feat_sz = int(cfg.DATA.SEARCH.SIZE / stride)
    # channel = getattr(cfg.MODEL, "HEAD_DIM", 384)
    # head_bbox = MulSearchHead(embedding=cfg.MODEL.HIDDEN_DIM, channel=channel,
    #                                                    feat_sz=feat_sz, stride=stride, num_search=cfg.DATA.SEARCH.NUMBER)
    # head_score = MulScoreDecoder(pool_size=4, hidden_dim=cfg.MODEL.HIDDEN_DIM, num_heads=cfg.MODEL.HIDDEN_DIM // 64,
    #                              num_search=6)

    # head_score = MulScoreDecoder(pool_size=4, hidden_dim=cfg.MODEL.HIDDEN_DIM, num_heads=cfg.MODEL.HIDDEN_DIM//64, num_search=cfg.DATA.SEARCH.NUMBER)
    head_score = None
    # checkpoint = backbone.box_head.state_dict()
    # head_bbox.load_state_dict(checkpoint, strict= False)
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

