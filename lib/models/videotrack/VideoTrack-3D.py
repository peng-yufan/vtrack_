# -*-coding:utf-8-*-
from functools import partial

import torch
import torch.nn as nn
# from lib.models.videotrack.DeformableBlock_3D import *
import os
from ..mixformer_vit.mixformer_online import build_mixformer_vit_online_score
from lib.models.videotrack.score_decoder_mul import MulScoreDecoder
from lib.utils.box_ops import box_xyxy_to_cxcywh, box_cxcywh_to_xyxy
from lib.models.mixformer_cvt.head import *
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
    def __init__(self, embedding=768, backbone=None, neck=None, head_bbox=None, head_score=None):
        super().__init__()
        self.embedding = embedding
        self.backbone = backbone
        self.neck = neck
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
        search_tokens = search_tokens.permute([1, 2, 0, 3, 4]) # b, c, n, h, w
        search_tokens = self.neck(search_tokens)
        search_tokens = search_tokens.permute([0, 2, 1, 3, 4]).reshape([B * N, C, Hs, Ws])
        out_bb = self.head_bbox(search_tokens)
        out_bb = out_bb.view(B, N, -1)
        # search_tokens = search_tokens.reshape(B, self.num_search, int(NC/self.num_search), Hs, Ws)\
        #     .permute([1, 0, 2, 3, 4]).contiguous()
        if gts is None:
            gts = out_bb.permute([1, 0, 2])
        out_score = self.head_score(search_tokens.view(B,N,C,Hs,Ws).permute([1,0,2,3,4]).contiguous(), template_tokens, gts)

        out_dict = {'pred_boxes': box_xyxy_to_cxcywh(out_bb.permute([1, 0, 2])), #N, B, 4
                    'pred_scores': out_score.view(-1, B)
                    }
        # out_dict = {'pred_boxes': out_bb, #N, B, 4
        #             'pred_scores': out_score.view(-1, B)
        #             }
        return out_dict

class Neck_Block(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False):
        super(Neck_Block, self).__init__()
        # self.conv1 = DeformConv3d(in_channels, out_channels, kernel_size, stride, padding, bias)
        self.conv1 = nn.Conv3d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                               stride=stride, padding=padding, bias=bias)
        self.bn1 = nn.BatchNorm3d(in_channels)
        self.relu = nn.ReLU(inplace=True)
        # self.conv2 = DeformConv3d(in_channels, out_channels, kernel_size, stride, padding, bias)
        self.conv2 = nn.Conv3d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                               stride=stride, padding=padding, bias=bias)
        self.bn2 = nn.BatchNorm3d(in_channels)
    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += residual
        out = self.relu(out)
        return out
class Neck(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=True, N=3):
        super(Neck, self).__init__()
        self.blocks = nn.Sequential(*[Neck_Block(in_channels, out_channels, kernel_size, stride, padding, bias) for i in range(N)])
    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        return x

def build_videonet(cfg, settings=None, training=True):
    current_dir = os.path.dirname(os.path.abspath(__file__))  # This is your Project Root
    pretrained_path = os.path.join(current_dir, 'pretrained_models')
    if settings!=None:
        settings.stage1_model = pretrained_path + '/mixformer_vit_base_online.pth.tar'
    backbone = build_mixformer_vit_online_score(cfg, settings=settings, train=training)
    neck = Neck(in_channels=cfg.MODEL.HIDDEN_DIM, out_channels=cfg.MODEL.HIDDEN_DIM, N=5)
    # stride = 16
    stride = 4
    feat_sz = int(cfg.DATA.SEARCH.SIZE / stride)
    channel = getattr(cfg.MODEL, "HEAD_DIM", 384)
    # head_bbox = Pyramid_MulSearch_Predictor(inplanes=cfg.MODEL.HIDDEN_DIM, channel=channel,
    #                                                    feat_sz=feat_sz, stride=stride, num_search=cfg.DATA.SEARCH.NUMBER)
    head_bbox = Pyramid_Corner_Predictor(inplanes=cfg.MODEL.HIDDEN_DIM, channel=channel, feat_sz=feat_sz, stride=stride)
    # stride = 16
    # feat_sz = int(cfg.DATA.SEARCH.SIZE / stride)
    # channel = getattr(cfg.MODEL, "HEAD_DIM", 384)
    # head_bbox = MulSearchHead(embedding=cfg.MODEL.HIDDEN_DIM, channel=channel,
    #                                                    feat_sz=feat_sz, stride=stride, num_search=cfg.DATA.SEARCH.NUMBER)

    head_score = MulScoreDecoder(num_search=cfg.DATA.SEARCH.NUMBER)
    # checkpoint = backbone.box_head.state_dict()
    # head_bbox.load_state_dict(checkpoint, strict= False)
    model = VideoTrack(backbone=backbone, head_bbox=head_bbox, neck=neck, head_score=head_score)
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

