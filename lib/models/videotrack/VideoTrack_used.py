# -*-coding:utf-8-*-
from functools import partial
from lib.utils.box_ops import box_cxcywh_to_xyxy, box_xywh_to_xyxy
import torch
import torch.nn as nn
import os
from ..mixformer_vit.mixformer_online import build_mixformer_vit_online_score
from lib.models.videotrack.score_decoder_mul import MulScoreDecoder
from lib.models.mixformer_cvt.head import Mul_Pyramid_Corner_Predictor
from lib.models.videotrack.VideoTrack_stack import MulSearchHead
from lib.models.videotrack.VideoTrack_stack_py import Pyramid_MulSearch_Predictor
from lib.models.component.mlp import MultiLayerMlp
from matplotlib import pyplot as plt
import cv2
import numpy as np
import random
from timm.models.layers import Mlp


class VideoTrack(nn.Module):
    def __init__(self, embedding=768, backbone=None, head_bbox=None, head_score=None, num_search=1):
        super().__init__()
        self.embedding = embedding
        self.backbone = backbone
        self.head_bbox = head_bbox
        self.head_score = head_score
        self.num_search = num_search
    def forward(self, data):
        searches = []
        templates = []
        template0 = data['template_images'][0]
        with torch.no_grad():
            # self.net.module.backbone.forward(data['template_images'][0], data['template_images'][1])
            for template, search in zip(data['template_images'][:-1], data['search_images']):
                out, _ = self.backbone(template0, template, search)
                templates.append(out['template_feature'].detach())
                searches.append(out['search_feature'].detach())
            templates = torch.stack(templates, dim=0)
            searches = torch.stack(searches, dim=0)
            N, B, C, Hs, Ws = searches.shape
            # searches = self.random_mask_tokens(searches, mask_ratio=0.25)
            # bboxes head
            search = searches.permute([1, 0, 2, 3, 4]).reshape(B, -1, Hs, Ws)
        # out_dict = self.net(template, searches, gt)
        out_dict = self.forward_pass(templates, search)
        # out_dict = self.net(template.clone().detach(), search.clone().detach(), gts=gt.clone().detach())
        return out_dict
    def forward_pass(self, template_tokens, search_tokens, gts=None):
        B, NC, Hs, Ws = search_tokens.shape
        out_bb = self.head_bbox(search_tokens)
        search_tokens = search_tokens.reshape(B, self.num_search, int(NC/self.num_search), Hs, Ws)\
            .permute([1, 0, 2, 3, 4]).contiguous()
        if gts is None:
            gts = box_xywh_to_xyxy(out_bb.permute([2, 0, 1]))
        out_score = self.head_score(search_tokens, template_tokens, gts)

        out_dict = {'pred_boxes': out_bb.permute([2, 0, 1]), #N, B, 4
                    'pred_scores': out_score.view(-1, B)
                    }
        # out_dict = {'pred_boxes': out_bb, #N, B, 4
        #             'pred_scores': out_score.view(-1, B)
        #             }
        return out_dict
    # def forward(self, data):
    #     gt = box_xywh_to_xyxy(data['search_gt_bboxes'])
    #     searches = []
    #     template = []
    #     with torch.no_grad():
    #         self.backbone.set_online(data['template_images'][0].clone().detach(), data['template_images'][1])
    #         for search in data['search_images']:
    #             out, _ = self.backbone.forward_test(search, run_score_head=True)
    #             template.append(out['template_feature'])
    #             searches.append(out['search_feature'])
    #         template_tokens = torch.stack(template, dim=0).clone().detach()
    #         searches = torch.stack(searches, dim=0).clone().detach()
    #         N, B, C, Hs, Ws = searches.shape
    #         searches = self.random_mask_tokens(searches, mask_ratio=0.25)
    #         # bboxes head
    #         search_tokens = searches.permute([1, 0, 2, 3, 4]).reshape(B, -1, Hs, Ws)
    #     B, NC, Hs, Ws = search_tokens.shape
    #     out_bb = self.head_bbox(search_tokens)
    #     search_tokens = search_tokens.reshape(B, self.num_search, int(NC/self.num_search), Hs, Ws)\
    #         .permute([1, 0, 2, 3, 4]).contiguous()
    #     out_score = self.head_score(search_tokens, template_tokens, gt)
    #     out_dict = {'pred_boxes': out_bb.permute([2, 0, 1]), #N, B, 4
    #                 'pred_scores': out_score.view(-1, B)
    #                 }
    #     # out_dict = {'pred_boxes': out_bb, #N, B, 4
    #     #             'pred_scores': out_score.view(-1, B)
    #     #             }
    #     return out_dict
    # def random_mask_tokens(self, searches, mask_ratio=0.):
    #     N, B, C, Hs, Ws = searches.shape
    #     mask = torch.rand([N, B, 1, Hs, Ws])>mask_ratio
    #     mask = mask.expand(searches.shape)
    #     searches *= mask.cuda()
    #     return searches


class Conv2d(nn.Module):
    def __init__(self, in_embedding=768, out_embedding=768):
        super().__init__()
        self.net0 = nn.Sequential(
            nn.Linear(in_embedding, out_embedding),
            nn.BatchNorm1d(out_embedding),
            nn.ReLU(inplace=True),
        )
        self.net1 = nn.Sequential(
            nn.Conv2d(in_channels=out_embedding, out_channels=out_embedding
                      , kernel_size=3, padding=1, stride=1, groups=out_embedding),
            nn.BatchNorm2d(out_embedding),
            nn.ReLU(inplace=True),
        )
    def forward(self, x):
        x = x.permute([0, 2, 3, 1])
        shape = x.shape
        x = x.reshape(-1, shape[-1])
        x = self.net0(x)
        shape = [i for i in shape[:3]]
        shape.append(x.shape[-1])
        x = x.view(shape)
        x = x.permute([0, 3, 1, 2])

        x = self.net1(x)
        return x



def build_videonet(cfg, settings=None, training=True):
    current_dir = os.path.dirname(os.path.abspath(__file__))  # This is your Project Root
    pretrained_path = os.path.join(current_dir, 'pretrained_models')
    if settings!=None:
        settings.stage1_model = pretrained_path + '/mixformer_vit_base_online.pth.tar'
    backbone = build_mixformer_vit_online_score(cfg, settings=settings, train=training)
    # channel = getattr(cfg.MODEL, "HEAD_DIM", 384)
    # freeze_bn = getattr(cfg.MODEL, "HEAD_FREEZE_BN", False)
    # if cfg.MODEL.HEAD_TYPE == "CORNER_UP":
    #     stride = 4
    #     feat_sz = int(cfg.DATA.SEARCH.SIZE / stride)
        # head_bbox = Mul_Pyramid_Corner_Predictor(inplanes=cfg.MODEL.HIDDEN_DIM, channel=channel,
        #                                                feat_sz=feat_sz, stride=stride, freeze_bn=freeze_bn, num_search=cfg.DATA.SEARCH.NUMBER)
    stride = 16
    feat_sz = int(cfg.DATA.SEARCH.SIZE / stride)
    channel = getattr(cfg.MODEL, "HEAD_DIM", 256)
    # head_bbox = MulSearchHead(embedding=cfg.MODEL.HIDDEN_DIM, channel=channel,
    #                                                    feat_sz=feat_sz, stride=stride, num_search=cfg.DATA.SEARCH.NUMBER)
    head_bbox = Pyramid_MulSearch_Predictor(inplanes=cfg.MODEL.HIDDEN_DIM, channel=channel,
                                                       feat_sz=feat_sz, stride=stride, num_search=cfg.DATA.SEARCH.NUMBER)
    head_score = MulScoreDecoder(num_search=cfg.DATA.SEARCH.NUMBER)


    # missing_keys_bbox, unexpected_keys_bbox = head_bbox.load_state_dict(backbone.box_head.state_dict(), strict=False)
    # missing_keys_score, unexpected_keys_score = head_score.load_state_dict(backbone.score_branch.state_dict(), strict=False)

    model = VideoTrack(backbone=backbone, head_bbox=head_bbox, head_score=head_score, num_search=cfg.DATA.SEARCH.NUMBER)
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


