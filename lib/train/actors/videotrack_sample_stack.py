from . import BaseActor
from lib.utils.misc import NestedTensor
from lib.utils.box_ops import box_cxcywh_to_xyxy, box_xywh_to_xyxy
import torch
from lib.utils.loss import *
from lib.utils.merge import merge_template_search
# from ...utils.heapmap_utils import generate_heatmap
# from ...utils.ce_utils import generate_mask_cond, adjust_keep_rate
import copy
from torch import nn


class VideoActor(BaseActor):
    """ Actor for training OSTrack models """

    def __init__(self, net, objective, loss_weight, settings, cfg=None):
        super().__init__(net, objective)
        self.loss_weight = loss_weight
        self.settings = settings
        self.batch_size = self.settings.batchsize  # batch size
        self.cfg = cfg

        for p in net.module.backbone.parameters():
            p.require_grad = False
        self.net.module.backbone.eval()

        # for n, p in net.named_parameters():
        #     if "time" in n or 'score' in n:
        #         p.requires_grad = True
        #     else:
        #         p.requires_grad = False



    def __call__(self, data):
        """
        args:
            data - The input data, should contain the fields 'template', 'search', 'gt_bbox'.
            template_images: (N_t, batch, 3, H, W)
            search_images: (N_s, batch, 3, H, W)
        returns:
            loss    - the training loss
            status  -  dict containing detailed losses
        """
        # forward pass
        # loss0, out_dict = self.forward_pass(data, memory)
        out_dict = self.forward_pass(data)
        # from matplotlib import pyplot as plt
        # plt.figure(figsize=(128, 128))
        # plt.imshow(torch.mean(torch.abs(data['search_images'][0, 0].permute([1, 2, 0]).detach().to('cpu')), dim=-1))

        # compute losses
        loss1, status = self.compute_losses(out_dict, data)

        # return loss0+loss1, status, out_dict
        # return loss1, status, out_dict
        return loss1, status

    def forward_pass(self, data):
        # with torch.no_grad():
        searches = data['search_feature']
        template = data['template_feature']
        gt = box_xywh_to_xyxy(data['ground_truth'])

        N, B, C, Hs, Ws = searches.shape
        searches = self.random_mask_tokens(searches, mask_ratio=0.25)
        # bboxes head
        search = searches.permute([1, 0, 2, 3, 4]).reshape(B, -1, Hs, Ws)
        # out_dict = self.net(template, searches, gt)
        out_dict = self.net(template, search, gt)
        return out_dict
    def compute_losses(self, pred_dict, gt_dict, return_status=True):
        gt_bbox = gt_dict['ground_truth'] #N, B, 4
        # Get boxes
        # pred_scores = pred_dict['score']
        # background = pred_dict['memory']['x']
        pred_boxes = pred_dict['pred_boxes']
        pred_scores = pred_dict['pred_scores']
        # m_dict = pred_dict['memory']
        # gt_label = gt_dict['label'].view(-1)

        if torch.isnan(pred_boxes).any():
            raise ValueError("Network outputs is NAN! Stop Training")
        # num_queries = pred_boxes.size(1)
        pred_boxes_vec = box_cxcywh_to_xyxy(pred_boxes.reshape(-1, 4)).view(-1, 4)  # (B,N,4) --> (BN,4) (x1,y1,x2,y2)
        gt_boxes_vec = box_xywh_to_xyxy(gt_bbox.view(-1, 4)).clamp(min=0.0,max=1.0).view(-1, 4)
        iou, _ = box_iou(pred_boxes_vec, gt_boxes_vec) #
        gt_label = iou > 0.5
        gt_label = gt_label.float().cuda()
        # compute giou and iou
        try:
            iou_loss, iou = self.objective['iou'](pred_boxes_vec, gt_boxes_vec)  # (BN,4) (BN,4)
        except:
            iou_loss, iou = torch.tensor(1.0).cuda(), torch.tensor(0.0).cuda()

        # compute l1 loss
        l1_loss = self.objective['l1'](pred_boxes_vec, gt_boxes_vec)  # (BN,4) (BN,4)
        # compute score loss
        s_loss = self.objective['score'](pred_scores.view(-1), gt_label)
        if torch.isnan(s_loss):
            s_loss = torch.tensor(1.0).cuda()


        # weighted sum
        # back_loss = pred_dict['back_loss']
        loss = self.loss_weight['iou'] * iou_loss + self.loss_weight['l1'] * l1_loss \
               + self.loss_weight['score'] * s_loss \
               # + 1 * back_loss
               # + 0.5 * back_loss
                # + max_loss

        # loss = renew_loss
        if return_status:
            # status for log
            mean_iou = iou.detach().mean()
            status = {"Loss/total": loss.item(),
                      # "Loss/max_memory": max_loss.item(),
                      "Loss/score": s_loss.item(),
                      "Loss/iou": iou_loss.item(),
                      "Loss/l1": l1_loss.item(),
                      # 'Loss/background': back_loss.item(),
                      "IoU": mean_iou.item(),
                      }
            return loss, status
        else:
            return loss
    def random_mask_tokens(self, searches, mask_ratio=0.):
        N, B, C, Hs, Ws = searches.shape
        mask = torch.rand([N, B, 1, Hs, Ws])>mask_ratio
        mask = mask.expand(searches.shape)
        searches *= mask.cuda()
        return searches


