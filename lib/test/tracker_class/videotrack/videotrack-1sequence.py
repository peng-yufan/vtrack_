import numpy

from lib.test.tracker_class.basetracker import BaseTracker
import  matplotlib.pyplot as plt
import torch
from lib.train.data.util.processing_utils import sample_target
# for debug
import cv2
import os
from lib.models.videotrack.VideoTrack import build_videonet
from lib.test.utils.pre_processor import Preprocessor_wo_mask
from lib.utils.box_ops import clip_box
from lib.utils.loss import *
from lib.utils.box_ops import giou_loss
import time
from lib.train.data.util.processing_utils import *



class VideoTrack(BaseTracker):
    def __init__(self, params, dataset_name):
        super(VideoTrack, self).__init__(params)
        network = build_videonet(params.cfg, training=False)
        network.load_state_dict(torch.load(self.params.checkpoint, map_location='cpu')['net'], strict=True)
        print(f"Load checkpoint {self.params.checkpoint} successfully!")
        self.cfg = params.cfg
        self.network = network.cuda()
        self.network.eval()
        self.attn_weights = []
        self.search_factor = 1.0 * self.params.search_factor
        self.num_search = params.cfg.DATA.SEARCH.NUMBER

        self.preprocessor = Preprocessor_wo_mask()
        self.state = None
        # for debug
        self.debug = False
        self.frame_id = 0
        if self.debug:
            self.save_dir = "debug"
            if not os.path.exists(self.save_dir):
                os.makedirs(self.save_dir)
        # for save boxes from all queries
        self.save_all_boxes = params.save_all_boxes

        # Set the update interval
        self.aver_num = 1
        # self.aver_num = 2
        # self.inter = 2
        # self.inter = 5
        # self.inter = 10
        # self.inter = 15
        self.inter = 20
        self.sample_id = [self.inter*i for i in range(self.num_search-self.aver_num)]
        # self.sample_id = [self.inter*(self.num_search-self.aver_num)-2**(self.num_search-i) for i in range(self.num_search - self.aver_num)]
        # self.sample_id = [0, 39, 69, 89, 99]
        # self.inter = 1
        DATASET_NAME = dataset_name.upper()
        if hasattr(self.cfg.TEST.UPDATE_INTERVALS, DATASET_NAME):
            self.update_intervals = self.cfg.TEST.UPDATE_INTERVALS[DATASET_NAME]
            self.online_sizes = self.cfg.TEST.ONLINE_SIZES[DATASET_NAME]
        else:
            self.update_intervals = self.cfg.DATA.MAX_SAMPLE_INTERVAL
            self.online_sizes = [3]
            # self.online_sizes = [2]
        self.update_interval = self.update_intervals[0]
        self.online_size = self.online_sizes[0]
        # self.online_size = 1
        if hasattr(params, 'online_sizes'):
            self.online_size = params.online_sizes
        if hasattr(params, 'update_interval'):
            self.update_interval = params.update_interval
        if hasattr(params, 'max_score_decay'):
            self.max_score_decay = params.max_score_decay
        else:
            self.max_score_decay = 1.0
        if not hasattr(params, 'vis_attn'):
            self.params.vis_attn = 0
        # self.update_interval = int(self.update_interval/ self.online_size)
        print("Search scale is: ", self.params.search_factor)
        print("Online size is: ", self.online_size)
        print("Update interval is: ", self.update_interval)
        print("Max score decay is ", self.max_score_decay)

    def initialize(self, image, info: dict, seq=None):
    # def initialize(self, image, info: dict):
        # forward the template once
        # info['init_bbox'] = [591, 232, 36, 35]

        z_patch_arr, _, z_amask_arr = sample_target(image, info['init_bbox'], self.params.template_factor,
                                                    output_sz=self.params.template_size)
        template = self.preprocessor.process(z_patch_arr)
        self.template = template
        self.online_template = template

        if self.online_size > 1:
            with torch.no_grad():
                self.network.backbone.set_online(self.template, self.online_template)

        x_patch_arr, _, x_amask_arr = sample_target(image, info['init_bbox'], self.params.search_factor,
                                                    output_sz=self.params.search_size)
        search = self.preprocessor.process(x_patch_arr)
        out_dict, _ = self.network.backbone.forward_test(search)

        self.search_features = [out_dict['search_feature']] * (self.num_search-self.aver_num)*self.inter
        self.template_features = [out_dict['template_feature']] * (self.num_search-self.aver_num)*self.inter
        self.pred_boxes = [box_cxcywh_to_xyxy(out_dict['pred_boxes'].view(-1, 4))] * (self.num_search-self.aver_num)*self.inter

        self.online_state = info['init_bbox']

        self.online_image = image
        self.max_pred_score = -1.0
        self.online_max_template = template
        self.online_forget_id = 0


        # save states
        self.state = info['init_bbox']
        self.frame_id = 0
        if self.save_all_boxes:
            '''save all predicted boxes'''
            all_boxes_save = info['init_bbox'] * self.cfg.MODEL.NUM_OBJECT_QUERIES
            return {"all_boxes": all_boxes_save}

    def reset_search(self, image, state):
        # print(self.frame_id, state)
        # state = [float(i) for i in state ]
        # state = numpy.array(state)
        x_patch_arr, _, x_amask_arr = sample_target(image, state, self.params.search_factor,
                                                    output_sz=self.params.search_size)
        search = self.preprocessor.process(x_patch_arr)
        out_dict, _ = self.network.backbone.forward_test(search)

        self.search_features = [out_dict['search_feature']] * (self.num_search - self.aver_num) * self.inter



    def track(self, image, info: dict = None):
        # time.sleep(0.5)
        # import  matplotlib.pyplot as plt
        # plt.imshow(x_patch_arr)

        H, W, _ = image.shape
        self.frame_id += 1
        x_patch_arr, resize_factor, x_amask_arr = sample_target(image, self.state, self.search_factor,
                                                                output_sz=self.params.search_size)  # (x1, y1, w, h)
        # plt.imshow(x_patch_arr)
        # plt.imsave(str(self.frame_id)+'.jpg', x_patch_arr)
        search = self.preprocessor.process(x_patch_arr)
        with torch.no_grad():
            out, _ = self.network.backbone.forward_test(search)
            pred_boxes = box_cxcywh_to_xyxy(out['pred_boxes'].view(-1, 4))
            pred_box = (out['pred_boxes'].view(-1, 4).mean(dim=0) * self.params.search_size / resize_factor).tolist()
            pred_boxes_precise = out['pred_boxes'].view(-1, 4).clone()
            state = clip_box(self.map_box_back(pred_box, resize_factor), H, W, margin=10)
            # if self.frame_id % 20==0 and torch.sigmoid(out['pred_scores'])>0.8:
        with torch.no_grad():
            # searches = torch.stack(self.search_features + self.aver_num * [out['search_feature']], dim=1) # b, n, c, w, h
            # templates = torch.stack(self.template_features + self.aver_num * [out['template_feature']], dim=0) #n, b, c, w, h
            # pred_boxes = torch.stack(self.pred_boxes + self.aver_num * [pred_boxes], dim=0)
            searches = [self.search_features[i] for i in self.sample_id]
            templates = [self.template_features[i] for i in self.sample_id]
            pred_boxes_ = [self.pred_boxes[i] for i in self.sample_id]

            searches = torch.stack(searches+self.aver_num * [out['search_feature']], dim=1)
            templates = torch.stack(templates + self.aver_num * [out['template_feature']], dim=0)
            pred_boxes = torch.stack(pred_boxes_ + self.aver_num * [pred_boxes], dim=0)
            b, n, c, w, h = searches.shape
            # out_bb = self.network.head_bbox(searches.view(b, n*c, h, w))
            # out_bb = self.network.head_bbox(searches.permute([1, 0, 2, 3, 4]).view(b, -1, w, h))
            out_bb = self.network.head_bbox(searches.permute([1, 0, 2, 3, 4]))
            # out_bb = self.network.head_bbox(searches.permute([1, 0, 2, 3, 4])
            #                                 .view(-1, searches.shape[2], searches.shape[3],  searches.shape[4]))
            # out_bb = out_bb.unsqueeze(1).permute([1, 2, 0])
            # out_score = self.network.head_score(searches.permute([1, 0, 2, 3, 4]), templates, pred_boxes)
        # pred_boxes = torch.mean(out_bb[:, :, -self.aver_num:], dim=-1).view(-1, 4)
        pred_boxes = box_xyxy_to_cxcywh(torch.mean(out_bb[:, :, -self.aver_num:], dim=-1).view(-1, 4))
        # pred_boxes = out_bb[-1, :, :].view(-1, 4)
        # pred_score = out_score[-1].view(1).sigmoid().item()
        pred_boxes_robust = pred_boxes.clone()

        pred_boxes_vec_precise = box_cxcywh_to_xyxy(pred_boxes_precise)
        pred_boxes_vec_robust = box_cxcywh_to_xyxy(pred_boxes_robust)
        iou, _ = box_iou( pred_boxes_vec_precise,  pred_boxes_vec_robust)  #
        # iou = torch.tensor(iou > 0.5).float()
        iou = 0
        # iou = 1
        pred_boxes = iou*pred_boxes_precise+(1-iou)*pred_boxes_robust

        out['pred_scores'] = self.network.backbone.score_branch(out['search_feature'], out['template_feature'],
                                                                box_cxcywh_to_xyxy(pred_boxes).view(-1))
        pred_score = out['pred_scores'].view(1).sigmoid().item()
        # pred_score = out_score[-1].view(1).sigmoid().item()

        pred_box = (pred_boxes.mean(dim=0) * self.params.search_size / resize_factor).tolist()  # (cx, cy, w, h) [0,1]

        self.state = clip_box(self.map_box_back(pred_box, resize_factor), H, W, margin=10)

        self.max_pred_score = self.max_pred_score * self.max_score_decay

        if pred_score > 0.5:
            # self.search_features.append(self.mask(out['search_feature'], pred_boxes))
            self.search_features.append(out['search_feature'])
            self.template_features.append(out['template_feature'])
            self.pred_boxes.append(pred_boxes)
            self.search_features.pop(0)
            self.template_features.pop(0)
            self.pred_boxes.pop(0)
            if self.frame_id % int(self.num_search * self.inter * 7) == 0:
            # if self.frame_id % int(self.num_search * self.inter * 9) == 0:
            # if self.frame_id % int(self.num_search * self.inter * 10) == 0:
                self.reset_search(image, self.state)



        thr = 0.5
        # update template
        if pred_score > thr and pred_score > self.max_pred_score:
            self.max_pred_score = pred_score
            z_patch_arr, _, z_amask_arr = sample_target(image, self.state,
                                                        self.params.template_factor,
                                                        output_sz=self.params.template_size)  # (x1, y1, w, h)
            # z_patch_arr, _, z_amask_arr = sample_target(image, state,
            #                                             self.params.template_factor,
            #                                             output_sz=self.params.template_size)  # (x1, y1, w, h)
            self.online_max_template = self.preprocessor.process(z_patch_arr)

        if self.frame_id % self.update_interval == 0:
            if self.online_size == 1:
                self.online_template = self.online_max_template
                # self.online_search = self.online_max_search
            elif self.online_template.shape[0] < self.online_size:
                self.online_template = torch.cat([self.online_template, self.online_max_template])
                # self.online_search = torch.cat([self.online_search, self.online_max_search])
            else:
                self.online_template[self.online_forget_id:self.online_forget_id + 1] = self.online_max_template
                # self.online_search[self.online_forget_id:self.online_forget_id + 1] = self.online_max_search
                self.online_forget_id = (self.online_forget_id + 1) % self.online_size

            if self.online_size > 1:
                with torch.no_grad():
                    self.network.backbone.set_online(self.template, self.online_template)
            self.max_pred_score = -1
            self.online_max_template = self.template

        # print(self.frame_id, '  ', pred_score, ' ', self.state)
        # self.state = state
        if self.debug:
            x1, y1, w, h = self.state
            image_BGR = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            cv2.rectangle(image_BGR, (int(x1), int(y1)), (int(x1 + w), int(y1 + h)), color=(0, 0, 255), thickness=2)
            save_path = os.path.join(self.save_dir, "%04d.jpg" % self.frame_id)
            cv2.imwrite(save_path, image_BGR)
        if self.save_all_boxes:
            '''save all predictions'''
            all_boxes = self.map_box_back_batch(pred_boxes * self.params.search_size / resize_factor, resize_factor)
            all_boxes_save = all_boxes.view(-1).tolist()  # (4N, )
            return {"target_bbox": self.state,
                    "all_boxes": all_boxes_save}
            # return {"target_bbox": state,
            #         "all_boxes": all_boxes_save}
        else:
            return {"target_bbox": self.state}
            # return {"target_bbox": state}
    def map_box_back(self, pred_box: list, resize_factor: float):
        cx_prev, cy_prev = self.state[0] + 0.5 * self.state[2], self.state[1] + 0.5 * self.state[3]
        cx, cy, w, h = pred_box
        half_side = 0.5 * self.params.search_size / resize_factor
        cx_real = cx + (cx_prev - half_side)
        cy_real = cy + (cy_prev - half_side)
        return [cx_real - 0.5 * w, cy_real - 0.5 * h, w, h]

    def map_box_back_batch(self, pred_box: torch.Tensor, resize_factor: float):
        cx_prev, cy_prev = self.state[0] + 0.5 * self.state[2], self.state[1] + 0.5 * self.state[3]
        cx, cy, w, h = pred_box.unbind(-1)  # (N,4) --> (N,)
        half_side = 0.5 * self.params.search_size / resize_factor
        cx_real = cx + (cx_prev - half_side)
        cy_real = cy + (cy_prev - half_side)
        return torch.stack([cx_real - 0.5 * w, cy_real - 0.5 * h, w, h], dim=-1)
    def mask(self, feature, boxes):
        boxes = box_cxcywh_to_xyxy(boxes)
        B, C, H, W = feature.shape
        h_arange = (torch.arange(0, H) / H).cuda()
        w_arange = (torch.arange(0, W) / W).cuda()
        h_index, w_index = torch.meshgrid(h_arange, w_arange, indexing='ij')
        h_index, w_index = h_index.repeat(B, 1, 1).cuda(), w_index.repeat(B, 1, 1).cuda()
        object_index_h = (h_index.view(B, -1) > (boxes[:, 1]).view(B, -1)).view(B, H, W)
        object_index_h *= (h_index.view(B, -1) < (boxes[:, 3]).view(B, -1)).view(B, H, W)
        object_index_w = (w_index.view(B, -1) > (boxes[:, 0]).view(B, -1)).view(B, H, W)
        object_index_w *= (w_index.view(B, -1) < (boxes[:, 2]).view(B, -1)).view(B, H, W)
        object_index = object_index_w * object_index_h
        object_index = object_index.view(B, 1, H, W).repeat(1,C,1,1)
        object_index = 1 * object_index.float() + 0.5 * object_index.float()
        feature = feature * object_index
        return feature
def get_tracker_class():
    return VideoTrack
