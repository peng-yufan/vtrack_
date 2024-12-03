import torch
import torchvision.transforms as transforms
from lib.utils import TensorDict
import lib.train.data.util.processing_utils as prutils
import torch.nn.functional as F
from lib.utils.image import *
from lib.utils.box_ops import box_xywh_to_xyxy, box_iou
from  lib.train.data.util.processing_utils import *


def stack_tensors(x):
    if isinstance(x, (list, tuple)) and len(x) > 0 and isinstance(x[0], torch.Tensor):
        return torch.stack(x)
    return x


class SequenceProcessing():
    def __init__(self, search_area_factor, output_sz, center_jitter_factor, scale_jitter_factor, settings=None,
                 template_transform=None, search_transform=None,
                 joint_transform=None, transform=transforms.ToTensor()):

        self.search_area_factor = search_area_factor
        self.output_sz = output_sz
        self.center_jitter_factor = center_jitter_factor
        self.scale_jitter_factor = scale_jitter_factor
        self.settings = settings

        self.transform = {'template': transform if template_transform is None else template_transform,
                          'search': transform if search_transform is None else search_transform,
                          # 'sequence': transform if search_transform is None else search_transform,
                          'joint': joint_transform}

    # def _get_jittered_box(self, box, mode):
    #
    #     jittered_size = box[2:4] * torch.exp(torch.randn(2) * self.scale_jitter_factor[mode])
    #     max_offset = (jittered_size.prod().sqrt() * torch.tensor(self.center_jitter_factor[mode]).float())
    #     jittered_center = box[0:2] + 0.5 * box[2:4] + max_offset * (torch.rand(2) - 0.5)
    #
    #     return torch.cat((jittered_center - 0.5 * jittered_size, jittered_size), dim=0)
    def _get_jittered_box_sample(self, box, jittered_size, s):
        jittered_center = box[0:2] + 0.5 * box[2:4]
        jittered_size = jittered_size * torch.exp(torch.randn(2) * self.scale_jitter_factor[s]/2)
        jittered_size = box[2:4] * jittered_size
        max_offset = (jittered_size.prod().sqrt() * torch.tensor(self.center_jitter_factor[s]).float())
        offset = max_offset * (torch.rand(2) - 0.5)
        jittered_center = jittered_center + offset

        return torch.cat((jittered_center - 0.5 * jittered_size, jittered_size), dim=0)

    def __call__(self, data: TensorDict):
        if self.transform['joint'] is not None:
            if len(data["template_images"]) > 0:
                data['template_images'], data['template_bboxes'], data['template_masks'] = self.transform['joint'](
                    image=data['template_images'], bbox=data['template_bboxes'], mask=data['template_masks'])
            if len(data["search_images"]) > 0:
                data['search_images'], data['search_bboxes'], data['search_masks'], data['search_gt_bboxes'] = \
                self.transform['joint'](
                    image=data['search_images'], bbox=data['search_bboxes'], mask=data['search_masks'],
                    gt_bbox=data['search_gt_bboxes'], new_roll=False)

        sstr = []
        if len(data["template_images"]) > 0:
            sstr.append('template')
        if len(data["search_images"]) > 0:
            sstr.append('search')

        for s in sstr:
            # Add a uniform noise to the center pos
            jittered_size = torch.exp(torch.randn(2) * self.scale_jitter_factor[s])
            # jittered_size = data[s + '_bboxes'][0][2:4] * jittered_factor
            # max_offset = (jittered_size.prod().sqrt() * torch.tensor(self.center_jitter_factor[s]).float())
            # offset = max_offset * (torch.rand(2) - 0.5)
            jittered_anno = [self._get_jittered_box_sample(a, jittered_size, s)
                             for a in data[s + '_bboxes']]

            # Check whether data is valid. Avoid too small bounding boxes
            w, h = torch.stack(jittered_anno, dim=0)[:, 2], torch.stack(jittered_anno, dim=0)[:, 3]

            crop_sz = torch.ceil(torch.sqrt(w * h) * 2.0)
            if (crop_sz < 1).any():
                data['valid'] = False
                # print("Too small box is found. Replace it with new data.")
                return data

            # Crop image region centered at jittered_anno box and get the attention mask
            crops_resize_factors = [sample_target(f, a, self.search_area_factor[s], self.output_sz[s], m)
                                    for f, a, m in zip(data[s+'_images'], jittered_anno, data[s+'_masks'])]
            crops, resize_factors, att_mask, masks_crop = zip(*crops_resize_factors)
            att_box_mask = None
            output_size = torch.tensor([self.output_sz[s], self.output_sz[s]])
            box_crop = [transform_image_to_crop(a_gt, a_ex, rf, output_size, normalize=True)
                        for a_gt, a_ex, rf in zip(data[s + '_bboxes'], jittered_anno, resize_factors)]
            if s == 'search':
                gt_box_crop = [transform_image_to_crop(a_gt, a_ex, rf, output_size, normalize=True)
                            for a_gt, a_ex, rf in zip(data[s + '_gt_bboxes'], jittered_anno, resize_factors)]
            else:
                gt_box_crop = None

            # crops, boxes, att_mask, att_box_mask, mask_crops = prutils.jittered_center_crop(data[s + '_images'],
            #                                                                                 jittered_anno,
            #                                                                                 data[s + '_bboxes'],
            #                                                                                 self.search_area_factor[s],
            #                                                                                 self.output_sz[s],
            #                                                                                 masks=data[s + '_masks'])
            if s != 'search':
                data[s + '_images'], data[s + '_bboxes'], data[s + '_att'], data[s + '_masks'] = \
                    self.transform[s](image=crops,
                                      bbox=box_crop,
                                      att=att_mask,
                                      mask=masks_crop,
                                      joint=False)
            else:
                data[s + '_images'], data[s + '_bboxes'], data[s + '_att'], data[s + '_masks'], data[s + '_gt_bboxes'] = \
                    self.transform[s](image=crops,
                                      bbox=box_crop,
                                      att=att_mask,
                                      mask=masks_crop,
                                      gt_bbox=gt_box_crop,
                                      joint=False,
                                      )

            # Check whether elements in data[s + '_att'] is all 1
            # Note that type of data['att'] is tuple, type of ele is torch.tensor
            for ele in data[s + '_att']:
                if (ele == 1).all():
                    data['valid'] = False
                    # print("Values of original attention mask are all one. Replace it with new data.")
                    return data
            # # more strict conditions: require the donwsampled masks not to be all 1
            for ele in data[s + '_att']:
                feat_size = self.output_sz[s] // 16  # 16 is the backbone stride
                mask_down = F.interpolate(ele[None, None].float(), size=feat_size).to(torch.bool)[0]
                if (mask_down == 1).all():
                    data['valid'] = False
                    # print("Values of down-sampled attention mask are all one. "
                    #       "Replace it with new data.")
                    return data

        data['valid'] = True
        data = data.apply(stack_tensors)
        return data
