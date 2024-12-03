import torch
import torchvision.transforms as transforms
from lib.utils import TensorDict
import lib.train.data.util.processing_utils as prutils
import torch.nn.functional as F
from lib.utils.image import *
from lib.utils.box_ops import box_xywh_to_xyxy, box_iou


def stack_tensors(x):
	if isinstance(x, (list, tuple)) and len(x) > 0 and isinstance(x[0], torch.Tensor):
		return torch.stack(x)
	return x


class VideoProcessing():
	def __init__(self, settings=None,
	             template_transform=None, search_transform=None,
	             joint_transform=None, transform=transforms.ToTensor()):



		self.transform = {'template': transform if template_transform is None else template_transform,
		                  'search': transform if search_transform is None else search_transform,
		                  # 'sequence': transform if search_transform is None else search_transform,
		                  'joint': joint_transform}

	def __call__(self, data: TensorDict):
		if self.transform['joint'] is not None:
			if len(data["template_images"]) > 0:
				data['template_images'], data['template_bboxes'], data['template_masks'] = self.transform['joint'](
					image=data['template_images'], bbox=data['template_bboxes'], mask=data['template_masks'])
			# if len(data["sequence_images"]) > 0:
			# 	data['sequence_images'], data['sequence_bboxes'], data['sequence_masks'] = self.transform['joint'](
			# 		image=data['sequence_images'], bbox=data['sequence_bboxes'], mask=data['sequence_masks'])
			if len(data["search_images"]) > 0:
				data['search_images'], data['search_bboxes'], data['search_masks'] = self.transform['joint'](
					image=data['search_images'], bbox=data['search_bboxes'], mask=data['search_masks'], new_roll=False)
		data['valid'] = True
		return data
