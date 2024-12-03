import random
import torch.utils.data
from lib.utils import TensorDict
from lib.utils.random import random_choice
import numpy as np
from lib.utils.image import *

def no_processing(data):
	return data

class VideoDataset(torch.utils.data.Dataset):
	def __init__(self, datasets, p_datasets, samples_per_epoch,
	             num_search_frames, num_template_frames, processing=no_processing,
	             train_cls=False, pos_prob=0.5, max_inter=20):
		self.datasets = datasets
		self.train_cls = train_cls
		self.pos_prob = pos_prob

		if p_datasets is None:
			p_datasets = [len(d) for d in self.datasets]
		# Normalize
		p_total = sum(p_datasets)
		self.p_datasets = [x / p_total for x in p_datasets]

		self.samples_per_epoch = samples_per_epoch
		self.num_search_frames = num_search_frames
		self.num_template_frames = num_template_frames
		self.processing = processing

		self.max_inter = max_inter

	def __len__(self):
		return self.samples_per_epoch
	def _sample_visible_ids(self, visible, num_ids=1, min_id=None, max_id=None,
	                        allow_invisible=False, force_invisible=False):
		""" Samples num_ids frames between min_id and max_id for which target is visible

		args:
			visible - 1d Tensor indicating whether target is visible for each frame
			num_ids - number of frames to be samples
			min_id - Minimum allowed frame number
			max_id - Maximum allowed frame number

		returns:
			list - List of sampled frame numbers. None if not sufficient visible frames could be found.
		"""

		if num_ids == 0:
			return []
		if min_id is None or min_id < 0:
			min_id = 0
		if max_id is None or max_id > len(visible):
			max_id = len(visible)
		# get valid ids
		if force_invisible:
			valid_ids = [i for i in range(min_id, max_id) if not visible[i]]
		else:
			if allow_invisible:
				valid_ids = [i for i in range(min_id, max_id)]
			else:
				valid_ids = [i for i in range(min_id, max_id) if visible[i]]

		# No visible ids
		if len(valid_ids) == 0:
			return None

		return random_choice(valid_ids, k=num_ids, is_repeat=False)

	def _sample_sequence_ids(self, visible, num_ids=1, min_id=None, max_id=None):
		""" Samples num_ids frames between min_id and max_id for which target is visible

		args:
			visible - 1d Tensor indicating whether target is visible for each frame
			num_ids - number of frames to be samples
			min_id - Minimum allowed frame number
			max_id - Maximum allowed frame number

		returns:
			list - List of sampled frame numbers. None if not sufficient visible frames could be found.
		"""
		# if self.sample_inter == None:
		# max_inter = max(num_ids, min(int(len(visible)/num_ids/5), self.max_inter))
		max_inter = max(1, min(int(len(visible) / num_ids / 3), self.max_inter))
		sample_inter = random.randint(1, max_inter)
			# sample_inter = 10
		if num_ids == 0:
			return []
		if min_id is None or min_id < 0:
			min_id = 0
		if max_id is None or max_id > len(visible):
			max_id = len(visible)-num_ids * sample_inter
		# get valid ids
		if max_id < 0:
			return []
		start_id = random.randint(min_id, max_id)
		sample_ids = [i for i in range(start_id, start_id+num_ids*sample_inter, sample_inter) ]


		# No visible ids
		if len(sample_ids) == 0:
			return None

		return sample_ids

	def __getitem__(self, index):
		return self.getitem()

	def getitem(self):
		"""
		returns:
			TensorDict - dict containing all the data blocks
		"""

		valid = False
		while not valid:
			# Select a dataset
			dataset = random.choices(self.datasets, self.p_datasets)[0]
			is_video_dataset = dataset.is_video_sequence()

			# sample a sequence
			seq_id, visible, seq_info_dict = self.sample_seq_from_dataset(dataset, is_video_dataset)
			# seq_len = self.num_search_frames + self.num_template_frames + self.num_sequence_frames

			if is_video_dataset:
				# sample a visible frame
				template_frame_ids = self._sample_visible_ids(visible, num_ids=self.num_template_frames,
															  allow_invisible=False)
				# sequence_frame_ids = self._sample_visible_ids(visible, num_ids=self.num_sequence_frames,
				# 											  allow_invisible=False)
				search_frame_ids = self._sample_sequence_ids(visible, num_ids=self.num_search_frames+1)
				# change the frame_id sort randomly
				if random.random() < 0.5:
					template_frame_ids = sorted(template_frame_ids, reverse=False)
					# sequence_frame_ids = sorted(sequence_frame_ids, reverse=False)
					search_frame_ids = sorted(search_frame_ids, reverse=False)
				else:
					template_frame_ids = sorted(template_frame_ids, reverse=True)
					# sequence_frame_ids = sorted(sequence_frame_ids, reverse=True)
					search_frame_ids = sorted(search_frame_ids, reverse=True)
			else:
				template_frame_ids = [1] * template_frame_ids
				# sequence_frame_ids = [1] * sequence_frame_ids
				search_frame_ids = [1] * search_frame_ids

			# if template_frame_ids is None or sequence_frame_ids or search_frame_ids:
			# 	continue
			if template_frame_ids is None or search_frame_ids is None:
				continue
			try:
				frames_template, anno_template, meta_obj = dataset.get_frames(seq_id, template_frame_ids, seq_info_dict)
				H, W, _ = frames_template[0].shape
				masks_template = anno_template['mask'] if 'mask' in anno_template else [torch.zeros(
					(H, W))] * self.num_template_frames

				# frames_sequence, anno_sequence, meta_obj = dataset.get_frames(seq_id, sequence_frame_ids, seq_info_dict)
				# H, W, _ = frames_sequence[0].shape
				# masks_sequence = anno_sequence['mask'] if 'mask' in anno_sequence else [torch.zeros((H, W))] * self.num_sequence_frames

				frames_search, anno_search, meta_obj = dataset.get_frames(seq_id, search_frame_ids, seq_info_dict)
				H, W, _ = frames_search[0].shape
				masks_search = anno_search['mask'] if 'mask' in anno_search else [torch.zeros(
					(H, W))] * (self.num_search_frames + 1)

				data = TensorDict({'template_images': frames_template,
								   'template_bboxes': anno_template['bbox'],
								   'template_masks': masks_template,
								   'search_images': frames_search[1:],
								   'search_masks': masks_search[1:],
								   'search_gt_bboxes': anno_search['bbox'][1:],
								   'search_bboxes': anno_search['bbox'][:-1],
								   # 'search_gt_masks': masks_search[:-1],
								   'dataset': dataset.get_name(),
								   'test_class': meta_obj.get('object_class_name')})
				# make data augmentation
				if self.processing != None:
					data = self.processing(data)
				# check whether data is valid
				valid = data['valid']
			# valid = True
			except:
				print("dataset load fail: " + dataset.get_name() + ", seq_id: " + str(seq_id))
				valid = False

		return data



	def get_one_search(self):
		# Select a dataset
		dataset = random.choices(self.datasets, self.p_datasets)[0]

		is_video_dataset = dataset.is_video_sequence()
		# sample a sequence
		seq_id, visible, seq_info_dict = self.sample_seq_from_dataset(dataset, is_video_dataset)
		# sample a frame
		if is_video_dataset:
			search_frame_ids = self._sample_visible_ids(visible, num_ids=1, allow_invisible=True)
		else:
			search_frame_ids = [1]
		# get the image, bounding box and other info
		search_frames, search_anno, meta_obj_test = dataset.get_frames(seq_id, search_frame_ids, seq_info_dict)

		return search_frames, search_anno, meta_obj_test

	def sample_seq_from_dataset(self, dataset, is_video_dataset):

		# Sample a sequence with enough visible frames
		enough_visible_frames = False
		while not enough_visible_frames:
			# Sample a sequence
			seq_id = random.randint(0, dataset.get_num_sequences() - 1)
			# seq_id = 1

			# Sample frames
			seq_info_dict = dataset.get_sequence_info(seq_id)
			visible = seq_info_dict['visible']

			enough_visible_frames = visible.type(torch.int64).sum().item() > \
			                        2 * (self.num_search_frames + self.num_template_frames ) \
			                        and len(visible) >= 30

			enough_visible_frames = enough_visible_frames or not is_video_dataset
		return seq_id, visible, seq_info_dict

	def get_center_box(self, H, W, ratio=1 / 8):
		cx, cy, w, h = W / 2, H / 2, W * ratio, H * ratio
		return torch.tensor([int(cx - w / 2), int(cy - h / 2), int(w), int(h)])
