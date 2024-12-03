import random
import torch.utils.data
from lib.utils import TensorDict
from lib.utils.random import random_choice
import numpy as np
from lib.utils.image import *
from lib.utils.box_ops import *

def no_processing(data):
	return data

class SequenceDataset(torch.utils.data.Dataset):
	def __init__(self, path, len, processing=None, sequence_len=None):
		self.feature_path = path+'/search_feature'
		self.processing = processing
		self.len = len
		self.video_number = int((os.listdir(self.feature_path)).__len__() / 3)
		self.sequence_len = sequence_len
	def __len__(self):
		return self.len



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
			data_id = random.randint(0, self.len)
			search_feature_path = self.feature_path + '/search_feature_' + str(data_id) + '.pt'
			template_feature_path = self.feature_path + '/template_feature_' + str(data_id) + '.pt'
			ground_truth_path = self.feature_path + '/ground_truth_' + str(data_id) + '.pt'


			try:
				search_feature = torch.load(search_feature_path, map_location='cpu')
				template_feature = torch.load(template_feature_path, map_location='cpu')
				ground_truth = torch.load(ground_truth_path, map_location='cpu')

				if random.random() < 0.5:
					search_feature = torch.flip(search_feature, dims=[0])
					template_feature = torch.flip(template_feature, dims=[0])
					ground_truth = torch.flip(ground_truth, dims=[0])

				N, C, _, __ = search_feature.shape

				sequence_len = self.sequence_len
				if sequence_len == None:
					sequence_len = N


				offset = random.randint(0, N-sequence_len)
				choice = list(range(offset, offset + sequence_len))
				# for i in range(choice.__len__()):
				# 	choice[i] = choice[i] + offset

				data = TensorDict({'search_feature': search_feature[choice],
				                   'template_feature': template_feature[choice],
				                   'ground_truth': ground_truth[choice],
				                   })
				# data = TensorDict({'search_feature': search_feature,
				#                    'template_feature': template_feature,
				#                    'ground_truth': ground_truth,
				#                    })
				# make data augmentation
				if self.processing is not None:
					data = self.processing(data)
				# check whether data is valid
				valid = True
			except:
				print("sequence load fail: " + str(data_id) )
				valid = False

		return data

