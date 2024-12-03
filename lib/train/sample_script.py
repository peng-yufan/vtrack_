import os
import torch
import importlib
import multiprocessing
from torch.nn.functional import l1_loss
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.nn import BCEWithLogitsLoss

from lib.train.trainers import LTRTrainer
from lib.models.mixformer_vit.mixformer_online import build_mixformer_vit_online_score
from lib.train.data.dataset_sampler_loader import build_seq_dataloaders
from lib.train.utils.optim_factory import get_optimizer_tt,get_optimizer_mynet,get_optimizer
from lib.train.utils.schedule_factory import get_schedule
from lib.train.utils.set_params import update_settings
from lib.train.sample_sequence import VideoSampler
from lib.train.actors import *
from lib.test.tracker import Tracker

from lib.utils.box_ops import giou_loss, ciou_loss

def run(settings):

	settings.description = 'Sample sequence for training head!'

	# update the default configs with config file
	if not os.path.exists(settings.cfg_file):
		raise ValueError("%s doesn't exist." % settings.cfg_file)
	config_module = importlib.import_module("lib.config.%s.config" % settings.script_name)
	cfg = config_module.cfg
	config_module.update_config_from_file(settings.cfg_file)
	if settings.local_rank in [-1, 0]:
		print("New configuration is shown below.")
		for key in cfg.keys():
			print("%s configuration:" % key, cfg[key])
			print('\n')

	# update settings based on cfg
	update_settings(settings, cfg)

	# Build dataloaders
	multiprocessing.set_start_method('spawn', force=True)
	loader = build_seq_dataloaders(cfg, settings)
	# Create network

	if settings.script_name == "mixformer_vit_online":
		net = build_mixformer_vit_online_score(cfg, settings)
	else:
		raise ValueError("illegal script name")

	net.cuda()
	# model_path = os.getcwd()+'/lib/models/mixformer_vit/model/mixformer_vit_base_online.pth.tar'
	# ckpt = torch.load(model_path, map_location='cpu')
	# missing_keys, unexpected_keys = net.load_state_dict(ckpt['net'], strict=False)
	# print("missing keys:", missing_keys)
	# print("unexpected keys:", unexpected_keys)
	# print("Loading pretrained mixformer weights done.")
	net.eval()

	# wrap networks to distributed one
	if settings.local_rank != -1:
		net = DDP(net, device_ids=[settings.local_rank], find_unused_parameters=True)
		settings.device = torch.device("cuda:%d" % settings.local_rank)
	else:
		settings.device = torch.device("cuda:0")
	tracker =Tracker(settings.script_name, 'baseline', 'sample', run_id=1, report_name=None)
	sampler = VideoSampler(loader, net, tracker, save_path=settings.save_dir, config=cfg)
	# if not os.path.exists(settings.save_dir+'/sequence.txt'):
	# 	file = open(settings.save_dir+'/sequence.txt', 'w')
	# 	file.close()

	sampler.cycle_sample()

