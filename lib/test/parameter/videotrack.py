from lib.test.utils import TrackerParams
import os
from lib.test.evaluation.environment import env_settings
import importlib

def parameters(group_name:str, name: str, script_name: str):
	params = TrackerParams()
	prj_dir = env_settings().prj_dir
	save_dir = env_settings().save_dir
	# update default config from yaml file
	yaml_file = os.path.join(prj_dir, 'experiments', '%s' % name, '%s.yaml' % script_name)

	config_module = importlib.import_module('lib.config.{}.config'.format(name))
	config_module.update_config_from_file(yaml_file)
	cfg = config_module.cfg
	params.cfg = cfg
	print("test config: ", cfg)

	params.template_factor = cfg.DATA.TEMPLATE.FACTOR
	params.template_size = cfg.DATA.TEMPLATE.SIZE
	params.search_factor = cfg.DATA.SEARCH.FACTOR
	params.search_size = cfg.DATA.SEARCH.SIZE

	if name == 'videotrack':
		# params.checkpoint = os.path.join(save_dir, "checkpoints/train/videotrack/baseline/VideoTrack_ep0022.pth.tar")
		if 'baseline-test' in script_name:
			params.checkpoint = os.path.join(save_dir, "checkpoints/baseline_test/VideoTrack_ep0020.pth.tar")
			# if '20' in script_name:
			# 	params.checkpoint = os.path.join(save_dir, "checkpoints/baseline_test/VideoTrack_ep0020.pth.tar")
			# elif '25' in script_name:
			# 	params.checkpoint = os.path.join(save_dir, "checkpoints/baseline_test/VideoTrack_ep0025.pth.tar")
			# elif '30' in script_name:
			# 	params.checkpoint = os.path.join(save_dir, "checkpoints/baseline_test/VideoTrack_ep0030.pth.tar")
		elif 'baseline_large' in script_name:
			if 'ep25' in script_name:
				params.checkpoint = os.path.join(save_dir,"checkpoints/train/videotrack/baseline_large/VideoTrack_ep0025.pth.tar")
			elif 'ep30' in script_name:
				params.checkpoint = os.path.join(save_dir,"checkpoints/train/videotrack/baseline_large/VideoTrack_ep0030.pth.tar")
			elif 'ep15' in script_name:
				params.checkpoint = os.path.join(save_dir,"checkpoints/train/videotrack/baseline_large/VideoTrack_ep0015.pth.tar")
			else:
				params.checkpoint = os.path.join(save_dir, "checkpoints/train/videotrack/baseline_large/VideoTrack_ep0020.pth.tar")
		elif 'baseline' in script_name:
			params.checkpoint = os.path.join(save_dir, "checkpoints/train/videotrack/baseline/VideoTrack_ep0020.pth.tar")
			# params.checkpoint = os.path.join(save_dir,
			# 								 "checkpoints/train/videotrack/baseline/VideoTrack_ep0145.pth.tar")

	# whether to save boxes from all queries
	params.save_all_boxes = False
	return params
