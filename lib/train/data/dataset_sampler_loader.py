from torch.utils.data.distributed import DistributedSampler
# datasets related
from lib.train.data.dataset import Lasot, Got10k, MSCOCOSeq, ImagenetVID, TrackingNet
from lib.train.data.dataset import Lasot_lmdb, Got10k_lmdb, MSCOCOSeq_lmdb, ImagenetVID_lmdb, TrackingNet_lmdb
from lib.train.data.util import dataset, opencv_loader, processing_sample, LTRLoader, pil_loader, dataset_sample
import lib.train.data.util.transforms as tfm
import multiprocessing

def names2datasets(name_list: list, settings, image_loader):
	assert isinstance(name_list, list)
	datasets = []
	for name in name_list:
		assert name in ["LASOT", "GOT10K_vottrain", "GOT10K_votval", "GOT10K_train_full", "COCO17", "VID",
		                "TRACKINGNET"]
		if name == "LASOT":
			if settings.use_lmdb:
				print("Building lasot dataset from lmdb")
				datasets.append(Lasot_lmdb(settings.env.lasot_lmdb_dir, split='train', image_loader=image_loader))
			else:
				datasets.append(Lasot(settings.env.lasot_dir, split='train', image_loader=image_loader))
		if name == "GOT10K_vottrain":
			if settings.use_lmdb:
				print("Building got10k from lmdb")
				datasets.append(Got10k_lmdb(settings.env.got10k_lmdb_dir, split='vottrain', image_loader=image_loader))
			else:
				datasets.append(Got10k(settings.env.got10k_dir, split='vottrain', image_loader=image_loader))
		if name == "GOT10K_train_full":
			if settings.use_lmdb:
				print("Building got10k_train_full from lmdb")
				datasets.append(
					Got10k_lmdb(settings.env.got10k_lmdb_dir, split='train_full', image_loader=image_loader))
			else:
				datasets.append(Got10k(settings.env.got10k_dir, split='train_full', image_loader=image_loader))
		if name == "GOT10K_votval":
			if settings.use_lmdb:
				print("Building got10k from lmdb")
				datasets.append(Got10k_lmdb(settings.env.got10k_lmdb_dir, split='votval', image_loader=image_loader))
			else:
				datasets.append(Got10k(settings.env.got10k_dir, split='votval', image_loader=image_loader))
		if name == "COCO17":
			if settings.use_lmdb:
				print("Building COCO2017 from lmdb")
				datasets.append(MSCOCOSeq_lmdb(settings.env.coco_lmdb_dir, version="2017", image_loader=image_loader))
			else:
				datasets.append(MSCOCOSeq(settings.env.coco_dir, version="2017", image_loader=image_loader))
		if name == "VID":
			if settings.use_lmdb:
				print("Building VID from lmdb")
				datasets.append(ImagenetVID_lmdb(settings.env.imagenet_lmdb_dir, image_loader=image_loader))
			else:
				datasets.append(ImagenetVID(settings.env.imagenet_dir, image_loader=image_loader))
		if name == "TRACKINGNET":
			if settings.use_lmdb:
				print("Building TrackingNet from lmdb")
				datasets.append(TrackingNet_lmdb(settings.env.trackingnet_lmdb_dir, image_loader=image_loader))
			else:
				datasets.append(TrackingNet(settings.env.trackingnet_dir, image_loader=image_loader))
	return datasets


def build_seq_dataloaders(cfg, settings):
	# Data transform
	# Data transform

	joint_transform = tfm.Transform(tfm.ToTensor(),
	                                 tfm.Normalize(mean=cfg.DATA.MEAN, std=cfg.DATA.STD))
	template_transform = tfm.Transform(tfm.ToTensor(),
	                                 tfm.Normalize(mean=cfg.DATA.MEAN, std=cfg.DATA.STD))
	search_transform = tfm.Transform(tfm.ToTensor(),
	                                 tfm.Normalize(mean=cfg.DATA.MEAN, std=cfg.DATA.STD))

	data_processing = processing_sample.VideoProcessing(
	                                                      transform=template_transform,
	                                                      search_transform=search_transform,
	                                                      joint_transform=joint_transform,
	                                                      settings=settings)
	train_cls = getattr(cfg.TRAIN, "TRAIN_CLS", True)

	# Train sampler and loader
	dataset_train = dataset_sample.VideoDataset(
		datasets=names2datasets(cfg.DATA.TRAIN.DATASETS_NAME, settings, pil_loader),
		p_datasets=cfg.DATA.TRAIN.DATASETS_RATIO,
		# samples_per_epoch=cfg.DATA.TRAIN.SAMPLE_PER_EPOCH,
		samples_per_epoch= 10,
		num_search_frames=settings.num_search,
		num_template_frames=settings.num_template,
		processing=data_processing,
		# processing=None,
		train_cls=train_cls)

	train_sampler = DistributedSampler(dataset_train) if settings.local_rank != -1 else None
	# train_sampler = None
	shuffle = False if settings.local_rank != -1 else True
	loader = LTRLoader('train', dataset_train, training=True, batch_size=cfg.TRAIN.BATCH_SIZE, shuffle=shuffle,
	                         num_workers=10, drop_last=True, stack_dim=1, sampler=train_sampler)


	return loader

	# return dataset_train
