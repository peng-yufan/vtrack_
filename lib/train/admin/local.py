class EnvironmentSettings:
    def __init__(self):
        self.workspace_dir = '/home/yufan/videotrack'    # Base directory for saving network checkpoints.
        self.tensorboard_dir = '/home/yufan/videotrack/tensorboard'    # Directory for tensorboard files.
        self.pretrained_networks = '/home/yufan/videotrack/pretrained_networks'
        self.lasot_dir = '/data1/lasot'
        self.got10k_dir = '/data1/got10k/train'
        self.lasot_lmdb_dir = '/data1/lasot_lmdb'
        self.got10k_lmdb_dir = '/data1/got10k_lmdb'
        self.trackingnet_dir = '/data1/trackingnet'
        self.trackingnet_lmdb_dir = '/data1/trackingnet_lmdb'
        self.coco_dir = '/data1/coco'
        self.coco_lmdb_dir = '/data1/coco_lmdb'
        self.lvis_dir = ''
        self.sbd_dir = ''
        self.imagenet_dir = '/data1/vid'
        self.imagenet_lmdb_dir = '/data1/vid_lmdb'
        self.imagenetdet_dir = ''
        self.ecssd_dir = ''
        self.hkuis_dir = ''
        self.msra10k_dir = ''
        self.davis_dir = ''
        self.youtubevos_dir = ''
