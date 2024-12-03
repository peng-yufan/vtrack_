import numpy
import torch
import cv2 as cv
import os
from lib.train.data.util.processing_utils import *
import multiprocessing
import itertools
from  itertools import zip_longest
from itertools import product
class VideoSampler():
    def __init__(self, loader, net, tracker, save_path, config,
                 retain_feature=True, retain_bbox=True):
        self.loader = loader
        self.retain_feature = retain_feature
        self.retain_bbox = retain_bbox
        self.net = net
        self.tracker = tracker
        self.save_path = save_path
        self.config = config

    def sample_one_video(self, data, run_id):

        try:
            num_gpu = torch.cuda.device_count()
            worker_name = multiprocessing.current_process().name
            worker_id = int(worker_name[worker_name.find('-') + 1:]) - 1
            gpu_id = worker_id % num_gpu
            torch.cuda.set_device(gpu_id)
        except:
            torch.set_num_threads(1)
            pass

        path = self.save_path+'/'
        path_s = path + '/search_feature'
        path_b = path + '/target_bbox'

        if not os.path.exists(path_s):
            os.makedirs(path_s)
        if not os.path.exists(path_b):
            os.makedirs(path_b)
        if os.path.exists(self.save_path+'/search_feature/ground_truth_' + str(run_id) + '.pt'):
            print('skip'+str(run_id))
            return
        output_boxes = []
        output_feature_search = []
        output_feature_template = []
        output_gt = []
        frame = numpy.array(data['template_images'][0].squeeze(1).permute([1, 2, 0]))
        init_state = [int(s) for s in data['template_bboxes'][0]]
        init_info = {'init_bbox': init_state}
        params = self.tracker.get_parameters()
        params.search_factor *= torch.exp(torch.randn(1) * 0.5).item()
        tracker = self.tracker.create_tracker(params)
        tracker.initialize(frame, init_info)
        state = [int(s) for s in data['search_bboxes'][0]]
        tracker.state = state
        for i in range(data['search_images'].__len__()):
            frame = numpy.array(data['search_images'][i].squeeze(1).permute([1, 2, 0]))
            if frame is None:
                continue
            # Draw box
            out = tracker.track(frame)
            box_crop = transform_image_to_crop(data['search_bboxes'][i].view(-1), torch.tensor(state),
                                               out['resize_factor'],
                                               torch.tensor([tracker.params.search_size, tracker.params.search_size]),
                                               normalize=True)
            output_gt.append(box_crop.view(-1, 4))
            state = [int(s) for s in out['target_bbox']]
            output_boxes.append(state)
            output_feature_search.append(out['search_feature'])
            output_feature_template.append(out['template_feature'])
        output_feature_search = torch.cat(output_feature_search, dim=0)
        output_feature_template = torch.cat(output_feature_template, dim=0)
        output_gt = torch.cat(output_gt, dim=0)
        output_boxes = torch.tensor(output_boxes)
        torch.save(output_gt, path + 'search_feature/ground_truth_' + str(run_id) + '.pt')
        assert self.retain_feature or self.retain_bbox
        if self.retain_feature:
            torch.save(output_feature_search, path + 'search_feature/search_feature_' + str(run_id) + '.pt')
            torch.save(output_feature_template, path + 'search_feature/template_feature_' + str(run_id) + '.pt')
        if self.retain_bbox:
            torch.save(output_boxes, path+'target_bbox/target_bbox_'+str(run_id)+'.pt')
    def cycle_sample(self):
        try:
            i = int(len(os.listdir(self.save_path + '/search_feature')) / 3)
        except:
            i = 0
        for _ in range(int(self.config.DATA['TRAIN']['SAMPLE_PER_EPOCH'] / self.loader.__len__())):
            for data in self.loader:
                try:
                    self.sample_one_video(data, i)
                    i = i + 1
                except:
                    print('wrong in sequence:'+str(i))

        # multiprocessing.set_start_method('spawn', force=True)
        # with multiprocessing.Pool(processes=10) as pool:
        #     # param_list = [zip(self.loader, range(i, i+self.loader.__len__()))]
        #     for _ in range(int(self.config.DATA['TRAIN']['SAMPLE_PER_EPOCH']/self.loader.__len__())):
        #         if i > self.config.DATA['TRAIN']['SAMPLE_PER_EPOCH']:
        #             break
        #         try:
        #             param_list = itertools.zip_longest(self.loader, range(i, i+self.loader.__len__()))
        #             pool.starmap(self.sample_one_video, param_list)
        #             i = i + self.loader.__len__()
        #         except:
        #             print('something wrong in sequence ' + str(i))

