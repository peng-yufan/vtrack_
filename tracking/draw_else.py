import matplotlib
import numpy as np
import torch
import matplotlib.pyplot as plt
import os

dataset_name = '/home/yufan/PycharmProjects/CTTrack-main/output/test/tracking_results/mynet_mixformer/baseline/got10k_val_else/GOT-10k_Val_000'
save_dir ='/home/yufan/PycharmProjects/CTTrack-main/output/test/tracking_results/mynet_mixformer/baseline/got10k_val_else_plot' # This is your Project Root
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
for i in range(180):
    try:
        if i < 10-1:
         path = dataset_name+'00'+str(i+1)
        elif i <100-1:
            path = dataset_name + '0' + str(i+1)
        elif i < 1000-1:
            path = dataset_name + str(i+1)
        giou_file = path+'_giou.txt'
        score_nm_file = path+'_score_nm.txt'
        score_m_file = path + '_score_m.txt'
        giou = np.loadtxt(giou_file)
        score_nm = np.loadtxt(score_nm_file)
        score_m = np.loadtxt(score_m_file)
        frame_id = list(range(giou.shape[0]))
        plt.title('got10k_val_'+str(i+1))
        plt.xlabel('frame_id')
        # plt.legend(['GIOU', 'score_wo', 'score_w'])
        plt.plot(frame_id, giou, 'y--', frame_id, score_nm,'b-', frame_id, score_m, 'r-')
        plt.legend(['GIOU', 'score_wo', 'score_w'])
        plt.savefig(save_dir+'/got10k_val'+str(i+1)+'.jpg')
        plt.cla()
    except:
        print('got10k_val_'+str(i+1)+' can not read!')
        continue
