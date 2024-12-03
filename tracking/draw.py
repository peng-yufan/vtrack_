import torch
import cv2 as cv
import math
import numpy as np
def get_frame(path,id):
    frame_path = path+get_id(id)+'.jpg'
    return cv.imread(frame_path)
def get_id(id):
    k = int(math.log(id,10)) if id > 0 else 0
    zero_num = 7-k
    id_str = '0'
    for i in range(zero_num-1):
        id_str = id_str + '0'
    return id_str+str(id)
def get_anno(path):
    file_path = path
    gts = np.loadtxt(file_path,delimiter=',')
    return gts
def get_result(path):
    file_path = path
    result = np.loadtxt(file_path,delimiter='\t')
    return result
def draw_frame(frame,boxes, name):
    rgbs = [(255,0,0),(0,0,255),(0,255,0)]
    for state,rgb in zip(boxes, rgbs):
        cv.rectangle(frame, (state[0], state[1]), (state[2] + state[0], state[3] + state[1]),
                     rgb, 5)
        # cv.imshow(name, frame)
        cv.imwrite(name+'.jpg', frame)
def draw():
    seq_name = 'sheep-5'
    frame_path = '/data1/LaSOT/'+seq_name+'/img/'
    result_path = '/home/yufan/pysot-master/result/LaSOT-/'
    result_path1 = result_path+'MixVit/'+seq_name+'.txt'
    result_path2 = result_path + 'Video-MixVit/'+seq_name+'.txt'
    result1 = get_result(result_path1)
    result2 = get_result(result_path2)
    gt = get_anno('/data1/LaSOT/'+seq_name+'/groundtruth.txt')
    for id in range(700, 1000, 20):
        frame = get_frame(frame_path, id)
        box1 = [int(s) for s in result1[id]]
        box2 = [int(s) for s in result2[id]]
        boxgt = [int(s) for s in gt[id]]
        boxes = [box1,box2,boxgt]
        draw_frame(frame, boxes, seq_name+'-'+str(id))
def main():
    draw()
if __name__ == '__main__':
    main()




