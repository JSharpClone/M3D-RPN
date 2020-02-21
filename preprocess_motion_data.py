import numpy as np
from PIL import Image, ImageDraw
import os
from lib.core import iou as pairwise_iou
from lib.raw_data import RawKitti
import matplotlib.pyplot as plt
import lapsolver
from skimage.measure import compare_nrmse
import pandas as pd
import json
from tqdm import tqdm

phase = 'validation'
if phase == 'training':
    raw_kitti = RawKitti(mode='train')
    num_data = 3712
else:
    raw_kitti = RawKitti(mode='val')
    num_data = 3769

align_gt = False
motion_test = True

save_root = f'./data/kitti_split1/{phase}/motion_M3D_proj_test/'
os.makedirs(save_root, exist_ok=True)

def bbox2d_alignment(curr_bboxes, prev_bboxes, curr_image, prev_image):
    iou = pairwise_iou(curr_bboxes, prev_bboxes)
    vis = np.zeros_like(iou)
    for i, curr_box in enumerate(curr_bboxes):
        for j, prev_box in enumerate(prev_bboxes):
            curr_visual = np.array(curr_image.crop(curr_box).resize((32, 32))).astype(float)/255
            prev_visual = np.array(prev_image.crop(prev_box).resize((32, 32))).astype(float)/255
    
            vis_error = np.absolute(curr_visual - prev_visual).sum() / (32*32)
            vis[i, j] = vis_error

    iou[iou < 0.2] = np.nan
    cost = (1 - iou) + 10 * vis
    rids, cids = lapsolver.solve_dense(cost)
    return rids, cids

def get_label_bbox(filename, idx, gt=False):
    labels = pd.read_csv(filename, header=None, delimiter=' ')
    if not gt:
        labels.columns = ['class', 'truncate', 'occ', 'alpha', 'x1', 'y1', 'x2', 'y2', \
            'h3d', 'w3d', 'l3d', 'x3d', 'y3d', 'z3d', 'ry3d', 'score', 'x2d', 'y2d']
    else:
        labels.drop(columns=labels.columns[15:], inplace=True)
        labels.columns = ['class', 'truncate', 'occ', 'alpha', 'x1', 'y1', 'x2', 'y2', 'h3d', 'w3d', 'l3d', 'x3d', 'y3d', 'z3d', 'ry3d']
    labels = labels[(labels['class']=='Car') | (labels['class']=='Van')]
    labels = labels[labels['occ'] < 2]
    bbox_2d = labels[['x1', 'y1', 'x2', 'y2']].values
    bbox_3d = labels[['x3d', 'y3d', 'z3d', 'l3d', 'h3d', 'w3d', 'ry3d', 'alpha']].values
    bbox_3d[:, 1] -= bbox_3d[:, 4] / 2
    if not gt:
        bbox_proj_center = labels[['x2d', 'y2d']].values
    else:
        curr_p2, _, _ = raw_kitti.get_previous_p2(idx, previous_num=1)
        bbox_proj_center_list = []
        for center_3d in bbox_3d[:, 0:3].copy():
            center_3d = np.concatenate([center_3d, np.ones(1)], axis=0)
            center_3d = np.expand_dims(center_3d, axis=1)
            center_2d = (curr_p2 @ center_3d).squeeze(axis=1)
            center_2d[:2] /= center_2d[2]
            bbox_proj_center_list.append(center_2d[:2])

        if bbox_proj_center_list:
            bbox_proj_center = np.stack(bbox_proj_center_list, axis=0)
        else:
            bbox_proj_center = None
    return bbox_2d, bbox_3d, bbox_proj_center




anns = []
for i in tqdm(range(num_data)):
    idx = f'{i:06d}'
    data_root = f'/home/jsharp/M3D-RPN/data/kitti_split1/{phase}/'

    curr_img_path = os.path.join(data_root, 'image_2', f'{idx}.png')
    curr_img = Image.open(curr_img_path)
    w, h = curr_img.size

    if align_gt:
        true_label_path = os.path.join(data_root, 'label_2', f'{idx}.txt')
        pred_label_path = os.path.join(data_root, 'label_2_M3D', f'{idx}.txt')
    else:
        if phase == 'training' or motion_test:
            curr_label_path = os.path.join(data_root, 'label_2', f'{idx}.txt')
        else:
            curr_label_path = os.path.join(data_root, 'label_2_M3D_proj', f'{idx}.txt')

    prev_img_path = os.path.join(data_root, 'prev_2', f'{idx}_01.png')
    prev_label_path = os.path.join(data_root, 'prev_label_2_M3D_proj', f'{idx}_01.txt')
    if not os.path.exists(prev_label_path):
        continue
    if align_gt:
        if os.stat(prev_label_path).st_size == 0 or os.stat(pred_label_path).st_size == 0 or os.stat(true_label_path).st_size == 0:
            continue

        true_bboxes_2d, true_bboxes_3d, true_bboxes_proj_center  = get_label_bbox(true_label_path, idx, gt=True)
        pred_bboxes_2d, pred_bboxes_3d, pred_bboxes_proj_center  = get_label_bbox(pred_label_path, idx, gt=False)   
        rids, cids = bbox2d_alignment(true_bboxes_2d, pred_bboxes_2d, curr_img, curr_img)
        if len(rids) == 0:
            continue
        # true 2d, pred 3d
        curr_bboxes_2d = true_bboxes_2d[rids]
        curr_bboxes_3d = pred_bboxes_3d[cids]
        true_bboxes_3d = true_bboxes_3d[rids]
        curr_bboxes_proj_center = pred_bboxes_proj_center[cids]
    else:
        if os.stat(prev_label_path).st_size == 0 or os.stat(curr_label_path).st_size == 0:
            continue
        if phase == 'training' or motion_test:
            curr_bboxes_2d, curr_bboxes_3d, curr_bboxes_proj_center  = get_label_bbox(curr_label_path, idx, gt=True)
        else:
            curr_bboxes_2d, curr_bboxes_3d, curr_bboxes_proj_center  = get_label_bbox(curr_label_path, idx, gt=False)
         

    prev_img = Image.open(prev_img_path)
    prev_bboxes_2d, prev_bboxes_3d, prev_bboxes_proj_center = get_label_bbox(prev_label_path, idx, gt=False)
    
    rids, cids = bbox2d_alignment(curr_bboxes_2d, prev_bboxes_2d, curr_img, prev_img)

    if len(rids) == 0:
        continue

    boxes1 = curr_bboxes_2d[rids]
    boxes2 = prev_bboxes_2d[cids]
    boxes_3d = curr_bboxes_3d[rids]
    if align_gt:
        boxes_3d_gt = true_bboxes_3d[rids]
    else:
        boxes_3d_gt = [None]*len(boxes_3d)

    boxes1_proj_center = curr_bboxes_proj_center[rids]
    boxes2_proj_center = prev_bboxes_proj_center[cids]

    boxes1 = np.floor(boxes1).astype(int)
    boxes2 = np.floor(boxes2).astype(int)
    boxes1[:, 0::2] = np.clip(boxes1[:, 0::2], 0, w-1)
    boxes2[:, 1::2] = np.clip(boxes2[:, 1::2], 0, h-1)

    for box1, box2, box_3d, box_3d_gt, box1_proj_center, box2_proj_center in zip(boxes1, boxes2, boxes_3d, boxes_3d_gt, boxes1_proj_center, boxes2_proj_center):
        output_idx = len(anns)
        curr_output = Image.new('RGB', (w, h))
        prev_output = Image.new('RGB', (w, h))
        curr_output.paste(curr_img.crop(box1), box1)
        prev_output.paste(prev_img.crop(box2), box2)
        curr_output.save(os.path.join(save_root, f'{output_idx:06d}_0.jpg'))
        prev_output.save(os.path.join(save_root, f'{output_idx:06d}_1.jpg'))

        d = {
            'src_idx': idx,
            'dst_idx': f'{output_idx:06d}',
            'box1': box1.tolist(),
            'box2': box2.tolist(),
            'box_3d': box_3d.tolist(),
            'box1_proj_center': box1_proj_center.tolist(),
            'box2_proj_center': box2_proj_center.tolist(),
        }
        if align_gt:
            d['box_3d_gt'] = box_3d_gt.tolist()

        anns.append(d)

with open(f'./data/kitti_split1/{phase}/motion_M3D_proj_test.json', 'w') as f:
    json.dump(anns, f, indent=2)




    