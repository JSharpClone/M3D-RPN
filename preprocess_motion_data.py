import numpy as np
from PIL import Image, ImageDraw
import os
from lib.core import iou as pairwise_iou
import matplotlib.pyplot as plt
import lapsolver
from skimage.measure import compare_nrmse
import pandas as pd
import json
from tqdm import tqdm


def get_label_bbox(filename, proj_center=False):
    labels = pd.read_csv(filename, header=None, delimiter=' ')
    if proj_center:
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
    if proj_center:
        bbox_proj_center = labels[['x2d', 'y2d']].values
    else:
        bbox_proj_center = None
    return bbox_2d, bbox_3d, bbox_proj_center

phase = 'validation'

if phase == 'training':
    num_data = 3712
    proj_center = False
else:
    num_data = 3769
    proj_center = True

anns = []
for i in tqdm(range(num_data)):
    idx = f'{i:06d}'
    data_root = f'/home/jsharp/M3D-RPN/data/kitti_split1/{phase}/'

    curr_img_path = os.path.join(data_root, 'image_2', f'{idx}.png')
    curr_img = Image.open(curr_img_path)
    w, h = curr_img.size

    curr_label_path = os.path.join(data_root, 'my_label_2', f'{idx}.txt')

    prev_img_path = os.path.join(data_root, 'prev_2', f'{idx}_01.png')
    prev_label_path = os.path.join(data_root, 'prev_label_2', f'{idx}_01.txt')
    if not os.path.exists(prev_label_path):
        continue
    if os.stat(prev_label_path).st_size == 0 or os.stat(curr_label_path).st_size == 0:
        continue

    curr_bboxes_2d, curr_bboxes_3d, curr_bboxes_proj_center  = get_label_bbox(curr_label_path, proj_center=proj_center)
    
    prev_img = Image.open(prev_img_path)
    prev_bboxes_2d, _, _ = get_label_bbox(prev_label_path, proj_center=proj_center)

    # if len(curr_bboxes) < 5 or len(prev_bboxes) < 5:
    #     continue
    
    iou = pairwise_iou(curr_bboxes_2d, prev_bboxes_2d)
    vis = np.zeros_like(iou)
    for i, curr_box in enumerate(curr_bboxes_2d):
        for j, prev_box in enumerate(prev_bboxes_2d):
            curr_visual = np.array(curr_img.crop(curr_box).resize((32, 32))).astype(float)/255
            prev_visual = np.array(prev_img.crop(prev_box).resize((32, 32))).astype(float)/255
    
            vis_error = np.absolute(curr_visual - prev_visual).sum() / (32*32)
            vis[i, j] = vis_error

    iou[iou < 0.2] = np.nan
    cost = (1 - iou) + 10 * vis
    rids, cids = lapsolver.solve_dense(cost)

    if len(rids) == 0:
        continue

    boxes1 = curr_bboxes_2d[rids]
    boxes2 = prev_bboxes_2d[cids]
    boxes_3d = curr_bboxes_3d[rids]
    if curr_bboxes_proj_center is not None:
        boxes_proj_center = curr_bboxes_proj_center[rids]
    else:
        boxes_proj_center = [None]*len(boxes_3d)

    boxes1 = np.floor(boxes1).astype(int)
    boxes2 = np.floor(boxes2).astype(int)
    boxes1[:, 0::2] = np.clip(boxes1[:, 0::2], 0, w-1)
    boxes2[:, 1::2] = np.clip(boxes2[:, 1::2], 0, h-1)

    for box1, box2, box_3d, box_proj_center in zip(boxes1, boxes2, boxes_3d, boxes_proj_center):
        output_idx = len(anns)
        curr_output = Image.new('RGB', (w, h))
        prev_output = Image.new('RGB', (w, h))
        curr_output.paste(curr_img.crop(box1), box1)
        prev_output.paste(prev_img.crop(box2), box2)
        curr_output.save(f'./data/kitti_split1/{phase}/motion/{output_idx:06d}_0.jpg')
        prev_output.save(f'./data/kitti_split1/{phase}/motion/{output_idx:06d}_1.jpg')

        d = {
            'src_idx': idx,
            'dst_idx': f'{output_idx:06d}',
            'box1': box1.tolist(),
            'box2': box2.tolist(),
            'box_3d': box_3d.tolist()
        }
        if box_proj_center is not None:
            d['box_proj_center'] = box_proj_center.tolist()

        anns.append(d)

with open(f'./data/kitti_split1/{phase}/motion.json', 'w') as f:
    json.dump(anns, f, indent=2)



    