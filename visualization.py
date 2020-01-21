import numpy as np
from PIL import Image, ImageDraw
import os
from lib.core import iou as pairwise_iou
import matplotlib.pyplot as plt
import lapsolver
from skimage.measure import compare_nrmse
import pandas as pd

def draw_boxes_2d(box, img, color='red'):
    draw = ImageDraw.Draw(img)
    x1, y1, x2, y2 = box
    draw.rectangle((x1, y1, x2, y2), outline=color, width=5)
    return img

def get_label_bbox(filename):
    labels = pd.read_csv(filename, header=None, delimiter=' ')
    labels.drop(columns=labels.columns[8:], inplace=True)
    labels.columns = ['class', 'truncate', 'occ', 'alpha', 'x1', 'y1', 'x2', 'y2']
    labels = labels[(labels['class']=='Car') | (labels['class']=='Van')]
    labels = labels[labels['occ'] < 2]
    bbox_2d = labels[['x1', 'y1', 'x2', 'y2']].values
    return bbox_2d

import matplotlib as mpl
colors = plt.cm.tab20(np.linspace(0, 1, 20))
colors = [mpl.colors.rgb2hex(c[:3]) for c in colors]
while len(colors) < 100:
    colors.extend(colors)

if not os.path.exists('./visual'):
    os.mkdir('./visual')

for i in range(3000):
    idx = f'{i:06d}'
    print(idx)
    data_root = '/home/jsharp/M3D-RPN/data/kitti_split1/training/'

    curr_img_path = os.path.join(data_root, 'image_2', f'{idx}.png')
    curr_img = Image.open(curr_img_path)

    curr_label_path = os.path.join(data_root, 'label_2', f'{idx}.txt')
    curr_bboxes = get_label_bbox(curr_label_path)

    prev_img_path = os.path.join(data_root, 'prev_2', f'{idx}_01.png')
    prev_label_path = os.path.join(data_root, 'prev_label_2', f'{idx}_01.txt')
    if not os.path.exists(prev_label_path):
        continue
    if os.stat(prev_label_path).st_size == 0:
        continue

    prev_img = Image.open(prev_img_path)
    prev_bboxes = get_label_bbox(prev_label_path)

    if len(curr_bboxes) < 5 or len(prev_bboxes) < 5:
        continue
    
    iou = pairwise_iou(curr_bboxes, prev_bboxes)
    vis = np.zeros_like(iou)
    for i, curr_box in enumerate(curr_bboxes):
        for j, prev_box in enumerate(prev_bboxes):
            curr_visual = np.array(curr_img.crop(curr_box).resize((32, 32))).astype(float)/255
            prev_visual = np.array(prev_img.crop(prev_box).resize((32, 32))).astype(float)/255
            # nrmse = compare_nrmse(curr_visual, prev_visual)
            # print(nrmse)
            vis_error = np.absolute(curr_visual - prev_visual).sum() / (32*32)
            vis[i, j] = vis_error

    iou[iou < 0.2] = np.nan
    cost = (1 - iou) + 10 * vis
    rids, cids = lapsolver.solve_dense(cost)

    if len(rids) == 0:
        continue

    boxes1 = curr_bboxes[rids]
    boxes2 = prev_bboxes[cids]

    for box1, box2, color in zip(boxes1, boxes2, colors):
        curr_result = draw_boxes_2d(box1, curr_img, color=color)
        prev_result = draw_boxes_2d(box2, prev_img, color=color)
    fig, ax = plt.subplots(2, 1)
    ax[0].imshow(curr_result)
    ax[1].imshow(prev_result)
    fig.savefig(os.path.join('./visual', idx + '.png'))
    plt.show()
    input()
    