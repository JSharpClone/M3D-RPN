import numpy as np
import os
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw

import torch

from lib.loss.motion_loss import get_corners, get_2d_corners_from_3d, iou

SCALE_TRANSLATION = 0.01

def dense_alignment(motion, data, max_depth=80, min_depth=0, steps=200, device='cuda:1', phase='training'):
    batch_size = motion.size(0)
    data_root = f'/home/jsharp/M3D-RPN/data/kitti_split1/{phase}'
    src_id = data['src_id']
    dst_id = data['dst_id']
    curr_box = data['curr_box']
    prev_box = data['prev_box']
    curr_p2 = data['curr_p2']
    prev_p2 = data['prev_p2']
    motion = motion.detach() / SCALE_TRANSLATION
    motion_x = motion[:, 0]
    motion_y = motion[:, 1]
    motion_z = motion[:, 2]
    box_3d = data['box_3d']
    x2d = (curr_box[:, 2] + curr_box[:, 0]) / 2
    y2d = (curr_box[:, 1] + curr_box[:, 3]) / 2

    x3d_list = []
    y3d_list = []
    z3d_list = []
    score_list = []
    for i in range(batch_size):
        l3d = box_3d[i:i+1, 3]
        h3d = box_3d[i:i+1, 4]
        w3d = box_3d[i:i+1, 5]
        ry3d = box_3d[i:i+1, 6]
        curr_image = Image.open(os.path.join(data_root, 'image_2', f'{src_id[i]:06d}.png'))
        prev_image = Image.open(os.path.join(data_root, 'prev_2', f'{src_id[i]:06d}_01.png'))
        w, h = curr_image.size
        
        max_iou = 0
        min_error = 1
        best_z3d = torch.tensor(0)
        for depth in torch.linspace(min_depth, max_depth, steps):
            center_3d_proj = torch.tensor([x2d[i] * depth, y2d[i] * depth, depth,  1]).unsqueeze(1).to(device)
            inv_p2 = torch.inverse(curr_p2[i])
            center_3d = inv_p2 @ center_3d_proj
            x3d, y3d, z3d = center_3d[:3]
            corners_3d = get_corners(ry3d, l3d, w3d, h3d, x3d, y3d, z3d)

            curr_box_proj = get_2d_corners_from_3d(corners_3d, curr_p2[i:i+1])
            corners_3d[:, 0, :] += motion_x[i]
            corners_3d[:, 1, :] += motion_y[i]
            corners_3d[:, 2, :] += motion_z[i]
            prev_box_proj = get_2d_corners_from_3d(corners_3d, prev_p2[i:i+1])

            curr_iou = iou(curr_box_proj, curr_box[i:i+1])[0]
            prev_iou = iou(prev_box_proj, prev_box[i:i+1])[0]

            if curr_iou + prev_iou < 0.2:
                continue
            
            curr_box_proj = curr_box_proj.squeeze(0).cpu().numpy()
            prev_box_proj = prev_box_proj.squeeze(0).cpu().numpy()
            
            if curr_iou + prev_iou > max_iou:
                max_iou = curr_iou + prev_iou
                best_z3d = depth
                best_x3d = x3d[0]
                best_y3d = y3d[0]
                best_curr = curr_box_proj
                best_prev = prev_box_proj
                best_iou = max_iou / 2

        x3d_list.append(best_x3d)
        y3d_list.append(best_y3d)
        z3d_list.append(best_z3d)
        score_list.append(best_iou)
    return torch.stack(x3d_list, dim=0), torch.stack(y3d_list, dim=0), torch.stack(z3d_list, dim=0), torch.stack(score_list, dim=0)


