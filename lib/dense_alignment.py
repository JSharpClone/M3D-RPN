import numpy as np
import os
import math
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw

import torch

from lib.loss.motion_loss import get_corners, get_2d_corners_from_3d, iou

SCALE_TRANSLATION = 0.01
SCALE_LHW = 0.1

def convertAlpha2Rot(alpha, z3d, x3d):
    
    ry3d = alpha + torch.atan2(-z3d, x3d) + 0.5 * math.pi
    #ry3d = alpha + math.atan2(x3d, z3d)# + 0.5 * math.pi

    # while ry3d > math.pi: ry3d -= math.pi * 2
    # while ry3d < (-math.pi): ry3d += math.pi * 2

    return ry3d

def dense_alignment(data, max_depth=80, min_depth=0, steps=200, device='cuda:1', phase='training', use_box=True):
    data_root = f'/home/jsharp/M3D-RPN/data/kitti_split1/{phase}'

    motion = data['motion']
    motion = motion.detach() / SCALE_TRANSLATION
    motion = torch.cat([motion]*steps, dim=0).unsqueeze(2)
    motion_x = motion[:, 0]
    motion_y = motion[:, 1]
    motion_z = motion[:, 2]

    box_3d = data['box_3d']
    # box_3d_gt = data['box_3d_gt']
    # box_3d[:] = box_3d_gt[:]

    alpha = torch.stack([box_3d[0, 7]]*steps, dim=0).unsqueeze(1)
    # ry3d = torch.stack([box_3d[0, 6]]*steps, dim=0).unsqueeze(1)

    l3d = torch.stack([box_3d[0, 3]]*steps, dim=0)
    h3d = torch.stack([box_3d[0, 4]]*steps, dim=0)
    w3d = torch.stack([box_3d[0, 5]]*steps, dim=0)
    # lhw = data['lhw'] / SCALE_LHW
    # lhw = torch.cat([lhw]*steps, dim=0)
    # l3d = lhw[:, 0]
    # h3d = lhw[:, 1]
    # w3d = lhw[:, 2]

    curr_box = data['curr_box']
    prev_box = data['prev_box']
    curr_box = torch.cat([curr_box]*steps, dim=0)
    prev_box = torch.cat([prev_box]*steps, dim=0)

    curr_p2 = data['curr_p2']
    prev_p2 = data['prev_p2']

    # xyz = box_3d[:, 0:3].unsqueeze(2)
    # ones = torch.cuda.FloatTensor(xyz.size(0), 1, xyz.size(2)).fill_(1)
    # xyz = torch.cat([xyz, ones], dim=1) # [batch_size, 4, 1]
    # xyz_proj = torch.bmm(curr_p2, xyz)
    # xyz_proj = (xyz_proj[:, :2, :] / xyz_proj[:, 2:3, :]).squeeze(2)
    # xyz_proj = torch.cat([xyz_proj]*steps, dim=0)
    # x2d = xyz_proj[:, 0]
    # y2d = xyz_proj[:, 1]
    box1_proj_center = data['box1_proj_center']
    box1_proj_center = torch.cat([box1_proj_center]*steps, dim=0)
    x2d = box1_proj_center[:, 0]
    y2d = box1_proj_center[:, 1]

    box2_proj_center = data['box2_proj_center']
    box2_proj_center = torch.cat([box2_proj_center]*steps, dim=0)

    # x2d = (curr_box[:, 2] + curr_box[:, 0]) / 2
    # y2d = (curr_box[:, 1] + curr_box[:, 3]) / 2
    
    inv_p2 = torch.inverse(curr_p2[0])
    inv_p2 = torch.stack([inv_p2]*steps, dim=0)
    curr_p2 = torch.cat([curr_p2]*steps, dim=0)
    prev_p2 = torch.cat([prev_p2]*steps, dim=0)


    depth = torch.linspace(min_depth, max_depth, steps).to(device)
    
    center_3d_proj = torch.stack([x2d * depth, y2d * depth, depth,  torch.ones(200).to(device)], dim=1).unsqueeze(2).to(device)
    
    center_3d = torch.bmm(inv_p2, center_3d_proj)
    x3d = center_3d[:, 0]
    y3d = center_3d[:, 1]
    z3d = center_3d[:, 2]
    ry3d = alpha + torch.atan2(-z3d, x3d) + 0.5 * math.pi
    x3d = x3d.squeeze(1)
    y3d = y3d.squeeze(1)
    z3d = z3d.squeeze(1)
    ry3d = ry3d.squeeze(1)

    if use_box:
        corners_3d = get_corners(ry3d, l3d, w3d, h3d, x3d, y3d, z3d)
        curr_box_proj = get_2d_corners_from_3d(corners_3d, curr_p2)
        corners_3d[:, 0, :] += motion_x
        corners_3d[:, 1, :] += motion_y
        corners_3d[:, 2, :] += motion_z
        
        prev_box_proj = get_2d_corners_from_3d(corners_3d, prev_p2)

        curr_iou = iou(curr_box_proj, curr_box)
        prev_iou = iou(prev_box_proj, prev_box)

        iou_sum = curr_iou + prev_iou
        best_idx = torch.argmax(iou_sum)

        best_z3d = depth[best_idx]
        best_x3d = x3d[best_idx]
        best_y3d = y3d[best_idx]
        best_ry3d = ry3d[best_idx]
        best_curr = curr_box_proj[best_idx]
        best_prev = prev_box_proj[best_idx]
        best_iou = iou_sum[best_idx] / 2
    else: # use only projected center
        prev_center_3d = center_3d[:, 0:3] + motion
        prev_center_3d = torch.cat([prev_center_3d, torch.ones((steps, 1, 1)).to(device)], dim=1)
        prev_proj_center = torch.bmm(prev_p2, prev_center_3d)
        prev_proj_center[:, 0:2, :] /= prev_proj_center[:, 2:3, :]
        prev_proj_center = prev_proj_center[:, 0:2, :].squeeze(2)
        x_error = prev_proj_center[:, 0] - box2_proj_center[:, 0]
        y_error = prev_proj_center[:, 1] - box2_proj_center[:, 1]
        dis_error = torch.sqrt(x_error**2 + y_error**2)
        best_idx = torch.argmin(dis_error)

        best_z3d = depth[best_idx]
        best_x3d = x3d[best_idx]
        best_y3d = y3d[best_idx]
        best_ry3d = ry3d[best_idx]
        
        corners_3d = get_corners(ry3d, l3d, w3d, h3d, x3d, y3d, z3d)
        curr_box_proj = get_2d_corners_from_3d(corners_3d, curr_p2)
        best_curr = curr_box_proj[best_idx]
        curr_iou = iou(curr_box_proj, curr_box)
        best_iou = curr_iou[best_idx]

    return best_x3d, best_y3d, best_z3d, best_ry3d, best_curr, best_iou


