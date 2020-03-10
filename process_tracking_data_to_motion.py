from PIL import Image, ImageDraw
import os
from lib.core import iou as pairwise_iou
import matplotlib.pyplot as plt
import lapsolver
from skimage.measure import compare_nrmse
import pandas as pd
import json
from tqdm import tqdm
import numpy as np
import math
from lib.kitti_raw_loader import *
from tqdm import tqdm

def get_IMU_pose(phase, clip_id, frame_num):
    metadata = np.genfromtxt(f'/mnt/data/KITTI_tracking/{phase}/oxts/{clip_id}.txt')[frame_num]
    lat = metadata[0]
    scale = np.cos(lat * np.pi / 180.)
    pose_matrix = pose_from_oxts_packet(metadata[:6], scale)
    return pose_matrix

def get_IMU_to_CAM_and_K(phase, clip_id):
    imu2velo = read_calib_file(f'/mnt/data/KITTI_tracking/{phase}/calib/{clip_id}/calib_imu_to_velo.txt')
    velo2cam = read_calib_file(f'/mnt/data/KITTI_tracking/{phase}/calib/{clip_id}/calib_velo_to_cam.txt')
    cam2cam = read_calib_file(f'/mnt/data/KITTI_tracking/{phase}/calib/{clip_id}/calib_cam_to_cam.txt')
    
    k = cam2cam['K_02'].reshape(3, 3)
    velo2cam_mat = transform_from_rot_trans(velo2cam['R'], velo2cam['T'])
    imu2velo_mat = transform_from_rot_trans(imu2velo['R'], imu2velo['T'])
    cam_2rect_mat = transform_from_rot_trans(cam2cam['R_rect_00'], np.zeros(3))

    imu2cam = cam_2rect_mat @ velo2cam_mat @ imu2velo_mat

    return imu2cam, k

def get_K_and_relative_pose(phase, clip_id, frame_num):
    imu2cam, K = get_IMU_to_CAM_and_K(phase, clip_id)
    ref_pose = get_IMU_pose(phase, clip_id, frame_num)
    tar_pose = get_IMU_pose(phase, clip_id, frame_num-1)
    relative_pose = imu2cam @ np.linalg.inv(tar_pose) @ ref_pose @ np.linalg.inv(imu2cam)
    return K, relative_pose

def get_P2(calib_path):
    with open(calib_path, 'r') as f:
        p2 = f.readlines()[2]
    p2 = list(map(float, p2.split()[1:]))
    p2 = np.array(p2).reshape(3, 4)
    return p2

def get_box(label):
    # x1, y1 x2, y2
    box_2d = label[:, 2:6]

    # x3d, y3d, z3d, w3d, h3d, l3d, ry3d
    x3d, y3d, z3d, l3d, h3d, w3d, ry3d = label[:, 10], label[:, 11]-label[:, 7]/2 , label[:, 12], label[:, 9], label[:, 7], label[:, 8], label[:, 13]
    box_3d = np.stack([x3d, y3d, z3d, l3d, h3d, w3d, ry3d], axis=1)
    return box_2d, box_3d

def project_3d(p2, box_3d):
    """
    Projects a 3D box into 2D vertices

    Args:
        p2 (nparray): projection matrix of size 4x3
        x3d: x-coordinate of center of object
        y3d: y-coordinate of center of object
        z3d: z-cordinate of center of object
        w3d: width of object
        h3d: height of object
        l3d: length of object
        ry3d: rotation w.r.t y-axis
    """
    x3d, y3d, z3d, l3d, h3d, w3d, ry3d = box_3d

    # compute rotational matrix around yaw axis
    R = np.array([[+math.cos(ry3d), 0, +math.sin(ry3d)],
                  [0, 1, 0],
                  [-math.sin(ry3d), 0, +math.cos(ry3d)]])

    # 3D bounding box corners
    x_corners = np.array([0, l3d, l3d, l3d, l3d,   0,   0,   0])
    y_corners = np.array([0, 0,   h3d, h3d,   0,   0, h3d, h3d])
    z_corners = np.array([0, 0,     0, w3d, w3d, w3d, w3d,   0])

    x_corners += -l3d / 2
    y_corners += -h3d / 2
    z_corners += -w3d / 2

    # bounding box in object co-ordinate
    corners_3d = np.array([x_corners, y_corners, z_corners])

    # rotate
    corners_3d = R.dot(corners_3d)

    # translate
    corners_3d += np.array([x3d, y3d, z3d]).reshape((3, 1))

    corners_3D_1 = np.vstack((corners_3d, np.ones((corners_3d.shape[-1]))))
    corners_2D = p2.dot(corners_3D_1)
    corners_2D = (corners_2D / corners_2D[2])[:3]

    bb3d_lines_verts_idx = [0, 1, 2, 3, 4, 5, 6, 7, 0, 5, 4, 1, 2, 7, 6, 3]

    verts3d = (corners_2D[:, bb3d_lines_verts_idx][:2]).astype(float).T

    x1 = np.round(min(verts3d[:, 0]))
    y1 = np.round(min(verts3d[:, 1]))
    x2 = np.round(max(verts3d[:, 0]))
    y2 = np.round(max(verts3d[:, 1]))

    return verts3d, (x1, y1, x2, y2)

def draw_boxes_3d(p2, boxes_3d, image, outline='red'):
    draw = ImageDraw.Draw(image)
    for box_3d in boxes_3d:
        corners_3d, corners_2d = project_3d(p2, box_3d)
        draw.rectangle(corners_2d, outline=outline, width=5)
    return image

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

def generate_motion_data_from_tracking(clip_id, save_path):
    phase = 'training'
    image_root = f'/mnt/data/KITTI_tracking/{phase}/image_02/{clip_id}/'
    calib_path = f'/mnt/data/KITTI_tracking/{phase}/calib/{clip_id}.txt'
    label_path = f'/mnt/data/KITTI_tracking/{phase}/car_3d_det_val/{clip_id}.txt'

    image_save_path = os.path.join(save_path, 'image')
    os.makedirs(image_save_path, exist_ok=True)

    p2 = get_P2(calib_path)
    labels = np.loadtxt(label_path, delimiter=',')
    labels = labels[labels[:, 6] > 5]
    if not labels.tolist():
        return

    per_frame_labels = []
    max_frame_num = int(labels[-1, 0])
    for i in range(0, max_frame_num):
        mask = labels[:, 0].astype(int) == i
        per_frame_labels.append(labels[mask])

    anns = []
    prev_image = None

    for curr_label in tqdm(per_frame_labels):
        if not curr_label.tolist():
            continue
        frame_num =  int(curr_label[0, 0])
        frame_id = f'{frame_num:06d}.png'
        curr_image = Image.open(os.path.join(image_root, frame_id))
        w, h = curr_image.size

        # skip first iteration
        if prev_image is None:
            prev_image = curr_image
            prev_label = curr_label
            continue

        # get box
        prev_box_2d, prev_box_3d = get_box(prev_label)
        curr_box_2d, curr_box_3d = get_box(curr_label)

        # box alignment 
        rids, cids = bbox2d_alignment(prev_box_2d, curr_box_2d, prev_image, curr_image)

        if len(rids) == 0:
            prev_image = curr_image
            prev_label = curr_label
            continue

        prev_box_2d = prev_box_2d[rids]
        prev_box_3d = prev_box_3d[rids]
        curr_box_2d = curr_box_2d[cids]
        curr_box_3d = curr_box_3d[cids]

        # curr_image = draw_boxes_3d(p2, curr_box_3d, curr_image, outline='green')

        # get calib
        K, relative_pose = get_K_and_relative_pose(phase, clip_id, frame_num)
        ego_motion_t = relative_pose[0:3, 3]
        
        curr_rt = np.linalg.inv(K) @ p2
        curr_rt = np.vstack([curr_rt, np.array([0, 0, 0, 1])])
        inv_curr_rt = np.linalg.inv(curr_rt)

        prev_rt = relative_pose @ curr_rt
        prev_p2 = K @ prev_rt[0:3]

        # move c3d from curr camera coordinate to prev camera coordinate
        curr_c3d = np.concatenate([curr_box_3d[:, 0:3], np.ones((curr_box_3d.shape[0], 1))], axis=1).T
        curr_on_prev_c3d = prev_rt @ inv_curr_rt @ curr_c3d
        curr_on_prev_c3d = curr_on_prev_c3d.T[:, 0:3]

        # calculate motion
        motion = prev_box_3d[:, 0:3] - curr_on_prev_c3d
    
        # save motion image data
        curr_box_2d = np.floor(curr_box_2d).astype(int)
        prev_box_2d = np.floor(prev_box_2d).astype(int)
        for box1, box2, box_motion in zip(curr_box_2d, prev_box_2d, motion):
            output_id = len(anns) 
            curr_output = Image.new('RGB', (w, h))
            prev_output = Image.new('RGB', (w, h))
            curr_output.paste(curr_image.crop(box1), box1)
            prev_output.paste(prev_image.crop(box2), box2)
            curr_output.save(os.path.join(image_save_path, f'{output_id:06d}_0.jpg'))
            prev_output.save(os.path.join(image_save_path, f'{output_id:06d}_1.jpg'))

            anns.append(
                {
                'clip_id': clip_id,
                'src_idx': f'{frame_num:06d}',
                'dst_idx': f'{output_id:06d}',
                'box1': box1.tolist(),
                'box2': box2.tolist(),
                'ego_motion_t': ego_motion_t.tolist(),
                'motion': box_motion.tolist()
                }
            )
        
        prev_image = curr_image
        prev_label = curr_label
        
    with open(os.path.join(save_path, 'motion.json'), 'w') as f:
        json.dump(anns, f, indent=2)

        # curr_image_copy = draw_boxes_3d(p2, curr_box_3d, curr_image.copy(), outline='red')
        # prev_image_copy = draw_boxes_3d(prev_p2, curr_box_3d, prev_image.copy(), outline='green')

        # curr_on_prev_3d = curr_box_3d.copy()
        # curr_on_prev_3d[:, 0:3] += motion
        # prev_image_copy = draw_boxes_3d(prev_p2, curr_on_prev_3d, prev_image_copy, outline='red')

        # fig, ax = plt.subplots(2)
        # ax[0].imshow(prev_image_copy)
        # ax[1].imshow(curr_image_copy)
        # plt.show()
        # input()


if __name__ == "__main__":
    for i in range(21):
        clip_id = f'{i:04d}'
        save_path = f'/mnt/data/KITTI_tracking/training/motion/{clip_id}'
        generate_motion_data_from_tracking(clip_id, save_path)
    

