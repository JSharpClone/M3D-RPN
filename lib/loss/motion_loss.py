import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from lib.motion_data import MotionDataset
from lib.rpn_util import iou

SCALE_TRANSLATION = 0.01
HEIGHT = 128 # 512
WIDTH = 416 # 1760

def intersect(box_a, box_b, sign=False):
    max_xy = torch.min(box_a[:, 2:], box_b[:, 2:])
    min_xy = torch.max(box_a[:, :2], box_b[:, :2])
    if sign:
        inter = max_xy - min_xy
        mask = (inter[:, 0] < 0) & (inter[:, 1] < 0)
        output = inter[:, 0] * inter[:, 1]
        output[mask] = -output[mask]
        return output
    else:
        inter = torch.clamp((max_xy - min_xy), 0)
        return inter[:, 0] * inter[:, 1]


def iou(box_a, box_b, sign=False):
    inter = intersect(box_a, box_b, sign=sign)
    area_a = (box_a[:, 2] - box_a[:, 0]) * (box_a[:, 3] - box_a[:, 1])
    area_b = (box_b[:, 2] - box_b[:, 0]) * (box_b[:, 3] - box_b[:, 1])

    union = area_a + area_b - inter
    return inter / union

def get_corners(ry3d, l, w, h, x, y, z):
    batch_size = len(ry3d)
    c = torch.cos(ry3d) # [batch_size]
    s = torch.sin(ry3d) # [batch_size]
    zeros = torch.cuda.FloatTensor(batch_size).fill_(0)
    ones = torch.cuda.FloatTensor(batch_size).fill_(1)
    R = torch.stack([torch.stack([c, zeros, s], dim=1), # [batch_size, 3, 3]
                        torch.stack([zeros, ones, zeros], dim=1),
                        torch.stack([-s, zeros, c], dim=1)], dim=1)

    x_corners = torch.stack([-l/2, l/2, l/2, l/2, l/2, -l/2, -l/2, -l/2], dim=1) # [batch_size, 8]
    y_corners = torch.stack([-h/2, -h/2, h/2, h/2, -h/2, -h/2, h/2, h/2], dim=1) # [batch_size, 8]
    z_corners = torch.stack([-w/2, -w/2, -w/2, w/2, w/2,-w/2, w/2, -w/2], dim=1) # [batch_size, 8]
    corners = torch.stack([x_corners, y_corners, z_corners], dim=1) # [batch_size, 3, 8]

    corners = torch.bmm(R, corners)
    corners[:, 0, :] += x.unsqueeze(1)
    corners[:, 1, :] += y.unsqueeze(1)
    corners[:, 2, :] += z.unsqueeze(1)

    return corners

def get_2d_corners_from_3d(corners_3d, p2):
    # ones = torch.ones((corners_3d.size()[0], 1, corners_3d.size()[2]))
    ones = torch.cuda.FloatTensor(corners_3d.size(0), 1, corners_3d.size(2)).fill_(1)
    corners_3d = torch.cat([corners_3d, ones], dim=1) # [batch_size, 4, 8]
    corners_2d = torch.bmm(p2, corners_3d)
    corners_2d = corners_2d[:, :2, :] / corners_2d[:, 2:3, :] # [batch_size, 4, 8]

    x = corners_2d[:, 0, :] # [batch_size, 8]
    min_x = torch.min(x, dim=1)[0] # [batch_size]
    max_x = torch.max(x, dim=1)[0] # [batch_size]
    y = corners_2d.clone()[:, 1, :] # [batch_size, 8]
    min_y = torch.min(y, dim=1)[0] # [batch_size]
    max_y = torch.max(y, dim=1)[0] # [batch_size]

    return torch.stack([min_x, min_y, max_x, max_y], dim=1) # [batch_size, 4]


class MotionLoss(nn.Module):
    def __init__(self):
        super(MotionLoss, self).__init__()
    
    def forward(self, motion, data):
        box_3d = data['box_3d']
        prev_box = data['prev_box']
        prev_p2 = data['prev_p2']
        prev_image = data['prev_image']
        h_scale = data['h_scale']
        w_scale = data['w_scale']

        h, w = prev_image.size(2), prev_image.size(3)
        raw_h = h * h_scale
        raw_w = w * w_scale

        motion = motion / SCALE_TRANSLATION
        motion_x = motion[:, 0]
        motion_y = motion[:, 1]
        motion_z = motion[:, 2]

        x3d = box_3d[:, 0]
        y3d = box_3d[:, 1]
        z3d = box_3d[:, 2]
        l3d = box_3d[:, 3]
        h3d = box_3d[:, 4]
        w3d = box_3d[:, 5]
        ry3d = box_3d[:, 6]
        
        corners = get_corners(ry3d, l3d, w3d, h3d, x3d, y3d, z3d)
        corners[:, 0, :] += motion_x.unsqueeze(1)
        corners[:, 1, :] += motion_y.unsqueeze(1)
        corners[:, 2, :] += motion_z.unsqueeze(1)
        pred_box = get_2d_corners_from_3d(corners, prev_p2) # [batch_size, 4]

        sign_iou = iou(prev_box, pred_box, sign=True)
        iou_2d_loss = 1-sign_iou
        iou_2d_loss = iou_2d_loss.mean()


        # l1_loss = F.l1_loss(pred_box, prev_box, reduction='none')
        # l1_loss[:, 0::2] = l1_loss[:, 0::2] # / raw_w.unsqueeze(1)
        # l1_loss[:, 1::2] = l1_loss[:, 1::2] # / raw_h.unsqueeze(1)

        return iou_2d_loss, pred_box, prev_box


if __name__ == "__main__":
    # dataset = MotionDataset()
    # dataloader = DataLoader(dataset, batch_size=2, num_workers=4, shuffle=False)
    # criterion = MotionLoss()
    # for data in iter(dataloader):
    #     loss = criterion(torch.rand((2, 3)), data)
    #     print(loss)
    #     input()
    box_a = torch.tensor([[10, 10, 20, 20], [10, 10, 20, 20], [10, 10, 20, 20]], dtype=torch.float32)
    box_b = torch.tensor([[12, 12, 18, 18], [25, 5, 35, 15], [15, 15, 25, 25]], dtype=torch.float32)
    print(iou(box_a, box_b, sign=True))

        

