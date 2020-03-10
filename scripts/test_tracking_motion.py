import torch
import torch.nn.functional as F
from torchvision import transforms

import os
import json
from tqdm import tqdm
from PIL import Image
import matplotlib.pyplot as plt

HEIGHT = 128 # 512
WIDTH = 416 # 1760

@torch.no_grad()
def compute_motion_error(clip_id):
    anns_path = f'/mnt/data/KITTI_tracking/training/motion/{clip_id}/motion.json'
    if not os.path.exists(anns_path):
        return 'File not found!'

    with open(anns_path, 'r') as f:
        anns = json.load(f)

    device = 'cuda:0'
    model = torch.load(f'/home/jsharp/M3D-RPN/output/motion/xyz/model_199.pth').to(device)
    model.eval()
    transform = transforms.Compose([
                transforms.Resize((HEIGHT, WIDTH)),
                transforms.ToTensor(),
                # transforms.Normalize(image_means, image_stds),
                ])

    losses = []
    for ann in anns:
        data_id = ann['dst_idx']
        curr_image_path = f'/mnt/data/KITTI_tracking/training/motion/{clip_id}/image/{data_id}_0.jpg'
        prev_image_path = f'/mnt/data/KITTI_tracking/training/motion/{clip_id}/image/{data_id}_1.jpg'
        curr_image = Image.open(curr_image_path)
        prev_image = Image.open(prev_image_path)
        
        curr_image = transform(curr_image).unsqueeze(0)
        prev_image = transform(prev_image).unsqueeze(0)

        ego_motion_t = torch.tensor(ann['ego_motion_t']).unsqueeze(0)
        motion_gt = torch.tensor(ann['motion']).unsqueeze(0)
        
        data = {
            'curr_image': curr_image,
            'prev_image': prev_image,
            'ego_motion_t': ego_motion_t,
            'motion_gt': motion_gt
        }

        for k, v in data.items():
            data[k] = v.to(device)

        data = model(data)
        loss = F.l1_loss(data['motion'], data['motion_gt'], reduction='none')
        losses.append(loss)

    motion_error = torch.cat(losses, dim=0)
    motion_error = motion_error.mean(dim=0)
    motion_error = motion_error.detach().cpu().numpy()
    return motion_error

if __name__ == "__main__":
    for i in range(21):
        clip_id = f'{i:04d}'
        print(f'clip_id: {clip_id}')
        motion_error = compute_motion_error(clip_id)
        print(motion_error)
    