import torch
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
from torchvision import transforms

from PIL import Image
import json
import os

from lib.raw_data import RawKitti

image_means = [0.485, 0.456, 0.406]
image_stds = [0.229, 0.224, 0.225]
HEIGHT = 128 # 512
WIDTH = 416 # 1760

class MotionDataset(Dataset):

    def __init__(self, phase='training', use_predict_val=False):
        super(MotionDataset, self).__init__()

        if phase == 'training':
            self.raw_kitti = RawKitti(mode='train')
        else:
            self.raw_kitti = RawKitti(mode='val')

        data_root = os.path.join('/home/jsharp/M3D-RPN/data/kitti_split1/', phase)
        if phase == 'validation' and use_predict_val == False:
            anns_path = os.path.join(data_root, 'motion_M3D_proj_test.json')
            self.motion_root = os.path.join(data_root, 'motion_M3D_proj_test')
        else:
            anns_path = os.path.join(data_root, 'motion_M3D_proj_3.json')
            self.motion_root = os.path.join(data_root, 'motion_M3D_proj_3')

        with open(anns_path, 'r') as f:
            self.anns = json.load(f)
        
        self.transfrom = transforms.Compose([
            transforms.Resize((HEIGHT, WIDTH)),
            transforms.ToTensor(),
            # transforms.Normalize(image_means, image_stds),
        ])

    def __getitem__(self, idx):
        ann = self.anns[idx]
        src_id = ann['src_idx']
        image_id = ann['dst_idx']
        curr_box = torch.tensor(ann['box1'], dtype=torch.float32)
        prev_box = torch.tensor(ann['box2'], dtype=torch.float32)

        curr_image = Image.open(os.path.join(self.motion_root, f'{image_id}_0.jpg'))
        prev_image = Image.open(os.path.join(self.motion_root, f'{image_id}_1.jpg'))

        w, h = curr_image.size
        w_scale = torch.tensor(w / WIDTH, dtype=torch.float32)
        h_scale = torch.tensor(h / HEIGHT, dtype=torch.float32)

        curr_image = self.transfrom(curr_image)
        prev_image = self.transfrom(prev_image)

        box_3d = torch.tensor(ann['box_3d'], dtype=torch.float32)

        box1_proj_center = torch.tensor(ann['box1_proj_center'], dtype=torch.float32)
        box2_proj_center = torch.tensor(ann['box2_proj_center'], dtype=torch.float32)

        curr_p2, prev_p2, relative_pose = self.raw_kitti.get_previous_p2(src_id, previous_num=1)
        curr_p2 = torch.from_numpy(curr_p2).float()
        prev_p2 = torch.from_numpy(prev_p2).float()
        ego_motion_t = torch.from_numpy(relative_pose[:, 3]).float()

        src_id = torch.tensor(int(src_id), dtype=torch.int)
        dst_id = torch.tensor(int(image_id), dtype=torch.int)

        labels = {
            'curr_image': curr_image,
            'prev_image': prev_image,
            'h_scale': h_scale,
            'w_scale': w_scale,
            'curr_p2': curr_p2,
            'prev_p2': prev_p2,
            'ego_motion_t': ego_motion_t,
            'curr_box': curr_box,
            'prev_box': prev_box,
            'box1_proj_center': box1_proj_center,
            'box2_proj_center': box2_proj_center,
            'box_3d': box_3d,
            'src_id': src_id,
            'dst_id': dst_id,
        }
        if 'box_3d_gt' in ann.keys():
            box_3d_gt = torch.tensor(ann['box_3d_gt'], dtype=torch.float32)
            labels['box_3d_gt'] =  box_3d_gt

        return labels
        

    def __len__(self):
        return len(self.anns)

if __name__ == "__main__":
    data = MotionDataset()
    dataloader = DataLoader(data, batch_size=2, num_workers=4, shuffle=False)
    for label in iter(dataloader):
        for k, v in label.items():
            print(k, v)
        input()

    