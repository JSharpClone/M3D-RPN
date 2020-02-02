import torch
from torch.utils.data import DataLoader

import os
from tqdm import tqdm

from lib.dense_alignment import dense_alignment
from lib.loss.motion_loss import MotionLoss
from lib.motion_data import MotionDataset

BATCH_SIZE = 1

results_path = '/home/jsharp/M3D-RPN/output/dense_alignment/data'
device = 'cuda:1'
model = torch.load(f'/home/jsharp/M3D-RPN/output/motion/model_199.pth').to(device)
valid_dataset = MotionDataset(phase='validation')
valid_dataloader = DataLoader(valid_dataset, batch_size=BATCH_SIZE, num_workers=0, shuffle=False)
criterion = MotionLoss()

for i in range(3769):
    file = open(os.path.join(results_path, f'{i:06d}.txt'), 'w')
    file.close()

last_id = '-1'
file = None
model.eval()
for  data in tqdm(iter(valid_dataloader)):
    for k, v in data.items():
        data[k] = v.to(device)
    
    motion = model(data)
    _, pred_box, prev_box = criterion(motion, data)
    x3d, y3d, z3d, score = dense_alignment(motion, data, device=device, phase='validation')
    src_id = data['src_id']
    curr_box = data['curr_box']
    box_3d = data['box_3d']

    curr_id =  f'{src_id[0]:06d}'
    if  curr_id != last_id:
        if file is not None:
            file.write(text_to_write)
            file.close()
        last_id = curr_id
        text_to_write = ''
        file = open(os.path.join(results_path, curr_id + '.txt'), 'w')

    alpha = -1
    x1 = curr_box[0, 0]
    y1 = curr_box[0, 1]
    x2 = curr_box[0, 2]
    y2 = curr_box[0, 3]
    h3d = box_3d[0, 4]
    w3d = box_3d[0, 5]
    l3d = box_3d[0, 3]
    ry3d = box_3d[0, 6]
    x3d = x3d[0]
    y3d = y3d[0]
    z3d = z3d[0]
    score = score[0]
    y3d += h3d/2
    text_to_write += ('Car -1 -1 {:.6f} {:.6f} {:.6f} {:.6f} {:.6f} {:.6f} {:.6f} {:.6f} {:.6f} {:.6f} {:.6f} '
                        + '{:.6f} {:.6f}\n').format(alpha, x1, y1, x2, y2, h3d, w3d, l3d, x3d, y3d, z3d, ry3d, score)

file.close()