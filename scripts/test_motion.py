import torch
from torch.utils.data import DataLoader

import os
from tqdm import tqdm

from lib.dense_alignment import dense_alignment
from lib.loss.motion_loss import MotionLoss
from lib.motion_data import MotionDataset

BATCH_SIZE = 1
STEPS = 200
SCALE_LHW = 0.1

# results_path = f'/home/jsharp/M3D-RPN/output/dense_alignment/steps_{STEPS}_xyz_car/data'
results_path = f'/home/jsharp/M3D-RPN/output/dense_alignment/steps_{STEPS}_M3D_without_motion/data'
if not os.path.exists(results_path):
    os.makedirs(results_path)
device = 'cuda:1'
model = torch.load(f'/home/jsharp/M3D-RPN/output/motion/xyz/model_199.pth').to(device)
# model = torch.load(f'/home/jsharp/M3D-RPN/output/motion/proj_center_loss/xyz/model_39.pth').to(device)
valid_dataset = MotionDataset(phase='validation', use_predict_val=True)
valid_dataloader = DataLoader(valid_dataset, batch_size=BATCH_SIZE, num_workers=0, shuffle=False)
# criterion = MotionLoss()

for i in range(3769):
    file = open(os.path.join(results_path, f'{i:06d}.txt'), 'w')
    file.close()

last_id = '-1'
file = None
model.eval()
for  data in tqdm(iter(valid_dataloader)):
    for k, v in data.items():
        data[k] = v.to(device)
    
    data = model(data)
    # _, pred_box, prev_box = criterion(data)
    x3d, y3d, z3d, ry3d, box, score = dense_alignment(data, steps=STEPS, device=device, phase='validation', use_box=True, use_motion=False)
    src_id = data['src_id']
    # curr_box = data['curr_box']
    box_3d = data['box_3d']

    curr_id =  f'{src_id[0]:06d}'
    if  curr_id != last_id:
        if file is not None:
            file.write(text_to_write)
            file.close()
        last_id = curr_id
        text_to_write = ''
        file = open(os.path.join(results_path, curr_id + '.txt'), 'w')

    x1 = box[0]
    y1 = box[1]
    x2 = box[2]
    y2 = box[3]

    l3d = box_3d[0, 3]
    h3d = box_3d[0, 4]
    w3d = box_3d[0, 5]
    # lhw = data['lhw'] / SCALE_LHW
    # l3d = lhw[0, 0]
    # h3d = lhw[0, 1]
    # w3d = lhw[0, 2]

    # alpha = -1
    alpha = box_3d[0, 7]

    ry3d = ry3d
    x3d = x3d
    y3d = y3d
    z3d = z3d
    score = score
    y3d += h3d/2
    text_to_write += ('Car -1 -1 {:.6f} {:.6f} {:.6f} {:.6f} {:.6f} {:.6f} {:.6f} {:.6f} {:.6f} {:.6f} {:.6f} '
                        + '{:.6f} {:.6f}\n').format(alpha, x1, y1, x2, y2, h3d, w3d, l3d, x3d, y3d, z3d, ry3d, score)

file.close()