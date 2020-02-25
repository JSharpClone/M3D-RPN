import torch
from torch.utils.data import DataLoader, Subset, ConcatDataset
from torch.utils.tensorboard import SummaryWriter

import numpy as np
from tqdm import tqdm
import os
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt

from lib.motion_data import MotionDataset
from lib.loss.motion_loss import MotionLoss
from lib.dense_alignment import dense_alignment
from models.motion import Motion

np.random.seed(107062513)
torch.manual_seed(107062513)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

EPOCH = 200
BATCH_SIZE = 16
SCALE_TRANSLATION = 0.01

def draw_boxes_2d(box, img, color='red'):
    draw = ImageDraw.Draw(img)
    x1, y1, x2, y2 = box
    draw.rectangle((x1, y1, x2, y2), outline=color, width=5)
    return img

def draw_text_xyz(img, motion):
    # x, y, z = np.round(motion, decimals=2)
    x, z = np.round(motion, decimals=2)
    img = img.convert('RGBA')
    txt = Image.new('RGBA', img.size, (255,255,255,0))
    # get a font
    fnt = ImageFont.load_default()
    # get a drawing context
    d = ImageDraw.Draw(txt)
    d.text((10,10), f'x: {x}', font=fnt, fill=(255,0,0,255))
    # d.text((10,60), f'y: {y}', font=fnt, fill=(0,255,0,255))
    d.text((10,110), f'z: {z}', font=fnt, fill=(255,255,0,255))

    out = Image.alpha_composite(img, txt)
    return out

def draw_circle(img, center, color):
    draw = ImageDraw.Draw(img)
    draw.ellipse((center[0]-10, center[1]-10, center[0]+10, center[1]+10), fill = color)
    return img

def visualize(data, pred_center, prev_center, epoch, phase, save_root):
    motion = data['motion']
    batch_size = motion.size(0)
    data_root = f'/home/jsharp/M3D-RPN/data/kitti_split1/{phase}'
    src_id = data['src_id'].cpu().numpy()
    dst_id = data['dst_id'].cpu().numpy()
    pred_center = pred_center.detach().cpu().numpy()
    prev_center = prev_center.detach().cpu().numpy()
    motion = motion.detach().cpu().numpy() / SCALE_TRANSLATION

    for i in range(batch_size):
        image_path = os.path.join(data_root, 'prev_2', f'{src_id[i]:06d}_01.png')
        image = Image.open(image_path)
        image = draw_text_xyz(image, motion[i])
        image = draw_circle(image, pred_center[i], 'yellow')
        image = draw_circle(image, prev_center[i], 'red')

        save_path = os.path.join(save_root, f'visual/{epoch}')
        os.makedirs(save_path, exist_ok=True)
        image.save(os.path.join(save_path, f'{dst_id[i]:06d}.png'))

# def visualize(data, pred_box, prev_box, epoch, phase, save_root):
#     motion = data['motion']
#     batch_size = motion.size(0)
#     data_root = f'/home/jsharp/M3D-RPN/data/kitti_split1/{phase}'
#     src_id = data['src_id'].cpu().numpy()
#     dst_id = data['dst_id'].cpu().numpy()
#     pred_box = pred_box.detach().cpu().numpy()
#     prev_box = prev_box.detach().cpu().numpy()
#     motion = motion.detach().cpu().numpy() / SCALE_TRANSLATION

#     for i in range(batch_size):
#         image_path = os.path.join(data_root, 'prev_2', f'{src_id[i]:06d}_01.png')
#         image = Image.open(image_path)
#         image = draw_boxes_2d(prev_box[i], image, color='green')
#         image = draw_boxes_2d(pred_box[i], image, color='red')
#         image = draw_text_xyz(image, motion[i])

#         # save_root = f'/home/jsharp/M3D-RPN/motion_visual/{epoch}'
#         save_path = os.path.join(save_root, f'visual/{epoch}')
#         if not os.path.exists(save_path):
#             os.makedirs(save_path)

#         image.save(os.path.join(save_path, f'{dst_id[i]:06d}.png'))

def main():
    device = 'cuda:0'
    save_path = '/home/jsharp/M3D-RPN/output/motion/proj_center_loss/xz'
    model = Motion().to(device)
    # model = torch.load(f'/home/jsharp/M3D-RPN/output/motion/xyzlhw/model_39.pth').to(device)
    train_dataset = MotionDataset(phase='training')
    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, num_workers=0, shuffle=True)
    valid_dataset = MotionDataset(phase='validation', use_predict_val=False)
    valid_dataloader = DataLoader(valid_dataset, batch_size=BATCH_SIZE, num_workers=0, shuffle=False)
    visual_dataset = Subset(valid_dataset, range(100))
    visual_dataloader = DataLoader(visual_dataset, batch_size=BATCH_SIZE, num_workers=0, shuffle=False)

    criterion = MotionLoss()
    writer = SummaryWriter()
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[50,100,150], gamma=0.1)

    step = 0
    with tqdm(total=EPOCH*len(train_dataset), ascii=True) as pbar:
        for e in range(EPOCH):  
            model.train()
            for  data in iter(train_dataloader):
                for k, v in data.items():
                    data[k] = v.to(device)

                optimizer.zero_grad()
                data = model(data)
                losses, _, _ = criterion(data)
                loss = losses['total_loss']
                loss.backward()
                optimizer.step()

                writer.add_scalar('train total loss', losses['total_loss'].item(), global_step=step)
                step += 1
                pbar.update(BATCH_SIZE)
                pbar.set_postfix({'Epoch': e})

            scheduler.step()

            if (e + 1) % 5 ==0:
                valid_total_loss = []
                model.eval()

                # validation
                for  data in iter(valid_dataloader):
                    for k, v in data.items():
                        data[k] = v.to(device)

                    data = model(data)
                    losses, pred_center, prev_center = criterion(data)
                    valid_total_loss.append(losses['total_loss'].item())
        
                writer.add_scalar('valid total loss', sum(valid_total_loss)/len(valid_total_loss), global_step=e)
            
                # visualization
                for  data in iter(visual_dataloader):
                    for k, v in data.items():
                        data[k] = v.to(device)
                    
                    data = model(data)
                    _, pred_center, prev_center = criterion(data)
                    visualize(data, pred_center, prev_center, e, 'validation', save_root=save_path)
        
                    # dense_alignment(motion, data, phase='validation')
                torch.save(model, os.path.join(save_path, f'model_{e:02d}.pth'))

    writer.close()

if __name__ == '__main__':
    main()