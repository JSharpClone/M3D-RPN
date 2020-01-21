import torch
from torch.utils.data import DataLoader, Subset
from torch.utils.tensorboard import SummaryWriter

import numpy as np
from tqdm import tqdm
import os
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt

from lib.motion_data import MotionDataset
from lib.loss.motion_loss import MotionLoss
from models.motion import Motion

np.random.seed(107062513)
torch.manual_seed(107062513)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

EPOCH = 50
BATCH_SIZE = 16

def draw_boxes_2d(box, img, color='red'):
    draw = ImageDraw.Draw(img)
    x1, y1, x2, y2 = box
    draw.rectangle((x1, y1, x2, y2), outline=color, width=5)
    return img

def visualize(motion, data, pred_box, prev_box, epoch, phase):
    batch_size = motion.size(0)
    data_root = f'/home/jsharp/M3D-RPN/data/kitti_split1/{phase}'
    src_id = data['src_id'].cpu().numpy()
    dst_id = data['dst_id'].cpu().numpy()
    pred_box = pred_box.detach().cpu().numpy()
    prev_box = prev_box.detach().cpu().numpy()

    for i in range(batch_size):
        image_path = os.path.join(data_root, 'prev_2', f'{src_id[i]:06d}_01.png')
        image = Image.open(image_path)
        image = draw_boxes_2d(prev_box[i], image, color='green')
        image = draw_boxes_2d(pred_box[i], image, color='red')
        fig, ax = plt.subplots()
        ax.imshow(image)

        save_root = f'/home/jsharp/M3D-RPN/motion_visual/{phase}/{epoch}'
        if not os.path.exists(save_root):
            os.makedirs(save_root)

        fig.savefig(os.path.join(save_root, f'{dst_id[i]:06d}.png'))
        plt.close(fig)

def main():
    device = 'cuda:0'
    model_path = '/home/jsharp/M3D-RPN/output/motion'
    model = Motion().to(device)
    # model = torch.load(f'/home/jsharp/M3D-RPN/output/motion/model_49.pth').to(device)
    train_dataset = MotionDataset(phase='training')
    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, num_workers=0, shuffle=True)
    visual_dataset = Subset(train_dataset, range(50))
    visual_dataloader = DataLoader(visual_dataset, batch_size=BATCH_SIZE, num_workers=0, shuffle=False)
    valid_dataset = MotionDataset(phase='validation')
    valid_dataset = Subset(valid_dataset, range(100))
    valid_dataloader = DataLoader(valid_dataset, batch_size=BATCH_SIZE, num_workers=0, shuffle=False)
    criterion = MotionLoss()
    writer = SummaryWriter()
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-5)

    step = 0
    with tqdm(total=EPOCH*len(train_dataset), ascii=True) as pbar:
        for e in range(EPOCH):  
            model.train()
            for  data in iter(train_dataloader):
                for k, v in data.items():
                    data[k] = v.to(device)

                optimizer.zero_grad()
                motion = model(data)
                loss, _, _ = criterion(motion, data)
                loss.backward()
                optimizer.step()

                writer.add_scalar('train loss', loss.item(), global_step=step)
                step += 1
                pbar.update(BATCH_SIZE)
                pbar.set_postfix({'Epoch': e})
    
            if (e + 1) % 5 ==0:
                valid_loss = []
                model.eval()

                # validation
                for  data in iter(valid_dataloader):
                    for k, v in data.items():
                        data[k] = v.to(device)
                    
                    motion = model(data)
                    loss, pred_box, prev_box = criterion(motion, data)
                    valid_loss.append(loss.item())
                    visualize(motion, data, pred_box, prev_box, e, 'validation')
                writer.add_scalar('valid loss', sum(valid_loss)/len(valid_loss), global_step=e)

                # visualization
                for  data in iter(visual_dataloader):
                    for k, v in data.items():
                        data[k] = v.to(device)
                    
                    motion = model(data)
                    _, pred_box, prev_box = criterion(motion, data)
                    visualize(motion, data, pred_box, prev_box, e, 'training')

            torch.save(model, os.path.join(model_path, f'model_{e:02d}.pth'))
    writer.close()

if __name__ == '__main__':
    main()