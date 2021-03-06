import torch
import torch.nn as nn

def conv_relu(in_channel, out_channel, kernel_size, stride):
    return nn.Sequential(
            nn.Conv2d(in_channel, out_channel, kernel_size, stride=stride, padding=int(kernel_size/2)),
            nn.ReLU(inplace=True),
            )
def fc_relu(in_channel, out_channel):
    return nn.Sequential(
            nn.Linear(in_channel, out_channel),
            nn.ReLU(inplace=True),
            )

VECTOR_SIZE = 2
SCALE_TRANSLATION = 0.01

class Motion(nn.Module):

    def __init__(self):
        super(Motion, self).__init__()

        self.conv1 = conv_relu(6, 16, 7, stride=2)
        self.conv2 = conv_relu(16, 32, 5, stride=2)
        self.conv3 = conv_relu(32, 64, 3, stride=2)
        self.conv4 = conv_relu(64, 128, 3, stride=2)
        self.conv5 = conv_relu(128, 256, 3, stride=2)
        self.conv6 = conv_relu(256, 256, 3, stride=2)
        self.conv7 = conv_relu(256, 256, 3, stride=2)
        # self.motion_predict = nn.Sequential(
        #     nn.Linear(1024+3, 512),
        #     nn.Linear(512, VECTOR_SIZE)
        # )
        self.motion_predict = nn.Linear(1024+3, VECTOR_SIZE)
        
        
    def forward(self, data):
        curr_image = data['curr_image']
        prev_image = data['prev_image']
        image = torch.cat([curr_image, prev_image], dim=1)

        ego_motion_t = data['ego_motion_t'] * SCALE_TRANSLATION

        batch_size = curr_image.size(0)
        # (128, 416)
        x = self.conv1(image)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.conv6(x)
        x = self.conv7(x)

        x = x.view(batch_size, -1)
        x = torch.cat([x, ego_motion_t], dim=1)
        motion = self.motion_predict(x)

        data['motion'] = motion

        return data

if __name__ == "__main__":
    device = 'cuda'
    model = Motion().to(device)
    image = torch.rand((1, 3, 512, 1760)).to(device)
    motion = model(image)
    print(motion)


