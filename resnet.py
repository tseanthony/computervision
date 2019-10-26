import torch
import torch.nn as nn
import torch.nn.functional as F

nclasses = 43 # GTSRB as 43 classes®

class Block(nn.Module):

    def __init__(self, n_feature_maps, down_sample=False):
        super(Block, self).__init__()

        self.bn1 = nn.BatchNorm2d(int(n_feature_maps/2 if down_sample else n_feature_maps))
        self.conv1 = nn.Conv2d(int(n_feature_maps/2 if down_sample else n_feature_maps),
                               n_feature_maps,
                               kernel_size=3,
                               stride=(2 if down_sample else 1),
                               padding=1, bias=False)

        self.bn2 = nn.BatchNorm2d(n_feature_maps)
        self.conv2 = nn.Conv2d(n_feature_maps, n_feature_maps, kernel_size=3, stride=1, padding=1, bias=False)

        self.down_sample = down_sample

        self.conv_down = None
        self.bn_down = None

        if down_sample:
            self.conv_down = nn.Conv2d(int(n_feature_maps/2), n_feature_maps, kernel_size=1, stride=2, bias=False)


    def forward(self, x):

        identity = x

        res = self.bn1(x)
        res = F.relu(res)

        if self.down_sample:
            identity = self.conv_down(res)

        res = self.conv1(res)

        res = self.bn2(res)
        res = F.relu(res)
        res = self.conv2(res)

        return identity + res


# RESNET (initial layers of resnet-34 omitted)
class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        # Layer 1
        self.conv0 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1, bias=False)

        # Layer 2 (32 x 32)
        self.block01 = Block(n_feature_maps=32)
        self.block02 = Block(n_feature_maps=32)

        # Layer 3 (16 x 16)
        self.block07 = Block(n_feature_maps=64, down_sample=True)
        self.block08 = Block(n_feature_maps=64)

        # Layer 4 (8 x 8)
        self.block13 = Block(n_feature_maps=128, down_sample=True)
        self.block14 = Block(n_feature_maps=128)

        # Layer 5 (4 x 4)
        self.block19 = Block(n_feature_maps=256, down_sample=True)
        self.block20 = Block(n_feature_maps=256)

        self.fc = nn.Linear(256, nclasses)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.conv0(x)

        x = self.block01(x)
        x = self.block02(x)

        x = self.block07(x)
        x = self.block08(x)

        x = self.block13(x)
        x = self.block14(x)

        x = self.block19(x)
        x = self.block20(x)

        x = F.avg_pool2d(x, kernel_size=(4, 4))

        x = torch.flatten(x, start_dim=1)
        x = self.fc(x)

        return F.log_softmax(x, dim=0)
