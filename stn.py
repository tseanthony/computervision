import torch
import torch.nn as nn
import torch.nn.functional as F

nclasses = 43 # GTSRB as 43 classes®
image_size = 48

class Stn(nn.Module):
    def __init__(self, ipt_dim, ipt_size, dims=list([200, 300, 200])):
        super(Stn, self).__init__()

        self.dims = dims

        # Spatial transformer localization-network
        self.localization = nn.Sequential(
            nn.Conv2d(ipt_dim, dims[0], padding=2, kernel_size=5),
            nn.ReLU(True),
            nn.MaxPool2d(2, stride=2),
            nn.BatchNorm2d(dims[0]),

            nn.Conv2d(dims[0], dims[1], padding=2, kernel_size=5),
            nn.ReLU(True),
            nn.MaxPool2d(2, stride=2),
            nn.BatchNorm2d(dims[1]),
        )

        # Regressor for the 3 * 2 affine matrix
        self.curr_size = (ipt_size // 2 // 2 // 2) ** 2

        self.fc_loc = nn.Sequential(
            nn.Linear(dims[1] * self.curr_size, dims[2]),
            nn.ReLU(True),
            nn.Linear(dims[2], 3 * 2)
        )

        # Initialize the weights/bias with identity transformation
        self.fc_loc[2].weight.data.zero_()
        self.fc_loc[2].bias.data.copy_(torch.tensor([1, 0, 0, 0, 1, 0],
                                                    dtype=torch.float))

    # Spatial transformer network forward function
    def forward(self, x):
        xs = F.max_pool2d(x, 2)
        xs = self.localization(xs)
        xs = torch.flatten(xs, start_dim=1)
        theta = self.fc_loc(xs)
        theta = theta.view(-1, 2, 3)

        grid = F.affine_grid(theta, x.size())
        x = F.grid_sample(x, grid)

        return x

class Net(nn.Module):
    def __init__(self):

        conv1_kernel_size = 7
        conv2_kernel_size = 4
        conv3_kernel_size = 4

        super(Net, self).__init__()

        self.stn1 = Stn(1, image_size)

        self.conv1 = nn.Conv2d(1, 150, padding=2,
                               kernel_size=conv1_kernel_size)
        self.bn2 = nn.BatchNorm2d(150)

        self.conv2 = nn.Conv2d(150, 200, padding=2,
                               kernel_size=conv2_kernel_size)
        self.bn3 = nn.BatchNorm2d(200)

        current_size = (image_size + 5 - conv1_kernel_size) // 2
        current_size = (current_size + 5 - conv2_kernel_size) // 2

        self.stn2 = Stn(200, current_size, [150, 150, 150])

        self.conv3 = nn.Conv2d(200, 300, padding=2,
                               kernel_size=conv3_kernel_size)
        self.bn4 = nn.BatchNorm2d(300)

        current_size = (current_size + 5 - conv3_kernel_size) // 2

        self.fc1 = nn.Linear(current_size ** 2 * 300, 350)
        self.fc2 = nn.Linear(350, nclasses)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight,
                                        mode='fan_out',
                                        nonlinearity='relu')

    def forward(self, x):
        x = self.stn1(x)
        x = self.bn2(F.max_pool2d(F.relu(self.conv1(x)), 2))
        x = self.bn3(F.max_pool2d(F.relu(self.conv2(x)), 2))

        x = self.stn2(x)

        x = self.bn4(F.max_pool2d(F.relu(self.conv3(x)), 2))
        x = torch.flatten(x, start_dim=1)

        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)



