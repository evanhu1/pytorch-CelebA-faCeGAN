import torch
import torch.nn as nn
import torch.nn.functional as F

class Generator(nn.Module):
    def __init__(self, classes, d=512):
        super(Generator, self).__init__()
        self.deconv1_1 = nn.ConvTranspose2d(100, d, 4, 1, 0)
        self.deconv1_1_bn = nn.BatchNorm2d(d)
        self.deconv1_2 = nn.ConvTranspose2d(classes, d, 4, 1, 0)
        self.deconv1_2_bn = nn.BatchNorm2d(d)
        self.deconv2 = nn.ConvTranspose2d(d*2, d, 4, 2, 1)
        self.deconv2_bn = nn.BatchNorm2d(d)
        self.deconv3 = nn.ConvTranspose2d(d, d//2, 4, 2, 1)
        self.deconv3_bn = nn.BatchNorm2d(d//2)
        self.deconv4 = nn.ConvTranspose2d(d//2, d//4, 4, 2, 1)
        self.deconv4_bn = nn.BatchNorm2d(d//4)
        self.deconv5 = nn.ConvTranspose2d(d//4, 3, 4, 2, 1)

    def forward(self, input, label):
        x = F.leaky_relu(self.deconv1_1_bn(self.deconv1_1(input)), 0.2)
        y = F.leaky_relu(self.deconv1_2_bn(self.deconv1_2(label)), 0.2)
        x = torch.cat([x, y], 1)
        x = F.leaky_relu(self.deconv2_bn(self.deconv2(x)), 0.2)
        x = F.leaky_relu(self.deconv3_bn(self.deconv3(x)), 0.2)
        x = F.leaky_relu(self.deconv4_bn(self.deconv4(x)), 0.2)
        x = torch.tanh(self.deconv5(x))
        return x

class Discriminator(nn.Module):
    def __init__(self, classes, d=128):
        super(Discriminator, self).__init__()
        self.conv1_1 = nn.Conv2d(3, int(d/2), 4, 2, 1)
        self.conv1_2 = nn.Conv2d(classes, int(d/2), 4, 2, 1)
        self.conv2 = nn.Conv2d(d, d*2, 4, 2, 1)
        self.conv2_bn = nn.BatchNorm2d(d*2)
        self.conv3 = nn.Conv2d(d*2, d*4, 4, 2, 1)
        self.conv3_bn = nn.BatchNorm2d(d*4)
        self.conv4 = nn.Conv2d(d*4, d*8, 4, 2, 1)
        self.conv4_bn = nn.BatchNorm2d(d*8)
        self.conv5 = nn.Conv2d(d*8, 1, 4, 1, 0)

    def forward(self, input, label):
        x = F.leaky_relu(self.conv1_1(input), 0.2)
        y = F.leaky_relu(self.conv1_2(label), 0.2)
        x = torch.cat([x, y], 1)
        x = F.leaky_relu(self.conv2_bn(self.conv2(x)), 0.2)
        x = F.leaky_relu(self.conv3_bn(self.conv3(x)), 0.2)
        x = F.leaky_relu(self.conv4_bn(self.conv4(x)), 0.2)
        x = torch.sigmoid(self.conv5(x))
        return x
