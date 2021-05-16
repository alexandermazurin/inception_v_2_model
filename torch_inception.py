import torch
import torch.nn as nn
import torch.nn.functional as F


class Inception_Block(nn.Module):
    def __init__(self, input_channels, output_channels):
        super(Inception_Block, self).__init__()
        self.branch_1_conv1 = nn.Conv2d(input_channels, output_channels[0], 1)
        self.branch_1_norm1 = nn.BatchNorm2d(output_channels[0])

        self.branch_2_conv1 = nn.Conv2d(input_channels, output_channels[1], 1)
        self.branch_2_norm1 = nn.BatchNorm2d(output_channels[1])
        self.branch_2_conv2 = nn.Conv2d(output_channels[1], output_channels[2], 3, padding=1)
        self.branch_2_norm2 = nn.BatchNorm2d(output_channels[2])

        self.branch_3_conv1 = nn.Conv2d(input_channels, output_channels[3], 1)
        self.branch_3_norm1 = nn.BatchNorm2d(output_channels[3])
        self.branch_3_conv2 = nn.Conv2d(output_channels[3], output_channels[4], 3, padding=1)
        self.branch_3_norm2 = nn.BatchNorm2d(output_channels[4])
        self.branch_3_conv3 = nn.Conv2d(output_channels[4], output_channels[4], 3, padding=1)
        self.branch_3_norm3 = nn.BatchNorm2d(output_channels[4])

        self.branch_4_pool = nn.MaxPool2d(3, stride=1, padding=1)
        self.branch_4_conv1 = nn.Conv2d(input_channels, output_channels[5], 1)
        self.branch_4_norm1 = nn.BatchNorm2d(output_channels[5])

    def forward(self, x):
        part_1 = self.branch_1_conv1(x)
        part_1 = F.relu(self.branch_1_norm1(part_1))

        part_2 = self.branch_2_conv1(x)
        part_2 = F.relu(self.branch_2_norm1(part_2))
        part_2 = self.branch_2_conv2(part_2)
        part_2 = F.relu(self.branch_2_norm2(part_2))

        part_3 = self.branch_3_conv1(x)
        part_3 = F.relu(self.branch_3_norm1(part_3))
        part_3 = self.branch_3_conv2(part_3)
        part_3 = F.relu(self.branch_3_norm2(part_3))
        part_3 = self.branch_3_conv3(part_3)
        part_3 = F.relu(self.branch_3_norm3(part_3))

        part_4 = self.branch_4_pool(x)
        part_4 = self.branch_4_conv1(part_4)
        part_4 = F.relu(self.branch_4_norm1(part_4))

        x = torch.cat((part_1, part_2, part_3, part_4), dim=1)
        return x


class Inception_Block_Shortened(nn.Module):
    def __init__(self, input_channels, output_channels):
        super(Inception_Block_Shortened, self).__init__()
        self.branch_1_conv1 = nn.Conv2d(input_channels, output_channels[0], 1)
        self.branch_1_norm1 = nn.BatchNorm2d(output_channels[0])
        self.branch_1_conv2 = nn.Conv2d(output_channels[0], output_channels[1], 3, stride=2, padding=1)
        self.branch_1_norm2 = nn.BatchNorm2d(output_channels[1])

        self.branch_2_conv1 = nn.Conv2d(input_channels, output_channels[2], 1)
        self.branch_2_norm1 = nn.BatchNorm2d(output_channels[2])
        self.branch_2_conv2 = nn.Conv2d(output_channels[2], output_channels[3], 3, padding=1)
        self.branch_2_norm2 = nn.BatchNorm2d(output_channels[3])
        self.branch_2_conv3 = nn.Conv2d(output_channels[3], output_channels[3], 3, stride=2, padding=1)
        self.branch_2_norm3 = nn.BatchNorm2d(output_channels[3])

        self.branch_3_pool = nn.MaxPool2d(3, stride=2, padding=1)

    def forward(self, x):
        part_1 = self.branch_1_conv1(x)
        part_1 = F.relu(self.branch_1_norm1(part_1))
        part_1 = self.branch_1_conv2(part_1)
        part_1 = F.relu(self.branch_1_norm2(part_1))

        part_2 = self.branch_2_conv1(x)
        part_2 = F.relu(self.branch_2_norm1(part_2))
        part_2 = self.branch_2_conv2(part_2)
        part_2 = F.relu(self.branch_2_norm2(part_2))
        part_2 = self.branch_2_conv3(part_2)
        part_2 = F.relu(self.branch_2_norm3(part_2))

        part_3 = self.branch_3_pool(x)
        x = torch.cat((part_1, part_2, part_3), dim=1)
        return x


def create_network_structure():
    layers = ['nn.Conv2d(3, 64, 7, stride=2, padding=3)',
              'nn.BatchNorm2d(64)',
              'nn.MaxPool2d(3, stride=2, padding=1)',
              'nn.Conv2d(64, 64, 1)',
              'nn.BatchNorm2d(64)',
              'nn.Conv2d(64, 192, 3, padding=1)',
              'nn.BatchNorm2d(192)',
              'nn.MaxPool2d(3, stride=2, padding=1)',
              'Inception_Block(input_channels=192, output_channels=[64, 64, 64, 64, 96, 32])',
              'Inception_Block(input_channels=256, output_channels=[64, 64, 96, 64, 96, 64])',
              'Inception_Block_Shortened(input_channels=320, output_channels=[128, 160, 64, 96])',
              'Inception_Block(input_channels=576, output_channels=[224, 64, 96, 96, 128, 128])',
              'Inception_Block(input_channels=576, output_channels=[192, 96, 128, 96, 128, 128])',
              'Inception_Block(input_channels=576, output_channels=[160, 128, 160, 128, 160, 96])',
              'Inception_Block(input_channels=576, output_channels=[96, 128, 192, 160, 192, 96])',
              'Inception_Block_Shortened(input_channels=576, output_channels=[128, 192, 192, 256])',
              'Inception_Block(input_channels=1024, output_channels=[352, 192, 320, 160, 224, 128])',
              'Inception_Block(input_channels=1024, output_channels=[352, 192, 320, 192, 224, 128])',
              'nn.AvgPool2d(7)',
              'nn.Dropout(dropout_rate)',
              'nn.Conv2d(1024, num_classes, 1)']
    return layers


class Inception_Net(nn.Module):
    def __init__(self, dropout_rate, num_classes):
        super(Inception_Net, self).__init__()
        self.structure = create_network_structure()
        self.layers = []
        for i in range(len(self.structure)):
            exec('self.conv_' + str(i) + '=' + self.structure[i])
            exec('self.layers.append(self.conv_' + str(i) + ')')

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        x = x.view(x.shape[0], x.shape[1])
        return x
