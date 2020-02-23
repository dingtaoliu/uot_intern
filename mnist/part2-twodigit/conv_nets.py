import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from train_utils import Flatten

num_classes = 10


# Split CNN model with separate weights for the top and bottom digit
class SCNN(nn.Module):

    def __init__(self, input_dimension):
        super(SCNN, self).__init__()

        # conv layer 1 for digit 1: 1 x 28 x 28 -> 32 x 28 x 28
        self.conv1_1 = nn.Conv2d(1, 32, 3, stride=1, padding=1)
        # conv layer 1 for digit 2: 1 x 28 x 28 -> 32 x 28 x 28
        self.conv1_2 = nn.Conv2d(1, 32, 3, stride=1, padding=1)

        # conv layer 2 for digit 1: 32 x 28 x 28 -> 64 x 28 x 28
        self.conv2_1 = nn.Conv2d(32, 64, 3, stride=1, padding=1)
        # conv layer 2 for digit 2: 32 x 28 x 28 -> 64 x 28 x 28
        self.conv2_2 = nn.Conv2d(32, 64, 3, stride=1, padding=1)

        # flatten feature maps
        self.flatten = Flatten()

        # linear layer for digit 1: 28 * 28 * 64 -> 10
        self.linear1 = nn.Linear(28 * 28 * 64, num_classes)
        # linear layer for digit 2: 28 * 28 * 64 -> 10
        self.linear2 = nn.Linear(28 * 28 * 64, num_classes)

        num_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print("Model contains {} trainable parameters".format(num_params))

    def forward(self, x):
        # pass the top 28 rows of the image to conv layer 1
        # apply relu and max pool
        h1 = F.relu(self.conv1_1(x[:, :, :28, :]))
        h1 = F.max_pool2d(h1, kernel_size=3, stride=1, padding=1)

        # pass the bottom 28 rows of the image to conv layer 1
        # apply relu and max pool
        h2 = F.relu(self.conv1_2(x[:, :, 14:, :]))
        h2 = F.max_pool2d(h2, kernel_size=3, stride=1, padding=1)

        # pass the top 28 rows of the image to conv layer 2
        # apply relu and max pool
        h1 = F.relu(self.conv2_1(h1))
        h1 = F.max_pool2d(h1, kernel_size=3, stride=1, padding=1)

        # pass the top 28 rows of the image to conv layer 2
        # apply relu and max pool
        h2 = F.relu(self.conv2_2(h2))
        h2 = F.max_pool2d(h2, kernel_size=3, stride=1, padding=1)

        # flatten the feature maps for top and bottom crops 
        h1_flat = self.flatten(h1)
        h2_flat = self.flatten(h2)

        # compute logits separately for top and bottom crops
        logits1 = self.linear1(h1_flat)
        logits2 = self.linear2(h2_flat)

        # normalize the logits to probabilities
        out_first_digit = F.softmax(logits1, dim=1)
        out_second_digit = F.softmax(logits2, dim=1)

        return out_first_digit, out_second_digit


# Split CNN model with shared weights for top and bottom digits
class SCNN2(nn.Module):

    def __init__(self, input_dimension):
        super(SCNN2, self).__init__()

        # conv layer 1: 1 x 28 x 28 -> 32 x 28 x 28
        self.conv1 = nn.Conv2d(1, 32, 3, stride=1, padding=1)

        # conv layer 2: 32 x 28 x 28 -> 64 x 28 x 28
        self.conv2 = nn.Conv2d(32, 64, 3, stride=1, padding=1)

        # conv layer 3: 64 x 28 x 28 -> 128 x 28 x 28
        self.conv3 = nn.Conv2d(64, 128, 3, stride=1, padding=1)

        # flatten layer
        self.flatten = Flatten()

        # linear layer 1: 14 * 14 * 128 -> 128
        self.linear1 = nn.Linear(14 * 14 * 128, 128)
        # linear layer 2: 128 -> 10
        self.linear2 = nn.Linear(128, num_classes)

        num_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print("Model contains {} trainable parameters".format(num_params))

    def forward(self, x):
        # pass the top 28 rows of the image to conv layer 1
        # apply relu and max pool
        h1 = F.relu(self.conv1(x[:, :, :28, :]))
        h1 = F.max_pool2d(h1, kernel_size=3, stride=1, padding=1)

        # pass the bottom 28 rows of the image to conv layer 1
        # apply relu and max pool
        h2 = F.relu(self.conv1(x[:, :, 14:, :]))
        h2 = F.max_pool2d(h2, kernel_size=3, stride=1, padding=1)

        # pass the top 28 rows of the image to conv layer 2
        # apply relu and downsample with max pool 
        h1 = F.relu(self.conv2(h1))
        h1 = F.max_pool2d(h1, kernel_size=2, stride=2, padding=0)

        # pass the bottom 28 rows of the image to conv layer 2
        # apply relu and downsample with max pool 
        h2 = F.relu(self.conv2(h2))
        h2 = F.max_pool2d(h2, kernel_size=2, stride=2, padding=0)

        # pass the top and bottom crops to conv layer 3
        # apply relu
        h1 = F.relu(self.conv3(h1))
        h2 = F.relu(self.conv3(h2))

        # flatten the top and bottom crops
        h1 = self.flatten(h1)
        h2 = self.flatten(h2)

        # compute the first linear activations
        h1 = torch.sigmoid(self.linear1(h1))
        h2 = torch.sigmoid(self.linear1(h2))

        # compute logits for top and bottom digits
        logits1 = self.linear2(h1)
        logits2 = self.linear2(h2)

        # apply softmax to compute probabilities
        out_first_digit = F.softmax(logits1, dim=1)
        out_second_digit = F.softmax(logits2, dim=1)

        return out_first_digit, out_second_digit

# Split CNN model with depthwise separable filters
class SCNN3(nn.Module):

    def __init__(self, input_dimension):
        super(SCNN3, self).__init__()

        # conv layer 1: 1 x 28 x 28 -> 32 x 28 x 28
        self.conv1 = nn.Conv2d(1, 32, 3, stride=1, padding=1)

        # depthwise separable layer 2
        # depthwise 1: 32 x 28 x 28 -> 32 x 28 x 28
        self.depthwise1 = nn.Conv2d(32, 32, 3, padding=1, groups=32)
        # pointwise 1: 32 x 28 x 28 -> 64 x 28 x 28
        self.pointwise1 = nn.Conv2d(32, 64, 1)

        # depthwise separable layer 3
        # depthwise 2: 64 x 28 x 28 -> 64 x 28 x 28
        self.depthwise2 = nn.Conv2d(64, 64, 3, padding=1, groups=64)
        # pointwise 2: 64 x 28 x 28 -> 128 x 28 x 28
        self.pointwise2 = nn.Conv2d(64, 128, 1)

        # flatten layer
        self.flatten = Flatten()

        # linear layer 1: 14 * 14 * 128 -> 128
        self.linear1 = nn.Linear(14 * 14 * 128, 128)
        # linear layer 2: 128 -> 10
        self.linear2 = nn.Linear(128, num_classes)

        num_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print("Model contains {} trainable parameters".format(num_params))


    def forward(self, x):
        # pass the top 28 rows of the image to conv layer 1
        # apply relu and max pool
        h1 = F.relu(self.conv1(x[:, :, :28, :]))
        h1 = F.max_pool2d(h1, kernel_size=3, stride=1, padding=1)

        # pass the bottom 28 rows of the image to conv layer 1
        # apply relu and max pool
        h2 = F.relu(self.conv1(x[:, :, 14:, :]))
        h2 = F.max_pool2d(h2, kernel_size=3, stride=1, padding=1)

        # apply depthwise and pointwise convolution
        # apply relu and max pool with downsampling
        h1 = F.relu(self.pointwise1(self.depthwise1(h1)))
        h1 = F.max_pool2d(h1, kernel_size=2, stride=2)

        # apply depthwise and pointwise convolution
        # apply relu and max pool with downsampling
        h2 = F.relu(self.pointwise1(self.depthwise1(h2)))
        h2 = F.max_pool2d(h2, kernel_size=2, stride=2)

        # apply depthwise and pointwise convolution
        # apply relu
        h1 = F.relu(self.pointwise2(self.depthwise2(h1)))
        h2 = F.relu(self.pointwise2(self.depthwise2(h2)))

        # flatten top and bottom crop feature maps
        h1 = self.flatten(h1)
        h2 = self.flatten(h2)

        # compute linear 1 activations
        h1 = torch.sigmoid(self.linear1(h1))
        h2 = torch.sigmoid(self.linear1(h2))

        # compute logits for top and bottom crop
        logits1 = self.linear2(h1)
        logits2 = self.linear2(h2)

        # compute probabilities from logits
        out_first_digit = F.softmax(logits1, dim=1)
        out_second_digit = F.softmax(logits2, dim=1)

        return out_first_digit, out_second_digit

# Baseline CNN model with depthwise separable filters
class CNN2(nn.Module):

    def __init__(self, input_dimension):
        super(CNN2, self).__init__()

        # conv layer 1: 1 x 42 x 28 -> 32 x 42 x 28
        self.conv1 = nn.Conv2d(1, 32, 3, stride=1, padding=1)

        # depthwise separable layer 2
        # depthwise 1: 32 x 42 x 28 -> 32 x 42 x 28
        self.depthwise1 = nn.Conv2d(32, 32, 3, padding=1, groups=32)
        # pointwise 1: 32 x 42 x 28 -> 64 x 42 x 28
        self.pointwise1 = nn.Conv2d(32, 64, 1)

        # depthwise separable layer 3
        # depthwise 1: 64 x 42 x 28 -> 64 x 42 x 28
        self.depthwise2 = nn.Conv2d(64, 64, 3, padding=1, groups=64)
        # pointwise 1: 64 x 42 x 28 -> 128 x 42 x 28
        self.pointwise2 = nn.Conv2d(64, 128, 1)
        
        # flatten layer
        self.flatten = Flatten()

        # linear layer 1: 21 * 14 * 128 -> 20
        self.linear = nn.Linear(21 * 14 * 128, num_classes * 2)

        num_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print("Model contains {} trainable parameters".format(num_params))

    def forward(self, x):
        # pass the image to conv layer 1
        # apply relu and max pool
        h1 = F.relu(self.conv1(x))
        h1 = F.max_pool2d(h1, kernel_size=3, stride=1, padding=1)

        # apply depthwise and pointwise convolution
        # apply relu and max pool
        h2 = F.relu(self.pointwise1(self.depthwise1(h1)))
        h2 = F.max_pool2d(h2, kernel_size=3, stride=1, padding=1)

        # apply depthwise and pointwise convolution
        # apply relu and max pool with downsampling
        h3 = F.relu(self.pointwise2(self.depthwise2(h2)))
        h3 = F.max_pool2d(h3, kernel_size=2, stride=2)

        # flatten the feature maps
        flat = self.flatten(h3)

        # compute logits from linear layer
        logits = self.linear(flat)

        # compute probabilities from the first and last 10 logits
        out_first_digit = F.softmax(logits[:, :num_classes], dim=1)
        out_second_digit = F.softmax(logits[:, num_classes:], dim=1)

        return out_first_digit, out_second_digit