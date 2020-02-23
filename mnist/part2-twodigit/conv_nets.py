import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from train_utils import Flatten

num_classes = 10


# Split CNN model
class SCNN(nn.Module):

    def __init__(self, input_dimension):
        super(SCNN, self).__init__()

        # input layer: preserve the original dimensions
        self.conv1_1 = nn.Conv2d(1, 32, 3, stride=1, padding=1)
        self.conv1_2 = nn.Conv2d(1, 32, 3, stride=1, padding=1)

        # conv layer 
        self.conv2_1 = nn.Conv2d(32, 64, 3, stride=1, padding=1)
        self.conv2_2 = nn.Conv2d(32, 64, 3, stride=1, padding=1)

        self.flatten = Flatten()

        self.linear1 = nn.Linear(28 * 28 * 64, num_classes)
        self.linear2 = nn.Linear(28 * 28 * 64, num_classes)

    def forward(self, x):

        h1 = F.relu(self.conv1_1(x[:, :, :28, :]))
        h1 = F.max_pool2d(h1, kernel_size=3, stride=1, padding=1)

        h2 = F.relu(self.conv1_2(x[:, :, 14:, :]))
        h2 = F.max_pool2d(h2, kernel_size=3, stride=1, padding=1)

        h1 = F.relu(self.conv2_1(h1))
        h1 = F.max_pool2d(h1, kernel_size=3, stride=1, padding=1)

        h2 = F.relu(self.conv2_2(h2))
        h2 = F.max_pool2d(h2, kernel_size=3, stride=1, padding=1)

        h1_flat = self.flatten(h1)
        h2_flat = self.flatten(h2)

        logits1 = self.linear1(h1_flat)
        logits2 = self.linear2(h2_flat)

        out_first_digit = F.softmax(logits1, dim=1)
        out_second_digit = F.softmax(logits2, dim=1)

        return out_first_digit, out_second_digit


# Split CNN model
class SCNN2(nn.Module):

    def __init__(self, input_dimension):
        super(SCNN2, self).__init__()

        # input layer: preserve the original dimensions
        self.conv1 = nn.Conv2d(1, 32, 3, stride=1, padding=1)

        # conv layer 
        self.conv2 = nn.Conv2d(32, 64, 3, stride=1, padding=1)

        self.conv3 = nn.Conv2d(64, 128, 3, stride=1, padding=1)

        self.flatten = Flatten()

        self.linear1 = nn.Linear(14 * 14 * 128, 128)
        self.linear2 = nn.Linear(128, num_classes)

        num_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print("Model contains {} trainable parameters".format(num_params))



    def forward(self, x):

        h1 = F.relu(self.conv1(x[:, :, :28, :]))
        h1 = F.max_pool2d(h1, kernel_size=3, stride=1, padding=1)

        h2 = F.relu(self.conv1(x[:, :, 14:, :]))
        h2 = F.max_pool2d(h2, kernel_size=3, stride=1, padding=1)

        h1 = F.relu(self.conv2(h1))
        h1 = F.max_pool2d(h1, kernel_size=2, stride=2, padding=0)

        h2 = F.relu(self.conv2(h2))
        h2 = F.max_pool2d(h2, kernel_size=2, stride=2, padding=0)

        h1 = F.relu(self.conv3(h1))
        h2 = F.relu(self.conv3(h2))

        h1 = self.flatten(h1)
        h2 = self.flatten(h2)

        h1 = torch.sigmoid(self.linear1(h1))
        h2 = torch.sigmoid(self.linear1(h2))

        logits1 = self.linear2(h1)
        logits2 = self.linear2(h2)

        out_first_digit = F.softmax(logits1, dim=1)
        out_second_digit = F.softmax(logits2, dim=1)

        return out_first_digit, out_second_digit

# Split CNN model
class SCNN3(nn.Module):

    def __init__(self, input_dimension):
        super(SCNN3, self).__init__()

        # depthwise separable layer 1
        self.conv1 = nn.Conv2d(1, 32, 3, stride=1, padding=1)

        # depthwise separable layer 2
        self.depthwise1 = nn.Conv2d(32, 32, 3, padding=1, groups=32)
        self.pointwise1 = nn.Conv2d(32, 64, 1)

        # depthwise separable layer 3
        self.depthwise2 = nn.Conv2d(64, 64, 3, padding=1, groups=64)
        self.pointwise2 = nn.Conv2d(64, 128, 1)

        self.flatten = Flatten()

        self.linear1 = nn.Linear(14 * 14 * 128, 128)
        self.linear2 = nn.Linear(128, num_classes)

        num_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print("Model contains {} trainable parameters".format(num_params))


    def forward(self, x):
        h1 = F.relu(self.conv1(x[:, :, :28, :]))
        h1 = F.max_pool2d(h1, kernel_size=3, stride=1, padding=1)

        h2 = F.relu(self.conv1(x[:, :, 14:, :]))
        h2 = F.max_pool2d(h2, kernel_size=3, stride=1, padding=1)

        h1 = F.relu(self.pointwise1(self.depthwise1(h1)))
        h1 = F.max_pool2d(h1, kernel_size=2, stride=2)

        h2 = F.relu(self.pointwise1(self.depthwise1(h2)))
        h2 = F.max_pool2d(h2, kernel_size=2, stride=2)

        h1 = F.relu(self.pointwise2(self.depthwise2(h1)))
        h2 = F.relu(self.pointwise2(self.depthwise2(h2)))

        h1 = self.flatten(h1)
        h2 = self.flatten(h2)

        h1 = torch.sigmoid(self.linear1(h1))
        h2 = torch.sigmoid(self.linear1(h2))

        logits1 = self.linear2(h1)
        logits2 = self.linear2(h2)

        out_first_digit = F.softmax(logits1, dim=1)
        out_second_digit = F.softmax(logits2, dim=1)

        return out_first_digit, out_second_digit

# Split CNN model
class CNN2(nn.Module):

    def __init__(self, input_dimension):
        super(CNN2, self).__init__()

        # input layer: preserve the original dimensions
        self.conv1 = nn.Conv2d(1, 32, 3, stride=1, padding=1)

        # conv layer 
        self.depthwise1 = nn.Conv2d(32, 32, 3, padding=1, groups=32)
        self.pointwise1 = nn.Conv2d(32, 64, 1)

        self.depthwise2 = nn.Conv2d(64, 64, 3, padding=1, groups=64)
        self.pointwise2 = nn.Conv2d(64, 128, 1)
        

        self.flatten = Flatten()

        self.linear = nn.Linear(21 * 14 * 128, num_classes * 2)

        num_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print("Model contains {} trainable parameters".format(num_params))


    def forward(self, x):

        h1 = F.relu(self.conv1(x))
        h1 = F.max_pool2d(h1, kernel_size=3, stride=1, padding=1)

        h2 = F.relu(self.pointwise1(self.depthwise1(h1)))
        h2 = F.max_pool2d(h2, kernel_size=3, stride=1, padding=1)

        h3 = F.relu(self.pointwise2(self.depthwise2(h2)))
        h3 = F.max_pool2d(h3, kernel_size=2, stride=2)
        #print(h3.shape)

        flat = self.flatten(h3)

        logits = self.linear(flat)

        out_first_digit = F.softmax(logits[:, :num_classes], dim=1)
        out_second_digit = F.softmax(logits[:, num_classes:], dim=1)

        return out_first_digit, out_second_digit