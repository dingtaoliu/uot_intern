import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from train_utils import batchify_data, run_epoch, train_model, Flatten
import utils_multiMNIST as U

from conv_nets import *
path_to_data_dir = '../Datasets/'
use_mini_dataset = True

batch_size = 32
nb_classes = 10
nb_epoch = 30
num_classes = 10
img_rows, img_cols = 42, 28 # input image dimensions


# Baseline CNN model
class CNN(nn.Module):

    def __init__(self, input_dimension):
        super(CNN, self).__init__()

        # input layer: preserve the original dimensions
        self.conv1 = nn.Conv2d(1, 32, 3, stride=1, padding=1)

        # conv layer 
        self.conv2 = nn.Conv2d(32, 64, 3, stride=1, padding=1)

        # conv layer
        self.conv3 = nn.Conv2d(64, 64, 3, stride=1, padding=1)

        self.flatten = Flatten()

        self.linear = nn.Linear(42 * 28 * 64, num_classes * 2)



    def forward(self, x):

        h = F.relu(self.conv1(x))
        h = F.max_pool2d(h, kernel_size=3, stride=1, padding=1)
        #print(h.shape)

        h = F.relu(self.conv2(h))
        h = F.max_pool2d(h, kernel_size=3, stride=1, padding=1)
        #print(h.shape)

        h = F.relu(self.conv3(h))
        h = F.max_pool2d(h, kernel_size=3, stride=1, padding=1)

        h_flat = self.flatten(h)

        logits = self.linear(h_flat)

        out_first_digit = F.softmax(logits[:, :num_classes], dim=1)
        out_second_digit = F.softmax(logits[:, num_classes:], dim=1)

        return out_first_digit, out_second_digit



def main():
    X_train, y_train, X_test, y_test = U.get_data(path_to_data_dir, use_mini_dataset)

    # Split into train and dev
    dev_split_index = int(9 * len(X_train) / 10)
    X_dev = X_train[dev_split_index:]
    y_dev = [y_train[0][dev_split_index:], y_train[1][dev_split_index:]]
    X_train = X_train[:dev_split_index]
    y_train = [y_train[0][:dev_split_index], y_train[1][:dev_split_index]]

    permutation = np.array([i for i in range(len(X_train))])
    np.random.shuffle(permutation)
    X_train = [X_train[i] for i in permutation]
    y_train = [[y_train[0][i] for i in permutation], [y_train[1][i] for i in permutation]]

    # Split dataset into batches
    train_batches = batchify_data(X_train, y_train, batch_size)
    dev_batches = batchify_data(X_dev, y_dev, batch_size)
    test_batches = batchify_data(X_test, y_test, batch_size)

    # Load model
    input_dimension = img_rows * img_cols
    model = SCNN3(input_dimension) # TODO add proper layers to CNN class above

    # Train
    train_model(train_batches, dev_batches, model, n_epochs=10, opt=torch.optim.Adam, lr=0.001)

    ## Evaluate the model on test data
    loss, acc = run_epoch(test_batches, model.eval(), None)
    print('Test loss1: {:.6f}  accuracy1: {:.6f}  loss2: {:.6f}   accuracy2: {:.6f}'.format(loss[0], acc[0], loss[1], acc[1]))

if __name__ == '__main__':
    # Specify seed for deterministic behavior, then shuffle. Do not change seed for official submissions to edx
    np.random.seed(12321)  # for reproducibility
    torch.manual_seed(12321)  # for reproducibility
    main()
