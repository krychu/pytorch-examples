# Author: https://github.com/krychu
#
# Problem:   Multiclass image classification. Classify grayscale images
#            of digits into 10 classes.
# Dataset:   MNIST
# Solution:  Feed-forward network with a single hidden layer
#
# MNIST dataset consists of 70_000 grayscale images of handwritten 0-9 digits.
# Each sample in the dataset is a pair: image and the target digit. We access
# MNIST dataset through torchvision.

import torch
import torch.nn as nn
import torch.utils.data
import torchvision
import torchvision.transforms as transforms
import math

# Create two datasets, one for training (60_000 samples) and one for testing
# (10_000 samples).
#
# Each element in a dataset is a SAMPLE: a tuple of input and output.
#
# If we didn't use `transforms.ToTensor()` a dataset sample would be:
#
# ( PILImage, 5 )
#   |         |
#   |         +--> Output: digit (int)
#   |
#   +--> Input: image (pixel values are in the [0,1] range)
#
# But because we use ToTensor() PyTorch transforms PILImages into tensors:
#
# ( tensor(...), 5 )
#   |            |
#   |            +--> Digit is still an int
#   |
#   +--> NEW: PILImage became a tensor of shape (1, 28, 28)
#                                                |  |   |
#                                 One channel <--+  |   +--> 28 values per line
#                                                   v
#                                           28 lines per image
#
# MNIST images are 28x28 grayscale. Hence, they only have a single set of 28x28
# values. In ohter words, each image has a single channel. Consequently, the
# size of tensor representing MNIST image is (1, 28, 28). Contrast this with
# RGB images which have three channels and require *three* sets of 28x28
# values, one set per R, G, and B. The size of such image is (3, 28, 28).
def load_datasets():
    train_dataset = torchvision.datasets.MNIST(
        # MNIST data will be downloaded into ./data directory.
        root = "./data",
        train = True,
        # Originally, MNIST samples include PILImages that are not suitable for
        # PyTorch. We ask to transform them to tensors.
        transform = transforms.ToTensor(),
        download = True
    )

    test_dataset = torchvision.datasets.MNIST(
        root = "./data",
        train = False,
        transform = transforms.ToTensor(),
        # download = True # Not necessary, the entire dataset is downloaded above
    )

    return train_dataset, test_dataset

# DataLoader wraps an iterable over dataset. This is done to facilitate
# batching, shuffling etc. We create data loader for each of the two datasets:
#
# Each element in a data loader is a BATCH (contrast this with a dataset where
# each element is a sample):
#
#
#      +--> Inputs: (batch_size, 1, 28, 28)
#      |
# [ tensor(...), tensor(...) ] <-- samples are lists of two values
#                   |
#                   +--> Outputs: (batch_size), int64
#
# Output of a dataset element was an int. But output of a data loader element
# is not a list of ints. It's a tensor of int64. This is a transformation that
# data loader did for us.
def create_dataloaders(train_dataset, test_dataset, batch_size):
    train_dataloader = torch.utils.data.DataLoader(
        dataset = train_dataset,
        batch_size = batch_size,
        shuffle = True
    )

    test_dataloader = torch.utils.data.DataLoader(
        dataset = test_dataset,
        batch_size = batch_size,
        shuffle = False
    )

    return train_dataloader, test_dataloader

class Model(nn.Module):
    def __init__(self, input_size, hidden_size, class_cnt):
        super(Model, self).__init__()
        self.flatten = nn.Flatten()
        # Each single output scale each single input by a separate weight
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(hidden_size, class_cnt)

    # The computations here are tracked and create computational graph. Each
    # tensor "remembers" the operation and arguments (other tensors) that were
    # used to compute it. This allows PyTorch to back propagate Loss gradients
    # and compute gradient of the Loss with respect to weights that are "deep"
    # down in the network.
    def forward(self, x):
        out = self.flatten(x)
        out = self.linear1(out)
        out = self.relu(out)
        out = self.linear2(out) # logits
        # We don't apply Softmax here because in case of PyTorch it's done
        # implicitly by the CrossEntropyLoss
        return out

def train(cfg):
    for epoch_idx in range(cfg["epoch_cnt"]):
        batch_cnt = len(cfg["train_dataloader"])
        interval_loss = 0
        for batch_idx, (x, y) in enumerate(cfg["train_dataloader"]):
            x = x.to(cfg["device"])
            y = y.to(cfg["device"])

            y_pred = cfg["model"](x)

            batch_loss = cfg["criterion"](y_pred, y)

            interval_loss += batch_loss.item()

            cfg["optimizer"].zero_grad()
            batch_loss.backward()
            cfg["optimizer"].step()

            if (batch_idx+1) % cfg["log_interval"] == 0:
                interval_loss_per_batch = interval_loss / cfg["log_interval"]
                interval_loss = 0
                print(" epoch: {:2d}/{:d}, batch: {:5d}/{:5d}, loss: {:5.3f}, ppl: {:8.2f}".format(
                    epoch_idx+1,
                    cfg["epoch_cnt"],
                    batch_idx+1,
                    batch_cnt,
                    interval_loss_per_batch,
                    math.exp(interval_loss_per_batch)
                ))

        test_loss, accuracy = evaluate(cfg, cfg["test_dataloader"])

        print()
        print("end of epoch: {:2d}, test loss: {:5.3f}, test ppl: {:4.2f}, accuracy: {:3.2f}%".format(
            epoch_idx+1,
            test_loss,
            math.exp(test_loss),
            accuracy
        ))
        print()

def evaluate(cfg, dataloader):
    total_loss = 0
    batch_cnt = len(dataloader)
    accuracy = 0
    with torch.no_grad():
        correct_cnt = 0
        sample_cnt = 0
        for x, y in dataloader:
            x = x.reshape(-1, cfg["input_size"]).to(cfg["device"])
            y = y.to(cfg["device"])

            y_pred = cfg["model"](x)
            batch_loss = cfg["criterion"](y_pred, y).item()
            total_loss += batch_loss

            # values, indexes
            _, y_pred = torch.max(y_pred, 1)
            sample_cnt += y.shape[0]

            correct_cnt += (y == y_pred).sum().item()

        accuracy = 100.0 * (correct_cnt / sample_cnt)

    return total_loss / batch_cnt, accuracy

def create_config():
    train_dataset, test_dataset = load_datasets()

    cfg = {
        # Input of a neural network must match the input of a sample. In the case of
        # MNIST each input (image) is (1, 28, 28). We will set the size of the network
        # input to 784 (28*28). Both sizes are the same but are shaped differently. We
        # will make sure that our network "flattens" image into a flat 784-value
        # tensor.
        "input_size": 28*28,
        "hidden_size": 250, # Number of outputs in the hidden layer
        "class_cnt": 10, # Number of outputs in the network

        # Calculations are performed on a device (e.g., GPU or CPU). In PyTorch we move
        # tensors to a device before performing any operations.
        "device": torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        "epoch_cnt": 10,
        "lr": 0.001,
        # Batch is a group of samples. Doing things in batches is more performant
        # because multiple samples can be processed in a single matrix operation.
        # Estimating "true" gradient from multiple samples is also more accurate.
        "batch_size": 64,
        # CrossEntropyLoss() will apply LogSoftmax (and NLLLoss) and this is why we
        # don't include it in the forward pass
        "criterion": nn.CrossEntropyLoss(),

        "log_interval": 100,
    }

    train_dataloader, test_dataloader = create_dataloaders(train_dataset, test_dataset, cfg["batch_size"])
    cfg["train_dataloader"] = train_dataloader
    cfg["test_dataloader"] = test_dataloader

    cfg["model"] = Model(cfg["input_size"], cfg["hidden_size"], cfg["class_cnt"])
    # Model registers internally parameters of layers created in
    # NeuralNetwork.__init__(). Because of this we can request here all trainable
    # model parameters and pass them to the optimizer.
    cfg["optimizer"] = torch.optim.Adam(cfg["model"].parameters(), lr=cfg["lr"])

    return cfg

if __name__ == "__main__":
    cfg = create_config()
    train(cfg)
