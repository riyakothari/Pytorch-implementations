import os
import torch
import argparse
from torchvision.datasets import MNIST
from torch.nn import Module
from torch.nn import Conv2d
from torch.nn import MaxPool2d
from torch.nn.init import kaiming_uniform_
from torch.nn.init import xavier_uniform_
from torch.nn import Softmax
from torch.nn import Linear
from torch.nn import ReLU
from torchvision.transforms import Compose
from torchvision.transforms import ToTensor
from torchvision.transforms import Normalize
from torch.utils.data import DataLoader
from torch.nn import CrossEntropyLoss
from torch.optim import SGD
from sklearn.metrics import accuracy_score
from numpy import argmax
from numpy import vstack

class CNN(Module):
    def __init__(self, n_channels):
        super(CNN, self).__init__()
        self.hidden1 = Conv2d(n_channels, 32, (3,3))
        kaiming_uniform_(self.hidden1.weight, nonlinearity='relu')
        self.act1 = ReLU()
        self.pool1 = MaxPool2d((2,2), stride=(2,2))

        self.hidden2 = Conv2d(32, 32, (3,3))
        kaiming_uniform_(self.hidden2.weight, nonlinearity='relu')
        self.act2 = ReLU()
        self.pool2 = MaxPool2d((2,2), stride=(2,2))

        self.hidden3 = Linear(5*5*32, 100)
        kaiming_uniform_(self.hidden3.weight, nonlinearity='relu')
        self.act3 = ReLU()

        self.hidden4 = Linear(100, 10)
        xavier_uniform_(self.hidden4.weight)
        self.act4 = Softmax(dim=1)

    def forward(self, X):
        X = self.hidden1(X)
        X = self.act1(X)
        X = self.pool1(X)

        X = self.hidden2(X)
        X = self.act2(X)
        X = self.pool2(X)

        X = X.view(-1, 4*4*50)
        X = self.hidden3(X)
        X = self.act3(X)

        X = self.hidden4(X)
        X = self.act4(X)

        return X

def prepare_data(path):
    trans = Compose([ToTensor(), Normalize((0.1307,), (0.3081,))])
    train = MNIST(path, train=True, download=True, transform=trans)
    test = MNIST(path, train=False, download=True, transform=trans)

    train_dl = DataLoader(train, batch_size=64, shuffle=True)
    test_dl = DataLoader(test, batch_size=1024, shuffle=False)

    return train_dl, test_dl

def train_model(train_dl, model):
    criterion = CrossEntropyLoss()
    optimizer = SGD(model.parameters(), lr=0.01, momentum=0.9)
    for epoch in range(10):
        for i, (inputs, targets) in enumerate(train_dl):
            optimizer.zero_grad()
            y_pred = model(inputs)
            loss = criterion(y_pred, targets)
            loss.backward()
            optimizer.step()

def evaluate_model(test_dl, model):
    preds, actuals = list(), list()
    for i, (inputs, targets) in enumerate(test_dl):
        y_pred = model(inputs)
        y_pred = y_pred.detach().numpy()
        y_pred = argmax(y_pred, axis=1)
        y_pred = y_pred.reshape((len(y_pred),1))
        preds.append(y_pred)
        actual = targets.numpy()
        actual = actual.reshape((len(actual),1))
        actuals.append(actual)
    preds, actuals = vstack(preds), vstack(actuals)
    return accuracy_score(preds, actuals)

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', type=str, default='~/.torch/datasets/mnist')
    opt = parser.parse_args()
    train_dl, test_dl = prepare_data(opt.path)
    model = CNN(1)
    train_model(train_dl, model)
    acc = evaluate_model(test_dl, model)
    print("Accuracy ", acc)
