import os
import torch
import pandas as pd
import argparse
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import random_split
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.nn import Module
from torch.nn import Linear
from torch.nn import ReLU
from torch.nn import Softmax
from torch.nn import MSELoss
from torch.optim import SGD
from torch.nn.init import xavier_uniform_
from sklearn.metrics import mean_squared_error
from numpy import vstack
from numpy import argmax
from numpy import sqrt
from torch import Tensor

class bostonDataset(Dataset):
    def __init__(self, path, header = None):
        boston_frame = pd.read_csv(path)
        self.X = boston_frame.values[:, :-1]
        self.y = boston_frame.values[:, -1]

        self.X = self.X.astype('float32')
        self.y = self.y.astype('float32')
        self.y = self.y.reshape((len(self.y),1))

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return [self.X[idx], self.y[idx]]

    def get_splits(self, ratio):
        test_size = round(ratio*len(self.X))
        train_size = len(self.X) - test_size
        return random_split(self, [train_size, test_size])

class MLP(Module):
    def __init__(self, n_inputs, n_classes):
        super(MLP, self).__init__()
        self.hidden1 = Linear(n_inputs, 10)
        xavier_uniform_(self.hidden1.weight)
        self.act1 = ReLU()

        self.hidden2 = Linear(10, 8)
        xavier_uniform_(self.hidden2.weight)
        self.act2 = ReLU()

        self.hidden3 = Linear(8,n_classes)
        xavier_uniform_(self.hidden3.weight)

    def forward(self, X):
        X = self.hidden1(X)
        X = self.act1(X)

        X = self.hidden2(X)
        X = self.act2(X)

        return self.hidden3(X)

def prepare_data(path, ratio):
    boston_frame = bostonDataset(path)
    train, test = boston_frame.get_splits(ratio)
    train_dl = DataLoader(train, batch_size=32, shuffle=True)
    test_dl = DataLoader(test, batch_size=1024, shuffle=False)
    return train_dl, test_dl

def train_model(train_dl, model):
    criterion = MSELoss()
    optimizer = SGD(model.parameters(), lr=0.01, momentum=0.9)
    for epoch in range(100):
        for i, (inputs, targets) in enumerate(train_dl):
            optimizer.zero_grad()
            y_pred = model(inputs)
            # print(y_pred.type())
            # print("taarfget", targets.type())
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
    return mean_squared_error(preds, actuals)

def predict(row, model):
    row = Tensor([row])
    y_pred = model(row)
    return y_pred.detach().numpy()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', type=str, default='bostonhousing.csv')
    parser.add_argument('--test_ratio', type=float, default=0.33)
    opt = parser.parse_args()
    train_dl, test_dl = prepare_data(opt.path, opt.test_ratio)
    model = MLP(13, 1)
    train_model(train_dl, model)
    mse = evaluate_model(test_dl, model)
    print("MSE ", mse, " RMSE ",sqrt(mse))
    row = [0.00632,18.00,2.310,0,0.5380,6.5750,65.20,4.0900,1,296.0,15.30,396.90,4.98]
    y_pred = predict(row, model)
    print("Predicted ", y_pred)
