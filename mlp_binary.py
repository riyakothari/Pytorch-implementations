import os
import torch
import pandas as pd
import argparse
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
from torch.utils.data import random_split
from torch.utils.data import DataLoader
from torch import Tensor
from torch.nn import Module
from torch.nn import Linear
from torch.nn import ReLU
from torch.nn import Sigmoid
from torch.nn import BCELoss
from torch.optim import SGD
from torch.nn.init import kaiming_uniform_
from torch.nn.init import xavier_uniform_
from sklearn.metrics import accuracy_score
from numpy import vstack

class ionosphereDataset(Dataset):

    def __init__(self, path, header = None):
        self.ionosphere_data = pd.read_csv(path)
        # self.root_dir = root_dir
        self.X = self.ionosphere_data.values[:,:-1]
        self.y = self.ionosphere_data.values[:,-1]
        self.X = self.X.astype('float32')
        self.y = LabelEncoder().fit_transform(self.y)
        self.y = self.y.astype('float32')
        self.y = self.y.reshape((len(self.y), 1))

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return [self.X[idx], self.y[idx]]

    def get_splits(self, ratio=0.25):
        test_size = round(ratio * len(self.X))
        train_size = len(self.X) - test_size
        return random_split(self, [train_size, test_size])

def prepare_data(path):
    dataset = ionosphereDataset(path)
    train, test = dataset.get_splits()
    train_dl = DataLoader(train, batch_size=32, shuffle=True)
    test_dl = DataLoader(test, batch_size=1024, shuffle=False)
    return train_dl, test_dl

class MLP(Module):
    def __init__(self, n_inputs):
        super(MLP, self).__init__()
        self.hidden1 = Linear(n_inputs, 10)
        kaiming_uniform_(self.hidden1.weight, nonlinearity='relu')
        self.act1 = ReLU()

        self.hidden2 = Linear(10, 8)
        kaiming_uniform_(self.hidden2.weight, nonlinearity='relu')
        self.act2 = ReLU()

        self.hidden3 = Linear(8,1)
        xavier_uniform_(self.hidden3.weight)
        self.act3 = Sigmoid()

    def forward(self, X):
        X = self.hidden1(X)
        X = self.act1(X)

        X = self.hidden2(X)
        X = self.act2(X)

        X = self.hidden3(X)
        return self.act3(X)


def train_model(train_dl, model):
    criterion = BCELoss()
    optimizer = SGD(model.parameters(), lr=0.01, momentum=0.9)
    for epoch in range(100):
        for i, (inputs, target) in enumerate(train_dl):
            optimizer.zero_grad()
            y_pred = model(inputs)
            loss = criterion(y_pred, target)
            loss.backward()
            optimizer.step()

def evaluate_model(test_dl, model):
    preds, actuals = list(), list()
    for i, (inputs, target) in enumerate(test_dl):
        y_pred = model(inputs).detach().numpy()
        actual = target.numpy()
        actual = actual.reshape((len(actual), 1))
        y_pred = y_pred.round()
        preds.append(y_pred)
        actuals.append(actual)
    preds, actuals = vstack(preds), vstack(actuals)
    return accuracy_score(actuals, preds)

def predict(row, model):
    row = Tensor([row])
    y_pred = model(row)
    return y_pred.detach().numpy()


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', type=str, default='ionosphere.csv')
    opt = parser.parse_args()
    train_dl, test_dl = prepare_data(opt.path)
    model = MLP(34)
    train_model(train_dl, model)
    acc = evaluate_model(test_dl, model)
    print("Accuracy ", acc)
    row = [1,0,0.99539,-0.05889,0.85243,0.02306,0.83398,-0.37708,1,0.03760,0.85243,-0.17755,0.59755,-0.44945,\
    0.60536,-0.38223,0.84356,-0.38542,0.58212,-0.32192,0.56971,-0.29674,0.36946,-0.47357,0.56811,-0.51171,0.41078,\
    -0.46168,0.21266,-0.34090,0.42267,-0.54487,0.18641,-0.45300]
    y_pred = predict(row, model)
    print("Predicted ", y_pred)
