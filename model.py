import os
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
os.environ['TORCH_USE_CUDA_DSA'] = "1"

import pandas as pd
import numpy as np
import random
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.optim as optim
import math
from bsor.Bsor import make_bsor
from tqdm import tqdm
from sklearn.utils import shuffle, resample
import time
import matplotlib.pyplot as plt
import json

SEQ_LEN = 1024
SEQ_PER_USR = 100
TEST_USR_PER_CLS = 10

truth = pd.read_csv('bool-final.csv')
users = np.load('users-1024.npy')
names = np.load('names-1024.npy')

def split(users, names, attr):
    pos = []
    neg = []
    for item in list(names):
        value = truth.loc[truth['User'] == int(item)].iloc[0][attr]
        if (type(value) == bool or type(value) == np.bool_):
            idx = names.index(item)
            if (value): pos.append(users[idx])
            else: neg.append(users[idx])
    random.shuffle(pos)
    random.shuffle(neg)
    testX = np.vstack(pos[0:TEST_USR_PER_CLS] + neg[0:TEST_USR_PER_CLS])
    trainX = np.vstack(pos[TEST_USR_PER_CLS:] + neg[TEST_USR_PER_CLS:])
    testY = np.array([1] * TEST_USR_PER_CLS * SEQ_PER_USR + [0] * TEST_USR_PER_CLS * SEQ_PER_USR)
    trainY = np.array([1] * len(pos[TEST_USR_PER_CLS:]) * SEQ_PER_USR + [0] * len(neg[TEST_USR_PER_CLS:]) * SEQ_PER_USR)
    return trainX, trainY, testX, testY

loss_train=[]
loss_test=[]
acc_train=[]
acc_test=[]

def accuracy(output, target, threshold):
    diff = torch.abs(output - target)
    predicted = (diff <= threshold).float()
    acc = predicted.mean().item()
    return acc

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        self.dropout = nn.Dropout(p=0.25)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:,:x.size(1), :]
        return self.dropout(x)

class TAMER(nn.Module):
    def __init__(self,input_size,embedding_dim, hidden_size, num_layers, output_size, max_len=5000):
        super().__init__()
        self.projection = nn.Linear(input_size, embedding_dim)
        self.pe = PositionalEncoding(embedding_dim, max_len=max_len)
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=embedding_dim, nhead=4),
            num_layers=num_layers
        )
        self.ffn = nn.Sequential(
            nn.LayerNorm(embedding_dim),
            nn.Linear(embedding_dim, output_size),
            nn.Sigmoid()
        )

    def init_weights(self):
        init.normal_(self.projection.weight, mean=0, std=1)
        init.constant_(self.ffn[1].bias, 0)
        init.xavier_normal_(self.ffn[1].weight)

    def forward(self, x):
        x= self.projection(x)
        x = self.pe(x)
        x = self.transformer(x)
        x = torch.mean(x, dim=1)
        x = self.ffn(x)
        return x

def train(model, train_data, train_label, batch_size, epochs, lr, testX, testY):
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.BCELoss()
    tacc = []

    for epoch in range(epochs):
        total_loss = 0
        acc = 0
        for i in range(len(train_data)//batch_size):
            input=train_data[i*batch_size:(i+1)*batch_size]
            target=train_label[i*batch_size:(i+1)*batch_size]
            optimizer.zero_grad()
            input_tensor = torch.tensor(input, dtype=torch.float).cuda()
            target_tensor = torch.tensor(target, dtype=torch.float).cuda()
            output = model(input_tensor)
            loss = criterion(output.squeeze(), target_tensor.squeeze())
            predicted_labels = torch.round(output)
            acc += (predicted_labels.squeeze() == target_tensor.squeeze()).sum().item()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / (len(train_data)//batch_size)
        loss_train.append(avg_loss)
        acc_train.append(acc/(len(train_data)//batch_size*batch_size))
        print(f"Train  : Epoch {epoch+1}/{epochs} - Loss: {avg_loss:.4f} -acc {(acc/(len(train_data)//batch_size*batch_size)):.4f} ")
        tacc.append((acc/(len(train_data)//batch_size*batch_size)))

    test_loss, test_acc, user_acc=test(model, batch_size, testX, testY)

    print("Training complete!")
    print(f"Test  - Loss: {test_loss:.4f} -acc {test_acc} -uacc {user_acc}")

    return np.max(tacc), user_acc, test_acc

def test(model, batch_size, test_data, test_label):
    criterion = nn.BCELoss()
    total_loss = 0
    acc = 0
    preds = []
    batch_size = SEQ_PER_USR
    with torch.no_grad():
        for i in range(len(test_label)//batch_size):
            input=test_data[batch_size*i:batch_size*(i+1)]
            target=test_label[batch_size*i:batch_size*(i+1)]
            input_tensor = torch.tensor(input, dtype=torch.float).cuda()
            target_tensor = torch.tensor(target, dtype=torch.float).cuda()
            output = model(input_tensor)
            loss = criterion(output.squeeze(), target_tensor.squeeze())
            preds.append(output)
            predicted_labels = torch.round(output)
            acc += (predicted_labels.squeeze() == target_tensor.squeeze()).sum().item()
            total_loss += loss.item()
    preds = torch.concat(preds)
    preds = preds.detach().cpu().numpy()
    pred = []
    actual = []
    for i in range(TEST_USR_PER_CLS * 2):
        t = preds[i*SEQ_PER_USR:i*SEQ_PER_USR+SEQ_PER_USR]
        pred.append(np.mean(t))
        actual.append(test_label[i*SEQ_PER_USR])
    tacc = sum(np.round(pred) == actual) / 20

    return total_loss/(len(test_data)//batch_size),acc/(len(test_label)//batch_size*batch_size),tacc

def run_attr(
    attr,
    input_shape = (1024, 21),
    embedding_size = 24,
    hidden_size = 128,
    num_layers = 2,
    output_size = 1,
    lr = 0.00001,
    epochs = 100,
    batch_size = 32,
):
    trainX, trainY, testX, testY = split(users, list(names), attr)
    trainX0, trainY0 = resample(trainX[trainY == 0], trainY[trainY == 0], replace=True, n_samples=10000)
    trainX1, trainY1 = resample(trainX[trainY == 1], trainY[trainY == 1], replace=True, n_samples=10000)
    trainX = np.vstack((trainX0, trainX1))
    trainY = np.hstack((trainY0, trainY1))
    trainX, trainY = shuffle(trainX, trainY)
    trainX[np.isnan(trainX)] = 0
    testX[np.isnan(testX)] = 0
    model = TAMER(input_shape[-1], embedding_size, hidden_size, num_layers, output_size, input_shape[0]).cuda()
    start = time.time()
    sacc, uacc, tacc = train(model, trainX, trainY, batch_size, epochs, lr, testX, testY)
    end = time.time()
    took = end - start
    return attr, sacc, uacc, tacc, took

f = open('epochs.json')
epochs = json.load(f)

for col in truth.columns[2:]:
    print("Starting " + str(col))
    out = run_attr(col, lr = 0.00002, epochs=epochs[col])
    print("\t".join([str(r) for r in out]))
    with open("./results/" + str(col) + '.txt', 'w') as f:
        f.write("\t".join([str(r) for r in out]))
