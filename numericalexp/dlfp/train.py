'''
Author: Jiawen Wei
Date: 2022-11-24 18:32:47
LastEditTime: 2022-12-13 15:46:53
Description: 
'''
import torch
import torch.nn as nn
from torch.autograd import grad
from torch.utils.data import DataLoader, Dataset
import numpy as np
import pandas as pd


class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(NeuralNet, self).__init__()
        self.layer = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.Sigmoid(),
            nn.Linear(hidden_size, hidden_size),
            nn.Sigmoid(),
            nn.Linear(hidden_size, hidden_size),
            nn.Sigmoid(),
            nn.Linear(hidden_size, output_size)
        )

    def forward(self, x):
        return self.layer(x)


class PdeDataset(Dataset):
    def __init__(self, data):
        self.data = torch.tensor(data, dtype=torch.float)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


def grad_net(y, x, order=1):
    weights = torch.ones_like(y)
    if order == 1:
        g = grad(outputs=y, inputs=x, grad_outputs=weights, create_graph=True)[0]
        return g
    elif order == 2:
        g_1 = grad(outputs=y, inputs=x, grad_outputs=weights, create_graph=True)[0]
        g_2 = grad(outputs=g_1, inputs=x, grad_outputs=weights, create_graph=True)[0]
        return g_2
    else:
        raise NotImplementedError


def get_d_and_b(interval_low=-2.2, interval_up=2.2, dt=0.01):
    set_d = np.arange(interval_low, interval_up, dt)
    set_b = np.array([interval_low, interval_up])
    return set_d, set_b


def gen_train_data(dt=0.01, num=1):
    data_d, data_b = [], []
    for _ in range(num):
        d, b = get_d_and_b(dt=dt)
        data_d.append(d)
        data_b.append(b)
    return np.expand_dims(np.array(data_d), axis=2), np.expand_dims(np.array(data_b), axis=2)


a = 0.3
b = 0.5
sigma = 0.5
dt = 0.01
epochs = 30000
batch_size = 1

f_func = lambda x: a * x - b * x**3
d_f_func = lambda x : a - 3 * b * x ** 2

train_data = gen_train_data(dt)
train_data_1 = PdeDataset(data=train_data[0])
train_data_2 = PdeDataset(data=train_data[1])
train_loader_1 = DataLoader(dataset=train_data_1, batch_size=batch_size, shuffle=True)
train_loader_2 = DataLoader(dataset=train_data_2, batch_size=batch_size, shuffle=True)

model = NeuralNet(1, 20, 1)
optimizer = torch.optim.Adam(model.parameters(), lr=3e-2)

loss_record = []

for epoch in epochs:
    for x, y in zip(train_loader_1, train_loader_2):
        var_x = x.requires_grad_()
        var_y = y.requires_grad_()
        out1 = model(var_x)
        out2 = model(var_y)
        f_value = f_func(var_x)

        f_deriviate_value = d_f_func(var_x)
        e1 = ((-(f_value * grad_net(out1, var_x) + f_deriviate_value * out1) +
                ((sigma**2) / 2) * grad_net(out1, var_x, order=2))**2).mean()


        e2 = (torch.abs(dt * out1.sum() - 1)) ** 2
        e3 = (out2 ** 2).mean()
        # e1, e2, e3 = e1.cuda(), e2.cuda(), e3.cuda()
        loss = e1 + e2 + e3
        loss_record.append(loss.item())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (epoch % 100) == 0:
            print(f'epoch: {epoch}, loss:{loss.item()}')

pd.DataFrame(loss_record).to_csv('loss5.csv', header='loss')
torch.save(model, f'model_save/modeltest5_grad.pth')