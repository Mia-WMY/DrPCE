import numpy as np
import matplotlib.pyplot as plt
import re

from torch.utils.data import DataLoader, TensorDataset

from tools.physics import get_deriv_with_level
plt.style.use('seaborn')
import sys
import torch
torch.set_default_dtype(torch.float64)
from sklearn.preprocessing import MinMaxScaler
from tools.read_data import readconfig,exp_data_tensor
import torch.utils.data as data
def plaste_deriv(X_train,y_train,dataset,hgrad):
    lim=X_train.shape[0]
    i=0
    if dataset=='GAA_data':
        bt=210
    else:
        bt=312
    loader = data.DataLoader(
        dataset=data.TensorDataset(X_train,y_train),
        batch_size=bt,
        shuffle=False
    )
    for step, (batch_x, batch_y) in enumerate(loader):
        deriv = get_deriv_with_level(batch_x[:, -2:], batch_y, 3)
        print("deriv",deriv.shape)
        deriv1 = torch.Tensor(deriv[1])
        tdvg = deriv1[:, 0].unsqueeze(1)  # 一阶偏导 [210,1]
        tdvd = deriv1[:, 1].unsqueeze(1)  # 一阶偏导 [210,1]
        pgloss = torch.cat((tdvg, tdvd), dim=1)
        if hgrad == True:
            deriv2 = torch.Tensor(deriv[2])
            tdvg2 = deriv2[:, 0].unsqueeze(1)  # 一阶偏导 [210,1]
            tdvd2 = deriv2[:, 1].unsqueeze(1)  # 一阶偏导 [210,1]
            pgloss2 = torch.cat((tdvg2, tdvd2), dim=1)
            pgloss=torch.cat((pgloss, pgloss2),dim=1)
        # pgloss= torch.cat((tdvg, tdvd),dim=1)
        if step==0:
            New_train=torch.cat((batch_x, pgloss),dim=1)
        else:
            x=torch.cat((batch_x, pgloss),dim=1)
            New_train = torch.cat((New_train, x), dim=0)
    return New_train



def planar_deriv(data, bound, batch_size,level,vlim,trainset):
    lin_x, ytr = exp_data_tensor(data, bound,vlim,trainset)
    # print("?",lin_x[0:10,:])
    ytr=torch.log2(ytr)
    train_loader = DataLoader(TensorDataset(lin_x, ytr), batch_size)
    New_train = torch.zeros(312, 1)
    for batch, (X, y) in enumerate(train_loader):
        if (len(y) != batch_size):
            continue
        if (len(X) != batch_size):
            continue
        if level is True:
            deriv = get_deriv_with_level(X[:, -2:], y, 3)
            deriv1 = torch.Tensor(deriv[1])
            tdvg = deriv1[:, 0].unsqueeze(1)  # 一阶偏导 [210,1]
            tdvd = deriv1[:, 1].unsqueeze(1)  # 一阶偏导 [210,1]
            pgloss = torch.cat((tdvg, tdvd), dim=1)
            deriv2 = torch.Tensor(deriv[2])
            tdvg2 = deriv2[:, 0].unsqueeze(1)  # 一阶偏导 [210,1]
            tdvd2 = deriv2[:, 1].unsqueeze(1)  # 一阶偏导 [210,1]
            pgloss2 = torch.cat((tdvg2, tdvd2), dim=1)
            pgloss = torch.cat((pgloss, pgloss2), dim=1)
            if ( pgloss.shape[0]!= batch_size):
                continue
            b_train = torch.zeros(312, 1)
            # print(torch.equal(New_train, b_train))
            if torch.equal(New_train, b_train):
                New_train = torch.cat((X, pgloss), dim=1)
                New_y=y
            else:
                x = torch.cat((X, pgloss), dim=1)
                New_train = torch.cat((New_train, x), dim=0)
                New_y = torch.cat((New_y, y), dim=0)
            # print(New_train.shape)
    # yo=torch.pow(2,New_y)
    return New_train,New_y

def planar_deriv_shape(data, bound, batch_size,level,vlim,trainset,shape):
    lin_x, ytr = exp_data_tensor(data, bound,vlim,trainset)
    # print("!", lin_x.shape)
    if shape=='rectangle':# lg_nw,0,w_nw,h_nw,t_ox,N_sd,eps_ox,index
        zeros = torch.zeros_like(lin_x[:, :1])
        lin_x = torch.cat((lin_x[:, :1], zeros, lin_x[:, 1:]), dim=1)
        # print("?",lin_x.shape)
    else: # lg_nw,r_nw,0,0,t_ox,N_sd,eps_ox
        zeros = torch.zeros_like(lin_x[:, :1])
        lin_x = torch.cat((lin_x[:, :2], zeros, zeros, lin_x[:, 2:]), dim=1)
        # print("?", lin_x.shape)
    ytr=torch.log2(ytr)
    train_loader = DataLoader(TensorDataset(lin_x, ytr), batch_size)
    New_train = torch.zeros(312, 1)
    for batch, (X, y) in enumerate(train_loader):
        if (len(y) != batch_size):
            continue
        if (len(X) != batch_size):
            continue
        if level is True:
            deriv = get_deriv_with_level(X[:, -2:], y, 3)
            deriv1 = torch.Tensor(deriv[1])
            tdvg = deriv1[:, 0].unsqueeze(1)  # 一阶偏导 [210,1]
            tdvd = deriv1[:, 1].unsqueeze(1)  # 一阶偏导 [210,1]
            pgloss = torch.cat((tdvg, tdvd), dim=1)
            deriv2 = torch.Tensor(deriv[2])
            tdvg2 = deriv2[:, 0].unsqueeze(1)  # 一阶偏导 [210,1]
            tdvd2 = deriv2[:, 1].unsqueeze(1)  # 一阶偏导 [210,1]
            pgloss2 = torch.cat((tdvg2, tdvd2), dim=1)
            pgloss = torch.cat((pgloss, pgloss2), dim=1)
            if ( pgloss.shape[0]!= batch_size):
                continue
            b_train = torch.zeros(312, 1)
            # print(torch.equal(New_train, b_train))
            if torch.equal(New_train, b_train):
                New_train = torch.cat((X, pgloss), dim=1)
                New_y=y
            else:
                x = torch.cat((X, pgloss), dim=1)
                New_train = torch.cat((New_train, x), dim=0)
                New_y = torch.cat((New_y, y), dim=0)

    return New_train,New_y

def planar_deriv_shapeX(data, bound, batch_size,level,vlim,trainset,shape):
    lin_x, ytr = exp_data_tensor(data, bound,vlim,trainset)
    # print("!", lin_x.shape)
    if shape=='rectangle':# lg_nw,w_nw,h_nw,t_ox,N_sd,eps_ox,index
        w = lin_x[:, 1]
        h = lin_x[:, 2]
        area = w * h
        circ = 2* (w+h)
        area = area.unsqueeze(1)
        circ = circ.unsqueeze(1)
        lin_x = torch.cat((lin_x[:, :1],area,circ, lin_x[:, 3:]), dim=1)
    elif shape == 'triangle':
        r = lin_x[:, 1]
        area=torch.pi * r ** 2
        circ = 2 * torch.pi * r
        area = area.unsqueeze( 1)
        circ = circ.unsqueeze(1)
        lin_x = torch.cat((lin_x[:, :1], area,circ, lin_x[:, 2:]), dim=1)
    else: # lg_nw,r_nw,0,0,t_ox,N_sd,eps_ox
        r = lin_x[:, 1]
        circ = 6 * r
        area =  torch.sqrt(torch.tensor([3.0])) / 4 *  r ** 2
        # print("?", area.shape)
        # print("?", circ.shape)
        print("r,c", r, circ)
        area = area.unsqueeze(1)
        circ = circ.unsqueeze(1)
        lin_x = torch.cat((lin_x[:, :1], area,circ, lin_x[:, 2:]), dim=1)

    ytr=torch.log2(ytr)
    train_loader = DataLoader(TensorDataset(lin_x, ytr), batch_size)
    New_train = torch.zeros(312, 1)
    for batch, (X, y) in enumerate(train_loader):
        if (len(y) != batch_size):
            continue
        if (len(X) != batch_size):
            continue
        if level is True:
            deriv = get_deriv_with_level(X[:, -2:], y, 3)
            deriv1 = torch.Tensor(deriv[1])
            tdvg = deriv1[:, 0].unsqueeze(1)  # 一阶偏导 [210,1]
            tdvd = deriv1[:, 1].unsqueeze(1)  # 一阶偏导 [210,1]
            pgloss = torch.cat((tdvg, tdvd), dim=1)
            deriv2 = torch.Tensor(deriv[2])
            tdvg2 = deriv2[:, 0].unsqueeze(1)  # 一阶偏导 [210,1]
            tdvd2 = deriv2[:, 1].unsqueeze(1)  # 一阶偏导 [210,1]
            pgloss2 = torch.cat((tdvg2, tdvd2), dim=1)
            pgloss = torch.cat((pgloss, pgloss2), dim=1)
            if ( pgloss.shape[0]!= batch_size):
                continue
            b_train = torch.zeros(312, 1)
            # print(torch.equal(New_train, b_train))
            if torch.equal(New_train, b_train):
                New_train = torch.cat((X, pgloss), dim=1)
                New_y=y
            else:
                x = torch.cat((X, pgloss), dim=1)
                New_train = torch.cat((New_train, x), dim=0)
                New_y = torch.cat((New_y, y), dim=0)

    return New_train,New_y

def planar_deriv_shapeY(data, bound, batch_size,level,vlim,trainset,shape):
    lin_x, ytr = exp_data_tensor(data, bound,vlim,trainset)
    # print("!", lin_x.shape)
    if shape=='rectangle':# lg_nw,w_nw,h_nw,t_ox,N_sd,eps_ox,index
        w = lin_x[:, 1]
        h = lin_x[:, 2]
        # area = w * h
        # circ = 2* (w+h)
        # area = area.unsqueeze(1)
        # circ = circ.unsqueeze(1)
        lin_x = torch.cat((lin_x[:, :1], lin_x[:, 1:]), dim=1)
    elif shape == 'triangle':
        r = lin_x[:, 1]
        area=r
        area = area.unsqueeze(1)
        # circ = circ.unsqueeze(1)
        lin_x = torch.cat((lin_x[:, :1], area,lin_x[:, 1:]), dim=1)
    else: # lg_nw,r_nw,0,0,t_ox,N_sd,eps_ox
        r = lin_x[:, 1]
        area =r
        # area = torch.sqrt(torch.tensor([3.0])) / 4 *  r ** 2
        # # print("?", area.shape)
        # # print("?", circ.shape)
        # print("r,c", r, circ)
        area = area.unsqueeze(1)
        # circ = circ.unsqueeze(1)
        lin_x = torch.cat((lin_x[:, :1], area, lin_x[:, 1:]), dim=1)

    ytr=torch.log2(ytr)
    train_loader = DataLoader(TensorDataset(lin_x, ytr), batch_size)
    New_train = torch.zeros(312, 1)
    for batch, (X, y) in enumerate(train_loader):
        if (len(y) != batch_size):
            continue
        if (len(X) != batch_size):
            continue
        if level is True:
            deriv = get_deriv_with_level(X[:, -2:], y, 3)
            deriv1 = torch.Tensor(deriv[1])
            tdvg = deriv1[:, 0].unsqueeze(1)  # 一阶偏导 [210,1]
            tdvd = deriv1[:, 1].unsqueeze(1)  # 一阶偏导 [210,1]
            pgloss = torch.cat((tdvg, tdvd), dim=1)
            deriv2 = torch.Tensor(deriv[2])
            tdvg2 = deriv2[:, 0].unsqueeze(1)  # 一阶偏导 [210,1]
            tdvd2 = deriv2[:, 1].unsqueeze(1)  # 一阶偏导 [210,1]
            pgloss2 = torch.cat((tdvg2, tdvd2), dim=1)
            pgloss = torch.cat((pgloss, pgloss2), dim=1)
            if ( pgloss.shape[0]!= batch_size):
                continue
            b_train = torch.zeros(312, 1)
            # print(torch.equal(New_train, b_train))
            if torch.equal(New_train, b_train):
                New_train = torch.cat((X, pgloss), dim=1)
                New_y=y
            else:
                x = torch.cat((X, pgloss), dim=1)
                New_train = torch.cat((New_train, x), dim=0)
                New_y = torch.cat((New_y, y), dim=0)

    return New_train,New_y

def planar_deriv_shapeZ(data, bound, batch_size,level,vlim,trainset,shape):
    lin_x, ytr = exp_data_tensor(data, bound,vlim,trainset)
    # print("!", lin_x.shape)
    if shape=='rectangle':# lg_nw,w_nw,h_nw,t_ox,N_sd,eps_ox,index
        ones = torch.ones_like(lin_x[:, :1])
        # ones = ones.unsqueeze(1)
        lin_x = torch.cat((ones,lin_x[:, :1], lin_x[:, 1:]), dim=1)
    elif shape == 'triangle':
        r = lin_x[:, 1]
        area=r
        ones = torch.ones_like(lin_x[:, :1])
        twos=2*ones
        # twos = twos.unsqueeze(1)
        area = area.unsqueeze(1)
        lin_x = torch.cat((twos,lin_x[:, :1], area,lin_x[:, 1:]), dim=1)
    else: # lg_nw,r_nw,0,0,t_ox,N_sd,eps_ox
        r = lin_x[:, 1]
        area =r
        ones = torch.ones_like(lin_x[:, :1])
        threes = 3 * ones
        # print("threes", threes.shape)
        # print("lin_x", lin_x[:, :1].shape)
        # print("area", area.shape)
        area = area.unsqueeze(1)
        lin_x = torch.cat((threes,lin_x[:, :1], area, lin_x[:, 1:]), dim=1)
    print("lin_x",lin_x.shape)
    ytr=torch.log2(ytr)
    train_loader = DataLoader(TensorDataset(lin_x, ytr), batch_size)
    New_train = torch.zeros(312, 1)
    for batch, (X, y) in enumerate(train_loader):
        if (len(y) != batch_size):
            continue
        if (len(X) != batch_size):
            continue
        if level is True:
            deriv = get_deriv_with_level(X[:, -2:], y, 3)
            deriv1 = torch.Tensor(deriv[1])
            tdvg = deriv1[:, 0].unsqueeze(1)  # 一阶偏导 [210,1]
            tdvd = deriv1[:, 1].unsqueeze(1)  # 一阶偏导 [210,1]
            pgloss = torch.cat((tdvg, tdvd), dim=1)
            deriv2 = torch.Tensor(deriv[2])
            tdvg2 = deriv2[:, 0].unsqueeze(1)  # 一阶偏导 [210,1]
            tdvd2 = deriv2[:, 1].unsqueeze(1)  # 一阶偏导 [210,1]
            pgloss2 = torch.cat((tdvg2, tdvd2), dim=1)
            pgloss = torch.cat((pgloss, pgloss2), dim=1)
            if ( pgloss.shape[0]!= batch_size):
                continue
            b_train = torch.zeros(312, 1)
            # print(torch.equal(New_train, b_train))
            if torch.equal(New_train, b_train):
                New_train = torch.cat((X, pgloss), dim=1)
                New_y=y
            else:
                x = torch.cat((X, pgloss), dim=1)
                New_train = torch.cat((New_train, x), dim=0)
                New_y = torch.cat((New_y, y), dim=0)

    return New_train,New_y

