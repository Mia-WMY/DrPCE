import os
from random import random

import torch.nn as nn
import pandas as pd
import numpy as np
import torch.optim as opt
import torch
from torch.utils.data import DataLoader, TensorDataset
from tools.physics import  get_deriv, get_deriv_with_level

# set train epoch
EPO = 500
# set learning-rate
LR = 1e-2
# set gpu/cpu
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# Set the default floating point dtype to double
torch.set_default_dtype(torch.double)
eps=1e-14
class MNNet(nn.Module):
    def __init__(self, in_dim, out_dim,bound):
        super(MNNet, self).__init__()
        self.net1= nn.Linear(in_dim, 20)
        self.tanh1= nn.Tanh()
        self.net2=  nn.Linear(20,20)
        self.tanh2= nn.Tanh()
        self.net3=  nn.Linear(20, out_dim)
        self.bound=bound
        torch.nn.init.xavier_uniform_(self.net1.weight, gain=1)
        torch.nn.init.xavier_uniform_(self.net2.weight, gain=1)
        torch.nn.init.xavier_uniform_(self.net3.weight, gain=1)

    def forward(self, x):
         x=self.net1(x)
         x=self.tanh1(x)
         x= self.net2(x)
         x=self.tanh2(x)
         x= self.net3(x)
         return x

    def predict(self,x):
        if isinstance(x, list):
            x = torch.Tensor(x)
        with torch.no_grad():
            vg = x[:, -2:-1]
            vd = x[:, -1:]
            param = x[:, :-2]
            tparam = one_param_process(param, self.bound).reshape(1, -1)
            trx = torch.cat((tparam, vg, vd), 1)
            y_pred_revert = self(trx)
            y_pred_revert = torch.pow(2, y_pred_revert)
            return y_pred_revert

    def fit(self,data,epoch,dataset,batch_size,num,level=True):#312
        lin_x, ytr = exp_data_tensor(data,self.bound,num)
        train_loader = DataLoader(TensorDataset(lin_x, ytr), batch_size)
        optimizer = opt.Adam(self.parameters(), lr=1e-3)
        nnm = sum(param.numel() for param in self.parameters())
        print('# generator parameters:',  nnm)
        loss_f_mse = nn.MSELoss(reduction='mean')
        trainAcc_txt = f"{dataset}_loss.txt"
        criterion = torch.nn.MSELoss()
        # if dataset == 'GAA_data':
        #     a = 1e-4
        #     b = 1e-6
        #     c = 1e-8
        # else:
        #     a = 1e-6
        #     b = 1e-8
        #     c = 1e-10
        for e in range(epoch):
            epoch_loss=0
            for batch, (X, y) in enumerate(train_loader):
                optimizer.zero_grad()
                X.requires_grad = True
                # print(X[:,:-2].shape)
                if(len(y)!=batch_size):
                    continue

                optimizer.zero_grad()
                deriv = get_deriv_with_level(X[:, -2:], y, 3)
                n = len(y)
                # params = X[:, :-2]  # param
                # vg = X[:, -2:-1]  # vg
                # vd = X[:, -1:]
                y_pred = self(X)
                y.requires_grad = True
                mainloss = loss_f_mse(y, y_pred) / n  # (y_i-y)^2/n
                # pgloss1 = torch.autograd.grad(y_pred, X,  grad_outputs=torch.ones(y_pred.size()),create_graph=True, retain_graph=True,only_inputs=True,
                #                              allow_unused=True)[0]
                # # torch.autograd.grad(outputs=y_pred, inputs=X,
                # #                     #                                     grad_outputs=torch.ones(y_pred.size()),retain_graph=True,
                # #                     #                             create_graph=True, only_inputs=True)[0]
                # #
                # pgloss2 = torch.autograd.grad(pgloss1, X, grad_outputs=torch.ones(pgloss1.size()), create_graph=True,
                #                     retain_graph=True, only_inputs=True,
                #                     allow_unused=True)[0]
                #
                # pgloss3 = torch.autograd.grad(pgloss2, X, grad_outputs=torch.ones(pgloss2.size()), create_graph=True,
                #                               retain_graph=True, only_inputs=True,
                #                               allow_unused=True)[0]
                # if (torch.tensor(deriv[1]).shape[0] != pgloss1[:, -2:].shape[0]):
                #     continue
                # derivloss1 = loss_f_mse(pgloss1[:, -2:], torch.tensor(deriv[1])) / n
                # derivloss2 = loss_f_mse(pgloss2[:, -2:], torch.tensor(deriv[2])) / n
                # derivloss3 = loss_f_mse(pgloss3[:, -2:], torch.tensor(deriv[3])) / n
                loss = mainloss
                loss.backward()
                optimizer.step()
            # epoch_loss+=criterion(y,y_pred)
            # print(f"loss", epoch_loss)
            # output = "%f" % (epoch_loss)
            # with open(trainAcc_txt, "a+") as f:
            #     f.write(output + ',')
            #     f.close
                # one_order_grad= torch.autograd.grad(outputs=y_pred, inputs=X,
                #                                     grad_outputs=torch.ones(y_pred.size()),retain_graph=True,
                #                             create_graph=True, only_inputs=True)[0]
                # two_order_grad=torch.autograd.grad(outputs=one_order_grad, inputs=X,
                #                             grad_outputs=torch.ones(one_order_grad.size()),retain_graph=True,
                #                             create_graph=True,only_inputs=True)[0]
                # three_order_grad = torch.autograd.grad(outputs=two_order_grad, inputs=X,
                #                             grad_outputs=torch.ones(two_order_grad.size()),retain_graph=True,
                #                             create_graph=True,only_inputs=True)[0]
                # try_order_grad=torch.autograd.grad(outputs=y, inputs=X,
                #                                     grad_outputs=torch.ones(y_pred.size()),retain_graph=True,
                #                             create_graph=True, only_inputs=True)[0]
                # print("1",one_order_grad[:,-2:].shape)
                # print("2",two_order_grad[:,-2:].shape)
                # print("3",three_order_grad[:,-2:].shape)
                # vloss = torch.autograd.grad(y_pred.sum(), X, create_graph=True, retain_graph=True, allow_unused=True)
                # # vdloss = torch.autograd.grad(y_pred.sum(),  vd, create_graph=True, retain_graph=True,
                #                              allow_unused=True)
                # vgloss=(item.detach().numpy() for item in vgloss)
                # vdloss=list(item.detach().numpy() for item in vdloss)
                # print("?",vgloss[0].shape)
                # print("??",vloss[0][:,-2:])
                # print("??",type(deriv))




# def exp_data_tensor(train_file,bound):
#     """ get x(paramsets,vg,vg) and y split from the dataset where vd=0/vg=0 is excluded
#         :param train_file: train_file dir
#         :return: lin_x:(paramsets,vg,vg); ytr, read corresponding y data.
#     """
#     vg = torch.ones(1)
#     vd = torch.ones(1)
#     trparams = torch.ones(1)
#     ytr = torch.ones(1)
#     lot_x = []
#     lin_x = torch.ones(1)
#     i = 0
#     params, index_list = get_params(train_file)
#     dir_path = os.path.dirname(train_file)
#     for param, index in zip(params, index_list):
#         param = params_process(param,bound)
#         x1, x2, y = get_index_info(dir_path, index)
#         x1 = x1.reshape(-1, 1)
#         x2 = x2.reshape(-1, 1)
#         y = y.reshape(-1, 1)
#         if i == 0:
#             trparams = param.repeat(x1.shape[0], 1)
#             vg = x1
#             vd = x2
#             ytr = y
#             lot_x.append(torch.cat((trparams, x1, x2), 1))
#             lin_x = torch.cat((trparams, x1, x2), 1)
#         else:
#             param = param.repeat(x1.shape[0], 1)
#             lot_x.append(torch.cat((param, x1, x2), 1))
#             trparams = torch.cat((trparams, param), 0)
#             vg = torch.cat((vg, x1), 0)
#             vd = torch.cat((vd, x2), 0)
#             ytr = torch.cat((ytr, y), 0)
#             lin_x = torch.cat((lin_x, torch.cat((param, x1, x2), 1)), 0)
#         i = i + 1
#     return lin_x, ytr

def exp_data_tensor(train_file,bound,num):
    """ get x(paramsets,vg,vg) and y split from the dataset where vd=0/vg=0 is excluded
        :param train_file: train_file dir
        :return: lin_x:(paramsets,vg,vg); ytr, read corresponding y data.
    """
    vg = torch.ones(1)
    vd = torch.ones(1)
    trparams = torch.ones(1)
    ytr = torch.ones(1)
    lot_x = []
    lin_x = torch.ones(1)
    i = 0
    params, index_list = get_params(train_file)
    dir_path = os.path.dirname(train_file)
    # index_list = random.sample(index_list, int(num))
    index_list =index_list[0:int(num)]
    print("index",len(index_list))
    for param, index in zip(params, index_list):
        param = params_process(param,bound)
        x1, x2, y = get_index_info(dir_path, index)

        x1 = x1.reshape(-1, 1)
        x2 = x2.reshape(-1, 1)
        y = y.reshape(-1, 1)
        # print("x",x1.shape)
        if i == 0:
            trparams = param.repeat(x1.shape[0], 1)
            vg = x1
            vd = x2
            ytr = y
            lot_x.append(torch.cat((trparams, x1, x2), 1))
            lin_x = torch.cat((trparams, x1, x2), 1)
        else:
            param = param.repeat(x1.shape[0], 1)
            lot_x.append(torch.cat((param, x1, x2), 1))
            trparams = torch.cat((trparams, param), 0)
            vg = torch.cat((vg, x1), 0)
            vd = torch.cat((vd, x2), 0)
            ytr = torch.cat((ytr, y), 0)
            lin_x = torch.cat((lin_x, torch.cat((param, x1, x2), 1)), 0)
        i = i + 1
    return lin_x, ytr

def unexp_data(train_file, split,bound):
    """ choose one to split from origin dataset if its value 0
        :param train_file: train_file dir
        :param split: 'vg' or 'vd' ,choose one to split if the value is 0.
        :return lin_x, read split train X-data;
        :return ytr, read corresponding y data.
    """
    vg = torch.randn(1)
    vd = torch.randn(1)
    trparams = torch.randn(1)
    ytr = torch.randn(1)
    params, index_list = get_params(train_file)
    i = 0
    for param, index in zip(params, index_list):  # 对vg\vd特殊数据也进行了参数处理
        param = params_process(param,bound)
        dir_path = os.path.dirname(train_file)
        x1, x2, y = get_index_split(dir_path, index, split)
        if i == 0:
            vg = x1
            vd = x2
            ytr = y
            trparams = param.repeat(x1.shape[0], 1)
        else:
            vg = torch.cat((vg, x1), 0)
            vd = torch.cat((vd, x2), 0)
            ytr = torch.cat((ytr, y), 0)
            param = param.repeat(x1.shape[0], 1)
            trparams = torch.cat((trparams, param), 0)
        i = i + 1
    lin_x = torch.cat((trparams, vg, vd), 1)
    return lin_x, ytr


def get_index_split(dir_path, index, split):
    """choose one to split from specific txt if its value 0
        :param dir_path
        :param index
        :param split paramter
        :return the split vg,vd,ids of specific txt
    """
    txt_pth = dir_path + f'/{index}.txt'
    df = pd.read_table(txt_pth, sep='\t')
    #  drop ids<0
    df = df.drop(df[df['ids'] < 0.0].index)
    # locate [split] == 0
    vd0 = df.loc[df[split] == 0]
    vd = vd0['vd']
    vg = vd0['vg']
    ids = vd0['ids']
    vd = np.array(vd).flatten()
    vg = np.array(vg).flatten()
    ids = np.array(ids).flatten()
    vg = torch.tensor(vg).reshape(-1, 1)
    vd = torch.tensor(vd).reshape(-1, 1)
    ids = torch.tensor(ids).reshape(-1, 1)
    ids = torch.log10(ids)
    return vg, vd, ids


def get_params(train_file):
    """get the params and txt-index from train_file
        :param train_file: {'GAA', 'planar'}
        :return params of the selected dataset
        :return index_list of the selected dataset
    """
    txt_pth = train_file
    df = pd.read_table(txt_pth, sep=',')
    index_list = list(df.values[:, -1])
    index_list = [int(item) for item in index_list]
    params = torch.Tensor(df.values[:, :-1])
    return params, index_list


def params_process(params,bound):
    """Converting params  > 1e2 or params < 1e-1
        :param params
        :return log10(params) if params  > 1e2 or params < 1e-1
    """
    params=params.numpy()
    up = np.zeros([len(bound['upperbound'].keys())])
    low = np.zeros([len(bound['lowerbound'].keys())])
    boundnum = 0
    for key in bound['upperbound'].keys():
        up[boundnum] = bound['upperbound'][key]
        low[boundnum] = bound['lowerbound'][key]
        boundnum += 1
    out = np.zeros(np.shape(params))
    out= 1 / (up - low) * params - low / (up - low)
    tparams = torch.tensor(out)
    return tparams

def one_param_process(params,bound):
    params = np.array(params)
    up = np.zeros([len(bound['upperbound'].keys())])
    low = np.zeros([len(bound['lowerbound'].keys())])
    boundnum = 0
    for key in bound['upperbound'].keys():
        up[boundnum] = bound['upperbound'][key]
        low[boundnum] = bound['lowerbound'][key]
        boundnum += 1
    out = np.zeros(np.shape(params))
    out = 1 / (up - low) * params - low / (up - low)
    tparams = torch.tensor(out)
    tparams.reshape(1, -1)
    return tparams


def get_index_info(dir_path, index):
    """Reading the vg,vd,ids of  specific txt when vg!=0 and vd!=0
            :param dir_path
            :param index
            :return vg,vd,log10(ids)
        """
    txt_pth = dir_path + f'/{index}.txt'
    f_iv = open(txt_pth)
    iv_data = f_iv.readlines()
    ids=[]
    vg=[]
    vd=[]
    for iv_line in iv_data[1:]:
        iv_line = iv_line.split()  # vg vd id
        for j in range(3):
            iv_line[j] = float(iv_line[j])
        if iv_line[-1] < 0:
            iv_line[-1] = 0.0
        if iv_line[-2] == 0.0:
            iv_line[-1] = 0.0
        if(iv_line[-1]!=0):
            vg.append(iv_line[-3])
            vd.append(iv_line[-2])
            ids.append(iv_line[-1])
        # print("ids,vd", iv_line[-1], iv_line[-2])
    #
    # df = pd.read_table(txt_pth, sep='\t')
    # #  drop ids<0
    # df = df.drop(df[df['ids'] < 0.0].index)
    # df.drop(df[df['vd'] == 0].index, inplace=True)
    # df.drop(df[df['vg'] == 0].index, inplace=True)
    # vg = df['vg']
    # vd = df['vd']
    # ids = df['ids']
    # vg = np.array(vg).flatten()
    # vd = np.array(vd).flatten()
    # ids = np.array(ids).flatten()
    vg = torch.tensor(vg)
    vd = torch.tensor(vd)
    ids = torch.tensor(ids)
    ids = torch.log2(ids)
    # print(ids.shape)
    return vg, vd, ids



def Dloss(pred,true,weight):
    loss = torch.empty(len(pred[0]))
    true=torch.tensor(true)
    # pred=pred.detach().numpy()
    # print("true", true.shape)
    # print("pred", pred.shape)
    true=true[0:pred.shape[0],:]
    for i in range(len(pred[0])):
        loss[i] = torch.mean((pred[:,i] - true[:,i])) * weight[i]
    losstotal = torch.sum(loss)
    return losstotal

def lossfn(pred,true,reduction='mean',weight=None,sampleweight=None):
    if reduction == 'none' and weight == None:
        losstotal = torch.mean((pred - true) / true )
        return losstotal

def readconfig(config_file):
    f_config = open(config_file, "r")
    config_data = f_config.readlines()
    flag = 0
    vbound = {}
    parameterbound = {}
    parameterset = ''
    for config_line in config_data:
        config_line = config_line.replace(',', ' ')
        config_line = config_line.split()
        if flag == 0:
            try:
                vbound[config_line[0]] = float(config_line[1])
            except:
                flag = 1
        if flag == 1:
            if config_line[0] == 'parameter':
                parameterset = config_line[1:]
            else:
                boundname = config_line[0]
                config_line = config_line[1:]
                parameterbound[boundname] = {}
                for parameter in parameterset:
                    parameterbound[boundname][parameter] = float(config_line[0])
                    config_line = config_line[1:]
    f_config.close()
    return vbound, parameterbound