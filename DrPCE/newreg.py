import torch.utils.data as data
import os
import torch.nn as nn
from orthnet import Legendre
import pandas as pd
import numpy as np
import torch
import math
from torch.autograd import Variable
from torch.optim.lr_scheduler import MultiStepLR
from tools.physics import get_deriv_with_level
import torch.nn.functional as F
from sklearn.linear_model import LinearRegression
from tools.read_data import plot_loss_curve
# set train epoch
# set learning-rate
# set gpu/cpu
l1_loss = nn.L1Loss(reduction='mean')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# Set the default floating point dtype to double
torch.set_default_dtype(torch.double)
class NewRegression(nn.Module):
    def __init__(self,bound,epoch,poly,batch):
        super(NewRegression, self).__init__()
        self.bound=bound
        self.epoch=epoch
        self.batch_size =batch
        self.level = True
        self.poly = poly

    def forward(self, x):
         print(type(x),x.shape)
         print(self.poly)
         L = Legendre(x, self.poly)
         lin_x = L.tensor
         # print("!",self.pce.predict(lin_x))# numpy
         # print("!!",self.model(x).item())#tensor
         y_pred = self.pce.predict(lin_x)+self.model(torch.cat((x,lin_x),dim=1)).item()
         # y_pred = self.pce.predict(lin_x) + self.model(lin_x).item()
         # print(y_pred)
         return y_pred

    def predict(self, x):
        # use cmodel to predict
        x = torch.Tensor(x)
        with torch.no_grad():
            y_pred = self(x)
            return y_pred

    def fit(self, xtr, ytr,dataset,dimm):
            true_x=xtr[:,:-4]
            L = Legendre(true_x, self.poly)
            Ltrue_x = L.tensor
            self.pce= LinearRegression()
            self.pce.fit(Ltrue_x,ytr)
            a = 1e-4
            b = 1e-8
            c = 1e8
            self.model = ResidualBlock(dim=dimm+int(Ltrue_x.shape[1]))
            criterion = torch.nn.MSELoss()
            Lr = 0.01
            optimizer = torch.optim.Adam(self.model.parameters(), lr=Lr,weight_decay=1e-8)
            # # GAA 30 80
            scheduler = MultiStepLR(optimizer, milestones=[30,100,300,500], gamma=0.3)
            # # scheduler = MultiStepLR(optimizer, milestones=[80,300], gamma=0.1)
            combine_x=xtr
            for e in range(self.epoch):

                true_x=combine_x[:,:7]
                true_x.requires_grad = True
                L = Legendre(true_x, self.poly)
                lin_x = L.tensor
                coef = torch.Tensor(self.pce.coef_)
                exp = torch.matmul(lin_x,coef.T) + torch.tensor(self.pce.intercept_)
                modelx=torch.cat((true_x,lin_x),dim=1)
                ypred=self.model(modelx)+exp
                pgloss1_pce = torch.autograd.grad(exp, true_x, grad_outputs=torch.ones(exp.size()), create_graph=True,
                                                  retain_graph=True, only_inputs=True,
                                                  allow_unused=True)[0]
                pgloss2_pce = torch.autograd.grad(pgloss1_pce, true_x, grad_outputs=torch.ones(pgloss1_pce.size()),
                                                  create_graph=True, retain_graph=True, only_inputs=True,
                                                  allow_unused=True)[0]

                pgloss1_res=torch.autograd.grad(self.model(modelx), true_x, grad_outputs=torch.ones(self.model(modelx).size()), create_graph=True,
                                              retain_graph=True, only_inputs=True,
                                              allow_unused=True)[0]
                pgloss2_res = \
                torch.autograd.grad(pgloss1_res, true_x, grad_outputs=torch.ones( pgloss1_res.size()),
                                    create_graph=True,
                                    retain_graph=True, only_inputs=True,
                                    allow_unused=True)[0]
                true_loss_vg=combine_x[:, 7:8] - pgloss1_pce[:,-2:-1]
                true_loss_vd = combine_x[:, 8:9] - pgloss1_pce[:, -1:]
                true_loss_vg2 =combine_x[:, 9:10] - pgloss2_pce[:, -2:-1]
                true_loss_vd2= combine_x[:, 10:11] - pgloss2_pce[:, -1:]
                xx=c*criterion(torch.pow(2,ypred),torch.pow(2,ytr))
                # l1 regular
                regular_loss=0
                for param in self.model.parameters():
                    regular_loss+=torch.sum(abs(param))
                loss =xx+criterion(ypred,ytr)+a*criterion(pgloss1_res[:, -2:-1], true_loss_vg)+a*criterion(pgloss1_res[:, -1:],  true_loss_vd)\
                       + b * criterion(pgloss2_res[:, -2:-1],  true_loss_vg2) + b * criterion(pgloss2_res[:, -1:], true_loss_vd2)
                loss.backward()
                optimizer.step()
                mainloss= loss
                rmse=criterion(ypred,ytr).item()
                scheduler.step()
                print('Epoch [{}/{}], Loss: {:.4f}, RMSE:{:.4f}'.format(e + 1, self.epoch,mainloss,rmse))



def exp_data_tensor(train_file,bound):
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
    for param, index in zip(params, index_list):
        param = params_process(param,bound)
        x1, x2, y = get_index_info(dir_path, index)
        x1 = x1.reshape(-1, 1)
        x2 = x2.reshape(-1, 1)
        y = y.reshape(-1, 1)
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
    """ single param converting  > 1e2 or params < 1e-1
        :param params
        :return log10(params) if params  > 1e2 or params < 1e-1
    """
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

    vg = torch.tensor(vg)
    vd = torch.tensor(vd)
    ids = torch.tensor(ids)
    return vg, vd, ids



def Dloss(pred,true,weight):
    loss = torch.empty(len(pred[0]))
    true=torch.tensor(true)
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
class MNNet(nn.Module):
    def __init__(self,dim):
        super(MNNet, self).__init__()
        c=torch.rand(dim,1)
        self.c = torch.nn.Parameter(c)

    def forward(self, x):
         x=torch.matmul(x, self.c)
         return x

class NNet(nn.Module):
    def __init__(self):
        super(NNet, self).__init__()
        self.net1 = nn.Linear(2, 20)
        self.tanh1 = nn.Tanh()
        self.net2 = nn.Linear(20, 1)
    def forward(self, x):
         x=self.net1(x)
         x =self.tanh1(x)
         x= self.net2(x)
         return x

class ResidualBlock(nn.Module):
    def __init__(self,dim):
        super(ResidualBlock, self).__init__()

        self.netblock=nn.Sequential(
            # small
            nn.Linear(dim,20),
            nn.ReLU(),
            nn.Linear(20, 20),
            nn.ReLU(),
            nn.Linear(20, 1),
        )

    def forward(self, x):
        out=self.netblock(x)
        return out

