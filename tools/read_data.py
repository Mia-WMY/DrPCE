import os
import random
from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
import torch
def add_noise(x, mean, std):
    noise = torch.randn_like(x) * std + mean
    return x + noise

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
def exp_data_tensor(train_file,bound,vlim,num):
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
    random.seed(10)
    # index_list = random.sample(index_list, int(num))
    index_list =index_list[0:int(num)]
    # print("index",len(index_list))
    for param, index in zip(params, index_list):
        param = params_process(param,bound)
        x1, x2, y = get_index_info(dir_path, index)
        x1=x1/vlim
        x2=x2/vlim
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
        # print("??",txt_pth)
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
    # ids = torch.log2(ids)
    # print(ids.shape)
    return vg, vd, ids
def one_param_process(params,bound):
    """ single param converting  > 1e2 or params < 1e-1
        :param params
        :return log10(params) if params  > 1e2 or params < 1e-1
    """
    # print(params.shape)
    params = np.array(params)
    up = np.zeros([len(bound['upperbound'].keys())])
    low = np.zeros([len(bound['lowerbound'].keys())])
    boundnum = 0
    for key in bound['upperbound'].keys():
        up[boundnum] = bound['upperbound'][key]
        low[boundnum] = bound['lowerbound'][key]
        boundnum += 1
    out = np.zeros(np.shape(params))
    # print("??",params.shape)
    out = 1 / (up - low) * params - low / (up - low)
    tparams = torch.tensor(out)
    tparams.reshape(1, -1)
    return tparams


def plot_loss_curve(loss_list,poly,train):
    print(loss_list)
    plt.plot(loss_list)
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.savefig(f'{poly}_{train}_loss.pdf', bbox_inches='tight')
    plt.show()
