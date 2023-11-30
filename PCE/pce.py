import numpy as np
import matplotlib.pyplot as plt
import re
from sklearn.metrics import mean_squared_error as MSE
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score as R2
from tools.physics import get_deriv_with_level
plt.style.use('seaborn')
import sys
import torch
torch.set_default_dtype(torch.float64)
from sklearn.preprocessing import MinMaxScaler
from tools.read_data import readconfig,exp_data_tensor,add_noise
from reg import NewRegression
from tools.get_deriv import planar_deriv
from tools.seed import setup_seed
from orthnet import Legendre
from sklearn.linear_model import LinearRegression
# 12
dlist = []
# main program
for arg in sys.argv:
    if 'data' in arg:
        data = re.split('=',arg)[1]
    if 'model' in arg:
        model_file = re.split('=',arg)[1]
    if 'poly' in arg:
        poly_degree = re.split('=', arg)[1]
    if 'epoch' in arg:
        epoch = re.split('=', arg)[1]
    if 'set' in arg:
        dataset = re.split('=', arg)[1]
    if 'train' in arg:
        train = re.split('=', arg)[1]
    if 'ds' in arg:
        seed = re.split('=', arg)[1]
setup_seed(seed)
config_file=fr'../data/{dataset}/config.txt'
config, parameterbound =readconfig(config_file)
data=fr'../data/{dataset}/parametertrain.txt'

poly_degree=int(poly_degree)
epoch=500
X_train, y_train = planar_deriv(data, parameterbound, batch_size=210, level=True, vlim=0.7, trainset=train)
bt =500
if dataset=='rectangle':
    dim=8
else :
    dim=7
modelx= NewRegression(parameterbound,epoch,poly_degree,batch=bt)
modelx.fit(X_train,y_train,dataset,dimm=dim)
torch.save(modelx, model_file)
print("taining finished")