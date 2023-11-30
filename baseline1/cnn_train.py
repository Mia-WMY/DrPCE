import sys
import re
import os
import torch
from read_data_cnn import readconfig, MNNet
from tools.seed import setup_seed
## main program
for arg in sys.argv:
    if 'data' in arg:
        data = re.split('=',arg)[1]
    if 'model' in arg:
        model_file = re.split('=',arg)[1]
    if 'set' in arg:
        set = re.split('=', arg)[1]
    if 'num' in arg:
        num = re.split('=', arg)[1]
    if 'ds' in arg:
        seed = re.split('=', arg)[1]
setup_seed(seed)
config_file=fr'../data/{set}/config.txt'
config, parameterbound =readconfig(config_file)
data=fr'../data/{set}/parametertrain.txt'
if set=="rectangle":
    MNN = MNNet(8, 1, parameterbound)
    MNN.fit(data, 500, set, batch_size=210, num=num)
else :
    MNN = MNNet(7, 1, parameterbound)
    MNN.fit(data, 500, set,batch_size=210,num=num)
print("Training finished.")
model={}
model[0]=MNN
print(model_file)
torch.save(model, model_file)

