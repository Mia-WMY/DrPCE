import sys
import re
import os
#------------------------import--------------------#
import sys
import re
import os
import torch.nn as nn
import numpy as np
import torch
import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.ensemble import RandomForestRegressor
from tools.read_data import exp_data_tensor,readconfig,get_params,one_param_process
#------------------------import--------------------#
import sys
import re
import os
from orthnet import Legendre
import torch.nn as nn
import numpy as np
import torch
import pandas as pd
import numpy as np
import torch
from tools.seed import setup_seed
from torch.utils.data import DataLoader, TensorDataset
# main program
from decimal import Decimal,getcontext
getcontext().prec=100000000
# print("training finished")
for arg in sys.argv:
    if '=' not in arg:
        continue
    val = re.split('=', arg)[1]
    if 'tt' in arg:
        tt = val  # input_file
    if 'model' in arg:
        model_file = val  # model.pt
    if 'output' in arg:
        output = val  # output_file
    if 'dataset' in arg:
        dataset = val  # output_file
    if 'o' in arg:
        seed = val  # output_file
setup_seed(seed)
if dataset=='planar_data':
    vlim=1.2
else:
    vlim=0.7
config_file=f"../data/{dataset}/config.txt"
config, bound =readconfig(os.path.abspath(config_file))
model = torch.load(model_file)

with open(tt) as fdata:
    test_data = fdata.readlines()
skip_lines = test_data[:3]
skip_lines[-1] = skip_lines[-1].strip() + ',ids\n'
p_data = test_data[1].strip().split(',')
p_data = [float(p) for p in p_data]
v_data = [v_line.strip().split(',') for v_line in test_data[3:]]
for i in range(len(v_data)):
    v_data[i] = [float(v) for v in v_data[i]]
v_data = np.array(v_data)
v_mark = [vds < 0 for vds in v_data[:,1]]
testdata = [p_data + list(v_line) for v_line in v_data]
testdatap=[]
testdatan=[]
indexp=[]
indexn=[]
idsp=[]
idsn=[]
ids = np.zeros([len(testdata), 1])
for i in range(len(testdata)):
    tdata = torch.Tensor(testdata[i])
    tdata = torch.reshape(tdata, (1, -1))
    tdata = tdata.squeeze(-1)
    vg =  tdata[:, -2:-1]/vlim
    vd =  tdata[:, -1:]/vlim
    param =  tdata[:, :-2]
    tparam = one_param_process(param, bound).reshape(1, -1)
    trx = torch.cat((tparam, vg, vd), 1)
    x= model.predict(trx)
    if(x[0][0]>0):
        x[0][0]=-34
    ids[i, 0] =np.power(2,x[0][0])
ids = list(ids)
out_data = [list(v_line).copy() for v_line in v_data]
for i in range(len(out_data)):
    out_data[i] = out_data[i] + [ids[i]]

with open(output, 'w') as fout:
    for line in skip_lines:
        fout.write(line)
    for out_line in out_data:
        fout.write(str(out_line[0]) + ',' + str(out_line[1]) + ',' + str(out_line[2][0]) + '\n')
