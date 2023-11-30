#------------------------import--------------------#
# # set train epoch
# EPO = 1500
# # set learning-rate
# LR = 1e-2
# # set gpu/cpu
#------------------------import--------------------#
import sys
import re
import os
import numpy as np
import torch
from tools.read_data import readconfig
# main program
from tools.seed import setup_seed
for arg in sys.argv:
    if '=' not in arg:
        continue
    val = re.split('=', arg)[1]
    if 'tt' in arg:
        data = val  # input_file
    if 'model' in arg:
        model_file = val  # model.pt
    if 'output' in arg:
        output = val  # output_file
    if 'set' in arg:
        set = val  # output_file
    if 'ds' in arg:
        seed = val  # output_file

setup_seed(seed)
# print("set",set)
config_file=fr'../data/{set}/config.txt'
config, bound =readconfig(os.path.abspath(config_file))

model = torch.load(model_file)
with open(data) as fdata:
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
for i in range(len(testdata)):
    if testdata[i][-1]<=0:# vd<0
        testdatan.append(testdata[i].copy())
        testdatan[-1][-1]*=-1
        indexn.append(i)
    if testdata[i][-1]>=0:# vd>0
        testdatap.append(testdata[i].copy())
        indexp.append(i)
if len(indexp) > 0:
    for i in range(len(indexp)):
        tdata=torch.Tensor(testdata[i])
        tdata=torch.reshape(tdata,(1,-1))
        tdata=tdata.squeeze(-1)
        idsp.append(model[0].predict(tdata))
if len(indexn) > 0:
    for i in range(len(indexn)):
        tdata = torch.Tensor(testdata[i])
        tdata = torch.reshape(tdata, (1, -1))
        tdata = tdata.squeeze(-1)
        idsn.append(model[0].predict(tdata))
ids = np.zeros([len(testdata), 1])
ni = 0
pi = 0
for i in range(len(ids)):
    if indexp[pi] == i:
        ids[i, 0] = idsp[pi]
        pi += 1
    elif indexn[ni] == i:
        ids[i, 0] = idsn[ni]
        ni += 1
    else:
        print('combine np err')
ids = list(ids)
out_data = [list(v_line).copy() for v_line in v_data]
for i in range(len(out_data)):
    if v_mark[i]:
        out_data[i] = out_data[i] + [-ids[i]]
        out_data[i][1] = -out_data[i][1]
    else:
        out_data[i] = out_data[i] + [ids[i]]
with open(output, 'w') as fout:
    for line in skip_lines:
        fout.write(line)
    for out_line in out_data:
        fout.write(str(out_line[0]) + ',' + str(out_line[1]) + ',' + str(out_line[2][0]) + '\n')
