import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec
import torch
from matplotlib.patches import FancyArrowPatch
from tools.read_data import exp_data_tensor,readconfig,get_params,one_param_process
#------------------------import--------------------#
import sys
import re
import os

font_format = {'family': "Times New Roman",  # 字体名称
               'color': 'blue',   # 字体颜色
               'weight': 'bold',  # 字体粗细
               'size': 16        # 字体大小
              }
fontdict=font_format
vlim=0.7
dataset="circle"
model = torch.load("../results/DrPCE/modelx3.pt")
config_file=f"../data/{dataset}/config.txt"
config, bound =readconfig(os.path.abspath(config_file))
param1=[10,2,1.5,5.0e+19,3.9]
param2=[12,2,1.5,5.0e+19,3.9]
param3=[14,2,1.5,5.0e+19,3.9]
param4=[16,2,1.5,5.0e+19,3.9]
param5=[18,2,1.5,5.0e+19,3.9]
param6=[20,2,1.5,5.0e+19,3.9]
param=torch.Tensor(param6)
vg= 0.5
vd= 0.5
avg =torch.tensor([vg / vlim])
avd =torch.tensor([vd / vlim])
tparam = one_param_process(param, bound).reshape(1, -1)
avg= torch.unsqueeze(avg, dim=1)
avd=torch.unsqueeze(avd, dim=1)
trx = torch.cat((tparam, avg, avd), 1)
x = model.predict(trx)
ids = np.power(2, x)
# print(ids)
lg=[10,12,14,16,18,20]
ids1=[6.44589741e-06,6.36955935e-06,6.29849282e-06,6.23386507e-06,6.18773453e-06,6.17121241e-06]
ids2=[8.97421473e-06,8.70550814e-06,8.55022083e-06,8.44665779e-06,8.35159424e-06,8.21227707e-06]
ids3=[4.06314505e-06,3.97219557e-06,3.90677362e-06,3.85435677e-06,3.80429559e-06,3.74667149e-06]
# true_ids=[4.0123e-6,3.9557e-6,3.9133e-6,3.8767e-6,3.8426e-6,3.8101e-6]
# ids=[4.06314505e-06,3.97219557e-06,3.90677362e-06,3.85435677e-06,3.80429559e-06,3.74667149e-06]
# plt.ylim(-11,-13)
# plt.plot(lg,true_ids,color='red')
ids_c1=[]
for x in ids1:
    ids_c1.append(x*1e6)
ids_c2=[]
for x in ids2:
    ids_c2.append(x*1e6)
ids_c3=[]
for x in ids3:
    ids_c3.append(x*1e6)
# true_ids3=[8.7972e-06,8.5952e-06,8.4627e-06,8.3644e-06,8.2827e-06,8.2093e-06]
# true_ids2=[4.0123e-6,3.9557e-6,3.9133e-6,3.8767e-6,3.8426e-6,3.8101e-6]
# true_ids1 = [5.9852e-6, 5.9523e-6, 5.9203e-6, 5.8887e-6, 5.8577e-6, 5.8272e-6]

# tids_c3=[]
# for x in true_ids3:
#     tids_c3.append(x*1e6)
# tids_c2=[]
# for x in true_ids2:
#     tids_c2.append(x*1e6)
# tids_c1=[]
# for x in true_ids1:
#     tids_c1.append(x*1e6)
# plt.plot(lg,ids_c,color='blue')
# # 创建数据
# x = np.linspace(0, 10, 100)
# y1 = np.sin(x)
# y2 = np.cos(x)

# 创建图形和子图布局
fig = plt.figure(figsize=(8, 6))
gs = GridSpec(1, 2, width_ratios=[1, 1])

# 创建第一个子图
ax1 = plt.subplot(gs[0])
ax1.plot(lg, ids_c1, color='blue',label="DrPCE",linewidth=2)
plt.annotate("2, 1, $1x10^{20}$, 7.5", xy=(lg[4], ids_c1[4]+0.1), xytext=(lg[1], ids_c1[1]+0.1),fontsize=14,
           )
plt.scatter(lg, ids_c1,marker='D', color='blue', s=30)
ax1.plot(lg, ids_c2, color='blue',label="DrPCE",linewidth=2)
plt.annotate("3, 1.5, $5x10^{19}$, 3.9", xy=(lg[4], ids_c2[4]+0.1), xytext=(lg[1], ids_c2[1]+0.1),fontsize=14,
             )
plt.scatter(lg, ids_c2,marker='D', color='blue', s=30)
ax1.plot(lg, ids_c3, color='blue',label="DrPCE",linewidth=2)
plt.annotate("2, 1.5, $5x10^{19}$, 3.9", xy=(lg[4], ids_c3[4]+0.1), xytext=(lg[1], ids_c3[1]+0.1),fontsize=14,
             )
# 创建一个空心箭头
# arrow = FancyArrowPatch( (lg[1], ids_c3[1]+0.3), (lg[4], ids_c3[4]+0.2),arrowstyle='-|>', mutation_scale=20, color='yellow', lw=12, fill=False,alpha=0.5)
#
# # 添加箭头到图中
# plt.gca().add_patch(arrow)
plt.scatter(lg, ids_c3,marker='D', color='blue', s=30)
# plt.scatter(lg,tids_c1,color='red')
# plt.scatter(lg,tids_c2,color='red')
# plt.scatter(lg,tids_c3,color='red')
# # 创建第二个子图
ax2 = plt.subplot(gs[1])
ss3=[0.06490672,0.07015175,0.07666006]
ss2=[0.09463154,0.1144671,0.13706631]
ss1=[0.1398239,0.17956237,0.22293721]
# true_ss=[0.1403034001,0.1818892855,0.2263764956]
lg=[0.5,1,1.5]
# true_ids=[4.0123e-6,3.9557e-6,3.9133e-6,3.8767e-6,3.8426e-6,3.8101e-6]
# ids=[4.06314505e-06,3.97219557e-06,3.90677362e-06,3.85435677e-06,3.80429559e-06,3.74667149e-06]
# # plt.ylim(-11,-13)
# plt.scatter(lg,true_ss,color='red')

ax2.plot(lg,ss1,color="blue",linewidth=2)
plt.scatter(lg, ss1,marker='D', color='blue', s=30)
plt.annotate("12, 4, $5x10^{19}$, 3.9", xy=(lg[1]+0.25, ss1[1]+0.015), xytext=(lg[0]+0.25, ss1[0]+0.015),fontsize=14,)
ax2.plot(lg,ss3,color="blue",linewidth=2)
plt.scatter(lg, ss3, marker='D', color='blue', s=30)
plt.annotate("10, 2, $5x10^{19}$, 3.9", xy=(lg[1]+0.25, ss3[1]+0.015), xytext=(lg[0]+0.25, ss3[0]+0.015),fontsize=14,)
ax2.plot(lg,ss2,color="blue",linewidth=2)
plt.scatter(lg, ss2,marker='D', color='blue', s=30)
plt.annotate("12, 3, $5x10^{19}$, 3.9", xy=(lg[1]+0.25, ss2[1]+0.036), xytext=(lg[0]+0.25, ss2[0]+0.036),fontsize=14,)
# ax2.plot(x, y2, 'g-')
#
# ax1.legend()
# ax2.legend()
ax1.tick_params(axis='x', labelsize=14)  # 增大x轴刻度的字体大小
ax1.tick_params(axis='y', labelsize=14)  # 增大y轴刻度的字体大小
ax2.tick_params(axis='x', labelsize=14)  # 增大x轴刻度的字体大小
ax2.tick_params(axis='y', labelsize=14)  # 增大y轴刻度的字体大小
ax1.set_xlabel("$l_g$ (nm)",fontsize=16)
ax1.set_ylabel("$I_{on}$ (uA)",fontsize=16)
ax2.set_xlabel('$t_{ox}$ (nm)',fontsize=16)
ax2.set_ylabel('$SS$ (V/dec)',fontsize=16)
desired_ticks = [0.5, 1, 1.5]  # 您想要显示的刻度值
ax2.set_xticks(desired_ticks)
ax1.text(0.41, -0.15, "(a)", transform=ax1.transAxes, fontsize=16, verticalalignment='center')
ax2.text(0.45, -0.15, "(b)", transform=ax2.transAxes, fontsize=16, verticalalignment='center')
# 调整子图之间的间距
plt.tight_layout()
plt.savefig("fig_sensitivity.pdf")
# 显示图形
plt.show()
